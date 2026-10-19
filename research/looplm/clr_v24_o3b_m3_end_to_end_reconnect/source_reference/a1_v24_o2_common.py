import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from a1_v22_storage_common import DEFAULT_CACHE, get_device, seed_all
from a1_v23_r2_common import load_checkpoint as load_r2_checkpoint
from clr_v24_o2 import AllSlotsVisibleBaseline, RetrievalController

ROOT = Path(r"D:\training")
DEFAULT_OUT = r"D:\training\checkpoints\clr_v24_o2_retrieval"
DEFAULT_R2 = ROOT / "checkpoints" / "clr_v23_oracle_r2" / "best_model.pt"
CAPACITIES = (1, 2, 4)


class O2Cache:
    def __init__(self, root=DEFAULT_CACHE):
        self.root = Path(root); self.data = {}
        self.meta = {}
        for split in ("train", "val", "testA", "testB"):
            p = self.root / f"{split}_data.npz"
            z = np.load(p, allow_pickle=False)
            self.data[split] = {k: z[k] for k in z.files}
            mp = self.root / f"{split}_meta.jsonl"
            if mp.exists():
                with mp.open("r", encoding="utf-8") as f:
                    self.meta[split] = [json.loads(line) for line in f if line.strip()]
        qh = self.root / "query_person_H.npz"
        qp = self.root / "query_persons.json"
        self.person_query_h = None
        self.person_to_index = {}
        if qh.exists() and qp.exists():
            self.person_query_h = np.load(qh, allow_pickle=False)["H"].astype(np.float32)
            persons = json.loads(qp.read_text(encoding="utf-8"))
            self.person_to_index = {str(name): i for i, name in enumerate(persons)}

    def rows_for_capacity(self, split, cap):
        return np.flatnonzero(self.data[split]["N"] == cap).astype(np.int64)

    def batch(self, split, rows, cap, device):
        d = self.data[split]; rows = np.asarray(rows, dtype=np.int64)
        off = d["fact_offsets"]
        facts = np.stack([d["fact_H"][int(off[r]):int(off[r + 1])] for r in rows], axis=0)
        labels = np.stack([d["fact_Y"][int(off[r]):int(off[r + 1])] for r in rows], axis=0).astype(np.int64)
        if facts.shape[1] != cap:
            raise ValueError(f"capacity mismatch: {facts.shape} vs {cap}")
        query = d["query_H"][rows].astype(np.float32)
        target_slot = d["query_idx"][rows].astype(np.int64)
        target_y = d["Y"][rows].astype(np.int64)
        if not np.array_equal(target_y, labels[np.arange(len(rows)), target_slot]):
            raise AssertionError("query target does not match fact label at query_idx")
        return (torch.from_numpy(facts).to(device), torch.from_numpy(labels).to(device),
                torch.from_numpy(query).to(device), torch.from_numpy(target_slot).to(device),
                torch.from_numpy(target_y).to(device))

    def person_query_batch(self, split, rows, positions, device):
        """Return query hidden for the person at each episode/position."""
        if self.person_query_h is None or split not in self.meta:
            raise FileNotFoundError("query_person_H.npz or split metadata is unavailable")
        rows = np.asarray(rows, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.int64)
        ids = []
        for r, p in zip(rows, positions):
            person = self.meta[split][int(r)]["persons"][int(p)]
            if person not in self.person_to_index:
                raise KeyError(f"person missing from query_persons.json: {person}")
            ids.append(self.person_to_index[person])
        return torch.from_numpy(self.person_query_h[np.asarray(ids, dtype=np.int64)]).to(device)


def rows_for(cache, split, cap):
    return cache.rows_for_capacity(split, cap)


def freeze_storage(core, heads):
    core.eval(); heads.eval()
    for p in core.parameters(): p.requires_grad_(False)
    for p in heads.parameters(): p.requires_grad_(False)


@torch.no_grad()
def encode_batch(core, cache, split, rows, cap, device):
    facts, labels, query, target_slot, target_y = cache.batch(split, rows, cap, device)
    state, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
    return state, labels, query, target_slot, target_y


def slot_logits_from_heads(heads, state):
    return torch.stack([heads[i](state[:, i, :]).float() for i in range(4)], dim=1)


def weighted_answer_logits(heads, state, weights):
    return (weights[:, :, None] * slot_logits_from_heads(heads, state)).sum(1)


def hard_weights(scores):
    idx = scores.argmax(1)
    one = F.one_hot(idx, num_classes=scores.shape[1]).float()
    soft = scores.softmax(1)
    return one.detach() - soft.detach() + soft, idx


def direct_selected_logits(heads, state, selected):
    all_logits = slot_logits_from_heads(heads, state)
    return all_logits[torch.arange(len(selected), device=selected.device), selected]


@torch.no_grad()
def evaluate_controller(core, heads, controller, cache, split, cap, mode="hard", rows=None, batch_size=128,
                        intervention=None, rng=None):
    core.eval(); heads.eval(); controller.eval()
    device = next(core.parameters()).device
    rows = rows_for(cache, split, cap) if rows is None else np.asarray(rows, dtype=np.int64)
    total = good_ret = good_ans = good_oracle = total_cond_good = total_cond_bad = cond_good = cond_bad = 0
    entropy_sum = 0.0
    per_position = {p: {"total": 0, "retrieval_correct": 0, "hard_correct": 0, "soft_correct": 0} for p in range(cap)}
    for i in range(0, len(rows), batch_size):
        rb = rows[i:i + batch_size]
        state, labels, query, target_slot, target_y = encode_batch(core, cache, split, rb, cap, device)
        if intervention is not None:
            state, query, target_slot = intervention(state, query, target_slot, cap)
        scores = controller(query, state, cap)
        probs = scores.softmax(1)
        selected = scores.argmax(1)
        if rng is None:
            random_selected = torch.remainder(torch.arange(len(selected), device=device) + 1, cap)
        else:
            random_selected = torch.from_numpy(rng.integers(0, cap, size=len(selected))).to(device)
        hard_logits = direct_selected_logits(heads, state, selected)
        soft_logits = weighted_answer_logits(heads, state, probs)
        oracle_logits = direct_selected_logits(heads, state, target_slot)
        random_logits = direct_selected_logits(heads, state, random_selected)
        hard_pred = hard_logits.argmax(1); soft_pred = soft_logits.argmax(1)
        oracle_pred = oracle_logits.argmax(1); random_pred = random_logits.argmax(1)
        ret_ok = selected == target_slot
        hard_ok = hard_pred == target_y; soft_ok = soft_pred == target_y
        for p in range(cap):
            m = target_slot == p
            if m.any():
                rec = per_position[p]; rec["total"] += int(m.sum()); rec["retrieval_correct"] += int(ret_ok[m].sum())
                rec["hard_correct"] += int(hard_ok[m].sum()); rec["soft_correct"] += int(soft_ok[m].sum())
        total += len(target_y); good_ret += int(ret_ok.sum()); good_ans += int(hard_ok.sum()); good_oracle += int((oracle_pred == target_y).sum())
        total_cond_good += int(ret_ok.sum()); total_cond_bad += int((~ret_ok).sum())
        cond_good += int(hard_ok[ret_ok].sum()); cond_bad += int(hard_ok[~ret_ok].sum())
        entropy_sum += float((-(probs * (probs.clamp_min(1e-8).log())).sum(1)).sum().item())
        result_random = locals().get("random_good", 0) + int((random_pred == target_y).sum())
        random_good = result_random
    out = {"retrieval_top1_acc": good_ret / max(total, 1), "hard_answer_acc": good_ans / max(total, 1),
           "soft_answer_acc": float("nan"), "oracle_answer_acc": good_oracle / max(total, 1),
           "random_answer_acc": random_good / max(total, 1), "total": total,
           "P_answer_correct_given_retrieval_correct": cond_good / max(total_cond_good, 1),
           "P_answer_correct_given_retrieval_wrong": cond_bad / max(total_cond_bad, 1),
           "routing_entropy": entropy_sum / max(total, 1), "by_target_position": per_position}
    # Recompute soft accuracy in a separate light pass to keep the counters clear.
    soft_good = 0
    for i in range(0, len(rows), batch_size):
        rb = rows[i:i + batch_size]
        state, _, query, _, target_y = encode_batch(core, cache, split, rb, cap, device)
        if intervention is not None:
            state, query, _ = intervention(state, query, torch.zeros(len(rb), dtype=torch.long, device=device), cap)
        probs = controller(query, state, cap).softmax(1)
        soft_good += int((weighted_answer_logits(heads, state, probs).argmax(1) == target_y).sum())
    out["soft_answer_acc"] = soft_good / max(total, 1)
    return out


def load_storage(device, r2_path=DEFAULT_R2):
    core, heads, ck = load_r2_checkpoint(r2_path, device, n_heads=4)
    freeze_storage(core, heads)
    return core, heads, ck


def save_controller(path, controller, optimizer=None, step=0, mode="hard", lam=1.0, metrics=None, extra=None):
    d = {"controller": controller.state_dict(), "step": int(step), "mode": mode, "lambda": float(lam),
         "metrics": metrics or {}, "config": {"query_dim": 2048, "slot_dim": 64, "proj_dim": 64,
                                                 "storage": "frozen CLR-v2.3-OA-r2"}}
    if optimizer is not None: d["optimizer"] = optimizer.state_dict()
    if extra: d.update(extra)
    torch.save(d, path)


def json_dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json), encoding="utf-8")


def _json(x):
    if isinstance(x, (np.integer, np.floating)): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)
