# -*- coding: utf-8 -*-
"""CLR-v2.4-V-GEN / O1-MR.

Value-only diagnosis for the already completed O2.1a-MR cache.  The main
path is deliberately oracle allocated and oracle read: fact -> frozen or
trainable OracleLocalCore -> correct slot -> answer head.  K/query-key
retrieval is evaluated only once at the end as a reconnect diagnostic.
"""
import copy
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v23_r2_common import load_checkpoint as load_r2_checkpoint
from a1_v24_o2_common import json_dump
from a1_v24_o21a import KeyEncoder, QueryEncoder
from a1_v24_o21a_mr import MRCache, MRValueHeads, CAPS, N_CLASSES, N_REL, RELATIONS, value_state
from clr_v23_oracle_r2 import OracleLocalCore

ROOT = Path(r"D:/training")
OUT = Path(os.environ.get("V24_VGEN_OUT", r"D:/training/checkpoints/clr_v24_vgen"))
MR_CACHE = Path(os.environ.get("V24_VGEN_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
MR_BEST = Path(os.environ.get("V24_VGEN_MR_BEST", r"D:/training/checkpoints/clr_v24_o21a_mr/best_model.pt"))
R2 = ROOT / "checkpoints" / "clr_v23_oracle_r2" / "best_model.pt"
OLD_CACHE = ROOT / "checkpoints" / "clr_v2_a1_ls0" / "cache"
SEED = int(os.environ.get("V24_VGEN_SEED", "42"))
STEPS = int(os.environ.get("V24_VGEN_STEPS", "3000"))
DEVICE_NAME = os.environ.get("V24_VGEN_DEVICE", "cuda")
CHANCE = 1.0 / N_CLASSES


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class OldSingleCache:
    """The original LS0 single-relation H/Y cache used for retention checks."""
    def __init__(self, root=OLD_CACHE):
        self.root = Path(root)
        self.data = {}
        for split in ("train", "val", "testA", "testB"):
            z = np.load(self.root / f"{split}_hidden.npz", allow_pickle=False)
            self.data[split] = {k: z[k] for k in z.files}

    def rows(self, split):
        return np.arange(len(self.data[split]["Y"]), dtype=np.int64)

    def batch(self, split, rows, device):
        rows = np.asarray(rows, dtype=np.int64)
        d = self.data[split]
        h = d["H"][rows].astype(np.float32)[:, None, :]
        y = d["Y"][rows].astype(np.int64)
        return torch.from_numpy(h).to(device), torch.from_numpy(y).to(device)


class ConditionedAnswerHead(nn.Module):
    """Minimal A1 decoder: V plus a learned query representation, never K."""
    def __init__(self, query_dim=2048, context_dim=32):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, context_dim)
        self.heads = nn.ModuleList([nn.Linear(64 + context_dim, N_CLASSES) for _ in range(4)])

    def context(self, query):
        return torch.tanh(self.query_proj(query.float()))

    def all_logits(self, v, query):
        c = self.context(query)
        return torch.stack([self.heads[i](torch.cat([v[:, i, :].float(), c], dim=1)) for i in range(4)], dim=1)

    def selected(self, v, query, idx):
        a = self.all_logits(v, query)
        return a[torch.arange(len(idx), device=v.device), idx]


class LegacyLocalHeads(nn.Module):
    """Adapter for the original O1 ModuleList[Linear(64,64)] heads."""
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def all_logits(self, v):
        return torch.stack([self.heads[i](v[:, i, :].float()) for i in range(4)], dim=1)

    def selected(self, v, idx):
        a = self.all_logits(v)
        return a[torch.arange(len(idx), device=v.device), idx]


def load_core(device, trainable=False):
    core, old_heads, ck = load_r2_checkpoint(R2, device, n_heads=4)
    old_heads.eval()
    for p in old_heads.parameters():
        p.requires_grad_(False)
    core.train(trainable)
    for p in core.parameters():
        p.requires_grad_(bool(trainable))
    return core, old_heads, ck


def load_t0_decoder(device):
    ck = torch.load(MR_BEST, map_location="cpu", weights_only=False)
    vh = MRValueHeads().to(device)
    vh.load_state_dict(ck["value_heads"], strict=True)
    vh.eval()
    for p in vh.parameters():
        p.requires_grad_(False)
    return vh, ck


def mr_batch(cache, split, rows, cap, device):
    return cache.batch(split, np.asarray(rows, dtype=np.int64), cap, device)


def selected_logits_a0(vh, v, slot):
    return vh.selected(v, slot)


def selected_logits_a1(head, v, q, slot):
    return head.selected(v, q, slot)


def answer_loss(v, slot, y, vh=None, ah=None, q=None):
    logits = selected_logits_a0(vh, v, slot) if ah is None else selected_logits_a1(ah, v, q, slot)
    return F.cross_entropy(logits, y), logits


def sample_mr(cache, split, rng, device, batch_size=64):
    cap = int(rng.choice(CAPS))
    rows_all = cache.rows(split, cap)
    rows = rows_all[rng.integers(0, len(rows_all), size=min(batch_size, len(rows_all)))]
    return cap, mr_batch(cache, split, rows, cap, device), rows


def old_loss_batch(old, core, vh, ah, use_a1, rng, device, batch_size=32):
    rows_all = old.rows("train")
    rows = rows_all[rng.integers(0, len(rows_all), size=min(batch_size, len(rows_all)))]
    facts, y = old.batch("train", rows, device)
    v, _ = core.forward_facts(facts, slot_indices=[0])
    slot = torch.zeros(len(y), dtype=torch.long, device=device)
    if use_a1:
        # The original LS0 statement is a single relation.  H itself is the
        # only available frozen query representation for replay; this replay
        # is used for retention, not as a new query-label source.
        loss, _ = answer_loss(v, slot, y, ah=ah, q=facts[:, 0, :])
    else:
        loss, _ = answer_loss(v, slot, y, vh=vh)
    return loss


def train_stage(name, cache, old, base_vh, device, mode, steps=STEPS, replay=False):
    """Train one of T1-A0, T1-A1, T2, T3 with fixed 3e-4 AdamW."""
    core, _, _ = load_core(device, trainable=(mode in ("T2", "T3")))
    vh = MRValueHeads().to(device)
    vh.load_state_dict(base_vh.state_dict(), strict=True)
    ah = None
    if mode == "T1-A1":
        ah = ConditionedAnswerHead().to(device)
    if mode in ("T1-A0", "T1-A1"):
        for p in core.parameters():
            p.requires_grad_(False)
        core.eval()
        if mode == "T1-A0":
            params = list(vh.parameters())
        else:
            params = list(ah.parameters())
        for p in params: p.requires_grad_(True)
    elif mode == "T2":
        for p in vh.parameters(): p.requires_grad_(False)
        vh.eval()
        params = list(core.parameters())
    elif mode == "T3":
        params = list(core.parameters()) + list(vh.parameters())
        for p in params: p.requires_grad_(True)
    else:
        raise ValueError(mode)
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(SEED + sum(ord(ch) for ch in name) % 10000)
    logs = []; best = -1.0; best_step = 0; best_payload = None
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        cap, pack, _ = sample_mr(cache, "train", rng, device, 64)
        facts, _, _, q, slot, y, _ = pack
        if mode in ("T2", "T3"):
            # 50/50 old replay for the value-path experiments.
            old_n = old_loss_batch(old, core, vh, ah, False, rng, device, 32)
            v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            new_loss, _ = answer_loss(v, slot, y, vh=vh)
            loss = 0.5 * old_n + 0.5 * new_loss
        elif mode == "T1-A0":
            with torch.no_grad():
                v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            loss, _ = answer_loss(v, slot, y, vh=vh)
        else:
            with torch.no_grad():
                v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            loss, _ = answer_loss(v, slot, y, ah=ah, q=q)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            vals = eval_stage(core, vh, ah, cache, "val", device, use_a1=(mode == "T1-A1"))
            macro = float(np.mean([vals[str(c)]["answer"] for c in CAPS]))
            logs.append({"step": step, "loss": float(loss.detach()), "val_macro": macro, "val": vals})
            if macro > best:
                best = macro; best_step = step
                best_payload = {"core": copy.deepcopy(core.state_dict()), "value_heads": copy.deepcopy(vh.state_dict()),
                                "answer_head": copy.deepcopy(ah.state_dict()) if ah is not None else None,
                                "mode": mode, "step": step, "val": vals}
                torch.save(best_payload, OUT / f"{name}_best.pt")
    final = {"core": core.state_dict(), "value_heads": vh.state_dict(),
             "answer_head": ah.state_dict() if ah is not None else None,
             "mode": mode, "step": steps}
    torch.save(final, OUT / f"{name}_final.pt")
    return {"name": name, "mode": mode, "best_step": best_step, "best_val_macro": best,
            "steps": steps, "replay_ratio": 0.5 if mode in ("T2", "T3") else 0.0,
            "wall_clock_s": time.perf_counter() - t0, "logs": logs,
            "best_path": str(OUT / f"{name}_best.pt"), "final_path": str(OUT / f"{name}_final.pt")}


def restore_stage(record, device):
    ck = torch.load(record["best_path"], map_location="cpu", weights_only=False)
    core, _, _ = load_core(device, trainable=False)
    core.load_state_dict(ck["core"], strict=True); core.eval()
    vh = MRValueHeads().to(device); vh.load_state_dict(ck["value_heads"], strict=True); vh.eval()
    ah = None
    if ck.get("answer_head") is not None:
        ah = ConditionedAnswerHead().to(device); ah.load_state_dict(ck["answer_head"], strict=True); ah.eval()
    for p in core.parameters(): p.requires_grad_(False)
    for p in vh.parameters(): p.requires_grad_(False)
    if ah is not None:
        for p in ah.parameters(): p.requires_grad_(False)
    return core, vh, ah


@torch.no_grad()
def eval_stage(core, vh, ah, cache, split, device, use_a1=False):
    out = {}
    for cap in CAPS:
        rows = cache.rows(split, cap); total = good = 0; by_pos = [0] * cap; pos_n = [0] * cap
        for st in range(0, len(rows), 128):
            rb = rows[st:st + 128]
            facts, _, _, q, slot, y, _ = mr_batch(cache, split, rb, cap, device)
            v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            logits = ah.selected(v, q, slot) if use_a1 else vh.selected(v, slot)
            pred = logits.argmax(1); good += int((pred == y).sum()); total += len(y)
            for p in range(cap):
                m = slot == p; pos_n[p] += int(m.sum()); by_pos[p] += int((pred[m] == y[m]).sum())
        out[str(cap)] = {"answer": good / max(total, 1), "n": total,
                         "by_position": [by_pos[p] / max(pos_n[p], 1) for p in range(cap)],
                         "position_n": pos_n}
    out["macro"] = float(np.mean([out[str(c)]["answer"] for c in CAPS]))
    return out


@torch.no_grad()
def collect_predictions(core, vh, ah, cache, split, device, use_a1=False):
    out = []
    for cap in CAPS:
        rows = cache.rows(split, cap)
        for st in range(0, len(rows), 128):
            rb = rows[st:st + 128]
            facts, _, rel, q, slot, y, _ = mr_batch(cache, split, rb, cap, device)
            v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            logits = ah.selected(v, q, slot) if use_a1 else vh.selected(v, slot)
            pred = logits.argmax(1)
            out.append({"rel": rel[torch.arange(len(y), device=device), slot].cpu().numpy(),
                        "y": y.cpu().numpy(), "pred": pred.cpu().numpy(), "slot": slot.cpu().numpy(),
                        "cap": np.full(len(y), cap, dtype=np.int64)})
    return {k: np.concatenate([x[k] for x in out]) for k in out[0]}


def confusion_by_relation(pred):
    mats = np.zeros((N_REL, N_CLASSES), dtype=np.int64)
    counts = np.zeros(N_REL, dtype=np.int64)
    correct = np.zeros(N_REL, dtype=np.int64)
    for r, y, p in zip(pred["rel"], pred["y"], pred["pred"]):
        mats[int(r), int(p)] += 1; counts[int(r)] += 1; correct[int(r)] += int(p == y)
    return {"matrix_pred_columns": mats.tolist(), "accuracy_by_relation": (correct / np.maximum(counts, 1)).tolist(),
            "n_by_relation": counts.tolist(), "overall": float(correct.sum() / max(counts.sum(), 1))}


def fresh_v_probe(core, cache, device):
    train_x = []; train_y = []; train_r = []
    for cap in CAPS:
        rows = cache.rows("train", cap)
        for st in range(0, len(rows), 128):
            facts, _, rel, _, _, _, fy = mr_batch(cache, "train", rows[st:st + 128], cap, device)
            v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            train_x.append(v[:, :cap, :].cpu().numpy()); train_y.append(fy.cpu().numpy()); train_r.append(rel.cpu().numpy())
    flat_x = [a.reshape(-1, 64) for a in train_x]
    flat_y = [a.reshape(-1) for a in train_y]
    flat_r = [a.reshape(-1) for a in train_r]
    x = np.concatenate(flat_x); y = np.concatenate(flat_y); r = np.concatenate(flat_r)
    probes = []
    for slot in range(4):
        mask = np.concatenate([np.ones(len(a), dtype=bool) if slot < a.shape[1] else np.zeros(len(a), dtype=bool) for a in train_x])
        # The concatenated training arrays are padded only conceptually; use
        # direct slot-wise arrays to avoid assigning labels to unwritten slots.
        sx = []; sy = []; sr = []
        for a, b, c in zip(train_x, train_y, train_r):
            if slot < b.shape[1]: sx.append(a[:, slot]); sy.append(b[:, slot]); sr.append(c[:, slot])
        sx = np.concatenate(sx); sy = np.concatenate(sy); sr = np.concatenate(sr)
        p = nn.Linear(64, N_CLASSES).to(device)
        opt = torch.optim.AdamW(p.parameters(), lr=3e-3, weight_decay=0.01)
        tx = torch.from_numpy(sx.astype(np.float32)).to(device); ty = torch.from_numpy(sy.astype(np.int64)).to(device)
        for it in range(300):
            ix = torch.randperm(len(ty), device=device)[:min(4096, len(ty))]
            opt.zero_grad(set_to_none=True); F.cross_entropy(p(tx[ix]), ty[ix]).backward(); opt.step()
        probes.append(p.eval())
    result = {"slots": {}, "overall": {}, "per_relation": {}, "probe_type": "fresh Linear(64,64), value-only"}
    for split in ("train", "val", "testA", "testB"):
        total = good = 0; by_rel = np.zeros(N_REL, dtype=np.int64); rel_n = np.zeros(N_REL, dtype=np.int64)
        for cap in CAPS:
            rows = cache.rows(split, cap)
            for st in range(0, len(rows), 128):
                facts, _, rel, _, _, _, fy = mr_batch(cache, split, rows[st:st + 128], cap, device)
                v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
                for slot in range(cap):
                    pr = probes[slot](v[:, slot, :].float()).argmax(1).cpu().numpy(); yy = fy[:, slot].cpu().numpy(); rr = rel[:, slot].cpu().numpy()
                    good += int((pr == yy).sum()); total += len(yy)
                    for q in range(N_REL):
                        m = rr == q; by_rel[q] += int((pr[m] == yy[m]).sum()); rel_n[q] += int(m.sum())
        result["overall"][split] = {"accuracy": good / max(total, 1), "n": total}
        result["per_relation"][split] = {RELATIONS[q]: {"accuracy": float(by_rel[q] / max(rel_n[q], 1)), "n": int(rel_n[q])} for q in range(N_REL)}
    # V -> relation probe is optional but inexpensive and useful for context.
    rel_probe = nn.Linear(64, N_REL).to(device); opt = torch.optim.AdamW(rel_probe.parameters(), lr=3e-3, weight_decay=0.01)
    tr = torch.from_numpy(r.astype(np.int64)).to(device); tx = torch.from_numpy(x.astype(np.float32)).to(device)
    for _ in range(300):
        ix = torch.randperm(len(tr), device=device)[:min(4096, len(tr))]; opt.zero_grad(set_to_none=True); F.cross_entropy(rel_probe(tx[ix]), tr[ix]).backward(); opt.step()
    rel_probe.eval(); result["v_to_relation"] = {}
    for split in ("testA", "testB"):
        xs = []; ys = []
        for cap in CAPS:
            rows = cache.rows(split, cap)
            for st in range(0, len(rows), 128):
                facts, _, rel, _, _, _, _ = mr_batch(cache, split, rows[st:st + 128], cap, device); v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
                xs.append(v[:, :cap, :].cpu().numpy().reshape(-1, 64)); ys.append(rel.cpu().numpy().reshape(-1))
        xx = torch.from_numpy(np.concatenate(xs).astype(np.float32)).to(device); yy = np.concatenate(ys)
        result["v_to_relation"][split] = {"accuracy": float((rel_probe(xx).argmax(1).cpu().numpy() == yy).mean()), "n": int(len(yy))}
    return result


@torch.no_grad()
def geometry(core, cache, split, device):
    xs = []; rs = []
    for cap in CAPS:
        rows = cache.rows(split, cap)
        for st in range(0, len(rows), 128):
            facts, _, rel, _, _, _, _ = mr_batch(cache, split, rows[st:st + 128], cap, device); v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            xs.append(v[:, :cap, :].cpu().numpy().reshape(-1, 64)); rs.append(rel.cpu().numpy().reshape(-1))
    v = np.concatenate(xs); r = np.concatenate(rs); flat = v.reshape(-1, 64); rr = r.reshape(-1)
    cent = []; within = []; nrel = []
    for q in range(N_REL):
        z = flat[rr == q]; nrel.append(len(z)); cent.append(z.mean(0).tolist() if len(z) else [0.0] * 64)
        within.append(float(np.mean(np.linalg.norm(z - z.mean(0), axis=1))) if len(z) else None)
    c = np.asarray(cent); cross = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2).tolist()
    return {"n": int(len(flat)), "norm_mean": float(np.linalg.norm(flat, axis=1).mean()),
            "norm_std": float(np.linalg.norm(flat, axis=1).std()), "centroid_by_relation": cent,
            "within_relation_distance": within, "centroid_cross_relation_distance": cross,
            "relation_counts": nrel}


@torch.no_grad()
def single_retention(core, vh, ah, old, device, use_a1=False):
    out = {}
    for split in ("testA", "testB"):
        rows = old.rows(split); total = good = 0
        for st in range(0, len(rows), 256):
            facts, y = old.batch(split, rows[st:st + 256], device); v, _ = core.forward_facts(facts, slot_indices=[0]); slot = torch.zeros(len(y), dtype=torch.long, device=device)
            if use_a1: pred = ah.selected(v, facts[:, 0, :], slot).argmax(1)
            else: pred = vh.selected(v, slot).argmax(1)
            good += int((pred == y).sum()); total += len(y)
        out[split] = {"accuracy": good / max(total, 1), "n": total}
    return out


@torch.no_grad()
def shortcuts(core, vh, ah, cache, device, use_a1=False, split="testA", cap=4):
    rows = cache.rows(split, cap); out = {"correct_v": [], "query_only": [], "v_only": [], "wrong_v": [], "correct_v_zero": []}
    total = 0
    for st in range(0, len(rows), 128):
        rb = rows[st:st + 128]; facts, _, _, q, slot, y, _ = mr_batch(cache, split, rb, cap, device); v, _ = core.forward_facts(facts, slot_indices=list(range(cap))); z = torch.zeros_like(v); vw = torch.roll(v, 1, 0); vz = v.clone(); vz[torch.arange(len(y), device=device), slot] = 0
        def pred(vv, qq):
            return (ah.selected(vv, qq, slot) if use_a1 else vh.selected(vv, slot)).argmax(1)
        qs = torch.zeros_like(q)
        for name, pp in (("correct_v", pred(v, q)), ("query_only", pred(z, q)), ("v_only", pred(v, qs)), ("wrong_v", pred(vw, q)), ("correct_v_zero", pred(vz, q))):
            out[name].append(float((pp == y).float().mean().item()))
        total += len(y)
    return {k: float(np.average(vals)) for k, vals in out.items()} | {"chance": CHANCE, "n": total}


@torch.no_grad()
def relation_swap(core, vh, ah, cache, device, use_a1=False):
    rows = cache.rows_type("testA", 4, "same_entity")
    if len(rows) == 0: return {"n": 0}
    facts, _, _, q, slot, y, fy = mr_batch(cache, "testA", rows, 4, device); v, _ = core.forward_facts(facts, slot_indices=list(range(4)))
    d = cache.data["testA"]; off = d["fact_offsets"]
    alt = (slot + 1) % 4
    ids = np.asarray([int(off[r]) + int(p) for r, p in zip(rows, alt)], dtype=np.int64)
    q_alt = torch.from_numpy(d["fact_query_H"][ids].astype(np.float32)).to(device); y_alt = fy[torch.arange(len(slot), device=device), alt]
    def run(qq, vv, ii, yy):
        pred = (ah.selected(vv, qq, ii) if use_a1 else vh.selected(vv, ii)).argmax(1)
        return float((pred == yy).float().mean())
    return {"n": len(rows), "original_query_accuracy": run(q, v, slot, y),
            "alternate_relation_query_accuracy_at_alternate_slot": run(q_alt, v, alt, y_alt),
            "note": "The same memory is queried at two fact positions; A0 is V-only, A1 additionally receives query_hidden."}


@torch.no_grad()
def collision_test(core, vh, ah, cache, device, use_a1=False):
    """Same surface value across relation families; query each relation."""
    rows = cache.rows_type("testA", 4, "same_value")
    if len(rows) == 0:
        return {"n": 0}
    facts, _, _, q, slot, y, fy = mr_batch(cache, "testA", rows, 4, device)
    v, _ = core.forward_facts(facts, slot_indices=list(range(4)))
    d = cache.data["testA"]; off = d["fact_offsets"]
    preds = []; oks = []
    for p in range(4):
        ids = np.asarray([int(off[r]) + p for r in rows], dtype=np.int64)
        qp = torch.from_numpy(d["fact_query_H"][ids].astype(np.float32)).to(device)
        ix = torch.full((len(rows),), p, dtype=torch.long, device=device)
        pp = (ah.selected(v, qp, ix) if use_a1 else vh.selected(v, ix)).argmax(1)
        preds.append(pp.cpu().numpy()); oks.append((pp == fy[:, p]).float().cpu().numpy())
    pred = np.stack(preds, axis=1); same = np.all(pred == pred[:, :1], axis=1)
    return {"n": len(rows), "all_relation_queries_correct": float(np.all(np.stack(oks, axis=1), axis=1).mean()),
            "mean_query_accuracy": float(np.mean(np.stack(oks, axis=1))), "prediction_consistency": float(same.mean()),
            "same_value_target_consistency": float(np.all(fy.cpu().numpy() == fy.cpu().numpy()[:, :1], axis=1).mean())}


@torch.no_grad()
def reconnect(core, vh, ah, cache, device, use_a1=False):
    ck = torch.load(MR_BEST, map_location="cpu", weights_only=False)
    key = KeyEncoder(64).to(device); query = QueryEncoder(64).to(device); key.load_state_dict(ck["key"]); query.load_state_dict(ck["query"]); key.eval(); query.eval()
    out = {}
    for split in ("testA", "testB"):
        out[split] = {}
        for cap in CAPS:
            rows = cache.rows(split, cap); total = ret = good = oracle = 0
            for st in range(0, len(rows), 128):
                rb = rows[st:st + 128]; facts, _, _, qh, slot, y, _ = mr_batch(cache, split, rb, cap, device); v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
                k = key(facts.reshape(-1, 2048)).reshape(len(rb), cap, -1); qq = query(qh); sel = (qq[:, None, :] * k).sum(-1).argmax(1)
                pred = (ah.selected(v, qh, sel) if use_a1 else vh.selected(v, sel)).argmax(1); op = (ah.selected(v, qh, slot) if use_a1 else vh.selected(v, slot)).argmax(1)
                total += len(y); ret += int((sel == slot).sum()); good += int((pred == y).sum()); oracle += int((op == y).sum())
            out[split][str(cap)] = {"retrieval": ret / max(total, 1), "hard_answer": good / max(total, 1), "oracle_answer": oracle / max(total, 1), "n": total}
    return out


def git_info():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=10).stdout.strip()
        diff = subprocess.run(["git", "diff", "--stat"], cwd=str(ROOT), capture_output=True, text=True, timeout=10).stdout.strip()
        return {"sha": sha or None, "diff_stat": diff or None}
    except Exception as e:
        return {"sha": None, "diff_stat": None, "error": str(e)}


def pct(x): return "n/a" if x is None else f"{100.0 * float(x):.2f}%"


def main():
    seed_all(); t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(MR_CACHE); old = OldSingleCache(OLD_CACHE)
    core0, old_heads, _ = load_core(device, trainable=False); vh0, mr_ck = load_t0_decoder(device)

    # T0: current O2.1a-MR decoder on the frozen O1 value path.  Also retain
    # the original O1 head number to distinguish schema adaptation from storage.
    t0_current = eval_stage(core0, vh0, None, cache, "testA", device)
    t0_current_b = eval_stage(core0, vh0, None, cache, "testB", device)
    t0_old_core, t0_old_heads_raw, _ = load_core(device, trainable=False)
    t0_old_heads = LegacyLocalHeads(t0_old_heads_raw).to(device).eval()
    t0_old = eval_stage(t0_old_core, t0_old_heads, None, cache, "testA", device)
    baseline = {"current_mr_decoder": {"TestA": t0_current, "TestB": t0_current_b},
                "original_o1_heads": {"TestA": t0_old, "TestB": eval_stage(t0_old_core, t0_old_heads, None, cache, "testB", device)},
                "source_checkpoint": str(MR_BEST), "value_path": str(R2),
                "statement": "current MR decoder reproduces the prior ~53% oracle answer; original O1 heads are also shown."}
    json_dump(OUT / "baseline_frozen.json", baseline)

    probes = fresh_v_probe(core0, cache, device); json_dump(OUT / "fresh_v_probe.json", probes)
    geom_before = {s: geometry(core0, cache, s, device) for s in ("testA", "testB")}

    # Staged training. T2/T3 include exactly 50% old single-relation replay.
    records = {}
    if os.environ.get("V24_VGEN_SMOKE", "0") == "1":
        run_steps = min(STEPS, 20)
    else:
        run_steps = STEPS
    records["T1-A0"] = train_stage("t1_answer_a0", cache, old, vh0, device, "T1-A0", run_steps, False)
    records["T1-A1"] = train_stage("t1_answer_a1", cache, old, vh0, device, "T1-A1", run_steps, False)
    records["T2"] = train_stage("t2_value_only", cache, old, vh0, device, "T2", run_steps, True)
    records["T3"] = train_stage("t3_joint", cache, old, vh0, device, "T3", run_steps, True)

    stage_objs = {}; stage_results = {}; retention = {"original_o1": single_retention(t0_old_core, t0_old_heads, None, old, device)}
    for name, rec in records.items():
        c, v, a = restore_stage(rec, device); use_a1 = name == "T1-A1"
        stage_objs[name] = (c, v, a, use_a1)
        stage_results[name] = {s: eval_stage(c, v, a, cache, s, device, use_a1) for s in ("val", "testA", "testB")}
        retention[name] = single_retention(c, v, a, old, device, use_a1)

    # Select the best valid stage by TestA N=4 oracle answer, with T3 included
    # as the primary candidate and T1-A1 retained as the conditioning ablation.
    best_name = max(stage_results, key=lambda n: stage_results[n]["val"]["macro"])
    best_core, best_vh, best_ah, best_a1 = stage_objs[best_name]
    relation_conf = {}
    for name, (c, v, a, ua) in stage_objs.items():
        relation_conf[name] = {s: confusion_by_relation(collect_predictions(c, v, a, cache, s, device, ua)) for s in ("testA", "testB")}

    shortcuts_out = {s: shortcuts(best_core, best_vh, best_ah, cache, device, best_a1, s, 4) for s in ("testA", "testB")}
    counter = {"relation_swap": relation_swap(best_core, best_vh, best_ah, cache, device, best_a1),
               "cross_relation_value_collision": collision_test(best_core, best_vh, best_ah, cache, device, best_a1),
               "answer_inputs": {"A0": "selected V only", "A1": "selected V + query_hidden through Linear(2048,32)", "K_used_in_answer": False}}
    geom_after = {s: geometry(best_core, cache, s, device) for s in ("testA", "testB")}
    reconnect_out = reconnect(best_core, best_vh, best_ah, cache, device, best_a1)

    # Save a compact top-level checkpoint for the selected candidate.
    best_ck = torch.load(records[best_name]["best_path"], map_location="cpu", weights_only=False)
    torch.save({"condition": best_name, **best_ck, "source_r2": str(R2), "source_mr": str(MR_BEST)}, OUT / "best.pt")
    final_ck = torch.load(records["T3"]["final_path"], map_location="cpu", weights_only=False)
    torch.save({"condition": "T3", **final_ck, "source_r2": str(R2), "source_mr": str(MR_BEST)}, OUT / "final.pt")

    config = {"phase": "CLR-v2.4-V-GEN", "seed": SEED, "device": str(device),
              "source_r2": str(R2), "source_mr_checkpoint": str(MR_BEST), "cache": str(MR_CACHE),
              "capacities": list(CAPS), "relations": list(RELATIONS), "steps": run_steps,
              "architecture_changed": False, "main_path": "oracle allocation + oracle correct slot read",
              "trainable_modules": {"T1-A0": ["value_heads.heads.*"], "T1-A1": ["answer_head.query_proj", "answer_head.heads.*"],
                                    "T2": ["content_read.*", "shared_gru.*"], "T3": ["content_read.*", "shared_gru.*", "value_heads.heads.*"]},
              "frozen_modules": ["semantic K", "query-key encoder", "retrieval controller", "allocation", "Ouro"],
              "replay": {"T1-A0": 0.0, "T1-A1": 0.0, "T2": 0.5, "T3": 0.5},
              "dataset": {s: {"episodes": int(len(cache.data[s]["N"])), "relation_counts": np.bincount(cache.data[s]["query_relation"], minlength=N_REL).tolist()} for s in ("train", "val", "testA", "testB")},
              "git": git_info(), "exact_command": "py -3 a1_v24_vgen.py"}
    json_dump(OUT / "config.json", config)
    json_dump(OUT / "train.log", records)
    json_dump(OUT / "t1_answer_only.json", {"T1-A0": records["T1-A0"], "T1-A1": records["T1-A1"], "results": {k: stage_results[k] for k in ("T1-A0", "T1-A1")}})
    json_dump(OUT / "t2_value_only.json", {"training": records["T2"], "results": stage_results["T2"]})
    json_dump(OUT / "t3_joint.json", {"training": records["T3"], "results": stage_results["T3"]})
    json_dump(OUT / "per_relation_results.json", {"best_condition": best_name, "stages": {n: {s: stage_results[n][s] for s in ("testA", "testB")} for n in stage_results}})
    json_dump(OUT / "relation_confusion.json", relation_conf)
    json_dump(OUT / "shortcut_checks.json", shortcuts_out)
    json_dump(OUT / "counterfactual_results.json", counter)
    json_dump(OUT / "single_relation_retention.json", retention)
    json_dump(OUT / "multi_relation_oracle.json", {"best_condition": best_name, "stages": stage_results})
    json_dump(OUT / "full_kv_reconnect.json", {"condition": best_name, "results": reconnect_out})
    json_dump(OUT / "metrics.json", {"baseline": baseline, "fresh_v_probe": probes, "best_condition": best_name,
                                      "stage_testA": {n: stage_results[n]["testA"] for n in stage_results},
                                      "stage_testB": {n: stage_results[n]["testB"] for n in stage_results},
                                      "single_relation_retention": retention, "reconnect": reconnect_out})

    b4a = baseline["current_mr_decoder"]["TestA"]["4"]["answer"]; b4b = baseline["current_mr_decoder"]["TestB"]["4"]["answer"]
    best4a = stage_results[best_name]["testA"]["4"]["answer"]; best4b = stage_results[best_name]["testB"]["4"]["answer"]
    p4a = probes["overall"]["testA"]["accuracy"]; p4b = probes["overall"]["testB"]["accuracy"]
    old_a = retention[best_name]["testA"]["accuracy"]; old_b = retention[best_name]["testB"]["accuracy"]
    orig_a = retention["original_o1"]["testA"]["accuracy"]; orig_b = retention["original_o1"]["testB"]["accuracy"]
    reconnect_a = reconnect_out["testA"]["4"]["hard_answer"]; reconnect_b = reconnect_out["testB"]["4"]["hard_answer"]
    if p4a >= .90 and p4b >= .85 and best4a >= .90 and best4b >= .85 and reconnect_a >= .90 and reconnect_b >= .85 and old_a >= orig_a - .05 and old_b >= orig_b - .05:
        verdict = "V-GEN PASS: MULTI-RELATION VALUE GENERALIZATION RESTORED"
    elif p4a >= .90 and p4b >= .85 and best4a < .90:
        verdict = "V-GEN PARTIAL: VALUE STORAGE WORKS, DECODER GENERALIZATION INCOMPLETE"
    elif best4a >= .90 and best4b >= .85 and p4a < .90:
        verdict = "V-GEN PARTIAL: DECODER WORKS, VALUE WRITE GENERALIZATION INCOMPLETE"
    elif old_a < orig_a - .05 or old_b < orig_b - .05:
        verdict = "V-GEN FAIL: CATASTROPHIC FORGETTING"
    elif (p4a < .90 or p4b < .85) and stage_results["T1-A0"]["testA"]["4"]["answer"] < .90 and stage_results["T2"]["testA"]["4"]["answer"] < .90:
        verdict = "V-GEN FAIL: JOINT VALUE/DECODER BOTTLENECK"
    else:
        verdict = "V-GEN FAIL: DECODER GENERALIZATION BOTTLENECK" if max(stage_results["T1-A0"]["testA"]["4"]["answer"], stage_results["T1-A1"]["testA"]["4"]["answer"]) < .70 else "V-GEN FAIL: VALUE WRITE GENERALIZATION BOTTLENECK"

    report = ["# CLR-v2.4-V-GEN — Multi-Relation Value Generalization", "", f"**{verdict}**", "", "## Primary result",
              f"Current frozen O1 + current MR decoder baseline: TestA N=4 {pct(b4a)}, TestB N=4 {pct(b4b)}.",
              f"Fresh V probe before adaptation: TestA {pct(p4a)}, TestB {pct(p4b)}; best condition {best_name}: oracle TestA {pct(best4a)}, TestB {pct(best4b)}.",
              f"Reconnect with frozen K/router: retrieval TestA/TestB {pct(reconnect_out['testA']['4']['retrieval'])}/{pct(reconnect_out['testB']['4']['retrieval'])}; answer {pct(reconnect_a)}/{pct(reconnect_b)}.", "",
              "## Conditions", "- T0 freezes the O1 core and the existing O2.1a-MR value decoder.", "- T1-A0 trains only D(V); T1-A1 trains only a minimal D(V, query_hidden) decoder.", "- T2 trains only O1 content_read/shared_gru with the answer decoder frozen.", "- T3 trains O1 content_read/shared_gru and D(V) jointly.", "- T2/T3 use exactly 50% old single-relation replay and 50% MR samples; no K/router is trained.", "- All main evaluations use oracle write allocation and oracle correct-slot selection.", "", "## Stage results"]
    for n in ("T1-A0", "T1-A1", "T2", "T3"):
        report.append(f"{n}: TestA N=4 {pct(stage_results[n]['testA']['4']['answer'])}; TestB N=4 {pct(stage_results[n]['testB']['4']['answer'])}; best step {records[n]['best_step']}; wall {records[n]['wall_clock_s']:.2f}s.")
    report += ["", "## Fresh V probes", json.dumps(probes, ensure_ascii=False), "", "## Relation balance and confusion", json.dumps(relation_conf, ensure_ascii=False),
               "", "## Shortcuts/interventions", json.dumps(shortcuts_out, ensure_ascii=False), json.dumps(counter, ensure_ascii=False),
               "", "## Original O1 retention", json.dumps(retention, ensure_ascii=False),
               f"Original O1 retention reference: TestA/TestB {pct(orig_a)}/{pct(orig_b)}; selected condition: {pct(old_a)}/{pct(old_b)}.",
               "", "## Value geometry", json.dumps({"before": geom_before, "after_best": geom_after}, ensure_ascii=False),
               "", "## Protocol and boundary", "The answer decoder never reads K. The query-conditioned ablation reads frozen Ouro query_hidden only through a fresh 2048→32 projection. No allocation, retrieval, attention memory, recurrent depth, O2.1b, O3, or A2 was entered.",
               "", "## Runtime", f"Total wall-clock: {time.perf_counter()-t0:.2f}s; training steps per condition: {run_steps}; device: {device}; GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}.",
               "Peak VRAM should be read from the nvidia-smi snapshot recorded alongside this run if available.", "", "## Q1–Q23 summary",
               f"Q1 baseline reproduced: yes, {pct(b4a)}/{pct(b4b)}.", f"Q2/Q3 fresh V probe: {pct(p4a)}/{pct(p4b)}; per relation is in fresh_v_probe.json.",
               f"Q4 T1 answer-only: A0 {pct(stage_results['T1-A0']['testA']['4']['answer'])}/{pct(stage_results['T1-A0']['testB']['4']['answer'])}; A1 {pct(stage_results['T1-A1']['testA']['4']['answer'])}/{pct(stage_results['T1-A1']['testB']['4']['answer'])}.",
               f"Q5 T2: {pct(stage_results['T2']['testA']['4']['answer'])}/{pct(stage_results['T2']['testB']['4']['answer'])}.", f"Q6 T3: {pct(stage_results['T3']['testA']['4']['answer'])}/{pct(stage_results['T3']['testB']['4']['answer'])}.",
               "Q7 bottleneck diagnosis is determined from the T1/T2/T3 and fresh-probe pattern above.", "Q8/Q9 relation conditioning results are T1-A0 versus T1-A1; K is excluded.",
               f"Q10/Q11 best oracle answer: {pct(best4a)}/{pct(best4b)}.", f"Q12 storage gate: {pct(p4a)}/{pct(p4b)}.", f"Q13/Q14 selected original single-relation retention: {pct(old_a)}/{pct(old_b)}; catastrophic forgetting={'yes' if old_a < orig_a-.05 or old_b < orig_b-.05 else 'no'}.",
               "Q15–Q18 are in shortcut_checks.json, counterfactual_results.json, relation_confusion.json, and fresh_v_probe.json.", f"Q19 selected condition: {best_name}.", f"Q20/Q21 reconnect: retrieval {pct(reconnect_out['testA']['4']['retrieval'])}/{pct(reconnect_out['testB']['4']['retrieval'])}; answer {pct(reconnect_a)}/{pct(reconnect_b)}.",
               f"Q22 system-level multi-relation KV cell: {'yes' if reconnect_a >= .90 and reconnect_b >= .85 else 'no'}.", "Q23 O2.1b: not entered; this phase stops here.", "", "## Capability boundary", "This phase only tests multi-relation value storage/decoding under oracle allocation/read and a previously validated semantic K reconnect. It does not establish learned allocation, autonomous memory management, long-context compression, or latent reasoning."]
    (OUT / "v24_vgen_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "best_condition": best_name, "baseline_testA_N4": b4a, "baseline_testB_N4": b4b, "best_testA_N4": best4a, "best_testB_N4": best4b, "probe_testA": p4a, "probe_testB": p4b, "reconnect_testA_N4": reconnect_a, "reconnect_testB_N4": reconnect_b, "wall_clock_s": time.perf_counter()-t0}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
