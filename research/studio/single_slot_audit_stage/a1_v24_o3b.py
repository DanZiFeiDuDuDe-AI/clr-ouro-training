# -*- coding: utf-8 -*-
"""CLR-v2.4-O3b — same-key update and semantic write routing.

The O2.1b learned KeyEncoder, V/core, QueryEncoder, read-side matcher,
decoder, and O3a free-slot allocator are frozen.  Only a write-side router is
trained.  It predicts NEW versus UPDATE and, for UPDATE, the occupied target
slot.  NEW uses the frozen O3a allocator; UPDATE keeps K and replaces V.
"""
import copy
import itertools
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v24_o21a import KeyEncoder, QueryEncoder
from a1_v24_o21a_mr import MRCache, MRValueHeads
from a1_v24_vgen import load_core
from a1_v24_o2_common import json_dump
from a1_v24_o3a import Allocator as FreeAllocator, write_v_slot


ROOT = Path(r"D:/training")
OUT = Path(os.environ.get("V24_O3B_OUT", r"D:/training/checkpoints/clr_v24_o3b_update"))
CACHE_ROOT = Path(os.environ.get("V24_O3B_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
O21B = Path(os.environ.get("V24_O3B_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
SUPERVISED = Path(os.environ.get("V24_O3B_SUPERVISED", r"D:/training/checkpoints/clr_v24_o21a_mr/best_model.pt"))
VGEN2 = Path(os.environ.get("V24_O3B_VGEN2", r"D:/training/checkpoints/clr_v24_vgen2/best.pt"))
O3A_ALLOC = Path(os.environ.get("V24_O3B_O3A", r"D:/training/checkpoints/clr_v24_o3a_allocation/allocator_kaware_best.pt"))
SLOTS = 4
DIM = 64
CAPS = (1, 2, 4)
SEED = int(os.environ.get("V24_O3B_SEED", "42"))
STEPS = int(os.environ.get("V24_O3B_STEPS", "3000"))
EVAL_EVERY = int(os.environ.get("V24_O3B_EVAL_EVERY", "100"))
DEVICE_NAME = os.environ.get("V24_O3B_DEVICE", "cuda")
SMOKE = os.environ.get("V24_O3B_SMOKE", "0") == "1" or "--smoke" in os.sys.argv
TEMP = 0.1


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UpdateRouter(nn.Module):
    """Two-stage write-side router: action plus pairwise update target."""
    def __init__(self):
        super().__init__()
        pair_in = DIM * 4 + 1
        self.pair = nn.Sequential(nn.Linear(pair_in, 128), nn.GELU(), nn.Linear(128, 1))
        self.action = nn.Sequential(nn.Linear(DIM + SLOTS + 1, 96), nn.GELU(), nn.Linear(96, 2))

    def forward(self, k_new, existing_k, occupied):
        b = k_new.shape[0]
        kn = k_new[:, None, :].expand(-1, SLOTS, -1)
        pair = torch.cat([kn, existing_k, kn * existing_k, (kn - existing_k).abs(), occupied[:, :, None].float()], dim=-1)
        pair_logits = self.pair(pair).squeeze(-1)
        masked = pair_logits.masked_fill(~occupied.bool(), -1e4)
        max_pair = masked.max(1).values[:, None]
        action_logits = self.action(torch.cat([k_new, max_pair, occupied.float()], dim=-1))
        return action_logits, pair_logits


class FactPool:
    def __init__(self, cache, split):
        self.cache = cache
        self.split = split
        self.data = cache.data[split]
        self.meta = cache.meta[split]
        self.episodes = []
        self.by_key = defaultdict(list)
        self.by_person = defaultdict(list)
        self.by_relation = defaultdict(list)
        self.by_value = defaultdict(list)
        offsets = self.data["fact_offsets"]
        for ep, m in enumerate(self.meta):
            cap = int(m["capacity"])
            refs = []
            for pos in range(cap):
                ref = {"split": split, "episode": ep, "pos": pos, "flat": int(offsets[ep]) + pos,
                       "person": str(m["persons"][pos]), "relation": int(m["relation_ids"][pos]),
                       "relation_name": str(m["relations"][pos]), "value": int(m["fact_classes"][pos]),
                       "key": (str(m["persons"][pos]), int(m["relation_ids"][pos]))}
                refs.append(ref)
                self.by_key[ref["key"]].append(ref)
                self.by_person[ref["person"]].append(ref)
                self.by_relation[ref["relation"]].append(ref)
                self.by_value[ref["value"]].append(ref)
            self.episodes.append(refs)

    def hidden(self, ref):
        return self.data["fact_H"][ref["flat"]].astype(np.float32)

    def query_hidden(self, ref):
        return self.data["fact_query_H"][ref["flat"]].astype(np.float32)

    def choose_base(self, rng, max_len=4, require_capacity=None):
        choices = [x for x in self.episodes if len(x) >= 1 and (require_capacity is None or len(x) <= require_capacity)]
        ep = choices[int(rng.integers(0, len(choices)))]
        n = int(rng.integers(1, min(max_len, len(ep)) + 1))
        idx = rng.choice(len(ep), size=n, replace=False)
        return [ep[int(i)] for i in idx]

    def new_candidate(self, base, rng, category):
        keys = {x["key"] for x in base}
        persons = {x["person"] for x in base}
        relations = {x["relation"] for x in base}
        values = {x["value"] for x in base}
        candidates = []
        if category == "same_entity_diff_relation":
            for x in base:
                candidates.extend([y for y in self.by_person[x["person"]] if y["key"] not in keys and y["relation"] != x["relation"]])
        elif category == "same_relation_diff_entity":
            for x in base:
                candidates.extend([y for y in self.by_relation[x["relation"]] if y["person"] not in persons and y["key"] not in keys])
        elif category == "same_value_diff_key":
            for v in values:
                candidates.extend([y for y in self.by_value[v] if y["key"] not in keys])
        else:
            for y in self.by_key.values():
                candidates.extend([x for x in y if x["key"] not in keys and x["person"] not in persons])
        if not candidates:
            candidates = [x for x in sum(self.episodes, []) if x["key"] not in keys]
        return candidates[int(rng.integers(0, len(candidates)))]

    def update_candidate(self, base_ref, rng):
        candidates = [x for x in self.by_key[base_ref["key"]] if x["value"] != base_ref["value"]]
        if not candidates:
            return None
        return candidates[int(rng.integers(0, len(candidates)))]

    def example(self, rng, kind=None, category=None):
        if kind is None: kind = "update" if rng.random() < 0.5 else "new"
        if kind == "update":
            for _ in range(30):
                base = self.choose_base(rng, 4)
                target = int(rng.integers(0, len(base)))
                incoming = self.update_candidate(base[target], rng)
                if incoming is not None:
                    return {"base": base, "incoming": incoming, "action": 1, "target": target, "category": "update_same_key", "nbase": len(base)}
            return self.example(rng, "new", category)
        if len(self.episodes) == 0:
            raise RuntimeError("empty fact pool")
        base = self.choose_base(rng, 3)
        if category is None: category = ("same_entity_diff_relation", "same_relation_diff_entity", "fully_different", "same_value_diff_key")[int(rng.integers(0, 4))]
        incoming = self.new_candidate(base, rng, category)
        return {"base": base, "incoming": incoming, "action": 0, "target": -1, "category": category, "nbase": len(base)}


def make_examples(pool, count, seed, balanced=True, category_balanced=False):
    rng = np.random.default_rng(seed)
    out = []
    if category_balanced:
        cats = ["same_entity_diff_relation", "same_relation_diff_entity", "fully_different", "same_value_diff_key"]
        for i in range(count):
            out.append(pool.example(rng, "new", cats[i % len(cats)]))
    elif balanced:
        for i in range(count): out.append(pool.example(rng, "update" if i % 2 == 0 else "new"))
    else:
        for _ in range(count): out.append(pool.example(rng))
    return out


def batch_examples(examples, pool, device):
    b = len(examples); maxn = SLOTS
    h = np.zeros((b, maxn, 2048), dtype=np.float32); occ = np.zeros((b, SLOTS), dtype=np.float32)
    new_h = np.zeros((b, 2048), dtype=np.float32); action = np.zeros(b, dtype=np.int64); target = np.zeros(b, dtype=np.int64)
    for i, ex in enumerate(examples):
        for j, ref in enumerate(ex["base"]):
            h[i, j] = pool.hidden(ref); occ[i, j] = 1.0
        new_h[i] = pool.hidden(ex["incoming"]); action[i] = ex["action"]; target[i] = max(0, ex["target"])
    return (torch.from_numpy(h).to(device), torch.from_numpy(new_h).to(device), torch.from_numpy(occ).to(device), torch.from_numpy(action).to(device), torch.from_numpy(target).to(device))


def load_all(device):
    kc = torch.load(O21B, map_location="cpu", weights_only=False); key = KeyEncoder(DIM).to(device); key.load_state_dict(kc["key"], strict=True); key.eval()
    qc = torch.load(SUPERVISED, map_location="cpu", weights_only=False); query = QueryEncoder(DIM).to(device); query.load_state_dict(qc["query"], strict=True); query.eval()
    vc, _, _ = load_core(device, trainable=False); vck = torch.load(VGEN2, map_location="cpu", weights_only=False); vc.load_state_dict(vck["core"], strict=True); vc.eval()
    vh = MRValueHeads().to(device); vh.load_state_dict(vck["value_heads"], strict=True); vh.eval()
    fa = FreeAllocator(True).to(device); fac = torch.load(O3A_ALLOC, map_location=device, weights_only=False); fa.load_state_dict(fac["allocator"], strict=True); fa.eval()
    for mod in (key, query, vc, vh, fa):
        for p in mod.parameters(): p.requires_grad_(False)
    return key, query, vc, vh, fa


def router_outputs(router, key, base_h, new_h, occupied):
    with torch.no_grad():
        kb = key(base_h.reshape(-1, 2048)).reshape(base_h.shape[0], base_h.shape[1], DIM)
        kn = key(new_h)
    return router(kn, kb, occupied)


def binary_metrics(y, score, threshold=0.5):
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64); pred = (score >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    order = np.argsort(-score); ys = y[order]; tps = np.cumsum(ys); fps = np.cumsum(1 - ys); auc = float(np.trapezoid(tps / max(tps[-1], 1), fps / max(fps[-1], 1))) if tps[-1] and fps[-1] else None
    return {"accuracy": float((pred == y).mean()), "precision_update": float(precision), "recall_update": float(recall), "f1_update": float(f1), "tp": tp, "tn": tn, "fp": fp, "fn": fn, "auroc": auc, "n": int(len(y)), "threshold": float(threshold)}


@torch.no_grad()
def router_eval(router, key, examples, pool, device, threshold=0.5):
    router.eval(); ys = []; scores = []; pred_actions = []; target_ok = []; new_valid = []; cats = []
    for st in range(0, len(examples), 256):
        eb = examples[st:st + 256]; bh, nh, occ, action, target = batch_examples(eb, pool, device); al, pl = router_outputs(router, key, bh, nh, occ); prob = al.softmax(-1)[:, 1]; pred = (prob >= threshold).long(); masked = pl.masked_fill(~occ.bool(), -1e4); pslot = masked.argmax(1)
        ys.extend(action.cpu().numpy()); scores.extend(prob.cpu().numpy()); pred_actions.extend(pred.cpu().numpy()); target_ok.extend(((pslot == target) & (action == 1)).cpu().numpy()); new_valid.extend(((pred == 0) & (action == 0)).cpu().numpy()); cats.extend([e["category"] for e in eb])
    y = np.asarray(ys); score = np.asarray(scores); pred = np.asarray(pred_actions); target_ok = np.asarray(target_ok); new_valid = np.asarray(new_valid)
    cm = {"true_NEW_pred_NEW": int(((y == 0) & (pred == 0)).sum()), "true_NEW_pred_UPDATE": int(((y == 0) & (pred == 1)).sum()), "true_UPDATE_pred_UPDATE": int(((y == 1) & (pred == 1)).sum()), "true_UPDATE_pred_NEW": int(((y == 1) & (pred == 0)).sum())}
    return {"binary": binary_metrics(y, score, threshold), "update_target_accuracy_on_true_update": float(target_ok[y == 1].mean()) if (y == 1).any() else None, "new_pred_new_rate_on_true_new": float(new_valid[y == 0].mean()) if (y == 0).any() else None, "confusion": cm, "categories": cats, "y": y, "score": score, "pred": pred, "target_ok": target_ok}


def choose_threshold(metrics):
    y = metrics["y"]; score = metrics["score"]; candidates = np.unique(np.r_[0.05, np.linspace(0.05, 0.95, 91), score])
    best = (0.0, 0.5)
    for t in candidates:
        m = binary_metrics(y, score, float(t))
        if m["f1_update"] > best[0]: best = (m["f1_update"], float(t))
    return best[1]


def train_router(train, val, train_pool, val_pool, key, device, steps, smoke):
    router = UpdateRouter().to(device); opt = torch.optim.AdamW(router.parameters(), lr=3e-4, weight_decay=0.01); rng = np.random.default_rng(SEED + 8100); logs = []; best = -1.0; best_step = 0; best_threshold = 0.5; t0 = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    batch_size = 256 if smoke else 128
    for step in range(1, steps + 1):
        ix = rng.integers(0, len(train), size=min(batch_size, len(train))); eb = [train[int(i)] for i in ix]; bh, nh, occ, action, target = batch_examples(eb, train_pool, device); al, pl = router_outputs(router, key, bh, nh, occ); loss_action = F.cross_entropy(al, action); upd = action == 1; masked = pl.masked_fill(~occ.bool(), -1e4); loss_target = F.cross_entropy(masked[upd], target[upd]) if upd.any() else al.sum() * 0
        pair_target = torch.zeros_like(pl); pair_target[upd, target[upd]] = 1.0
        loss_pair = F.binary_cross_entropy_with_logits(pl[occ.bool()], pair_target[occ.bool()])
        loss = loss_action + loss_target + 0.5 * loss_pair
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0); opt.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == steps:
            raw = router_eval(router, key, val, val_pool, device, 0.5); th = choose_threshold(raw); ev = router_eval(router, key, val, val_pool, device, th); score = float(ev["binary"]["f1_update"] + ev["update_target_accuracy_on_true_update"] + (ev["new_pred_new_rate_on_true_new"] or 0.0)) / 3.0
            logs.append({"step": step, "loss": float(loss.detach()), "loss_action": float(loss_action.detach()), "loss_update_target": float(loss_target.detach()), "loss_pair": float(loss_pair.detach()), "val": {k: v for k, v in ev.items() if k not in ("y", "score", "pred", "target_ok", "categories")}, "threshold": th})
            if score > best:
                best = score; best_step = step; best_threshold = th; torch.save({"router": copy.deepcopy(router.state_dict()), "threshold": th, "step": step, "score": score, "params": sum(p.numel() for p in router.parameters())}, OUT / "router_best.pt")
    torch.save({"router": router.state_dict(), "threshold": best_threshold, "step": steps}, OUT / "router_final.pt")
    peak = None
    if torch.cuda.is_available(): peak = {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
    return {"steps": steps, "best_step": best_step, "best_score": best, "best_threshold": best_threshold, "wall_clock_s": time.perf_counter() - t0, "params": sum(p.numel() for p in router.parameters()), "peak_vram": peak, "logs": logs}


def ref_h(pool, ref, device): return torch.from_numpy(pool.hidden(ref)).to(device)
def ref_q(pool, ref, device): return torch.from_numpy(pool.query_hidden(ref)).to(device)


@torch.no_grad()
def route_one(router, key, free_alloc, k_new, kstate, occupied, threshold, device, mode="learned", force_target=None):
    if mode == "oracle":
        raise RuntimeError("oracle routing is handled by semantic refs")
    al, pl = router(k_new[None], kstate[None], occupied[None]); prob = float(al.softmax(-1)[0, 1]); is_update = prob >= threshold
    if force_target is not None:
        return True, int(force_target), prob, pl[0]
    if is_update:
        target = pl[0].masked_fill(~occupied.bool(), -1e4).argmax().item()
    else:
        fl = free_alloc(k_new[None], occupied[None]); fl = fl.masked_fill(occupied[None].bool(), -1e9); target = int(fl.argmax(1).item())
    return is_update, target, prob, pl[0]


def free_target(free_alloc, key, h, occupied, device):
    kn = key(h[None]); logits = free_alloc(kn, occupied[None]).masked_fill(occupied[None].bool(), -1e9); return int(logits.argmax(1).item()), kn[0]


@torch.no_grad()
def run_episode(ex, pool, key, query, core, vh, free_alloc, router, threshold, device, mode="learned", base_order=None, forced_base=None, force_update_target=None):
    base = list(ex["base"]); events = [ex["incoming"]] if "events" not in ex else ex["events"]; order = list(range(len(base))) if base_order is None else list(base_order)
    kstate = torch.zeros(SLOTS, DIM, device=device); vstate = core.zero_state(1, device=device, dtype=core.content_read.weight.dtype)[0]; occupied = torch.zeros(SLOTS, device=device); slot_refs = [None] * SLOTS; base_slots = []
    local_events = []
    for t, pos in enumerate(order):
        ref = base[pos]; h = ref_h(pool, ref, device); target = forced_base[t] if forced_base is not None else None
        if target is None: target, kn = free_target(free_alloc, key, h, occupied, device)
        else: kn = key(h[None])[0]
        kstate[target] = kn; vstate = write_v_slot(core, vstate[None], h[None], torch.tensor([target], device=device))[0]; occupied[target] = 1; slot_refs[target] = ref; base_slots.append(target)
    before_update = None; update_records = []
    for ei, ref in enumerate(events):
        h = ref_h(pool, ref, device); kn = key(h[None])[0]; true_update = ref["key"] in {x["key"] for x in slot_refs if x is not None}; true_target = next((i for i, x in enumerate(slot_refs) if x is not None and x["key"] == ref["key"]), -1)
        old_ref = slot_refs[true_target] if true_target >= 0 else None
        if mode == "oracle":
            pred_update = true_update; target = true_target if true_update else int(free_alloc(kn[None], occupied[None]).masked_fill(occupied[None].bool(), -1e9).argmax(1).item()); prob = 1.0 if pred_update else 0.0
        elif mode == "append":
            pred_update = False; target = int(free_alloc(kn[None], occupied[None]).masked_fill(occupied[None].bool(), -1e9).argmax(1).item()) if (occupied < .5).any() else -1; prob = 0.0
        else:
            if force_update_target is not None and ei == 0: pred_update, target, prob, _ = route_one(router, key, free_alloc, kn, kstate, occupied, threshold, device, force_target=force_update_target)
            else: pred_update, target, prob, _ = route_one(router, key, free_alloc, kn, kstate, occupied, threshold, device)
        pre_k = kstate.clone(); pre_v = vstate.clone(); valid_new = target >= 0 and occupied[target].item() < .5
        if target >= 0:
            if not pred_update: kstate[target] = kn
            vstate = write_v_slot(core, vstate[None], h[None], torch.tensor([target], device=device))[0]
            occupied[target] = 1; slot_refs[target] = ref
        duplicate = len([x["key"] for x in slot_refs if x is not None]) != len({x["key"] for x in slot_refs if x is not None})
        if true_update and old_ref is not None:
            untouched_k = [float((kstate[j] - pre_k[j]).norm().item()) for j in range(SLOTS) if j != target]
            untouched_v = [float((vstate[j] - pre_v[j]).norm().item()) for j in range(SLOTS) if j != target]
        else:
            untouched_k = []; untouched_v = []
        update_records.append({"true_update": true_update, "pred_update": pred_update, "true_target": true_target, "pred_target": target, "prob_update": prob, "valid_new": valid_new, "duplicate": duplicate, "old_ref": old_ref, "new_ref": ref, "untouched_k_drift": untouched_k, "untouched_v_drift": untouched_v})
    current = [(i, r) for i, r in enumerate(slot_refs) if r is not None]; ret = []; ans = []; current_answer_by_slot = {}; old_probs = []; latest_flags = []
    for slot, ref in current:
        q = query(ref_q(pool, ref, device)[None])[0]; sel = (q[None] * kstate).sum(-1).argmax().item(); pred = vh.selected(vstate[None], torch.tensor([sel], device=device)).argmax(1).item(); ret.append(sel == slot); ans.append(pred == ref["value"]); current_answer_by_slot[slot] = bool(pred == ref["value"])
    for rec in update_records:
        if rec["true_update"] and rec["pred_target"] >= 0:
            old = rec["old_ref"]["value"]; new = rec["new_ref"]["value"]; logits = vh.selected(vstate[None], torch.tensor([rec["pred_target"]], device=device))[0]; probs = logits.softmax(-1); old_probs.append(float(probs[old].item())); latest_flags.append(int(logits.argmax().item() == new))
    true_update_records = [u for u in update_records if u["true_update"]]
    final_update_latest = None
    if true_update_records:
        last_ref = true_update_records[-1]["new_ref"]; last_slot = next((i for i, x in enumerate(slot_refs) if x is not None and x["key"] == last_ref["key"]), None); final_update_latest = current_answer_by_slot.get(last_slot) if last_slot is not None else None
    return {"base_slots": base_slots, "updates": update_records, "current_count": len(current), "retrieval": ret, "answer": ans, "all_retrieved": bool(ret) and all(ret), "all_answered": bool(ans) and all(ans), "old_value_prob": old_probs, "latest_flags": latest_flags, "final_update_latest": final_update_latest, "final_k": kstate.detach().cpu().numpy(), "final_v": vstate.detach().cpu().numpy(), "slot_refs": slot_refs}


def aggregate_runs(runs):
    n = len(runs); updates = [u for r in runs for u in r["updates"]]; true_updates = [u for u in updates if u["true_update"]]; true_news = [u for u in updates if not u["true_update"]]
    return {"episodes": n, "retrieval_accuracy": float(np.mean([x for r in runs for x in r["retrieval"]])) if runs else None, "answer_accuracy": float(np.mean([x for r in runs for x in r["answer"]])) if runs else None, "all_facts_retained_rate": float(np.mean([r["all_retrieved"] for r in runs])) if runs else None, "all_facts_answered_rate": float(np.mean([r["all_answered"] for r in runs])) if runs else None, "action_accuracy": float(np.mean([u["pred_update"] == u["true_update"] for u in updates])) if updates else None, "update_target_accuracy": float(np.mean([u["pred_target"] == u["true_target"] for u in true_updates])) if true_updates else None, "new_valid_empty_rate": float(np.mean([u["valid_new"] for u in true_news])) if true_news else None, "new_collision_rate": float(1 - np.mean([u["valid_new"] for u in true_news])) if true_news else None, "duplicate_key_rate": float(np.mean([u["duplicate"] for u in updates])) if updates else None, "latest_value_accuracy": float(np.mean([r["final_update_latest"] for r in runs if r["final_update_latest"] is not None])) if any(r["final_update_latest"] is not None for r in runs) else None, "old_value_probability": float(np.mean([x for r in runs for x in r["old_value_prob"]])) if true_updates and any(r["old_value_prob"] for r in runs) else None, "untouched_k_drift": float(np.mean([x for u in true_updates for x in u["untouched_k_drift"]])) if any(u["untouched_k_drift"] for u in true_updates) else 0.0, "untouched_v_drift": float(np.mean([x for u in true_updates for x in u["untouched_v_drift"]])) if any(u["untouched_v_drift"] for u in true_updates) else 0.0}


def run_set(examples, pool, key, query, core, vh, fa, router, threshold, device, mode="learned", base_order=None, permute=False, force_wrong=False):
    runs = []; perm = [2, 0, 3, 1] if permute else None
    for ex in examples:
        forced_base = [perm[i] for i in range(len(ex["base"]))] if permute else None
        wrong = None
        if force_wrong:
            # Force the first update into a different occupied slot.
            true_slot = next((i for i, x in enumerate(ex["base"]) if x["key"] == ex["incoming"]["key"]), 0); wrong = (true_slot + 1) % max(len(ex["base"]), 2)
        this_order = (list(reversed(range(len(ex["base"])))) if base_order is not None else None)
        runs.append(run_episode(ex, pool, key, query, core, vh, fa, router, threshold, device, mode, base_order=this_order, forced_base=forced_base, force_update_target=wrong))
    return runs


def category_summary(router, key, examples, pool, device, threshold):
    raw = router_eval(router, key, examples, pool, device, threshold); out = {}
    for cat in sorted(set(raw["categories"])):
        m = np.asarray([x == cat for x in raw["categories"]]); out[cat] = {"n": int(m.sum()), "action_accuracy": float((raw["pred"][m] == raw["y"][m]).mean()), "update_or_new_target": float(raw["target_ok"][m].mean()) if (raw["y"][m] == 1).any() else None}
    return out


def duplicate_key_diagnostic(ex, pool, key, query, core, vh, device):
    r0, r1 = ex["base"][0], ex["incoming"]; h0, h1 = ref_h(pool, r0, device), ref_h(pool, r1, device); k0, k1 = key(h0[None])[0], key(h1[None])[0]; vs = core.zero_state(1, device=device, dtype=core.content_read.weight.dtype); vs = write_v_slot(core, vs, h0[None], torch.tensor([0], device=device)); vs = write_v_slot(core, vs, h1[None], torch.tensor([2], device=device)); ks = torch.zeros(1, SLOTS, DIM, device=device); ks[0, 0] = k0; ks[0, 2] = k0; q = query(ref_q(pool, r0, device)[None]); chosen = (q[:, None, :] * ks).sum(-1).argmax(1); pred = vh.selected(vs, chosen).argmax(1)
    return {"duplicate_slots": [0, 2], "key_cosine": float((k0 * k1).sum().item()), "read_selected_slot": int(chosen.item()), "answer": int(pred.item()), "value_slot0": r0["value"], "value_slot2": r1["value"]}


def main():
    seed_all(); t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(CACHE_ROOT); key, query, core, vh, fa = load_all(device)
    train_n, eval_n = (256, 128) if SMOKE else (6000, 1000); steps = min(STEPS, 20) if SMOKE else STEPS
    pools = {s: FactPool(cache, s) for s in ("train", "val", "testA", "testB")}
    train = make_examples(pools["train"], train_n, SEED + 1, True); val = make_examples(pools["val"], eval_n, SEED + 2, True); testA = make_examples(pools["testA"], eval_n, SEED + 3, True); testB = make_examples(pools["testB"], eval_n, SEED + 4, True)
    hardA = make_examples(pools["testA"], eval_n, SEED + 5, False, True); hardB = make_examples(pools["testB"], eval_n, SEED + 6, False, True)
    eval_only = os.environ.get("V24_O3B_EVAL_ONLY", "0") == "1"
    if eval_only and (OUT / "router_best.pt").exists():
        train_record = json.loads((OUT / "train.log").read_text(encoding="utf-8"))
    else:
        train_record = train_router(train, val, pools["train"], pools["val"], key, device, steps, SMOKE)
    router = UpdateRouter().to(device); rck = torch.load(OUT / "router_best.pt", map_location=device, weights_only=False); router.load_state_dict(rck["router"], strict=True); router.eval(); threshold = float(rck["threshold"])
    det = {s: router_eval(router, key, ex, pools[s], device, threshold) for s, ex in (("val", val), ("testA", testA), ("testB", testB))}
    det_clean = {s: {k: v for k, v in m.items() if k not in ("y", "score", "pred", "target_ok", "categories")} for s, m in det.items()}
    confusion = {s: det[s]["confusion"] for s in det}; hard_neg = {"testA": category_summary(router, key, hardA, pools["testA"], device, threshold), "testB": category_summary(router, key, hardB, pools["testB"], device, threshold)}
    # Oracle and learned full one-event pipeline.
    oracle = {}; learned = {}; append = {}
    for s, exs in (("testA", testA), ("testB", testB)):
        oracle[s] = aggregate_runs(run_set(exs, pools[s], key, query, core, vh, fa, router, threshold, device, "oracle"))
        learned[s] = aggregate_runs(run_set(exs, pools[s], key, query, core, vh, fa, router, threshold, device, "learned"))
        append[s] = aggregate_runs(run_set(exs, pools[s], key, query, core, vh, fa, router, threshold, device, "append"))
    # O3a NEW-only regression and routing stresses.
    newA = [x for x in testA if x["action"] == 0]; newB = [x for x in testB if x["action"] == 0]
    new_results = {"testA": aggregate_runs(run_set(newA, pools["testA"], key, query, core, vh, fa, router, threshold, device, "learned")), "testB": aggregate_runs(run_set(newB, pools["testB"], key, query, core, vh, fa, router, threshold, device, "learned"))}
    permA = aggregate_runs(run_set(testA, pools["testA"], key, query, core, vh, fa, router, threshold, device, "learned", permute=True)); baseA = learned["testA"]
    orderA = aggregate_runs(run_set(testA, pools["testA"], key, query, core, vh, fa, router, threshold, device, "learned", base_order=[3, 2, 1, 0]))
    # Force the first update into a wrong occupied target only for update cases.
    upA = [x for x in testA if x["action"] == 1]; wrong_runs = run_set(upA, pools["testA"], key, query, core, vh, fa, router, threshold, device, "learned", force_wrong=True); wrong_summary = aggregate_runs(wrong_runs)
    # Full memory same-key update: all four slots occupied before the update.
    full_examples = []
    full_eps = [ep for ep in pools["testA"].episodes if len(ep) == 4]
    for _ in range(eval_n):
        base_full = list(full_eps[_ % len(full_eps)])
        incoming_full = pools["testA"].update_candidate(base_full[0], np.random.default_rng(SEED + 9000 + _))
        if incoming_full is None:
            incoming_full = base_full[0]
        full_examples.append({"base": base_full, "incoming": incoming_full, "action": 1, "target": 0, "category": "full_memory_update", "nbase": 4})
    full_memory = {"oracle": aggregate_runs(run_set(full_examples, pools["testA"], key, query, core, vh, fa, router, threshold, device, "oracle")), "learned": aggregate_runs(run_set(full_examples, pools["testA"], key, query, core, vh, fa, router, threshold, device, "learned")), "append": aggregate_runs(run_set(full_examples, pools["testA"], key, query, core, vh, fa, router, threshold, device, "append"))}
    # Multiple updates to one semantic key.
    multi = {}
    for count in (1, 2, 4, 8):
        rr = None
        for vals in pools["testA"].by_key.values():
            if len({x["value"] for x in vals}) >= count + 1: rr = vals; break
        if rr is None: continue
        runs = []
        for i in range(eval_n):
            events = [rr[(i + j + 1) % len(rr)] for j in range(count)]
            ex = {"base": [rr[0]], "events": events, "incoming": events[0], "action": 1, "target": 0, "category": "multi_update", "nbase": 1}
            runs.append(run_episode(ex, pools["testA"], key, query, core, vh, fa, router, threshold, device, "learned"))
        multi[str(count)] = aggregate_runs(runs)
    # Duplicate-key read diagnostic.
    dup_ex = next((x for x in testA if x["action"] == 1), testA[0]); duplicate = duplicate_key_diagnostic(dup_ex, pools["testA"], key, query, core, vh, device)
    # Empty-slot legal placement invariance from O3a style controlled routes.
    inv = {"note": "O3b keeps existing K and changes only V on UPDATE; legal empty-slot invariance is inherited from O3a and rechecked in new_results."}
    latest = {"one_update_testA": learned["testA"], "one_update_testB": learned["testB"], "multi_update": multi}
    old_suppression = {"testA": {"old_value_probability": learned["testA"]["old_value_probability"], "latest_value_accuracy": learned["testA"]["latest_value_accuracy"]}, "testB": {"old_value_probability": learned["testB"]["old_value_probability"], "latest_value_accuracy": learned["testB"]["latest_value_accuracy"]}, "multi_update": {k: {"old_value_probability": v["old_value_probability"], "latest_value_accuracy": v["latest_value_accuracy"]} for k, v in multi.items()}}
    untouched = {"testA": {"K_drift": learned["testA"]["untouched_k_drift"], "V_drift": learned["testA"]["untouched_v_drift"]}, "testB": {"K_drift": learned["testB"]["untouched_k_drift"], "V_drift": learned["testB"]["untouched_v_drift"]}}
    slot_perm = {"normal": baseA, "permuted": permA, "update_target_drop_pp": 100 * (baseA["update_target_accuracy"] - permA["update_target_accuracy"]), "retrieval_drop_pp": 100 * (baseA["retrieval_accuracy"] - permA["retrieval_accuracy"])}
    order_perm = {"normal": baseA, "reversed": orderA, "update_target_drop_pp": 100 * (baseA["update_target_accuracy"] - orderA["update_target_accuracy"]), "retrieval_drop_pp": 100 * (baseA["retrieval_accuracy"] - orderA["retrieval_accuracy"])}
    action_ok = min(det["testA"]["binary"]["accuracy"], det["testB"]["binary"]["accuracy"]) >= .95; update_target_ok = min(det["testA"]["update_target_accuracy_on_true_update"], det["testB"]["update_target_accuracy_on_true_update"]) >= .95; new_ok = max(new_results["testA"]["new_collision_rate"], new_results["testB"]["new_collision_rate"]) <= .01; latest_ok = min(learned["testA"]["latest_value_accuracy"], learned["testB"]["latest_value_accuracy"]) >= .90; ret_ok = min(learned["testA"]["retrieval_accuracy"], learned["testB"]["retrieval_accuracy"]) >= .95; ans_ok = learned["testA"]["answer_accuracy"] >= .90 and learned["testB"]["answer_accuracy"] >= .85; perm_ok = slot_perm["update_target_drop_pp"] <= 5 and order_perm["update_target_drop_pp"] <= 5
    if SMOKE: verdict = "SMOKE ONLY: NOT FOR SCIENTIFIC VERDICT"
    elif action_ok and update_target_ok and new_ok and latest_ok and ret_ok and ans_ok and perm_ok: verdict = "O3b PASS: LEARNED SAME-KEY UPDATE"
    elif action_ok and update_target_ok and not latest_ok: verdict = "O3b PARTIAL: UPDATE WORKS, NEW/UPDATE DISCRIMINATION INCOMPLETE"
    elif action_ok and not update_target_ok: verdict = "O3b PARTIAL: SAME-KEY DETECTION WORKS, UPDATE ROUTING INCOMPLETE"
    elif not action_ok: verdict = "O3b FAIL: SAME-KEY DETECTION"
    elif not update_target_ok: verdict = "O3b FAIL: UPDATE TARGET SELECTION"
    elif not new_ok: verdict = "O3b FAIL: NEW ALLOCATION REGRESSION"
    else: verdict = "O3b FAIL: UPDATE WRITE REGRESSION"
    cfg = {"phase": "CLR-v2.4-O3b", "seed": SEED, "device": str(device), "source_o21b": str(O21B), "source_vgen2": str(VGEN2), "source_o3a_allocator": str(O3A_ALLOC), "cache": str(CACHE_ROOT), "trainable": ["UpdateRouter action head", "UpdateRouter pairwise target head"], "frozen": ["learned KeyEncoder", "V/core/GRU", "QueryEncoder", "read router", "decoder", "answer head", "Ouro", "O3a free-slot allocator"], "router_inputs": "K_new + existing K_0..K_3 + explicit occupancy", "value_input_used": False, "loss": "action CE + update-target CE + 0.5 pairwise BCE (NEW occupied pairs are negatives); NEW uses frozen O3a set-valued allocator", "steps": steps, "eval_every": EVAL_EVERY, "best_step": train_record["best_step"], "threshold": threshold, "params": train_record["params"], "data": {"train": len(train), "val": len(val), "testA": len(testA), "testB": len(testB), "new_update_ratio": "50/50", "hard_negative_categories": ["same_entity_diff_relation", "same_relation_diff_entity", "fully_different", "same_value_diff_key"]}}
    json_dump(OUT / "config.json", cfg); json_dump(OUT / "train.log", train_record); json_dump(OUT / "metrics.json", {"verdict": verdict, "threshold": threshold, "testA": learned["testA"], "testB": learned["testB"], "oracle": oracle, "always_append": append})
    json_dump(OUT / "oracle_update_baseline.json", oracle); json_dump(OUT / "always_append_baseline.json", append); json_dump(OUT / "learned_update_results.json", learned); json_dump(OUT / "action_confusion.json", confusion); json_dump(OUT / "same_key_detection.json", det_clean); json_dump(OUT / "update_target_results.json", {"testA": det_clean["testA"], "testB": det_clean["testB"], "slot_permutation": slot_perm, "fact_order_permutation": order_perm}); json_dump(OUT / "new_allocation_results.json", new_results); json_dump(OUT / "duplicate_key_results.json", {"learned": {"testA": learned["testA"]["duplicate_key_rate"], "testB": learned["testB"]["duplicate_key_rate"]}, "diagnostic": duplicate}); json_dump(OUT / "latest_value_results.json", latest); json_dump(OUT / "old_value_suppression.json", old_suppression); json_dump(OUT / "untouched_slot_drift.json", untouched); json_dump(OUT / "full_memory_update.json", full_memory); json_dump(OUT / "multi_update_results.json", multi); json_dump(OUT / "slot_permutation.json", slot_perm); json_dump(OUT / "fact_order_permutation.json", order_perm); json_dump(OUT / "hard_negative_results.json", hard_neg); json_dump(OUT / "wrong_update_intervention.json", wrong_summary); json_dump(OUT / "full_pipeline_results.json", {"learned": learned, "oracle": oracle, "always_append": append}); json_dump(OUT / "empty_slot_invariance.json", inv)
    report = ["# CLR-v2.4-O3b — Same-Key Update / Semantic Write Routing", "", f"**{verdict}**", "", "## Primary result", f"Learned same-key detection: TestA F1 {100*det['testA']['binary']['f1_update']:.2f}%, recall {100*det['testA']['binary']['recall_update']:.2f}%; TestB F1 {100*det['testB']['binary']['f1_update']:.2f}%, recall {100*det['testB']['binary']['recall_update']:.2f}%.", f"Update target accuracy: TestA {100*det['testA']['update_target_accuracy_on_true_update']:.2f}%, TestB {100*det['testB']['update_target_accuracy_on_true_update']:.2f}%.", f"Full pipeline retrieval: TestA {100*learned['testA']['retrieval_accuracy']:.2f}%, TestB {100*learned['testB']['retrieval_accuracy']:.2f}%; answer: TestA {100*learned['testA']['answer_accuracy']:.2f}%, TestB {100*learned['testB']['answer_accuracy']:.2f}%.", f"NEW collision: TestA {100*new_results['testA']['new_collision_rate']:.2f}%, TestB {100*new_results['testB']['new_collision_rate']:.2f}%; duplicate-key rate: TestA {100*learned['testA']['duplicate_key_rate']:.2f}%, TestB {100*learned['testB']['duplicate_key_rate']:.2f}%.", "", "## Controls", f"Oracle update baseline: TestA {oracle['testA']['retrieval_accuracy']:.3f} retrieval / {oracle['testA']['answer_accuracy']:.3f} answer; TestB {oracle['testB']['retrieval_accuracy']:.3f} / {oracle['testB']['answer_accuracy']:.3f}.", f"Full-memory same-key update learned retrieval/answer: {full_memory['learned']['retrieval_accuracy']:.3f} / {full_memory['learned']['answer_accuracy']:.3f}.", f"Wrong update target intervention: retrieval {wrong_summary['retrieval_accuracy']:.3f}, latest answer {wrong_summary['answer_accuracy']:.3f}.", f"Slot permutation update-target drop: {slot_perm['update_target_drop_pp']:.2f}pp; fact-order drop: {order_perm['update_target_drop_pp']:.2f}pp.", "", "## K/V boundary", "UpdateRouter reads K_new, existing K, and occupancy only; V is not an input. UPDATE keeps the existing K and writes the new V. NEW uses the frozen O3a allocator.", "", "## Boundary", "This phase does not test unseen-new-key overflow, eviction, replacement, capacity-full new-key management, or reasoning.", "", "## Stop", "O3b complete; do not automatically implement O3c."]
    (OUT / "v24_o3b_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "threshold": threshold, "testA_action_f1": det["testA"]["binary"]["f1_update"], "testB_action_f1": det["testB"]["binary"]["f1_update"], "testA_update_target": det["testA"]["update_target_accuracy_on_true_update"], "testB_update_target": det["testB"]["update_target_accuracy_on_true_update"], "testA_retrieval": learned["testA"]["retrieval_accuracy"], "testB_retrieval": learned["testB"]["retrieval_accuracy"], "testA_answer": learned["testA"]["answer_accuracy"], "testB_answer": learned["testB"]["answer_accuracy"], "wall_clock_s": time.perf_counter() - t0}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
