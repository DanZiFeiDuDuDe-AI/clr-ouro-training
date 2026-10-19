# -*- coding: utf-8 -*-
"""CLR-v2.4-O3a — learned free-slot allocation.

O2.1b K and the V/decoder path are frozen.  The only trainable component is
an allocator which receives an explicit occupancy mask and emits a slot
distribution.  The primary condition is soft occupancy awareness: occupied
slots are not masked and the loss only rewards probability mass on any empty
slot.  A hard-mask condition is reported separately as a safety control.
"""
import copy
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v24_o21a import KeyEncoder, QueryEncoder
from a1_v24_o21a_mr import MRCache, MRValueHeads
from a1_v24_vgen import load_core
from a1_v24_o2_common import json_dump


ROOT = Path(r"D:/training")
OUT = Path(os.environ.get("V24_O3A_OUT", r"D:/training/checkpoints/clr_v24_o3a_allocation"))
CACHE_ROOT = Path(os.environ.get("V24_O3A_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
O21B = Path(os.environ.get("V24_O3A_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
SUPERVISED = Path(os.environ.get("V24_O3A_SUPERVISED", r"D:/training/checkpoints/clr_v24_o21a_mr/best_model.pt"))
VGEN2 = Path(os.environ.get("V24_O3A_VGEN2", r"D:/training/checkpoints/clr_v24_vgen2/best.pt"))
SLOTS = 4
DIM = 64
CAPS = (1, 2, 4)
SEED = int(os.environ.get("V24_O3A_SEED", "42"))
STEPS = int(os.environ.get("V24_O3A_STEPS", "3000"))
EVAL_EVERY = int(os.environ.get("V24_O3A_EVAL_EVERY", "100"))
DEVICE_NAME = os.environ.get("V24_O3A_DEVICE", "cuda")
SMOKE = os.environ.get("V24_O3A_SMOKE", "0") == "1" or "--smoke" in os.sys.argv
TEMP = 0.1


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Allocator(nn.Module):
    def __init__(self, k_aware=True):
        super().__init__()
        self.k_aware = bool(k_aware)
        inp = DIM + SLOTS if self.k_aware else SLOTS
        self.net = nn.Sequential(
            nn.Linear(inp, 128 if self.k_aware else 64),
            nn.GELU(),
            nn.Linear(128 if self.k_aware else 64, SLOTS),
        )

    def forward(self, k, occupied):
        x = torch.cat([k, occupied.float()], dim=-1) if self.k_aware else occupied.float()
        return self.net(x)


def load_frozen(device):
    kck = torch.load(O21B, map_location="cpu", weights_only=False)
    key = KeyEncoder(DIM).to(device)
    key.load_state_dict(kck["key"], strict=True)
    key.eval()
    for p in key.parameters():
        p.requires_grad_(False)

    qck = torch.load(SUPERVISED, map_location="cpu", weights_only=False)
    query = QueryEncoder(DIM).to(device)
    query.load_state_dict(qck["query"], strict=True)
    query.eval()
    for p in query.parameters():
        p.requires_grad_(False)

    vcore, _, _ = load_core(device, trainable=False)
    vck = torch.load(VGEN2, map_location="cpu", weights_only=False)
    vcore.load_state_dict(vck["core"], strict=True)
    vcore.eval()
    vh = MRValueHeads().to(device)
    vh.load_state_dict(vck["value_heads"], strict=True)
    vh.eval()
    for p in vcore.parameters():
        p.requires_grad_(False)
    for p in vh.parameters():
        p.requires_grad_(False)
    return key, query, vcore, vh


def rows_for(cache, split, cap, limit=None):
    rows = cache.rows(split, cap)
    if limit is not None:
        rows = rows[:min(int(limit), len(rows))]
    return rows


def sample_batch(cache, split, cap, rng, batch_size):
    rows = cache.rows(split, cap)
    rb = rows[rng.integers(0, len(rows), size=min(batch_size, len(rows)))]
    facts, _, _, _, _, _, _ = cache.batch(split, rb, cap, next(_DEVICE))
    return facts


def random_initial_occupancy(batch, max_occupied, rng, device):
    """Augmentation for allocator training; never exceeds free capacity."""
    occ = np.zeros((batch, SLOTS), dtype=np.float32)
    for i in range(batch):
        n = int(rng.integers(0, max_occupied + 1)) if max_occupied else 0
        if n:
            occ[i, rng.choice(SLOTS, size=n, replace=False)] = 1.0
    return torch.from_numpy(occ).to(device)


def allocator_sequence(allocator, key, facts, initial_occ=None, hard_mask=False,
                       order=None, slot_perm=None, policy="learned", rng=None):
    """Select slots for a batch of fact sequences without writing state."""
    b, n, _ = facts.shape
    if initial_occ is None:
        occupied = torch.zeros(b, SLOTS, device=facts.device)
    else:
        occupied = initial_occ.clone().float()
    actions = []
    valid = []
    if order is None:
        order = list(range(n))
    perm = None if slot_perm is None else torch.as_tensor(slot_perm, dtype=torch.long, device=facts.device)
    for orig_pos in order:
        k = key(facts[:, orig_pos, :])
        view_occ = occupied[:, perm] if perm is not None else occupied
        if policy == "first":
            order_idx = torch.arange(SLOTS, device=facts.device).expand(b, -1)
            sel_view = order_idx.masked_fill(view_occ.bool(), SLOTS + 1).argmin(1)
        elif policy == "random":
            sel_view = []
            for i in range(b):
                free = torch.where(view_occ[i] < 0.5)[0].cpu().numpy()
                sel_view.append(int(rng.choice(free)))
            sel_view = torch.as_tensor(sel_view, dtype=torch.long, device=facts.device)
        else:
            logits = allocator(k, view_occ)
            if hard_mask:
                logits = logits.masked_fill(view_occ.bool(), -1e9)
            sel_view = logits.argmax(1)
        sel = perm[sel_view] if perm is not None else sel_view
        ok = (occupied.gather(1, sel[:, None]).squeeze(1) < 0.5)
        valid.append(ok)
        actions.append(sel)
        occupied = occupied.scatter(1, sel[:, None], torch.ones(b, 1, device=facts.device))
    return torch.stack(actions, 1), torch.stack(valid, 1), occupied


def train_allocator(cache, key, device, k_aware, seed_offset, steps, smoke=False):
    allocator = Allocator(k_aware=k_aware).to(device)
    opt = torch.optim.AdamW(allocator.parameters(), lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(SEED + seed_offset)
    logs = []
    best = -1.0
    best_step = 0
    t0 = time.perf_counter()
    batch_size = 256 if smoke else 128
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        cap = int(rng.choice(CAPS))
        rows = cache.rows("train", cap)
        rb = rows[rng.integers(0, len(rows), size=min(batch_size, len(rows)))]
        facts, _, _, _, _, _, _ = cache.batch("train", rb, cap, device)
        # A mixture of empty-start and legal arbitrary occupancy teaches the
        # explicit occupancy interface without introducing slot targets.
        init = random_initial_occupancy(len(rb), SLOTS - cap, rng, device) if rng.random() < 0.5 else torch.zeros(len(rb), SLOTS, device=device)
        occupied = init
        losses = []
        collisions = []
        for t in range(cap):
            with torch.no_grad():
                kt = key(facts[:, t, :])
            logits = allocator(kt, occupied)
            p = logits.softmax(-1)
            empty_mass = (p * (1.0 - occupied)).sum(-1).clamp_min(1e-8)
            losses.append(-empty_mass.log().mean())
            sel = logits.detach().argmax(1)
            collisions.append((occupied.gather(1, sel[:, None]).squeeze(1) > 0.5).float().mean())
            occupied = occupied.scatter(1, sel[:, None], torch.ones(len(rb), 1, device=device))
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(allocator.parameters(), 1.0)
        opt.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == steps:
            val = allocation_eval(allocator, key, cache, "val", device, hard_mask=False, k_aware=k_aware, limit=(128 if smoke else None))
            macro = float(np.mean([val[str(c)]["valid_rate"] for c in CAPS]))
            rec = {"step": step, "loss_valid": float(loss.detach()), "train_collision_rate": float(torch.stack(collisions).mean()), "val_macro_valid": macro, "val": val}
            logs.append(rec)
            if macro > best:
                best = macro
                best_step = step
                torch.save({"allocator": copy.deepcopy(allocator.state_dict()), "k_aware": k_aware, "step": step, "val_macro_valid": macro, "seed": SEED}, OUT / ("allocator_kaware_best.pt" if k_aware else "allocator_kblind_best.pt"))
    final_name = "allocator_kaware_final.pt" if k_aware else "allocator_kblind_final.pt"
    torch.save({"allocator": allocator.state_dict(), "k_aware": k_aware, "step": steps, "seed": SEED}, OUT / final_name)
    peak = None
    if torch.cuda.is_available():
        peak = {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
    return {"k_aware": k_aware, "steps": steps, "best_step": best_step, "best_val_macro_valid": best, "wall_clock_s": time.perf_counter() - t0, "params": sum(p.numel() for p in allocator.parameters()), "peak_vram": peak, "logs": logs}


@torch.no_grad()
def allocation_eval(allocator, key, cache, split, device, hard_mask, k_aware, limit=None):
    allocator.eval(); out = {}
    for cap in CAPS:
        rows = rows_for(cache, split, cap, limit)
        total = valid = collisions = 0
        step_valid = np.zeros(cap, dtype=np.int64)
        selected_freq = np.zeros(SLOTS, dtype=np.int64)
        for st in range(0, len(rows), 128):
            rb = rows[st:st + 128]
            facts, _, _, _, _, _, _ = cache.batch(split, rb, cap, device)
            sel, ok, _ = allocator_sequence(allocator, key, facts, hard_mask=hard_mask, policy="learned")
            valid += int(ok.sum()); total += int(ok.numel()); collisions += int((~ok).sum())
            step_valid += ok.sum(0).cpu().numpy(); selected_freq += np.bincount(sel.cpu().numpy().reshape(-1), minlength=SLOTS)
        out[str(cap)] = {"n_episodes": int(len(rows)), "writes": total, "valid": valid, "collisions": collisions, "valid_rate": valid / max(total, 1), "collision_rate": collisions / max(total, 1), "per_step_valid_rate": (step_valid / max(len(rows), 1)).tolist(), "selected_frequency": selected_freq.tolist()}
    out["k_aware"] = k_aware; out["hard_mask"] = hard_mask
    return out


def write_v_slot(core, state, h, sel):
    z = core.content_read(h.to(device=state.device, dtype=state.dtype))
    candidates = torch.stack([core.shared_gru(z, state[:, j, :]) for j in range(SLOTS)], dim=1)
    b = torch.arange(len(sel), device=state.device)
    new_state = state.clone()
    new_state[b, sel] = candidates[b, sel]
    return new_state


@torch.no_grad()
def pipeline_batch(allocator, key, query, core, vh, cache, split, rows, cap, device, policy="learned", hard_mask=False, order=None, slot_perm=None, force_slots=None, rng=None):
    rows = np.asarray(rows, dtype=np.int64)
    facts, _, _, _, _, _, fy = cache.batch(split, rows, cap, device)
    b = len(rows)
    if order is None: order = list(range(cap))
    kfacts = key(facts.reshape(-1, 2048)).reshape(b, cap, DIM)
    kstate = torch.zeros(b, SLOTS, DIM, device=device)
    vstate = core.zero_state(b, device=device, dtype=core.content_read.weight.dtype)
    occupied = torch.zeros(b, SLOTS, device=device)
    assignments = torch.full((b, cap), -1, dtype=torch.long, device=device)
    alloc_valid = []; selected_steps = []
    perm = None if slot_perm is None else torch.as_tensor(slot_perm, dtype=torch.long, device=device)
    for t, orig_pos in enumerate(order):
        kt = kfacts[:, orig_pos, :]
        view_occ = occupied[:, perm] if perm is not None else occupied
        if force_slots is not None and force_slots[t] is not None:
            raw = force_slots[t]
            sel = torch.full((b,), int(raw), dtype=torch.long, device=device) if np.isscalar(raw) else torch.as_tensor(raw, dtype=torch.long, device=device)
        elif policy == "first":
            order_idx = torch.arange(SLOTS, device=device).expand(b, -1)
            sel_view = order_idx.masked_fill(view_occ.bool(), SLOTS + 1).argmin(1)
            sel = perm[sel_view] if perm is not None else sel_view
        elif policy == "random":
            vals = []
            for i in range(b):
                free = torch.where(view_occ[i] < 0.5)[0].cpu().numpy(); vals.append(int(rng.choice(free)))
            sel_view = torch.as_tensor(vals, dtype=torch.long, device=device); sel = perm[sel_view] if perm is not None else sel_view
        else:
            logits = allocator(kt, view_occ)
            if hard_mask: logits = logits.masked_fill(view_occ.bool(), -1e9)
            sel_view = logits.argmax(1); sel = perm[sel_view] if perm is not None else sel_view
        ok = occupied.gather(1, sel[:, None]).squeeze(1) < 0.5
        alloc_valid.append(ok); selected_steps.append(sel)
        kstate[torch.arange(b, device=device), sel] = kt
        vstate = write_v_slot(core, vstate, facts[:, orig_pos, :], sel)
        occupied = occupied.scatter(1, sel[:, None], torch.ones(b, 1, device=device))
        assignments[:, orig_pos] = sel
    ret = []; ans = []
    for p in range(cap):
        qh = cache.fact_query_batch(split, rows, np.full(b, p, dtype=np.int64), device)
        q = query(qh); scores = (q[:, None, :] * kstate).sum(-1) / TEMP; chosen = scores.argmax(1)
        target = assignments[:, p]
        pred = vh.selected(vstate, chosen).argmax(1)
        ret.append((chosen == target).cpu().numpy()); ans.append((pred == fy[:, p]).cpu().numpy())
    ret_arr = np.stack(ret, 1); ans_arr = np.stack(ans, 1); valid_arr = torch.stack(alloc_valid, 1).cpu().numpy()
    return {"alloc_valid": valid_arr, "assignments": torch.stack(selected_steps, 1).cpu().numpy(), "retrieval": ret_arr, "answer": ans_arr, "all_retrieved": ret_arr.all(1), "all_answered": ans_arr.all(1)}


def pipeline_eval(allocator, key, query, core, vh, cache, split, cap, device, policy="learned", hard_mask=False, limit=None, order=None, slot_perm=None, force_slots=None, seed_offset=0):
    rows = rows_for(cache, split, cap, limit); rng = np.random.default_rng(SEED + seed_offset)
    chunks = []
    for st in range(0, len(rows), 64):
        chunks.append(pipeline_batch(allocator, key, query, core, vh, cache, split, rows[st:st + 64], cap, device, policy, hard_mask, order, slot_perm, force_slots, rng))
    if not chunks: return {"episodes": 0}
    keys = ("alloc_valid", "retrieval", "answer", "all_retrieved", "all_answered")
    d = {k: np.concatenate([x[k] for x in chunks], axis=0) for k in keys}
    return {"episodes": int(len(rows)), "writes": int(d["alloc_valid"].size), "valid_allocation_rate": float(d["alloc_valid"].mean()), "collision_rate": float(1.0 - d["alloc_valid"].mean()), "per_fact_retrieval": d["retrieval"].mean(0).tolist(), "per_fact_answer": d["answer"].mean(0).tolist(), "retrieval_accuracy": float(d["retrieval"].mean()), "answer_accuracy": float(d["answer"].mean()), "all_facts_retained_rate": float(d["all_retrieved"].mean()), "all_facts_answered_rate": float(d["all_answered"].mean()), "raw": d}


def strip_raw(d):
    return {k: v for k, v in d.items() if k != "raw"}


@torch.no_grad()
def initial_pattern_results(allocator, key, cache, device, k_aware, hard_mask=False, limit=64):
    rows = rows_for(cache, "testA", 1, limit); facts, _, _, _, _, _, _ = cache.batch("testA", rows, 1, device); out = {}
    for empty_count in range(1, SLOTS + 1):
        entries = []
        for empty in itertools.combinations(range(SLOTS), empty_count):
            occ = torch.ones(len(rows), SLOTS, device=device)
            occ[:, list(empty)] = 0
            sel, ok, _ = allocator_sequence(allocator, key, facts, occ, hard_mask=hard_mask, policy="learned")
            entries.append({"empty_slots": list(empty), "valid_rate": float(ok[:, 0].float().mean()), "selected_empty_rate": float((sel[:, 0].cpu().numpy() == np.asarray(empty)[0:1]).mean()) if empty_count == 1 else float(ok[:, 0].float().mean()), "selected_frequency": np.bincount(sel[:, 0].cpu().numpy(), minlength=SLOTS).tolist()})
        out[str(empty_count)] = entries
    return out


def summarize_train_logs(a, b):
    return {"k_aware": {k: v for k, v in a.items() if k != "logs"}, "k_blind": {k: v for k, v in b.items() if k != "logs"}}


def main():
    global _DEVICE
    seed_all()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    _DEVICE = [device]
    cache = MRCache(CACHE_ROOT)
    key, query, core, vh = load_frozen(device)
    steps = min(STEPS, 20) if SMOKE else STEPS
    limit = 128 if SMOKE else None
    # B1: K-aware soft occupancy learner; B2: K-blind control.
    train_aware = train_allocator(cache, key, device, True, 3001, steps, SMOKE)
    train_blind = train_allocator(cache, key, device, False, 4001, steps, SMOKE)
    aware = Allocator(True).to(device); blind = Allocator(False).to(device)
    ck_a = torch.load(OUT / "allocator_kaware_best.pt", map_location=device, weights_only=False); ck_b = torch.load(OUT / "allocator_kblind_best.pt", map_location=device, weights_only=False)
    aware.load_state_dict(ck_a["allocator"], strict=True); blind.load_state_dict(ck_b["allocator"], strict=True); aware.eval(); blind.eval()

    hard_alloc = allocation_eval(aware, key, cache, "testA", device, True, True, limit)
    soft_alloc_a = allocation_eval(aware, key, cache, "testA", device, False, True, limit)
    soft_alloc_b = allocation_eval(aware, key, cache, "testB", device, False, True, limit)
    blind_alloc = {s: allocation_eval(blind, key, cache, s, device, False, False, limit) for s in ("testA", "testB")}

    policies = {}
    for name, model, hard, policy in (("random_free", None, False, "random"), ("first_free", None, False, "first"), ("learned_soft", aware, False, "learned"), ("learned_hard_mask", aware, True, "learned"), ("k_blind_soft", blind, False, "learned")):
        policies[name] = {s: {str(c): strip_raw(pipeline_eval(model, key, query, core, vh, cache, s, c, device, policy=policy, hard_mask=hard, limit=limit, seed_offset=700 + c)) for c in CAPS} for s in ("testA", "testB")}

    pattern_results = {"k_aware_soft": initial_pattern_results(aware, key, cache, device, True, False, 64 if SMOKE else 256), "k_aware_hard": initial_pattern_results(aware, key, cache, device, True, True, 64 if SMOKE else 256), "k_blind_soft": initial_pattern_results(blind, key, cache, device, False, False, 64 if SMOKE else 256)}
    one_empty = {name: {s: {str(c): strip_raw(pipeline_eval(model, key, query, core, vh, cache, s, c, device, policy=policy, hard_mask=hard, limit=limit, seed_offset=900 + c)) for c in (1, 2, 4)} for s in ("testA", "testB")} for name, model, hard, policy in (("learned_soft", aware, False, "learned"), ("learned_hard_mask", aware, True, "learned"))}

    normal = pipeline_eval(aware, key, query, core, vh, cache, "testA", 4, device, limit=limit, seed_offset=1200)
    perm = pipeline_eval(aware, key, query, core, vh, cache, "testA", 4, device, limit=limit, slot_perm=[2, 0, 3, 1], seed_offset=1201)
    reversed_order = pipeline_eval(aware, key, query, core, vh, cache, "testA", 4, device, limit=limit, order=[3, 2, 1, 0], seed_offset=1202)
    slot_perm_result = {"normal": strip_raw(normal), "permuted": strip_raw(perm), "retrieval_drop_pp": 100 * (normal["retrieval_accuracy"] - perm["retrieval_accuracy"]), "answer_drop_pp": 100 * (normal["answer_accuracy"] - perm["answer_accuracy"])}
    order_result = {"normal": strip_raw(normal), "reversed": strip_raw(reversed_order), "retrieval_drop_pp": 100 * (normal["retrieval_accuracy"] - reversed_order["retrieval_accuracy"]), "answer_drop_pp": 100 * (normal["answer_accuracy"] - reversed_order["answer_accuracy"])}

    # Force the final fact into slot 0 after facts 0..2 have occupied 0..2.
    forced = pipeline_eval(aware, key, query, core, vh, cache, "testA", 4, device, limit=limit, order=[0, 1, 2, 3], force_slots=[0, 1, 2, 0], seed_offset=1300)
    wrong_slot = {"normal": strip_raw(normal), "forced_last_into_slot0": strip_raw(forced), "old_fact0_retrieval": float(forced["raw"]["retrieval"][:, 0].mean()), "new_fact3_retrieval": float(forced["raw"]["retrieval"][:, 3].mean()), "old_fact0_answer": float(forced["raw"]["answer"][:, 0].mean()), "new_fact3_answer": float(forced["raw"]["answer"][:, 3].mean())}

    # Same facts, two legal physical placements; answer/query behavior should be equivalent.
    inv_a = pipeline_eval(aware, key, query, core, vh, cache, "testA", 2, device, limit=limit, force_slots=[0, 1], seed_offset=1400)
    inv_b = pipeline_eval(aware, key, query, core, vh, cache, "testA", 2, device, limit=limit, force_slots=[0, 3], seed_offset=1401)
    invariance = {"slot01": strip_raw(inv_a), "slot03": strip_raw(inv_b), "retrieval_abs_diff_pp": 100 * abs(inv_a["retrieval_accuracy"] - inv_b["retrieval_accuracy"]), "answer_abs_diff_pp": 100 * abs(inv_a["answer_accuracy"] - inv_b["answer_accuracy"])}

    full_answer = {name: {s: {str(c): policies[name][s][str(c)] for c in CAPS} for s in ("testA", "testB")} for name in ("learned_soft", "learned_hard_mask", "k_blind_soft", "random_free", "first_free")}
    retention = {name: {s: policies[name][s]["4"] for s in ("testA", "testB")} for name in policies}
    multi_query = {"learned_soft_testA_N4": policies["learned_soft"]["testA"]["4"], "learned_soft_testB_N4": policies["learned_soft"]["testB"]["4"]}
    dist = {"learned_soft_testA_N4": soft_alloc_a["4"], "learned_soft_testB_N4": soft_alloc_b["4"], "learned_hard_testA_N4": hard_alloc["4"], "k_blind_testA_N4": blind_alloc["testA"]["4"]}
    k_compare = {"k_aware": {"train": train_aware, "testA_allocation": soft_alloc_a, "testB_allocation": soft_alloc_b, "testA_pipeline_N4": policies["learned_soft"]["testA"]["4"], "testB_pipeline_N4": policies["learned_soft"]["testB"]["4"]}, "k_blind": {"train": train_blind, "testA_allocation": blind_alloc["testA"], "testB_allocation": blind_alloc["testB"], "testA_pipeline_N4": policies["k_blind_soft"]["testA"]["4"], "testB_pipeline_N4": policies["k_blind_soft"]["testB"]["4"]}}

    main_a = policies["learned_soft"]["testA"]["4"]; main_b = policies["learned_soft"]["testB"]["4"]
    one_empty_rates = []
    for ent in pattern_results["k_aware_soft"]["1"]: one_empty_rates.append(ent["valid_rate"])
    one_empty = float(np.mean(one_empty_rates)) if one_empty_rates else 0.0
    slot_drop = slot_perm_result["retrieval_drop_pp"]; order_drop = order_result["retrieval_drop_pp"]
    if SMOKE:
        verdict = "SMOKE ONLY: NOT FOR SCIENTIFIC VERDICT"
    elif soft_alloc_a["4"]["collision_rate"] <= .01 and soft_alloc_b["4"]["collision_rate"] <= .01 and one_empty >= .99 and main_a["retrieval_accuracy"] >= .95 and main_b["retrieval_accuracy"] >= .95 and main_a["answer_accuracy"] >= .90 and main_b["answer_accuracy"] >= .85 and slot_drop <= 5 and order_drop <= 5:
        verdict = "O3a PASS: LEARNED FREE-SLOT ALLOCATION"
    elif hard_alloc["4"]["collision_rate"] <= .001 and (soft_alloc_a["4"]["collision_rate"] > .01 or soft_alloc_b["4"]["collision_rate"] > .01):
        verdict = "O3a PARTIAL: HARD-MASK ALLOCATION ONLY"
    elif soft_alloc_a["4"]["collision_rate"] > .01 or soft_alloc_b["4"]["collision_rate"] > .01:
        verdict = "O3a FAIL: COLLISION AVOIDANCE"
    elif main_a["retrieval_accuracy"] < .95 or main_b["retrieval_accuracy"] < .95:
        verdict = "O3a FAIL: RETENTION REGRESSION"
    elif slot_drop > 5 or order_drop > 5:
        verdict = "O3a INVALID: POSITION SHORTCUT"
    else:
        verdict = "O3a FAIL: WRITE ACTION INTEGRATION"

    config = {"phase": "CLR-v2.4-O3a", "seed": SEED, "device": str(device), "source_o21b": str(O21B), "source_vgen2": str(VGEN2), "cache": str(CACHE_ROOT), "capacities": list(CAPS), "slots": SLOTS, "allocator_inputs": {"k_aware": "K_new + explicit occupied mask", "k_blind": "explicit occupied mask only"}, "occupancy_representation": "explicit binary occupied mask", "hard_mask_condition": "occupied logits masked only for safety control", "soft_loss": "-log(sum probability mass over empty slots)", "lr": 3e-4, "weight_decay": .01, "grad_clip": 1.0, "steps": steps, "eval_every": EVAL_EVERY, "trainable": ["Allocator only"], "frozen": ["learned KeyEncoder", "V/core/GRU", "QueryEncoder", "retrieval matcher", "decoder", "answer head", "Ouro", "physical memory dimensions"], "dataset": {s: {"episodes": int(sum(len(cache.rows(s, c)) for c in CAPS)), "by_capacity": {str(c): int(len(cache.rows(s, c))) for c in CAPS}} for s in ("train", "val", "testA", "testB")}}
    json_dump(OUT / "config.json", config)
    json_dump(OUT / "train.log", summarize_train_logs(train_aware, train_blind))
    json_dump(OUT / "metrics.json", {"verdict": verdict, "learned_soft_testA_N4": main_a, "learned_soft_testB_N4": main_b, "one_empty_correct_empty_rate": one_empty, "slot_permutation": slot_perm_result, "fact_order_permutation": order_result})
    json_dump(OUT / "hard_mask_results.json", {"allocation": hard_alloc, "pipeline": {s: policies["learned_hard_mask"][s] for s in ("testA", "testB")}, "interpretation": "validity is guaranteed by construction; not evidence of learned collision avoidance"})
    json_dump(OUT / "soft_occupancy_results.json", {"testA": soft_alloc_a, "testB": soft_alloc_b, "pipeline": {s: policies["learned_soft"][s] for s in ("testA", "testB")}})
    json_dump(OUT / "occupancy_pattern_results.json", pattern_results)
    json_dump(OUT / "one_empty_results.json", {"pattern_results": pattern_results["k_aware_soft"]["1"], "pipeline": one_empty})
    json_dump(OUT / "allocation_distribution.json", dist)
    json_dump(OUT / "slot_permutation.json", slot_perm_result)
    json_dump(OUT / "fact_order_permutation.json", order_result)
    json_dump(OUT / "retention_results.json", retention)
    json_dump(OUT / "multi_query_results.json", multi_query)
    json_dump(OUT / "full_kv_answer.json", full_answer)
    json_dump(OUT / "wrong_slot_intervention.json", wrong_slot)
    json_dump(OUT / "empty_slot_invariance.json", invariance)
    json_dump(OUT / "k_aware_vs_blind.json", k_compare)
    report = ["# CLR-v2.4-O3a — Learned Free-Slot Allocation", "", f"**{verdict}**", "", "## Result", f"Soft K-aware allocation N=4 collision rate: TestA {100*soft_alloc_a['4']['collision_rate']:.2f}%, TestB {100*soft_alloc_b['4']['collision_rate']:.2f}%.", f"Full pipeline retrieval N=4: TestA {100*main_a['retrieval_accuracy']:.2f}%, TestB {100*main_b['retrieval_accuracy']:.2f}%; full KV answer: TestA {100*main_a['answer_accuracy']:.2f}%, TestB {100*main_b['answer_accuracy']:.2f}%.", f"All-facts-retained: TestA {100*main_a['all_facts_retained_rate']:.2f}%, TestB {100*main_b['all_facts_retained_rate']:.2f}%; all-facts-answered: TestA {100*main_a['all_facts_answered_rate']:.2f}%, TestB {100*main_b['all_facts_answered_rate']:.2f}%.", "", "## Controls", f"Hard-mask N=4 validity is reported separately and is guaranteed by construction. One-empty correct-empty rate: {100*one_empty:.2f}%.", f"Slot permutation retrieval drop: {slot_drop:.2f} pp; reversed fact-order retrieval drop: {order_drop:.2f} pp.", f"Forced occupied-slot intervention: old fact0 retrieval {100*wrong_slot['old_fact0_retrieval']:.2f}%, new fact3 retrieval {100*wrong_slot['new_fact3_retrieval']:.2f}%.", f"Legal empty-slot invariance retrieval difference: {invariance['retrieval_abs_diff_pp']:.2f} pp.", "", "## K-aware versus K-blind", "The K-blind allocator is an explicit control. If its collision-free rate matches K-aware, this phase supports resource allocation rather than semantic placement; K is not required for choosing an arbitrary free slot.", "", "## Protocol", f"Only Allocator was trained; K/V/query/router/decoder/Ouro were frozen. K-aware parameters={train_aware['params']}; K-blind parameters={train_blind['params']}; steps={steps}; seed={SEED}.", "Occupancy was an explicit binary state. The primary loss rewarded total probability mass on any empty slot and did not use a slot-ID target.", "", "## Boundary", "This phase does not test same-key update, overwrite, replacement, eviction, capacity overflow, semantic placement, or latent reasoning.", "", "## Stop", "O3a complete; do not automatically implement O3b."]
    (OUT / "v24_o3a_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "soft_testA_N4_collision": soft_alloc_a["4"]["collision_rate"], "soft_testB_N4_collision": soft_alloc_b["4"]["collision_rate"], "testA_N4_retrieval": main_a["retrieval_accuracy"], "testB_N4_retrieval": main_b["retrieval_accuracy"], "testA_N4_answer": main_a["answer_accuracy"], "testB_N4_answer": main_b["answer_accuracy"], "one_empty": one_empty, "wall_clock_s": sum(x["wall_clock_s"] for x in (train_aware, train_blind))}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    _DEVICE = []
    main()
