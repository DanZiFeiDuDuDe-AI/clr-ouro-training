# -*- coding: utf-8 -*-
"""CLR-v2.4-O2.1b — Learned Semantic Key Writing.

Only fact -> K is learned.  The query encoder, value path, decoder, router,
oracle physical allocation, and Ouro are frozen.  The main loss uses only
the correct physical slot and query/key retrieval; entity/relation labels are
read only for post-hoc probes and diagnostics.
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

from a1_v24_o21a import KeyEncoder, QueryEncoder
from a1_v24_o21a_mr import MRCache, MRValueHeads, N_CLASSES, N_REL, RELATIONS
from a1_v24_vgen import OldSingleCache, load_core
from a1_v24_o2_common import json_dump

ROOT = Path(r"D:/training")
OUT = Path(os.environ.get("V24_O21B_OUT", r"D:/training/checkpoints/clr_v24_o21b"))
CACHE_ROOT = Path(os.environ.get("V24_O21B_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
SUPERVISED = Path(os.environ.get("V24_O21B_SUPERVISED", r"D:/training/checkpoints/clr_v24_o21a_mr/best_model.pt"))
VGEN2 = Path(os.environ.get("V24_O21B_VGEN2", r"D:/training/checkpoints/clr_v24_vgen2/best.pt"))
R2 = ROOT / "checkpoints" / "clr_v23_oracle_r2" / "best_model.pt"
OLD_CACHE = ROOT / "checkpoints" / "clr_v2_a1_ls0" / "cache"
SEED = int(os.environ.get("V24_O21B_SEED", "42"))
STEPS = int(os.environ.get("V24_O21B_STEPS", "3000"))
EVAL_EVERY = int(os.environ.get("V24_O21B_EVAL_EVERY", "100"))
DEVICE_NAME = os.environ.get("V24_O21B_DEVICE", "cuda")
TEMP = 0.1
CHANCE = 1.0 / 4.0


def seed_all(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_query_and_supervised(device):
    ck = torch.load(SUPERVISED, map_location="cpu", weights_only=False)
    q = QueryEncoder(64).to(device); q.load_state_dict(ck["query"], strict=True); q.eval()
    for p in q.parameters(): p.requires_grad_(False)
    k = KeyEncoder(64).to(device); k.load_state_dict(ck["key"], strict=True); k.eval()
    return k, q, ck


def key_query(key, query, facts, qh):
    k = key(facts.reshape(-1, 2048)).reshape(facts.shape[0], facts.shape[1], -1)
    q = query(qh)
    return k, q


def retrieval_loss(key, query, facts, qh, target_slot):
    k, q = key_query(key, query, facts, qh)
    score = (q[:, None, :] * k).sum(-1) / TEMP
    within = F.cross_entropy(score, target_slot)
    pos = k[torch.arange(len(target_slot), device=target_slot.device), target_slot]
    batch = F.cross_entropy((q @ pos.T) / TEMP, torch.arange(len(target_slot), device=target_slot.device))
    return within + 0.5 * batch, within, batch, k, q, score


def sample_rows(cache, rng, split="train", batch_size=128):
    cap = int(rng.choice((1, 2, 4)))
    rows_all = cache.rows(split, cap)
    rows = rows_all[rng.integers(0, len(rows_all), size=min(batch_size, len(rows_all)))]
    return cap, rows


def train_b1(cache, query, device, steps=STEPS, smoke=False):
    key = KeyEncoder(64).to(device)  # intentionally fresh random init
    query.eval()
    for p in query.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(key.parameters(), lr=1e-3, weight_decay=0.01)
    rng = np.random.default_rng(SEED + 2101)
    logs = []; best = -1.0; best_step = 0; best_payload = None
    t0 = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        cap, rows = sample_rows(cache, rng, "train", 128)
        facts, _, _, qh, slot, _, _ = cache.batch("train", rows, cap, device)
        loss, within, batch, _, _, _ = retrieval_loss(key, query, facts, qh, slot)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(key.parameters(), 1.0); opt.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == steps:
            val = evaluate_key(key, query, cache, "val", device)
            macro = float(np.mean([val[str(c)]["retrieval"] for c in (1, 2, 4)]))
            margin = score_geometry(key, query, cache, "val", device)
            logs.append({"step": step, "loss_retrieval": float(loss.detach()), "loss_within_slot": float(within.detach()), "loss_batch_contrastive": float(batch.detach()), "val": val, "val_macro": macro, "val_geometry": margin})
            if macro > best:
                best = macro; best_step = step
                best_payload = {"key": copy.deepcopy(key.state_dict()), "query": copy.deepcopy(query.state_dict()), "step": step, "val": val, "val_macro": macro, "seed": SEED, "loss": "within-episode retrieval CE + 0.5 batch positive InfoNCE", "explicit_semantic_supervision": False}
                torch.save(best_payload, OUT / "b1_best.pt")
    final = {"key": key.state_dict(), "query": query.state_dict(), "step": steps, "seed": SEED, "explicit_semantic_supervision": False}
    torch.save(final, OUT / "b1_final.pt")
    peak = None
    if torch.cuda.is_available(): peak = {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
    return {"steps": steps, "best_step": best_step, "best_val_macro": best, "wall_clock_s": time.perf_counter() - t0, "logs": logs, "peak_vram": peak, "best_path": str(OUT / "b1_best.pt"), "final_path": str(OUT / "b1_final.pt")}


@torch.no_grad()
def evaluate_key(key, query, cache, split, device, rows_override=None):
    key.eval(); query.eval(); out = {}
    for cap in (1, 2, 4):
        rows = cache.rows(split, cap) if rows_override is None else np.asarray(rows_override, dtype=np.int64)
        total = good = 0; selected = []; target = []; yall = []
        for st in range(0, len(rows), 128):
            rb = rows[st:st + 128]
            facts, _, _, qh, slot, y, _ = cache.batch(split, rb, cap, device); k, q = key_query(key, query, facts, qh)
            sel = ((q[:, None, :] * k).sum(-1) / TEMP).argmax(1)
            good += int((sel == slot).sum()); total += len(slot); selected.append(sel.cpu().numpy()); target.append(slot.cpu().numpy()); yall.append(y.cpu().numpy())
        out[str(cap)] = {"retrieval": good / max(total, 1), "n": total, "selected": np.concatenate(selected), "target": np.concatenate(target), "y": np.concatenate(yall)}
    out["macro"] = float(np.mean([out[str(c)]["retrieval"] for c in (1, 2, 4)]))
    return out


@torch.no_grad()
def evaluate_rows(key, query, cache, split, cap, rows, device, vcore=None, vh=None):
    rows = np.asarray(rows, dtype=np.int64); ret = ans = oracle = total = 0; selected = []; target = []; rels = []; ys = []; hard_preds = []
    for st in range(0, len(rows), 128):
        rb = rows[st:st + 128]; facts, _, rel, qh, slot, y, _ = cache.batch(split, rb, cap, device); k, q = key_query(key, query, facts, qh); sel = ((q[:, None, :] * k).sum(-1) / TEMP).argmax(1)
        ret += int((sel == slot).sum()); total += len(y); selected.append(sel.cpu().numpy()); target.append(slot.cpu().numpy()); rels.append(rel[torch.arange(len(y), device=device), slot].cpu().numpy()); ys.append(y.cpu().numpy())
        if vcore is not None and vh is not None:
            v, _ = vcore.forward_facts(facts, slot_indices=list(range(cap))); pred = vh.selected(v, sel).argmax(1); op = vh.selected(v, slot).argmax(1); ans += int((pred == y).sum()); oracle += int((op == y).sum()); hard_preds.append(pred.cpu().numpy())
    d = {"n": total, "retrieval": ret / max(total, 1), "selected": np.concatenate(selected), "target": np.concatenate(target), "y": np.concatenate(ys), "query_relation": np.concatenate(rels)}
    if hard_preds: d["hard_pred"] = np.concatenate(hard_preds)
    if vcore is not None: d.update({"hard_answer": ans / max(total, 1), "oracle_answer": oracle / max(total, 1)})
    return d


def stress_results(key, query, cache, device, vcore=None, vh=None):
    out = {}
    for name, kind in (("mr1_same_entity", "same_entity"), ("mr2_same_relation", "same_relation"), ("lexical_stress", "lexical_stress"), ("same_value", "same_value")):
        rows = cache.rows_type("testA", 4, kind); r = evaluate_rows(key, query, cache, "testA", 4, rows, device, vcore, vh); out[name] = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
    return out


@torch.no_grad()
def collect_k(key, query, cache, split, device):
    kk = []; ee = []; rr = []; vv = []; qq = []; qrel = []
    for cap in (1, 2, 4):
        rows = cache.rows(split, cap); d = cache.data[split]
        for st in range(0, len(rows), 128):
            rb = rows[st:st + 128]; facts, ent, rel, qh, slot, _, fy = cache.batch(split, rb, cap, device); k, q = key_query(key, query, facts, qh)
            kk.append(k.cpu().numpy().reshape(-1, 64)); ee.append(ent.cpu().numpy().reshape(-1)); rr.append(rel.cpu().numpy().reshape(-1)); vv.append(fy.cpu().numpy().reshape(-1)); qq.append(q.cpu().numpy()); qrel.append(rel[torch.arange(len(slot), device=device), slot].cpu().numpy())
    return {"K": np.concatenate(kk), "entity": np.concatenate(ee), "relation": np.concatenate(rr), "value": np.concatenate(vv), "Q": np.concatenate(qq), "query_relation": np.concatenate(qrel)}


def fit_probe(x, y, out_dim, device, steps=300, seed=700):
    torch.manual_seed(seed); x = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device); y = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device); p = nn.Linear(x.shape[1], out_dim).to(device); opt = torch.optim.AdamW(p.parameters(), lr=3e-3, weight_decay=0.01)
    for _ in range(steps):
        ix = torch.randperm(len(y), device=device)[:min(4096, len(y))]; opt.zero_grad(set_to_none=True); F.cross_entropy(p(x[ix]), y[ix]).backward(); opt.step()
    return p.eval()


@torch.no_grad()
def probe_metrics(p, x, y, mask=None):
    x = np.asarray(x); y = np.asarray(y); mask = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not mask.any(): return {"accuracy": None, "n": 0}
    pred = p(torch.from_numpy(x[mask].astype(np.float32)).to(next(p.parameters()).device)).argmax(1).cpu().numpy()
    return {"accuracy": float((pred == y[mask]).mean()), "n": int(mask.sum())}


def fresh_probes(key, query, cache, device):
    emb = {s: collect_k(key, query, cache, s, device) for s in ("train", "val", "testA", "testB")}
    tr = emb["train"]; nent = int(tr["entity"].max() + 1); joint_y = tr["entity"] * N_REL + tr["relation"]; njoint = int(joint_y.max() + 1)
    ep = fit_probe(tr["K"], tr["entity"], nent, device, 300, 701); rp = fit_probe(tr["K"], tr["relation"], N_REL, device, 300, 702); jp = fit_probe(tr["K"], joint_y, njoint, device, 300, 703); vp = fit_probe(tr["K"], tr["value"], N_CLASSES, device, 300, 704)
    out = {"entity": {}, "relation": {}, "joint": {}, "value": {}}
    for s, e in emb.items():
        known = e["entity"] >= 0
        out["entity"][s] = probe_metrics(ep, e["K"], e["entity"], known)
        out["relation"][s] = probe_metrics(rp, e["K"], e["relation"])
        out["joint"][s] = probe_metrics(jp, e["K"], e["entity"] * N_REL + e["relation"], known)
        out["value"][s] = probe_metrics(vp, e["K"], e["value"])
    out["probe_type"] = "fresh Linear probes trained after key writer freeze; no probe gradients enter key"
    return out


@torch.no_grad()
def score_geometry(key, query, cache, split, device):
    pos = []; neg = []; ent_cos = []; rel_cos = []; entropy = []
    e = collect_k(key, query, cache, split, device)
    # Query/fact positive matching on complete episodes.
    for cap in (1, 2, 4):
        rows = cache.rows(split, cap)
        for st in range(0, len(rows), 128):
            rb = rows[st:st + 128]; facts, _, _, qh, slot, _, _ = cache.batch(split, rb, cap, device); k, q = key_query(key, query, facts, qh); sc = (q[:, None, :] * k).sum(-1) / TEMP; pr = sc[torch.arange(len(slot), device=device), slot]; pos.extend(pr.cpu().tolist());
            if cap > 1:
                neg.extend(sc.masked_fill(F.one_hot(slot, cap).bool(), float("nan")).nanmean(1).cpu().tolist())
            entropy.extend((-(sc.softmax(1) * sc.softmax(1).clamp_min(1e-8).log()).sum(1)).cpu().tolist())
    # Same-entity/relation structure is descriptive only.
    K = e["K"]; E = e["entity"]; R = e["relation"]
    rng = np.random.default_rng(991); n = min(4000, len(K)); ix = rng.choice(len(K), n, replace=False)
    for a in ix[:min(1000, n)]:
        same_e = np.flatnonzero(E == E[a]); same_r = np.flatnonzero(R == R[a])
        if len(same_e) > 1:
            b = same_e[(a + 1) % len(same_e)]; ent_cos.append(float(np.dot(K[a], K[b]) / (np.linalg.norm(K[a]) * np.linalg.norm(K[b]) + 1e-9)))
        if len(same_r) > 1:
            b = same_r[(a + 1) % len(same_r)]; rel_cos.append(float(np.dot(K[a], K[b]) / (np.linalg.norm(K[a]) * np.linalg.norm(K[b]) + 1e-9)))
    return {"positive_score_mean": float(np.mean(pos)), "negative_score_mean": float(np.mean(neg)) if neg else None, "margin_mean": float(np.mean(pos) - np.mean(neg)) if neg else None, "routing_entropy_mean": float(np.mean(entropy)), "within_entity_cosine_mean": float(np.mean(ent_cos)) if ent_cos else None, "within_relation_cosine_mean": float(np.mean(rel_cos)) if rel_cos else None}


@torch.no_grad()
def key_interventions(key, query, vcore, vh, cache, split, cap, device):
    rows = cache.rows(split, cap); out = {"correct_k_zero": [], "wrong_k_swap": [], "full_kv_permutation": [], "fact_order_permutation": []}
    for st in range(0, len(rows), 128):
        rb = rows[st:st + 128]; facts, _, _, qh, slot, y, _ = cache.batch(split, rb, cap, device); k, q = key_query(key, query, facts, qh); v, _ = vcore.forward_facts(facts, slot_indices=list(range(cap))); ix = torch.arange(len(slot), device=device); alt = (slot + 1) % cap
        base_scores = (q[:, None, :] * k).sum(-1) / TEMP; base_sel = base_scores.argmax(1)
        kz = k.clone(); kz[ix, slot] = 0; zsel = ((q[:, None, :] * kz).sum(-1) / TEMP).argmax(1)
        kswap = k.clone(); tmp = k[ix, slot].clone(); kswap[ix, slot] = k[ix, alt]; kswap[ix, alt] = tmp; sswap = ((q[:, None, :] * kswap).sum(-1) / TEMP).argmax(1); swap_pred = vh.selected(v, sswap).argmax(1)
        perm = torch.arange(cap - 1, -1, -1, device=device); kp = k[:, perm, :]; vp = v[:, perm, :]; targetp = (perm[None, :] == slot[:, None]).nonzero(as_tuple=False)[:, 1]; sper = ((q[:, None, :] * kp).sum(-1) / TEMP).argmax(1); pp = vh.selected(vp, sper).argmax(1)
        fp = facts[:, perm, :]; kf, _ = key_query(key, query, fp, qh); targetf = targetp; sf = ((q[:, None, :] * kf).sum(-1) / TEMP).argmax(1)
        out["correct_k_zero"].append(float((zsel == slot).float().mean())); out["wrong_k_swap"].append({"retrieval": float((sswap == slot).float().mean()), "switch_to_alt": float((sswap == alt).float().mean()), "answer_original": float((swap_pred == y).float().mean())}); out["full_kv_permutation"].append({"retrieval": float((sper == targetp).float().mean()), "answer": float((pp == y).float().mean())}); out["fact_order_permutation"].append(float((sf == targetf).float().mean()))
    return {"correct_k_zero_retrieval": float(np.mean(out["correct_k_zero"])), "wrong_k_swap": {k: float(np.mean([x[k] for x in out["wrong_k_swap"]])) for k in ("retrieval", "switch_to_alt", "answer_original")}, "full_kv_permutation": {k: float(np.mean([x[k] for x in out["full_kv_permutation"]])) for k in ("retrieval", "answer")}, "fact_order_permutation_retrieval": float(np.mean(out["fact_order_permutation"]))}


@torch.no_grad()
def full_answer(key, query, vcore, vh, cache, split, device):
    out = {}
    for cap in (1, 2, 4):
        rows = cache.rows(split, cap); ret = hard = oracle = total = 0
        for st in range(0, len(rows), 128):
            rb = rows[st:st + 128]; facts, _, _, qh, slot, y, _ = cache.batch(split, rb, cap, device); k, q = key_query(key, query, facts, qh); sel = ((q[:, None, :] * k).sum(-1) / TEMP).argmax(1); v, _ = vcore.forward_facts(facts, slot_indices=list(range(cap))); ret += int((sel == slot).sum()); hard += int((vh.selected(v, sel).argmax(1) == y).sum()); oracle += int((vh.selected(v, slot).argmax(1) == y).sum()); total += len(y)
        out[str(cap)] = {"retrieval": ret / max(total, 1), "hard_answer": hard / max(total, 1), "oracle_answer": oracle / max(total, 1), "n": total}
    return out


def slot_and_order_diagnostics(key, query, cache, split, device):
    r = evaluate_key(key, query, cache, split, device); d = cache.data[split]
    out = {}
    for cap in (1, 2, 4):
        rows = cache.rows(split, cap); targets = d["query_idx"][rows]; sels = r[str(cap)]["selected"]; out[str(cap)] = {"target_frequency": np.bincount(targets, minlength=cap).tolist(), "selected_frequency": np.bincount(sels, minlength=cap).tolist(), "retrieval": r[str(cap)]["retrieval"]}
    return out


def git_info():
    try: return {"sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=10).stdout.strip() or None, "diff_stat": subprocess.run(["git", "diff", "--stat"], cwd=str(ROOT), capture_output=True, text=True, timeout=10).stdout.strip() or None}
    except Exception as e: return {"sha": None, "diff_stat": None, "error": str(e)}


def restore_key(path, query, device):
    ck = torch.load(path, map_location="cpu", weights_only=False); key = KeyEncoder(64).to(device); key.load_state_dict(ck["key"], strict=True); key.eval(); query.eval(); return key, ck


def main():
    seed_all(); t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(CACHE_ROOT)
    supervised_key, query, sup_ck = load_query_and_supervised(device)
    vcore, _, _ = load_core(device, trainable=False); vck = torch.load(VGEN2, map_location="cpu", weights_only=False); vcore.load_state_dict(vck["core"], strict=True); vcore.eval()
    vh = MRValueHeads().to(device); vh.load_state_dict(vck["value_heads"], strict=True); vh.eval()
    for p in vcore.parameters(): p.requires_grad_(False)
    for p in vh.parameters(): p.requires_grad_(False)
    smoke = os.environ.get("V24_O21B_SMOKE", "0") == "1" or "--smoke" in os.sys.argv; steps = min(STEPS, 20) if smoke else STEPS

    # B0 is a frozen upper-bound reference, never retrained here.
    b0 = {s: evaluate_key(supervised_key, query, cache, s, device) for s in ("val", "testA", "testB")}
    b1_record = train_b1(cache, query, device, steps, smoke)
    learned_key, learned_ck = restore_key(b1_record["best_path"], query, device)
    learned = {s: evaluate_key(learned_key, query, cache, s, device) for s in ("val", "testA", "testB")}
    stress_b0 = stress_results(supervised_key, query, cache, device, vcore, vh); stress_b1 = stress_results(learned_key, query, cache, device, vcore, vh)
    probes = fresh_probes(learned_key, query, cache, device)
    geometry_out = {"supervised": {s: score_geometry(supervised_key, query, cache, s, device) for s in ("val", "testA", "testB")}, "learned": {s: score_geometry(learned_key, query, cache, s, device) for s in ("val", "testA", "testB")}}
    interventions = {"supervised": key_interventions(supervised_key, query, vcore, vh, cache, "testA", 4, device), "learned": key_interventions(learned_key, query, vcore, vh, cache, "testA", 4, device)}
    slots = {"supervised": slot_and_order_diagnostics(supervised_key, query, cache, "testA", device), "learned": slot_and_order_diagnostics(learned_key, query, cache, "testA", device)}
    order = {"supervised": interventions["supervised"]["fact_order_permutation_retrieval"], "learned": interventions["learned"]["fact_order_permutation_retrieval"]}
    answer = {"supervised": {s: full_answer(supervised_key, query, vcore, vh, cache, s, device) for s in ("testA", "testB")}, "learned": {s: full_answer(learned_key, query, vcore, vh, cache, s, device) for s in ("testA", "testB")}}
    # Per-relation retrieval and answer, using query relation only for evaluation.
    per_rel = {}
    for label, key in (("supervised", supervised_key), ("learned", learned_key)):
        per_rel[label] = {}
        for split in ("testA", "testB"):
            rows = cache.rows(split, 4); r = evaluate_rows(key, query, cache, split, 4, rows, device, vcore, vh); rel = r["query_relation"]; per_rel[label][split] = {}
            for qrel in range(N_REL):
                m = rel == qrel; per_rel[label][split][RELATIONS[qrel]] = {"n": int(m.sum()), "retrieval": float((r["selected"][m] == r["target"][m]).mean()) if m.any() else None, "hard_answer": float((r["hard_pred"][m] == r["y"][m]).mean()) if m.any() else None}
    leak_path = ROOT / "checkpoints" / "clr_v24_o21a_mr" / "key_value_leakage.json"; old_leak = json.loads(leak_path.read_text(encoding="utf-8")) if leak_path.exists() else None
    config = {"phase": "CLR-v2.4-O2.1b", "seed": SEED, "device": str(device), "source_supervised_key": str(SUPERVISED), "source_vgen2": str(VGEN2), "cache": str(CACHE_ROOT), "capacities": [1, 2, 4], "relations": list(RELATIONS), "query_frozen": True, "explicit_entity_relation_supervision": False, "loss": "within-episode retrieval CE + 0.5 batch positive InfoNCE", "temperature": TEMP, "lr": 1e-3, "weight_decay": 0.01, "grad_clip": 1.0, "steps": steps, "eval_every": EVAL_EVERY, "trainable": ["KeyEncoder.net.*"], "frozen": ["QueryEncoder", "V write/core", "decoder", "answer head", "router/matcher", "allocation", "physical slot logic", "Ouro"], "key_writer_parameters": sum(p.numel() for p in KeyEncoder(64).parameters()), "dataset": {s: {"episodes": int(len(cache.data[s]["N"])), "query_relation_counts": np.bincount(cache.data[s]["query_relation"], minlength=N_REL).tolist()} for s in ("train", "val", "testA", "testB")}, "git": git_info(), "exact_command": "py -3 a1_v24_o21b.py"}
    learned_leak = {s: probes["value"][s] for s in ("val", "testA", "testB")}
    json_dump(OUT / "config.json", config); json_dump(OUT / "train.log", b1_record); json_dump(OUT / "baseline_supervised.json", {s: {k: v for k, v in b0[s].items() if not isinstance(v, np.ndarray)} for s in b0}); json_dump(OUT / "learned_key_results.json", {s: {k: v for k, v in learned[s].items() if not isinstance(v, np.ndarray)} for s in learned}); json_dump(OUT / "retrieval_comparison.json", {"supervised": {s: {k: v for k, v in b0[s].items() if not isinstance(v, np.ndarray)} for s in b0}, "learned": {s: {k: v for k, v in learned[s].items() if not isinstance(v, np.ndarray)} for s in learned}}); json_dump(OUT / "stress_results.json", {"supervised": stress_b0, "learned": stress_b1}); json_dump(OUT / "fresh_semantic_probes.json", probes); json_dump(OUT / "semantic_geometry.json", geometry_out); json_dump(OUT / "key_interventions.json", interventions); json_dump(OUT / "slot_order_diagnostics.json", slots); json_dump(OUT / "fact_order_permutation.json", order); json_dump(OUT / "full_kv_answer.json", answer); json_dump(OUT / "per_relation_results.json", per_rel); json_dump(OUT / "key_value_leakage.json", {"prior_supervised": old_leak, "learned": learned_leak})

    a4 = learned["testA"]["4"]["retrieval"]; b4 = learned["testB"]["4"]["retrieval"]; s_a4 = b0["testA"]["4"]["retrieval"]; s_b4 = b0["testB"]["4"]["retrieval"]; mr1 = stress_b1["mr1_same_entity"]["retrieval"]; mr2 = stress_b1["mr2_same_relation"]["retrieval"]; samev = stress_b1["same_value"]["retrieval"]; ans_a = answer["learned"]["testA"]["4"]["hard_answer"]; ans_b = answer["learned"]["testB"]["4"]["hard_answer"]; swap = interventions["learned"]["wrong_k_swap"]; perm = interventions["learned"]["full_kv_permutation"]; leak_a = (probes["value"]["testA"]["accuracy"] if probes["value"]["testA"]["accuracy"] is not None else 0.0)
    physical_ok = abs(np.mean(slots["learned"]["4"]["selected_frequency"]) - np.mean(slots["learned"]["4"]["target_frequency"])) < max(10, 0.25 * len(cache.rows("testA", 4)) / 4)
    if smoke:
        verdict = "SMOKE ONLY: NOT FOR SCIENTIFIC VERDICT"
    elif a4 >= .90 and b4 >= .85 and mr1 >= .90 and mr2 >= .90 and samev >= .90 and ans_a >= .90 and ans_b >= .85 and swap["switch_to_alt"] > .50 and perm["retrieval"] >= .90 and physical_ok:
        verdict = "O2.1b PASS: LEARNED SEMANTIC KEY WRITING"
    elif a4 >= .90 and b4 < .85:
        verdict = "O2.1b PARTIAL: IN-DISTRIBUTION LEARNED KEY ONLY"
    elif a4 >= .90 and (mr1 < .90 or mr2 < .90 or samev < .90):
        verdict = "O2.1b PARTIAL: COMPOSITIONALITY INCOMPLETE"
    elif b4 < .70:
        verdict = "O2.1b FAIL: UNSEEN ENTITY GENERALIZATION"
    elif a4 < .70:
        verdict = "O2.1b FAIL: QUERY-KEY ALIGNMENT"
    elif not physical_ok:
        verdict = "O2.1b FAIL: POSITION / ORDER SHORTCUT"
    else:
        verdict = "O2.1b FAIL: FACT-TO-KEY EXTRACTION"
    def pct(x): return f"{100*float(x):.2f}%"
    report = ["# CLR-v2.4-O2.1b — Learned Semantic Key Writing", "", f"**{verdict}**", "", "## Primary result", "B1 was trained from a fresh KeyEncoder using frozen QueryEncoder and retrieval-only supervision; entity/relation/joint labels never entered the key loss.", f"Supervised K upper bound N=4: TestA/TestB {pct(s_a4)}/{pct(s_b4)}.", f"Learned K N=4: TestA/TestB {pct(a4)}/{pct(b4)}; delta {100*(a4-s_a4):.2f}/{100*(b4-s_b4):.2f} percentage points.", f"Learned K full KV answer: TestA/TestB {pct(ans_a)}/{pct(ans_b)}.", "", "## Retrieval stress", f"MR1 same entity/different relation: {pct(mr1)}; MR2 same relation/different entity: {pct(mr2)}; same-value: {pct(samev)}; lexical: {pct(stress_b1['lexical_stress']['retrieval'])}.", f"Wrong-K swap: {json.dumps(swap, ensure_ascii=False)}; full KV permutation: {json.dumps(perm, ensure_ascii=False)}.", "", "## Probes and geometry", "Fresh K probes are in fresh_semantic_probes.json; K→value is diagnostic only and did not train the writer.", "", "## Protocol", f"KeyWriter parameters: {config['key_writer_parameters']}; trainable: KeyEncoder.net.*; frozen QueryEncoder/V/decoder/router/allocation/Ouro; steps={steps}; eval_every={EVAL_EVERY}; seed={SEED}.", "Oracle physical slot allocation was retained; allocation learning was not entered.", "", "## Boundary", "This result, even if PASS, only supports autonomous fact→K formation under oracle allocation, fixed schema, and frozen query/read paths. It does not establish learned allocation, eviction, long-context memory, KV-cache replacement, or latent reasoning.", "", "## Stop", "O2.1b complete; learned allocation is not implemented in this phase."]
    (OUT / "v24_o21b_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "supervised_testA_N4": s_a4, "supervised_testB_N4": s_b4, "learned_testA_N4": a4, "learned_testB_N4": b4, "mr1": mr1, "mr2": mr2, "same_value": samev, "full_kv_answer_testA": ans_a, "full_kv_answer_testB": ans_b, "wall_clock_s": time.perf_counter()-t0}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
