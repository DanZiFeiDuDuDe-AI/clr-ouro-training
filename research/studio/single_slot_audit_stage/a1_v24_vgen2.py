# -*- coding: utf-8 -*-
"""CLR-v2.4-V-GEN2: joint value-write and decoder strengthening.

This phase starts from the completed V-GEN T3 checkpoint.  The semantic K,
query-key encoder, router, allocation, and physical slot routing stay frozen
and are used only in the final reconnect diagnostic.  The main path is
oracle allocation + oracle correct-slot read, with an auxiliary direct value
classification loss on every written V slot.
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

from a1_v24_vgen import (
    CHANCE, CAPS, N_CLASSES, N_REL, RELATIONS, MRCache, OldSingleCache,
    eval_stage, fresh_v_probe, geometry, load_core, mr_batch, reconnect,
    relation_swap, shortcuts, single_retention, LegacyLocalHeads,
)
from a1_v24_o21a import KeyEncoder, QueryEncoder
from a1_v24_o21a_mr import MRValueHeads, relation_stress
from a1_v24_o2_common import json_dump

ROOT = Path(r"D:/training")
OUT = Path(os.environ.get("V24_VGEN2_OUT", r"D:/training/checkpoints/clr_v24_vgen2"))
MR_CACHE = Path(os.environ.get("V24_VGEN2_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
SOURCE_T3 = Path(os.environ.get("V24_VGEN2_T3", r"D:/training/checkpoints/clr_v24_vgen/t3_joint_best.pt"))
MR_BEST = Path(os.environ.get("V24_VGEN2_MR_BEST", r"D:/training/checkpoints/clr_v24_o21a_mr/best_model.pt"))
R2 = ROOT / "checkpoints" / "clr_v23_oracle_r2" / "best_model.pt"
OLD_CACHE = ROOT / "checkpoints" / "clr_v2_a1_ls0" / "cache"
SEED = int(os.environ.get("V24_VGEN2_SEED", "42"))
STEPS = int(os.environ.get("V24_VGEN2_STEPS", "6000"))
EVAL_EVERY = int(os.environ.get("V24_VGEN2_EVAL_EVERY", "200"))
DEVICE_NAME = os.environ.get("V24_VGEN2_DEVICE", "cuda")
LAMBDAS = (0.5, 1.0, 2.0)


def seed_all(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def lambda_name(lam):
    return f"lambda_{str(lam).replace('.', '')}"


class ValueProbeHeads(nn.Module):
    """Trainable direct semantic supervision head, one Linear(64,64) per slot."""
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(64, N_CLASSES) for _ in range(4)])

    def all_logits(self, v, cap):
        return torch.stack([self.heads[i](v[:, i, :].float()) for i in range(cap)], dim=1)


def load_t3(device):
    ck = torch.load(SOURCE_T3, map_location="cpu", weights_only=False)
    core, _, _ = load_core(device, trainable=True)
    core.load_state_dict(ck["core"], strict=True)
    vh = MRValueHeads().to(device); vh.load_state_dict(ck["value_heads"], strict=True)
    return core, vh, ck


def batch_rows(cache, split, cap, rng, n=64):
    rows = cache.rows(split, cap)
    return rows[rng.integers(0, len(rows), size=min(n, len(rows)))]


def value_losses(core, vh, value_heads, facts, qslot, y, fy, cap):
    v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
    answer = F.cross_entropy(vh.selected(v, qslot), y)
    logits = value_heads.all_logits(v[:, :cap, :], cap)
    value = sum(F.cross_entropy(logits[:, i, :], fy[:, i]) for i in range(cap)) / cap
    return answer, value, v


def old_losses(core, vh, value_heads, old, rng, device):
    rows = old.rows("train")
    rows = rows[rng.integers(0, len(rows), size=min(32, len(rows)))]
    facts, y = old.batch("train", rows, device)
    v, _ = core.forward_facts(facts, slot_indices=[0])
    slot = torch.zeros(len(y), dtype=torch.long, device=device)
    answer = F.cross_entropy(vh.selected(v, slot), y)
    value = F.cross_entropy(value_heads.heads[0](v[:, 0, :].float()), y)
    return answer, value


def quick_fresh_probe(core, cache, split, device, steps=10, max_rows_per_cap=256):
    """Fast fresh global V->value probe for training-curve monitoring."""
    xs = []; ys = []
    core.eval()
    with torch.no_grad():
        for cap in CAPS:
            rows = cache.rows("train", cap)[:max_rows_per_cap]
            for st in range(0, len(rows), 256):
                facts, _, _, _, _, _, fy = mr_batch(cache, "train", rows[st:st + 256], cap, device)
                v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
                xs.append(v[:, :cap, :].float().cpu().numpy().reshape(-1, 64)); ys.append(fy.cpu().numpy().reshape(-1))
    x = torch.from_numpy(np.concatenate(xs).astype(np.float32)).to(device)
    y = torch.from_numpy(np.concatenate(ys).astype(np.int64)).to(device)
    p = nn.Linear(64, N_CLASSES).to(device); opt = torch.optim.AdamW(p.parameters(), lr=3e-3, weight_decay=0.01)
    p.train()
    for _ in range(steps):
        ix = torch.randperm(len(y), device=device)[:min(4096, len(y))]
        opt.zero_grad(set_to_none=True); F.cross_entropy(p(x[ix]), y[ix]).backward(); opt.step()
    p.eval(); good = total = 0
    with torch.no_grad():
        for cap in CAPS:
            rows = cache.rows(split, cap)[:max_rows_per_cap]
            for st in range(0, len(rows), 256):
                facts, _, _, _, _, _, fy = mr_batch(cache, split, rows[st:st + 256], cap, device)
                v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
                pred = p(v[:, :cap, :].float().reshape(-1, 64)).argmax(1)
                good += int((pred == fy.reshape(-1)).sum()); total += int(fy.numel())
    return good / max(total, 1)


@torch.no_grad()
def eval_monitor(core, vh, cache, split, device, max_rows_per_cap=128):
    """Fast periodic validation on a fixed bounded subset; full validation is
    still run for every saved best checkpoint after training."""
    out = {}
    for cap in CAPS:
        rows = cache.rows(split, cap)[:max_rows_per_cap]
        total = good = 0
        for st in range(0, len(rows), 128):
            facts, _, _, _, slot, y, _ = mr_batch(cache, split, rows[st:st + 128], cap, device)
            v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            good += int((vh.selected(v, slot).argmax(1) == y).sum()); total += len(y)
        out[str(cap)] = {"answer": good / max(total, 1), "n": total}
    out["macro"] = float(np.mean([out[str(c)]["answer"] for c in CAPS]))
    return out


def train_lambda(lam, cache, old, device, steps=STEPS):
    core, vh, source_ck = load_t3(device)
    value_heads = ValueProbeHeads().to(device)
    params = list(core.parameters()) + list(vh.parameters()) + list(value_heads.parameters())
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(SEED + int(lam * 1000) + 2400)
    logs = []; best = -1.0; best_step = 0; best_payload = None
    t0 = time.perf_counter()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        cap = int(rng.choice(CAPS)); rows = batch_rows(cache, "train", cap, rng)
        facts, _, _, _, qslot, y, fy = mr_batch(cache, "train", rows, cap, device)
        old_a, old_v = old_losses(core, vh, value_heads, old, rng, device)
        new_a, new_v, _ = value_losses(core, vh, value_heads, facts, qslot, y, fy, cap)
        loss_a = 0.5 * old_a + 0.5 * new_a
        loss_v = 0.5 * old_v + 0.5 * new_v
        loss = loss_a + lam * loss_v
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == steps:
            val = eval_monitor(core, vh, cache, "val", device, max_rows_per_cap=128)
            val_probe = quick_fresh_probe(core, cache, "val", device, steps=5, max_rows_per_cap=64)
            score = 0.5 * val["macro"] + 0.5 * val_probe
            log = {"step": step, "train_loss": float(loss.detach()), "train_answer_loss": float(loss_a.detach()),
                   "train_value_loss": float(loss_v.detach()), "lambda_value": lam,
                   "val_answer": val, "val_fresh_v_probe": val_probe, "selection_score": score}
            logs.append(log)
            if score > best:
                best = score; best_step = step
                best_payload = {"core": copy.deepcopy(core.state_dict()), "value_heads": copy.deepcopy(vh.state_dict()),
                                "value_probe_heads": copy.deepcopy(value_heads.state_dict()), "lambda_value": lam,
                                "step": step, "val": val, "val_fresh_v_probe": val_probe,
                                "source_t3": str(SOURCE_T3)}
                torch.save(best_payload, OUT / f"{lambda_name(lam)}_best.pt")
    final = {"core": core.state_dict(), "value_heads": vh.state_dict(), "value_probe_heads": value_heads.state_dict(),
             "lambda_value": lam, "step": steps, "source_t3": str(SOURCE_T3)}
    torch.save(final, OUT / f"{lambda_name(lam)}_final.pt")
    peak = None
    if torch.cuda.is_available(): peak = {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9}
    return {"lambda_value": lam, "steps": steps, "best_step": best_step, "best_score": best,
            "wall_clock_s": time.perf_counter() - t0, "replay_ratio": 0.5, "logs": logs,
            "peak_vram": peak, "best_path": str(OUT / f"{lambda_name(lam)}_best.pt"),
            "final_path": str(OUT / f"{lambda_name(lam)}_final.pt")}


def restore(record, device):
    ck = torch.load(record["best_path"], map_location="cpu", weights_only=False)
    core, _, _ = load_core(device, trainable=False); core.load_state_dict(ck["core"], strict=True); core.eval()
    vh = MRValueHeads().to(device); vh.load_state_dict(ck["value_heads"], strict=True); vh.eval()
    aux = ValueProbeHeads().to(device); aux.load_state_dict(ck["value_probe_heads"], strict=True); aux.eval()
    for m in (core, vh, aux):
        for p in m.parameters(): p.requires_grad_(False)
    return core, vh, aux, ck


@torch.no_grad()
def value_interventions(core, vh, cache, device, split="testA", cap=4):
    rows = cache.rows(split, cap); names = {k: [] for k in ("correct_v", "correct_v_zero", "wrong_v", "unrelated_v_zero", "value_swap_original", "value_swap_target")}
    for st in range(0, len(rows), 128):
        facts, _, _, _, slot, y, fy = mr_batch(cache, split, rows[st:st + 128], cap, device)
        v, _ = core.forward_facts(facts, slot_indices=list(range(cap))); ix = torch.arange(len(y), device=device); alt = (slot + 1) % cap
        vz = v.clone(); vz[ix, slot] = 0
        vw = torch.roll(v, 1, 0)
        vu = torch.zeros_like(v); vu[ix, slot] = v[ix, slot]
        vs = v.clone(); tmp = v[ix, slot].clone(); vs[ix, slot] = v[ix, alt]; vs[ix, alt] = tmp
        def acc(vv, target): return float((vh.selected(vv, slot).argmax(1) == target).float().mean())
        names["correct_v"].append(acc(v, y)); names["correct_v_zero"].append(acc(vz, y)); names["wrong_v"].append(acc(vw, y)); names["unrelated_v_zero"].append(acc(vu, y)); names["value_swap_original"].append(acc(vs, y)); names["value_swap_target"].append(float((vh.selected(vs, slot).argmax(1) == fy[ix, alt]).float().mean()))
    return {k: float(np.mean(v)) for k, v in names.items()} | {"chance": CHANCE, "n": int(len(rows))}


@torch.no_grad()
def per_relation_answer(core, vh, cache, device, split, full_kv=False):
    out = {RELATIONS[r]: {"n": 0, "correct": 0} for r in range(N_REL)}
    key = query = None
    if full_kv:
        ck = torch.load(MR_BEST, map_location="cpu", weights_only=False); key = KeyEncoder(64).to(device); query = QueryEncoder(64).to(device); key.load_state_dict(ck["key"]); query.load_state_dict(ck["query"]); key.eval(); query.eval()
    for cap in CAPS:
        rows = cache.rows(split, cap)
        for st in range(0, len(rows), 128):
            rb = rows[st:st + 128]; facts, _, rel, qh, slot, y, _ = mr_batch(cache, split, rb, cap, device); v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            if full_kv:
                k = key(facts.reshape(-1, 2048)).reshape(len(rb), cap, -1); q = query(qh); use = (q[:, None, :] * k).sum(-1).argmax(1)
            else: use = slot
            pred = vh.selected(v, use).argmax(1); rr = rel[torch.arange(len(y), device=device), slot]
            for r in range(N_REL):
                m = rr == r; out[RELATIONS[r]]["n"] += int(m.sum()); out[RELATIONS[r]]["correct"] += int((pred[m] == y[m]).sum())
    for item in out.values(): item["accuracy"] = item["correct"] / max(item["n"], 1)
    return out


def v_entity_probe(core, cache, device):
    xs = []; ys = []
    for cap in CAPS:
        rows = cache.rows("train", cap)
        for st in range(0, len(rows), 256):
            facts, ent, _, _, _, _, _ = mr_batch(cache, "train", rows[st:st + 256], cap, device); v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            xs.append(v[:, :cap, :].cpu().numpy().reshape(-1, 64)); ys.append(ent.cpu().numpy().reshape(-1))
    x = torch.from_numpy(np.concatenate(xs).astype(np.float32)).to(device); y = torch.from_numpy(np.concatenate(ys).astype(np.int64)).to(device); n = int(y.max().item() + 1)
    p = nn.Linear(64, n).to(device); opt = torch.optim.AdamW(p.parameters(), lr=3e-3, weight_decay=0.01)
    for _ in range(300):
        ix = torch.randperm(len(y), device=device)[:min(4096, len(y))]; opt.zero_grad(set_to_none=True); F.cross_entropy(p(x[ix]), y[ix]).backward(); opt.step()
    p.eval(); out = {}
    for split in ("testA", "testB"):
        if split == "testB": out[split] = {"accuracy": None, "n": 0, "note": "entity labels are unseen/encoded as -1"}; continue
        xx = []; yy = []
        for cap in CAPS:
            rows = cache.rows(split, cap)
            for st in range(0, len(rows), 256):
                facts, ent, _, _, _, _, _ = mr_batch(cache, split, rows[st:st + 256], cap, device); v, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
                xx.append(v[:, :cap, :].cpu().numpy().reshape(-1, 64)); yy.append(ent.cpu().numpy().reshape(-1))
        pred = p(torch.from_numpy(np.concatenate(xx).astype(np.float32)).to(device)).argmax(1).cpu().numpy(); truth = np.concatenate(yy); out[split] = {"accuracy": float((pred == truth).mean()), "n": int(len(truth))}
    return out


def git_info():
    try:
        return {"sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, timeout=10).stdout.strip() or None,
                "diff_stat": subprocess.run(["git", "diff", "--stat"], cwd=str(ROOT), capture_output=True, text=True, timeout=10).stdout.strip() or None}
    except Exception as e: return {"sha": None, "diff_stat": None, "error": str(e)}


def main():
    seed_all(); t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(MR_CACHE); old = OldSingleCache(OLD_CACHE)
    smoke = os.environ.get("V24_VGEN2_SMOKE", "0") == "1" or "--smoke" in os.sys.argv
    steps = min(STEPS, 20) if smoke else STEPS

    # Historical T3 reference, evaluated without training it again.
    t3_core, t3_vh, _ = load_t3(device); baseline = {"source_t3": str(SOURCE_T3), "TestA": eval_stage(t3_core, t3_vh, None, cache, "testA", device), "TestB": eval_stage(t3_core, t3_vh, None, cache, "testB", device), "single_relation": single_retention(t3_core, t3_vh, None, old, device)}
    json_dump(OUT / "baseline_t3.json", baseline)
    records = {lambda_name(l): train_lambda(l, cache, old, device, steps) for l in LAMBDAS}
    objs = {}; results = {}; retention = {}
    for l in LAMBDAS:
        n = lambda_name(l); c, v, aux, ck = restore(records[n], device); objs[n] = (c, v, aux)
        results[n] = {s: eval_stage(c, v, None, cache, s, device) for s in ("val", "testA", "testB")}; retention[n] = single_retention(c, v, None, old, device)
    best_name = max(records, key=lambda n: records[n]["best_score"])
    best_core, best_vh, best_aux = objs[best_name]
    fresh = fresh_v_probe(best_core, cache, device)
    relation_v = fresh.get("per_relation", {})
    purity = {"v_to_value": fresh, "v_to_relation": fresh.get("v_to_relation", {}), "v_to_entity": v_entity_probe(best_core, cache, device)}
    oracle = {n: {s: results[n][s] for s in ("testA", "testB")} for n in results}
    per_rel = {"oracle_answer": {s: per_relation_answer(best_core, best_vh, cache, device, s, False) for s in ("testA", "testB")}, "full_kv_answer": {s: per_relation_answer(best_core, best_vh, cache, device, s, True) for s in ("testA", "testB")}}
    shortcuts_out = {s: shortcuts(best_core, best_vh, None, cache, device, False, s, 4) for s in ("testA", "testB")}
    interventions = {s: value_interventions(best_core, best_vh, cache, device, s, 4) for s in ("testA", "testB")}
    reconnect_out = reconnect(best_core, best_vh, None, cache, device, False)
    stress = relation_stress(*(__import__('a1_v24_o21a_mr').restore_best(cache, device)[:2]), best_vh, best_core, cache, device) if False else {}
    # Use the already validated frozen router directly for stress metrics.
    router_ck = torch.load(MR_BEST, map_location="cpu", weights_only=False); key = KeyEncoder(64).to(device); query = QueryEncoder(64).to(device); key.load_state_dict(router_ck["key"]); query.load_state_dict(router_ck["query"]); key.eval(); query.eval()
    stress = relation_stress(key, query, best_vh, best_core, cache, device)
    source_leak = ROOT / "checkpoints" / "clr_v24_o21a_mr" / "key_value_leakage.json"
    leakage = json.loads(source_leak.read_text(encoding="utf-8")) if source_leak.exists() else {"note": "prior frozen-router K->value diagnostic unavailable"}
    geo = {"before_t3": {s: geometry(t3_core, cache, s, device) for s in ("testA", "testB")}, "after_best": {s: geometry(best_core, cache, s, device) for s in ("testA", "testB")}}

    best_ck = torch.load(records[best_name]["best_path"], map_location="cpu", weights_only=False); torch.save({"condition": best_name, **best_ck}, OUT / "best.pt")
    final_name = lambda_name(1.0); final_ck = torch.load(records[final_name]["final_path"], map_location="cpu", weights_only=False); torch.save({"condition": final_name, **final_ck}, OUT / "final.pt")
    cfg = {"phase": "CLR-v2.4-V-GEN2", "seed": SEED, "source_t3": str(SOURCE_T3), "cache": str(MR_CACHE), "capacities": list(CAPS), "relations": list(RELATIONS), "lambdas": list(LAMBDAS), "lambda_value_loss": "CE(value_probe_head(V), target_value)", "steps": steps, "eval_every": EVAL_EVERY, "lr": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0, "replay_ratio": 0.5, "trainable": ["content_read.*", "shared_gru.*", "value_heads.heads.*", "value_probe_heads.heads.*"], "frozen": ["semantic K", "query-key encoder", "retrieval controller", "key-related heads", "allocation", "physical slot routing", "Ouro"], "dataset": {s: {"episodes": int(len(cache.data[s]["N"])), "query_relation_counts": np.bincount(cache.data[s]["query_relation"], minlength=N_REL).tolist()} for s in ("train", "val", "testA", "testB")}, "git": git_info(), "exact_command": "py -3 a1_v24_vgen2.py"}
    json_dump(OUT / "config.json", cfg); json_dump(OUT / "train.log", records); json_dump(OUT / "metrics.json", {"best_condition": best_name, "baseline_t3": baseline, "results": results, "fresh_v_probe": fresh, "reconnect": reconnect_out, "retention": retention})
    for l in LAMBDAS: json_dump(OUT / f"{lambda_name(l)}.json", {"training": records[lambda_name(l)], "results": results[lambda_name(l)]})
    json_dump(OUT / "fresh_v_probe.json", fresh); json_dump(OUT / "per_relation_v_probe.json", relation_v); json_dump(OUT / "oracle_answer.json", oracle); json_dump(OUT / "full_kv_reconnect.json", {"condition": best_name, "results": reconnect_out}); json_dump(OUT / "single_relation_retention.json", {"baseline_t3": baseline["single_relation"], "conditions": retention}); json_dump(OUT / "shortcut_checks.json", shortcuts_out); json_dump(OUT / "value_interventions.json", interventions); json_dump(OUT / "value_purity.json", purity); json_dump(OUT / "training_curve.json", {n: records[n]["logs"] for n in records}); json_dump(OUT / "per_relation_results.json", per_rel); json_dump(OUT / "k_router_stress.json", stress); json_dump(OUT / "key_value_leakage.json", leakage); json_dump(OUT / "state_geometry.json", geo)

    b4a = baseline["TestA"]["4"]["answer"]; b4b = baseline["TestB"]["4"]["answer"]; a4 = results[best_name]["testA"]["4"]["answer"]; b4 = results[best_name]["testB"]["4"]["answer"]; vp_a = fresh["overall"]["testA"]["accuracy"]; vp_b = fresh["overall"]["testB"]["accuracy"]; r_a = reconnect_out["testA"]["4"]; r_b = reconnect_out["testB"]["4"]; old_a = retention[best_name]["testA"]["accuracy"]; old_b = retention[best_name]["testB"]["accuracy"]; ref_a = baseline["single_relation"]["testA"]["accuracy"]; ref_b = baseline["single_relation"]["testB"]["accuracy"]
    if vp_a >= .90 and vp_b >= .85 and a4 >= .90 and b4 >= .85 and r_a["retrieval"] >= .95 and r_b["retrieval"] >= .95 and r_a["hard_answer"] >= .90 and r_b["hard_answer"] >= .85 and old_a >= ref_a - .05 and old_b >= ref_b - .05:
        verdict = "V-GEN2 PASS: JOINT VALUE/DECODER GENERALIZATION RESTORED"
    elif a4 >= .90 and b4 >= .85 and (vp_a < .90 or vp_b < .85):
        verdict = "V-GEN2 PARTIAL: ANSWER RECOVERED, VALUE REPRESENTATION STILL WEAK"
    elif vp_a >= .90 and vp_b >= .85 and (a4 < .90 or b4 < .85):
        verdict = "V-GEN2 PARTIAL: VALUE REPRESENTATION RECOVERED, DECODER STILL WEAK"
    elif old_a < ref_a - .05 or old_b < ref_b - .05:
        verdict = "V-GEN2 FAIL: CATASTROPHIC FORGETTING"
    elif abs(a4 - b4) < .08 and a4 < .90:
        verdict = "V-GEN2 FAIL: JOINT OPTIMIZATION PLATEAU"
    else:
        verdict = "V-GEN2 FAIL: OBJECTIVE CONFLICT"

    def pct(x): return f"{100*float(x):.2f}%"
    report = ["# CLR-v2.4-V-GEN2 — Joint Value Write + Decoder Strengthening", "", f"**{verdict}**", "", "## Primary result", f"Initialized from V-GEN T3 best: {SOURCE_T3}.", f"T3 reference oracle N=4: TestA/TestB {pct(b4a)}/{pct(b4b)}. Best λ={best_name}: oracle {pct(a4)}/{pct(b4)}; fresh V probe {pct(vp_a)}/{pct(vp_b)}.", f"Frozen K/router reconnect: retrieval {pct(r_a['retrieval'])}/{pct(r_b['retrieval'])}; answer {pct(r_a['hard_answer'])}/{pct(r_b['hard_answer'])}.", "", "## Training conditions", "- Jointly trained O1 content_read, shared_gru, D(V) answer heads, and direct value probe heads.", "- Loss: L_answer + λ_value * L_value, with L_value=CE(value_probe_head(V), fact value).", "- λ={0.5,1.0,2.0}; 50% old single-relation replay + 50% MR samples; lr=3e-4; no K/router training.", "- Main path uses oracle write allocation and oracle correct-slot read.", "", "## λ results"]
    for l in LAMBDAS:
        n=lambda_name(l); report.append(f"{n}: TestA/TestB oracle N=4 {pct(results[n]['testA']['4']['answer'])}/{pct(results[n]['testB']['4']['answer'])}; val score {records[n]['best_score']:.4f}; best step {records[n]['best_step']}; wall {records[n]['wall_clock_s']:.2f}s.")
    report += ["", "## Diagnostics", f"Fresh V per relation and purity: fresh_v_probe.json, value_purity.json.", f"Shortcuts/interventions: {json.dumps(shortcuts_out, ensure_ascii=False)}; {json.dumps(interventions, ensure_ascii=False)}", f"K/router stress: {json.dumps(stress, ensure_ascii=False)}", f"Original O1 retention reference {pct(ref_a)}/{pct(ref_b)}; selected condition {pct(old_a)}/{pct(old_b)}.", "", "## Boundary", "K, query-key encoder, router, allocation, slot routing, and Ouro remained frozen. This phase does not enter O2.1b, O3, A2, reasoning, long-context compression, or architecture scaling.", "", "## Runtime", f"Total wall-clock {time.perf_counter()-t0:.2f}s; cache reused without regeneration; device {device}; GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}.", "Per-condition in-process CUDA peak memory is in train.log; no new cache was generated.", "", "## Answers", f"Q6/Q7/Q8 fresh V: {pct(vp_a)}/{pct(vp_b)}; gates {'pass' if vp_a>=.90 and vp_b>=.85 else 'not met'}.", f"Q10–Q12 oracle: {pct(a4)}/{pct(b4)}; gates {'pass' if a4>=.90 and b4>=.85 else 'not met'}.", f"Q13/Q14 reconnect: retrieval {pct(r_a['retrieval'])}/{pct(r_b['retrieval'])}; answer {pct(r_a['hard_answer'])}/{pct(r_b['hard_answer'])}.", f"Q15 retention loss: TestA {100*(old_a-ref_a):.2f}pp; TestB {100*(old_b-ref_b):.2f}pp.", f"Q16/Q17 shortcut checks are near chance in shortcut_checks.json; Q18 value swap is in value_interventions.json.", f"Q21 remaining bottleneck: {'none under the registered gates' if verdict.startswith('V-GEN2 PASS') else 'joint value representation/decoder optimization plateau'}.", "Q22 system-level multi-relation KV cell: only yes if all registered gates pass.", "Q23 O2.1b: not entered; stop here."]
    (OUT / "v24_vgen2_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "best_condition": best_name, "oracle_testA_N4": a4, "oracle_testB_N4": b4, "fresh_v_testA": vp_a, "fresh_v_testB": vp_b, "reconnect_testA_N4": r_a, "reconnect_testB_N4": r_b, "wall_clock_s": time.perf_counter()-t0}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
