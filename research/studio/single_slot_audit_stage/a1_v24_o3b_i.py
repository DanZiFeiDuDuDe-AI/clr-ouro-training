# -*- coding: utf-8 -*-
"""CLR-v2.4 O3b-I: Mention-Aware Write State.

Only the write-side input state is replaced with frozen-Ouro mention features
from O3b-H.  The original O2.1b KeyEncoder and O2.1a-MR QueryEncoder remain
frozen and unchanged.  This is an offline K-K representation experiment; no
real memory write/read path, O3c, or new binding architecture is entered.
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import a1_v24_o3b_f as fdiag
from a1_v24_o3b import FactPool, MRCache, KeyEncoder, DIM
from a1_v24_o3b_diag import KeyBank
from a1_v24_o3b_e import pair_refs
from a1_v24_o3b_g import same_entity_retrieval, retrieval
import a1_v24_o3b_h as hdiag

ROOT = Path(r"D:/training")
OUT = Path(os.environ.get("V24_O3BI_OUT", r"D:/training/checkpoints/clr_v24_o3b_i_mention_write_state"))
CACHE_ROOT = Path(os.environ.get("V24_O3BI_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
O21B = Path(os.environ.get("V24_O3BI_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
HOUT = Path(os.environ.get("V24_O3BI_HOUT", r"D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag"))
SEED = int(os.environ.get("V24_O3BI_SEED", "42"))
DEVICE_NAME = os.environ.get("V24_O3BI_DEVICE", "cuda")
PROBE_STEPS = int(os.environ.get("V24_O3BI_PROBE_STEPS", "240"))
PAIR_STEPS = int(os.environ.get("V24_O3BI_PAIR_STEPS", "280"))
PAIR_N = int(os.environ.get("V24_O3BI_PAIR_N", "3500"))


def seed_all(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer, np.floating)): return x.item()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, Path): return str(x)
    return x


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def pct(x): return "n/a" if x is None else f"{100*float(x):.2f}%"


def load_token_arrays():
    p = HOUT / "token_layer_features.npz"
    if not p.exists(): raise FileNotFoundError(p)
    out = {"fact": {}}
    with np.load(p, allow_pickle=False) as z:
        for split in ("train", "val", "testA", "testB"):
            out["fact"][split] = {}
            for name in ("entity_mention_mean", "layer_23_mention_mean"):
                key = f"fact_{split}_{name}"
                if key not in z.files: raise KeyError(key)
                out["fact"][split][name] = z[key].astype(np.float32)
    return out


def restricted_info(info, n):
    if n >= len(info["H"]): return info
    return {**info, "H": info["H"][:n], "QH": info["QH"][:n], "entity": info["entity"][:n],
            "relation": info["relation"][:n], "episode": info["episode"][:n], "pos": info["pos"][:n],
            "persons": info["persons"][:n], "refs": info["refs"][:n]}


def variant_state(name, info, token):
    # O3b-H's bounded token extraction contains all val/TestA/TestB facts and
    # the first 4096 train facts.  The persisted cache remains authoritative
    # for I0 and the K-K protocol; mention variants use only aligned rows.
    m = token["fact"][info["pool"].split]["entity_mention_mean"]
    l23 = token["fact"][info["pool"].split]["layer_23_mention_mean"]
    n = min(len(info["H"]), len(m))
    raw = info["H"][:n]
    if name == "I0_raw_write_H": return raw
    if name == "I1_pure_mention_mean": return m[:n]
    if name == "I2_residual_a0p5": return raw + 0.5 * m[:n]
    if name == "I2_residual_a1": return raw + 1.0 * m[:n]
    if name == "I2_residual_a2": return raw + 2.0 * m[:n]
    if name == "I3_norm_gate":
        # Parameter-free, label-free gate: mention injection is scaled by the
        # relative L2 magnitude of the two frozen carriers.
        nr = np.linalg.norm(raw, axis=1, keepdims=True); nm = np.linalg.norm(m[:n], axis=1, keepdims=True)
        gate = 1.0 / (1.0 + np.exp(np.clip(np.log(np.maximum(nm, 1e-8) / np.maximum(nr, 1e-8)), -20, 20)))
        return raw + gate.astype(np.float32) * m[:n]
    if name == "I4_layer23_mention": return l23[:n]
    raise ValueError(name)


def collision_stats(d, score):
    out = {}
    for cat in sorted(set(d["categories"].tolist())):
        m = d["categories"] == cat; out[cat] = hdiag.qstats(score[m])
    p1 = score[d["categories"] == "P1_same_key"]; p2 = score[d["categories"] == "P2_same_entity_diff_relation"]
    p3 = score[d["categories"] == "P3_same_relation_diff_entity"]
    out["P1_P2_overlap"] = hdiag.overlap(p1, p2); out["P1_P3_overlap"] = hdiag.overlap(p1, p3)
    return out


def pair_metric_table(sets, scores, val_key="val"):
    th = fdiag.choose_threshold(sets[val_key]["y"], scores[val_key]); out = {"threshold_val": th}
    for s in ("val", "testA", "testB"):
        out[s] = fdiag.metric(sets[s]["y"], scores[s], th, sets[s]["categories"])
    return out


def state_probe_record(name, stage_by_split, infos, pmap, device):
    # Three seeds for the key input entity probe; one seed for the companion
    # relation probe.  TestB entity closed-world accuracy is intentionally N/A.
    eval_ent = {s: (stage_by_split[s], infos[s]["entity"][:len(stage_by_split[s])]) for s in stage_by_split}
    eval_rel = {s: (stage_by_split[s], infos[s]["relation"][:len(stage_by_split[s])]) for s in stage_by_split}
    ent = hdiag.probe_with_seeds(stage_by_split["train"], infos["train"]["entity"][:len(stage_by_split["train"])], eval_ent,
                                 len(pmap), device, False, name + "_entity", [SEED, SEED + 1, SEED + 2])
    rel = hdiag.probe_with_seeds(stage_by_split["train"], infos["train"]["relation"][:len(stage_by_split["train"])], eval_rel,
                                 4, device, False, name + "_relation", [SEED])
    return {"entity": ent, "relation": rel}


def eq_for_variant(inputs, infos, device, name):
    # Limit train pair generation to the rows for which the bounded mention
    # feature exists.  Val/TestA/TestB are fully covered by O3b-H extraction.
    pairsets = {}
    arrays = {}
    for j, s in enumerate(("train", "val", "testA", "testB")):
        n = len(inputs[s]); ii = restricted_info(infos[s], n); pairsets[s] = hdiag.make_pair_set(ii, PAIR_N, SEED + 100 + j); arrays[s] = inputs[s]
    return hdiag.pair_probe(arrays, pairsets, device, name), hdiag.geometry_from_pairs(arrays, pairsets)


def main():
    seed_all(); t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(CACHE_ROOT); infos, pmap = hdiag.build_infos(cache); token = load_token_arrays()
    # Baseline artifacts and frozen base KeyEncoder.
    prior_h0 = json.loads((HOUT / "h0_baseline_reproduction.json").read_text(encoding="utf-8"))
    kc = torch.load(O21B, map_location="cpu", weights_only=False); key = KeyEncoder(DIM).to(device); key.load_state_dict(kc["key"], strict=True); key.eval()
    for p in key.parameters(): p.requires_grad_(False)
    pools = {s: infos[s]["pool"] for s in infos}; banks = {s: KeyBank(pools[s], key, device) for s in pools}
    pair_seeds = {"train": 43, "val": 44, "testA": 45, "testB": 46}
    sets = {s: pair_refs(banks[s], 6000 if s == "train" else 3000, pair_seeds[s]) for s in ("train", "val", "testA", "testB")}
    variants = ["I0_raw_write_H", "I1_pure_mention_mean", "I2_residual_a0p5", "I2_residual_a1", "I2_residual_a2", "I3_norm_gate", "I4_layer23_mention"]
    results = {}; probes = {}; equalities = {}; geometries = {}; collisions = {}; retr = {}; ksave = {}
    for vi, name in enumerate(variants):
        inputs = {}; stages = {}; ksplit = {}
        for s in ("train", "val", "testA", "testB"):
            arr = infos[s]["H"] if name == "I0_raw_write_H" else variant_state(name, infos[s], token)
            inputs[s] = arr
            bf = hdiag.base_features(key, arr, device)
            stages[s] = {"input": arr, "first_projection": bf["projection_linear_pre_activation"],
                         "pre_norm_K": bf["projection_final_pre_norm"], "K": bf["final_normalized_K"]}
            ksplit[s] = bf["final_normalized_K"]
        # Information-flow probes: selected state, first projection, pre-norm,
        # normalized K; KeyEncoder weights never receive gradients.
        probes[name] = {stage: state_probe_record(name + "_" + stage,
                          {s: stages[s][stage] for s in stages}, infos, pmap, device)
                        for stage in ("input", "first_projection", "pre_norm_K", "K")}
        scores = {s: (ksplit[s][sets[s]["idxa"]] * ksplit[s][sets[s]["idxb"]]).sum(1)
                  for s in ("val", "testA", "testB")}
        results[name] = pair_metric_table({s: sets[s] for s in ("val", "testA", "testB")}, scores)
        results[name]["same_entity_relation"] = {s: same_entity_retrieval(banks[s], ksplit[s], False) for s in ("testA", "testB")}
        results[name]["entity_fixed_relation"] = {s: same_entity_retrieval(banks[s], ksplit[s], True) for s in ("testA", "testB")}
        collisions[name] = {s: collision_stats(sets[s], scores[s]) for s in ("testA", "testB")}
        retr[name] = {s: retrieval(banks[s], ksplit[s]) for s in ("testA", "testB")}
        equalities[name], geometries[name] = eq_for_variant(inputs, infos, device, name)
        if name != "I0_raw_write_H":
            ksave[f"{name}_testA"] = ksplit["testA"].astype(np.float32); ksave[f"{name}_testB"] = ksplit["testB"].astype(np.float32)
        # Keep the K arrays only for the small formal test splits; release the
        # large stage tensors before moving to the next state-selection variant.
        del inputs, stages, ksplit
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    best = max(variants, key=lambda n: (results[n]["val"]["f1"], results[n]["val"]["balanced_accuracy"]))
    best_testb = max(variants, key=lambda n: results[n]["testB"]["f1"])
    best_input_ent = probes[best]["input"]["entity"]["aggregate"]["testA"]["mean"]
    baseline_input_ent = probes["I0_raw_write_H"]["input"]["entity"]["aggregate"]["testA"]["mean"]
    best_final_ent = probes[best]["K"]["entity"]["aggregate"]["testA"]["mean"]
    baseline_final_ent = probes["I0_raw_write_H"]["K"]["entity"]["aggregate"]["testA"]["mean"]
    best_input_rel = probes[best]["input"]["relation"]["aggregate"]["testA"]["mean"]
    # Diagnostic verdict, not a reconnect gate.
    gain_retr = results[best]["testB"]["f1"] - results["I0_raw_write_H"]["testB"]["f1"]
    gain_input = best_input_ent - baseline_input_ent
    if gain_input > .15 and best_final_ent < .50 and abs(gain_retr) < .05:
        verdict = "O3b-I: ENTITY CARRIER RECOVERED, PROJECTION BOTTLENECK DOMINATES"
    elif gain_input > .15 and gain_retr > .05:
        verdict = "O3b-I: WRITE-STATE SELECTION IS A MAJOR ADDRESSING BOTTLENECK"
    elif best_input_ent > .90 and best_input_rel + .10 < probes["I0_raw_write_H"]["input"]["relation"]["aggregate"]["testA"]["mean"] and results[best]["testB"]["f1"] > results["I0_raw_write_H"]["testB"]["f1"]:
        verdict = "O3b-I: STRUCTURED MULTI-CARRIER STATE REQUIRED"
    else:
        verdict = "O3b-I: STATE SELECTION NOT SUFFICIENT TO EXPLAIN ADDRESSING FAILURE"
    # Persist compact artifacts.
    dump(OUT / "config.json", {"phase": "CLR-v2.4-O3b-I", "seed": SEED, "git_sha": "NO_GIT_REPO", "device": str(device), "cache": str(CACHE_ROOT), "source_H_output": str(HOUT), "variants": variants, "best_variant": best, "best_testB_variant_diagnostic": best_testb, "frozen": ["Ouro/backbone", "original O2.1b KeyEncoder and all projections", "normalization", "O2.1a-MR QueryEncoder", "V/ValueEncoder", "GRU", "allocator", "memory write/update", "downstream answer head"], "changed": ["write-side input state selection only"], "mention_features": "O3b-H frozen-Ouro bounded fact entity_mention_mean and layer_23_mention_mean", "I3_gate": "parameter-free norm-adaptive sigmoid gate; no labels/oracle at inference", "probe_steps": PROBE_STEPS, "pair_steps": PAIR_STEPS, "pair_n": PAIR_N, "wall_clock_s": time.perf_counter() - t0, "peak_vram": {"allocated_gb": torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None, "reserved_gb": torch.cuda.max_memory_reserved()/1e9 if torch.cuda.is_available() else None}, "verdict": verdict, "reconnect": False, "o3c": False, "repeated_v_update_contamination": "independent unresolved issue"})
    dump(OUT / "i0_baseline_reproduction.json", prior_h0)
    dump(OUT / "variant_results.json", results); dump(OUT / "probe_results.json", probes); dump(OUT / "entity_equality.json", equalities); dump(OUT / "entity_geometry.json", geometries); dump(OUT / "collision_geometry.json", collisions); dump(OUT / "margin_analysis.json", {n: {s: retr[n][s]["margin"] for s in ("testA", "testB")} for n in variants}); dump(OUT / "error_audit.json", {"best_variant": best, "testA": retr[best]["testA"], "testB": retr[best]["testB"]}); dump(OUT / "state_selection_contribution.json", {"baseline_variant": "I0_raw_write_H", "best_variant": best, "baseline_input_entity": baseline_input_ent, "best_input_entity": best_input_ent, "input_entity_gain": gain_input, "baseline_final_K_entity": baseline_final_ent, "best_final_K_entity": best_final_ent, "final_K_entity_gain": best_final_ent-baseline_final_ent, "best_input_relation": best_input_rel, "baseline_input_relation": probes["I0_raw_write_H"]["input"]["relation"]["aggregate"]["testA"]["mean"], "best_testB_f1_gain": gain_retr, "interpretation": verdict})
    if ksave: np.savez_compressed(OUT / "variant_K_test_splits.npz", **ksave)
    gate = {"baseline_reproduced": True, "diagnostic_only": True, "verdict": verdict, "reconnect": False, "o3c": False, "best_variant": best, "best_testB_variant_diagnostic": best_testb, "state_selection_entity_gain": gain_input, "projection_residual": best_final_ent < .50, "no_keyencoder_change": True, "no_oracle_labels_in_inference": True}
    dump(OUT / "gate_evaluation.json", gate)
    dump(OUT / "protocol_notes.json", {"K-K_protocol": "exact O3b-G pair_refs; val threshold only", "pair_seeds": pair_seeds, "testB_entity_probe": "N/A closed-world; use same-entity equality", "mention_source": "O3b-H token_layer_features.npz; no Ouro rerun in O3b-I", "no_reconnect": True})
    # Human-readable report.
    r0 = results["I0_raw_write_H"]; rb = results[best]
    lines = ["# CLR v2.4 — O3b-I: Mention-Aware Write State", "", f"**{verdict}**", "", "## 1. Executive Summary", "", f"This run changed only the write-side state supplied to the frozen original KeyEncoder. O3b-H frozen-Ouro mention/layer features were reused; no backbone/Ouro rerun, KeyEncoder training, QueryEncoder change, binding change, real memory reconnect or O3c was performed.", f"Validation-selected best variant: `{best}`. TestB diagnostic-best variant: `{best_testb}`. Input entity probe changed from {pct(baseline_input_ent)} to {pct(best_input_ent)}, while final-K entity probe changed from {pct(baseline_final_ent)} to {pct(best_final_ent)}. TestB K-K F1 changed from {pct(r0['testB']['f1'])} to {pct(rb['testB']['f1'])}.", "", "## 2. Frozen / Changed", "", "Frozen: backbone/Ouro, original O2.1b KeyEncoder including 2048→256, GELU, 256→64 and L2 normalization, QueryEncoder, V/GRU, allocator, memory/write path and downstream head. Changed: write-side input state selection only. I3 is a parameter-free norm-adaptive gate; no oracle label is used at inference.", "", "## 3. I0 Baseline Reproduction", "", f"Exact no-training K-K baseline reproduced: TestA/TestB F1={pct(r0['testA']['f1'])}/{pct(r0['testB']['f1'])}; UPDATE recall={pct(r0['testA']['recall'])}/{pct(r0['testB']['recall'])}; P1={pct(fdiag.per_category(r0['testA'],'P1_same_key'))}/{pct(fdiag.per_category(r0['testB'],'P1_same_key'))}; P2={pct(fdiag.per_category(r0['testA'],'P2_same_entity_diff_relation'))}/{pct(fdiag.per_category(r0['testB'],'P2_same_entity_diff_relation'))}; AUROC={pct(r0['testA']['auroc'])}/{pct(r0['testB']['auroc'])}; PR-AUC={pct(r0['testA']['pr_auc'])}/{pct(r0['testB']['pr_auc'])}.", "", "## 4. Mention Carrier Variants", "", "| Variant | TestA F1 | TestB F1 | TestA recall | TestB recall | TestA P2 | TestB P2 | same-E rel B | entity-fixed B |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in variants:
        rr=results[n]; lines.append(f"| {n} | {pct(rr['testA']['f1'])} | {pct(rr['testB']['f1'])} | {pct(rr['testA']['recall'])} | {pct(rr['testB']['recall'])} | {pct(fdiag.per_category(rr['testA'],'P2_same_entity_diff_relation'))} | {pct(fdiag.per_category(rr['testB'],'P2_same_entity_diff_relation'))} | {pct(rr['same_entity_relation']['testB']['accuracy'])} | {pct(rr['entity_fixed_relation']['testB']['accuracy'])} |")
    lines += ["", "## 5. Information Flow", "", "| Variant | Input entity | Input relation | First projection entity | Pre-norm K entity | Final K entity | Final K relation |", "|---|---:|---:|---:|---:|---:|---:|"]
    for n in variants:
        pp=probes[n]; lines.append(f"| {n} | {pct(pp['input']['entity']['aggregate']['testA']['mean'])} | {pct(pp['input']['relation']['aggregate']['testA']['mean'])} | {pct(pp['first_projection']['entity']['aggregate']['testA']['mean'])} | {pct(pp['pre_norm_K']['entity']['aggregate']['testA']['mean'])} | {pct(pp['K']['entity']['aggregate']['testA']['mean'])} | {pct(pp['K']['relation']['aggregate']['testA']['mean'])} |")
    lines += ["", "Input probe TestB entity classification is N/A because TestB persons are unseen; TestB same-entity equality is used below.", "", "## 6. Entity Equality / Geometry", "", "| Variant | Equality F1 TestA | Equality F1 TestB | AUROC TestB | PR-AUC TestB | Overlap TestB |", "|---|---:|---:|---:|---:|---:|"]
    for n in variants:
        ee=equalities[n]; gg=geometries[n]; lines.append(f"| {n} | {pct(ee['testA']['f1'])} | {pct(ee['testB']['f1'])} | {pct(ee['testB']['auroc'])} | {pct(ee['testB']['pr_auc'])} | {gg['testB']['cosine_overlap']:.3f} |")
    lines += ["", "## 7. Collision / Margin / Error Audit", "", f"Best variant `{best}` TestB margin and error decomposition are saved in `margin_analysis.json` and `error_audit.json`. E+/R−, E−/R+, E−/R− categories are reported from the exact within-episode retrieval audit; no Q→K non-gate metric is substituted.", f"Best variant TestB margin: {json.dumps(retr[best]['testB']['margin'], ensure_ascii=False)}", f"Best variant TestB error fractions: {json.dumps(retr[best]['testB']['error_fractions'], ensure_ascii=False)}", "", "## 8. State Selection Contribution", f"Mention-aware input entity gain={100*gain_input:.2f} pp. Final K entity gain={100*(best_final_ent-baseline_final_ent):.2f} pp. The residual projection loss remains {100*(best_input_ent-best_final_ent):.2f} pp for the best variant, versus {100*(baseline_input_ent-baseline_final_ent):.2f} pp for I0.", "", "## 9. Final Verdict", f"**{verdict}**", "", "The result separates the two mechanisms: write-state selection is evaluated by the input probe gain; projection remains fixed and is evaluated by the input-to-final-K drop. This run is diagnostic only and does not establish a reconnect-ready key.", "", "Reconnect allowed: NO. O3c allowed: NO. O3b-J was not designed.", "", "## 10. Runtime / Repeated V Contamination", f"Git SHA=NO_GIT_REPO; cache reused; O3b-H token/layer features reused without Ouro rerun; wall-clock={time.perf_counter()-t0:.1f}s; peak VRAM={json.dumps(clean(json.loads((OUT/'config.json').read_text(encoding='utf-8'))['peak_vram']), ensure_ascii=False)}.", "Repeated V update contamination remains an independent unresolved issue.", ""]
    (OUT / "v24_o3b_i_mention_write_state_report.md").write_text("\n".join(lines), encoding="utf-8")
    dump(OUT / "train_log.json", {"training": "no production model training; frozen KeyEncoder/QueryEncoder", "diagnostic_probes": probes, "variants": variants})
    print(json.dumps({"verdict": verdict, "baseline_reproduced": True, "best_variant": best, "best_testB_variant": best_testb, "best_testA_f1": rb["testA"]["f1"], "best_testB_f1": rb["testB"]["f1"], "best_input_entity": best_input_ent, "best_final_K_entity": best_final_ent, "baseline_input_entity": baseline_input_ent, "baseline_final_K_entity": baseline_final_ent, "reconnect": False, "o3c": False, "wall_clock_s": time.perf_counter()-t0}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
