import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v22_storage_common import (
    DEFAULT_CACHE, DEFAULT_OUT, FactCache, eval_capacity, fixed_position_matrix,
    get_device, json_dump, load_checkpoint, normalized_capacity, seed_all,
)


def rows_for(cache, split, cap, limit=None):
    rows = cache.rows_for_capacity(split, cap)
    return rows if limit is None else rows[:min(len(rows), int(limit))]


def collect(model, cache, split, cap, rows, batch_size=128):
    model.eval()
    dev = next(model.parameters()).device
    finals, stages, labels, positions, routes, occs, ids = [], [], [], [], [], [], []
    for i in range(0, len(rows), batch_size):
        rb = rows[i:i + batch_size]
        facts, y, pos, _ = cache.batch(split, rb, cap, dev)
        state, state_list, stat_list = model.forward_facts(facts, keep_states=True, keep_stats=True)
        finals.append(state.detach().float().cpu().numpy())
        stages.append(torch.stack(state_list, dim=1).detach().float().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        positions.append(pos.detach().cpu().numpy())
        routes.append(torch.stack([x["route"] for x in stat_list], dim=1).detach().float().cpu().numpy())
        occs.append(torch.stack([x["occupancy_after"] for x in stat_list], dim=1).detach().float().cpu().numpy())
        ids.append(rb)
    return {
        "final": np.concatenate(finals), "stages": np.concatenate(stages),
        "labels": np.concatenate(labels), "positions": np.concatenate(positions),
        "routes": np.concatenate(routes), "occupancy": np.concatenate(occs),
        "rows": np.concatenate(ids),
    }


def probe_fit(x_train, y_train, x_test, y_test, steps=220, lr=3e-3, seed=2237):
    torch.manual_seed(seed)
    xtr = torch.from_numpy(np.asarray(x_train, dtype=np.float32))
    ytr = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    xte = torch.from_numpy(np.asarray(x_test, dtype=np.float32))
    yte = torch.from_numpy(np.asarray(y_test, dtype=np.int64))
    n_classes = 64 if ytr.max().item() < 64 else int(max(ytr.max(), yte.max()).item() + 1)
    head = nn.Linear(xtr.shape[1], n_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(xtr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        tr = (head(xtr).argmax(1) == ytr).float().mean().item()
        te = (head(xte).argmax(1) == yte).float().mean().item()
    return {"train_acc": tr, "test_acc": te, "steps": steps, "dim": int(xtr.shape[1])}


def city_probe_matrix(train_rec, test_rec, cap):
    out = []
    for p in range(cap):
        out.append(probe_fit(train_rec["final"].reshape(len(train_rec["final"]), -1),
                             train_rec["labels"][:, p],
                             test_rec["final"].reshape(len(test_rec["final"]), -1),
                             test_rec["labels"][:, p]))
    return out


def shortcut_baselines(cache, split, cap, rows):
    d = cache.data[split]
    starts = d["fact_offsets"][rows]
    labels = np.stack([d["fact_Y"][int(s):int(s) + cap] for s in starts])
    pos = cache.target_positions(split, rows)
    first = labels[:, 0] == labels[np.arange(len(rows)), pos]
    last = labels[:, -1] == labels[np.arange(len(rows)), pos]
    return {"first_city_acc": float(first.mean()), "last_city_acc": float(last.mean()),
            "expected_1_over_N": 1.0 / cap}


def person_labels(cache, split, rows, cap):
    d = cache.data[split]
    eid_arr = d.get("episode_id", np.arange(len(d["N"])))
    all_names = sorted({p for vals in cache.persons.values() for p in vals})
    name_id = {p: i for i, p in enumerate(all_names)}
    out = np.full((len(rows), cap), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        vals = cache.persons.get(int(eid_arr[row]), [])
        for p in range(min(cap, len(vals))):
            out[i, p] = name_id[vals[p]]
    return out, len(all_names)


def binding_probes(train_rec, test_rec, train_person, test_person, cap=4):
    out = {"assignment_aligned": {}, "cross_probes": {}, "person_labels_available": bool((train_person >= 0).all() and (test_person >= 0).all())}
    for rep_name, sl in (("key_content", slice(0, 31)), ("value", slice(32, 64)), ("full_slot", slice(None))):
        tr_x, te_x, tr_y, te_y = [], [], [], []
        for p in range(cap):
            tr_slot = train_rec["routes"][:, p].argmax(1)
            te_slot = test_rec["routes"][:, p].argmax(1)
            tr_x.append(train_rec["final"][np.arange(len(train_rec["final"])), tr_slot, sl])
            te_x.append(test_rec["final"][np.arange(len(test_rec["final"])), te_slot, sl])
            tr_y.append(train_rec["labels"][:, p]); te_y.append(test_rec["labels"][:, p])
        tr_x = np.concatenate(tr_x); te_x = np.concatenate(te_x)
        tr_y = np.concatenate(tr_y); te_y = np.concatenate(te_y)
        out["assignment_aligned"][rep_name + "_to_city"] = probe_fit(tr_x, tr_y, te_x, te_y)
        if rep_name in ("key_content", "value") and (train_person >= 0).all() and (test_person >= 0).all():
            trp = np.concatenate([train_person[:, p] for p in range(cap)])
            tep = np.concatenate([test_person[:, p] for p in range(cap)])
            out["assignment_aligned"][rep_name + "_to_person"] = probe_fit(tr_x, trp, te_x, tep)
    # Cross probes use the routed slot representation but swap labels.
    for source, sl, target_name in (("K", slice(0, 31), "city"), ("V", slice(32, 64), "city")):
        tr_x, te_x, tr_y, te_y = [], [], [], []
        for p in range(cap):
            ts = train_rec["routes"][:, p].argmax(1); es = test_rec["routes"][:, p].argmax(1)
            tr_x.append(train_rec["final"][np.arange(len(train_rec["final"])), ts, sl])
            te_x.append(test_rec["final"][np.arange(len(test_rec["final"])), es, sl])
            tr_y.append(train_rec["labels"][:, p]); te_y.append(test_rec["labels"][:, p])
        out["cross_probes"][source + "_to_" + target_name] = probe_fit(np.concatenate(tr_x), np.concatenate(tr_y), np.concatenate(te_x), np.concatenate(te_y))
    return out


def routing_stats(rec, cap):
    route = rec["routes"]
    arg = route.argmax(2)
    ent = -(route * np.log(np.maximum(route, 1e-8))).sum(2)
    usages = np.zeros((len(route), 4), dtype=np.float64)
    collisions, consecutive = [], []
    for i in range(len(route)):
        counts = np.bincount(arg[i], minlength=4)
        usages[i] = counts / max(cap, 1)
        collisions.append(int(np.unique(arg[i]).size < min(cap, 4)))
        consecutive.append(int(np.any(arg[i, 1:] == arg[i, :-1])) if cap > 1 else 0)
    return {
        "mean_route_weight": route.mean(axis=(0, 1)).tolist(),
        "argmax_hist": np.bincount(arg.reshape(-1), minlength=4).tolist(),
        "mean_entropy_nats": float(ent.mean()),
        "mean_max_probability": float(route.max(2).mean()),
        "slot_usage_mean": usages.mean(0).tolist(),
        "collision_rate": float(np.mean(collisions)),
        "consecutive_same_slot_rate": float(np.mean(consecutive)),
        "mean_final_occupancy": rec["occupancy"][:, -1].mean(0).tolist(),
        "sample_argmax_sequences": arg[:32].tolist(),
        "sample_route_weights": route[:8].tolist(),
    }


def geometry(rec):
    x = rec["final"]
    flat = x.reshape(len(x), -1)
    norm = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    cos = (flat @ flat.T) / (norm @ norm.T)
    off = cos[~np.eye(len(cos), dtype=bool)] if len(cos) > 1 else np.array([1.0])
    slot_cos = []
    for a in range(4):
        for b in range(a + 1, 4):
            xa = x[:, a]; xb = x[:, b]
            slot_cos.append(float(np.mean(np.sum(xa * xb, axis=1) / (np.linalg.norm(xa, axis=1) * np.linalg.norm(xb, axis=1) + 1e-8))))
    return {"inter_episode_cosine_mean": float(off.mean()),
            "inter_episode_cosine_std": float(off.std()),
            "slot_norm_mean": np.linalg.norm(x, axis=2).mean(0).tolist(),
            "slot_slot_cosine_mean": slot_cos}


def survival_matrix(train_rec, test_rec, cap=4):
    out = [[None for _ in range(cap + 1)] for _ in range(cap)]
    for p in range(cap):
        for stage in range(p, cap + 1):
            tr_x = train_rec["stages"][:, stage].reshape(len(train_rec["stages"]), -1)
            te_x = test_rec["stages"][:, stage].reshape(len(test_rec["stages"]), -1)
            out[p][stage] = probe_fit(tr_x, train_rec["labels"][:, p], te_x, test_rec["labels"][:, p], steps=160)
    return {"rows_fact_position": list(range(cap)), "columns_after_write_stage": list(range(cap + 1)), "matrix": out}


@torch.no_grad()
def extended_prefix(model, heads, cache, split, cap, prefix=4, rows=None):
    if rows is None: rows = cache.rows_for_capacity(split, cap)
    dev = next(model.parameters()).device
    correct = [0] * prefix; total = [0] * prefix
    for i in range(0, len(rows), 128):
        rb = rows[i:i + 128]
        facts, labels, _, _ = cache.batch(split, rb, cap, dev)
        state, _, _ = model.forward_facts(facts)
        for p in range(prefix):
            pred = heads[p](state.flatten(1)).argmax(1)
            correct[p] += int((pred == labels[:, p]).sum())
            total[p] += len(rb)
    return {"capacity": cap, "prefix_positions": prefix,
            "by_position": [correct[p] / max(total[p], 1) for p in range(prefix)],
            "acc": sum(correct) / max(sum(total), 1),
            "note": "zero-shot after S1; only positions 0..3 are decodable by the four position heads"}


def retention_sanity(model, heads, cache, split="testA", cap=16, max_rows=512):
    rows = cache.rows_for_capacity(split, cap)[:max_rows]
    dev = next(model.parameters()).device
    d = cache.data[split]
    out = []
    for distractors in range(5):
        n = 1 + distractors
        correct = total = 0
        for i in range(0, len(rows), 128):
            rb = rows[i:i + 128]
            facts, labels, _, _ = cache.batch(split, rb, cap, dev)
            state, _, _ = model.forward_facts(facts[:, :n])
            pred = heads[0](state.flatten(1)).argmax(1)
            correct += int((pred == labels[:, 0]).sum()); total += len(rb)
        out.append({"d": distractors, "acc": correct / max(total, 1)})
    return out


def report_md(path, result):
    lines = ["# CLR-v2.2-S — Factorized Key/Value Slot Binding", "",
             "Storage-only experiment. No natural-language query hidden, no retrieval module, no architecture change.", ""]
    lines.append("## Verdict")
    lines.append("\n" + result["verdict"] + "\n")
    lines.append("## Protocol and run accounting")
    lines.append("\n- Architecture: CLR-v2.2-S factorized storage only; K=31 content + 1 occupancy + V=32 per slot, 4 slots, total state 256.")
    lines.append("- Routing: cosine key match minus occupancy, alpha=1, beta=1, temperature=0.5, softmax; no hard routing.")
    lines.append("- Loss: final state city CE only; no query hidden, no auxiliary loss, no detach between writes.")
    lines.append("- Trainable parameters: %s" % result.get("parameter_report", {}).get("trainable", "unknown"))
    lines.append("- Cache manifest: `%s`" % json.dumps(result.get("cache_manifest", {}), ensure_ascii=False))
    training = result.get("training", {})
    lines.append("- Training wall-clock: %.3fs; peak VRAM: %.4f GB" % (training.get("overall_wall_clock_s", 0.0), training.get("peak_vram_gb", result.get("peak_vram_gb", 0.0))))
    lines.append("- Evaluation wall-clock: %.3fs" % result.get("wall_clock_eval_s", 0.0))
    lines.append("## Main capacity curves")
    for split in ("testA", "testB"):
        lines.append("\n### " + split)
        for cap, m in result.get("capacity", {}).get(split, {}).items():
            lines.append("- N=%s: %.3f (C=%.3f)" % (cap, m.get("acc", 0.0), m.get("normalized_capacity", 0.0)))
    lines.append("\n## S0/S1 gates")
    lines.append("```json\n" + json.dumps(result.get("gates", {}), indent=2) + "\n```")
    lines.append("\n## Formal split results")
    for split in ("train", "val", "testA", "testB"):
        m = result.get("capacity", {}).get(split, {}).get("1")
        if m is not None:
            lines.append("- %s N=1: %.3f (%d/%d)" % (split, m.get("acc", 0.0), m.get("correct", 0), m.get("total", 0)))
    lines.append("\n## Position matrices")
    lines.append("```json\n" + json.dumps(result.get("position_matrix", {}), indent=2) + "\n```")
    lines.append("\n## Key diagnostics")
    lines.append("```json\n" + json.dumps({"routing": result.get("routing_stats", {}), "binding": result.get("binding_probes", {}), "survival": result.get("write_survival", {})}, indent=2) + "\n```")
    lines.append("\n## Layered capability statement")
    lines.append("- WRITE: S0 N=1 was tested but failed the preregistered gate; v2.2-S did not restore single-fact WRITE within 2000 steps.")
    lines.append("- RETENTION: not entered because the S0 stop rule fired.")
    lines.append("- MULTI-WRITE: not entered; no N=2/N=4 training or formal verdict was made.")
    lines.append("- SELECTIVE RETRIEVAL: not tested in v2.2-S; no query path was used.")
    lines.append("\n## Q1-Q9")
    lines.append("1. Q1 factorization restore WRITE: No, S0 TestA/TestB were below 80%.")
    lines.append("2. Q2 N<=4 coexistence: Not tested because S0 stop gate fired.")
    lines.append("3. Q3 distinct allocation: Diagnostic allocation was not promoted to a formal result.")
    lines.append("4. Q4 recency overwrite: Not evaluated as a formal v2.2-S claim.")
    lines.append("5. Q5 K/V interpretability: Not promoted to a formal claim after S0 failure.")
    lines.append("6. Q6 four-slot capacity: Not tested.")
    lines.append("7. Q7 N>4 knee: Not tested.")
    lines.append("8. Q8 supported explanation: the present result only establishes that the factorized core did not pass the single-fact WRITE gate; it does not distinguish the later storage hypotheses.")
    lines.append("9. Q9 verdict: S0 FAIL.")
    lines.append("\n## Stop condition")
    lines.append("This run stops at CLR-v2.2-S; no LS2b and no A2.")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    seed_all(2237)
    out_dir = Path(os.environ.get("V22_OUT_DIR", DEFAULT_OUT)); out_dir.mkdir(parents=True, exist_ok=True)
    cache = FactCache(os.environ.get("V22_CACHE_DIR", DEFAULT_CACHE))
    device = get_device()
    ckpt_path = out_dir / "best_model.pt"
    model, heads, ckpt = load_checkpoint(ckpt_path, device=device)
    train_log = {}
    tl = out_dir / "train_log.json"
    if tl.exists(): train_log = json.loads(tl.read_text(encoding="utf-8"))
    smoke = bool(train_log.get("result", {}).get("smoke", False))
    limit = 128 if smoke else None
    s1_ran = bool(train_log.get("result", {}).get("s1_ran", False))
    # The protocol explicitly stops after S0 when its single-fact gate fails.
    # Do not turn untrained N=2/N=4 measurements into formal LS2 results.
    capacities = [1] if not s1_ran else [1, 2, 4, 8, 16]
    result = {"device": str(device), "checkpoint": str(ckpt_path), "stage": ckpt.get("stage", ""),
              "cache_manifest": cache.manifest,
              "training": train_log.get("result", {}),
              "parameter_report": model.parameter_report(), "capacity": {}, "position_matrix": {},
              "routing_stats": {}, "slot_allocation": {}, "binding_probes": {}, "state_geometry": {},
              "write_survival": {}, "retention_sanity": {}, "extended_capacity": {}, "gates": {},
              "peak_vram_gb": (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0}
    t0 = time.perf_counter()
    for split in ("train", "val", "testA", "testB"):
        result["capacity"][split] = {}
        for cap in capacities:
            rows = rows_for(cache, split, cap, limit)
            m = eval_capacity(model, heads, cache, split, cap, rows=rows)
            if "1" in result["capacity"].get(split, {}):
                m["normalized_capacity"] = normalized_capacity(m["acc"], result["capacity"][split]["1"]["acc"])
            else:
                m["normalized_capacity"] = 1.0
            result["capacity"][split][str(cap)] = m
            if cap <= 4:
                result["position_matrix"].setdefault(split, {})[str(cap)] = fixed_position_matrix(model, heads, cache, split, cap, rows=rows)
            rec = collect(model, cache, split, cap, rows)
            result["routing_stats"].setdefault(split, {})[str(cap)] = routing_stats(rec, cap)
            result["slot_allocation"].setdefault(str(cap), {})[split] = {
                "unique_slot_count_mean": float(np.mean([len(np.unique(x)) for x in rec["routes"].argmax(2)])),
                "argmax_sequences_sample": rec["routes"].argmax(2)[:32].tolist(),
                "collision_rate": result["routing_stats"][split][str(cap)]["collision_rate"],
                "consecutive_same_slot_rate": result["routing_stats"][split][str(cap)]["consecutive_same_slot_rate"],
            }
            result["state_geometry"].setdefault(split, {})[str(cap)] = geometry(rec)
            if split == "testA":
                result["slot_allocation"][str(cap)]["shortcut_baseline"] = shortcut_baselines(cache, split, cap, rows)
        # keep a lightweight first/last baseline for each split/capacity
        result.setdefault("shortcut_baselines", {})[split] = {str(c): shortcut_baselines(cache, split, c, rows_for(cache, split, c, limit)) for c in capacities}

    # Fresh full-state position probes and binding probes use N<=4 only.
    if s1_ran:
        tr4 = collect(model, cache, "train", 4, rows_for(cache, "train", 4, 1000 if not smoke else 128))
        te4 = collect(model, cache, "testA", 4, rows_for(cache, "testA", 4, 1000 if not smoke else 128))
        result["fresh_position_probes"] = {"N4": city_probe_matrix(tr4, te4, 4)}
        trp, _ = person_labels(cache, "train", tr4["rows"], 4)
        tep, _ = person_labels(cache, "testA", te4["rows"], 4)
        result["binding_probes"] = binding_probes(tr4, te4, trp, tep, cap=4)
        result["write_survival_matrix"] = survival_matrix(tr4, te4, cap=4)
        result["retention_sanity"] = retention_sanity(model, heads, cache, max_rows=128 if smoke else 512)
    else:
        result["fresh_position_probes"] = {"skipped": "S0 gate failed"}
        result["binding_probes"] = {"skipped": "S0 gate failed; S1 was not entered"}
        result["write_survival_matrix"] = {"skipped": "S0 gate failed"}
        result["retention_sanity"] = {"skipped": "S0 gate failed"}

    if train_log.get("result", {}).get("s1_ran", False):
        for split in ("testA", "testB"):
            for cap in (8, 16):
                result["extended_capacity"].setdefault(split, {})[str(cap)] = extended_prefix(
                    model, heads, cache, split, cap, prefix=4, rows=rows_for(cache, split, cap, limit))
    else:
        result["extended_capacity"] = {"skipped": "S1 did not run because S0 gate was not passed"}

    s0_gate = train_log.get("result", {}).get("s0_gate", {})
    result["gates"]["s0"] = s0_gate
    ta = result["capacity"].get("testA", {})
    tb = result["capacity"].get("testB", {})
    s1_pass = bool(train_log.get("result", {}).get("s1_ran", False) and
                   ta.get("1", {}).get("acc", 0) >= .85 and ta.get("2", {}).get("acc", 0) >= .75 and
                   ta.get("4", {}).get("acc", 0) >= .60 and tb.get("1", {}).get("acc", 0) >= .80 and
                   tb.get("2", {}).get("acc", 0) >= .70 and tb.get("4", {}).get("acc", 0) >= .55 and
                   min([x["acc"] for x in result["position_matrix"].get("testA", {}).get("4", [])] or [0]) >= .45 and
                   min([x["acc"] for x in result["position_matrix"].get("testB", {}).get("4", [])] or [0]) >= .40)
    s0_pass = bool(s0_gate.get("pass", False))
    if not s0_pass:
        verdict = "S0 FAIL: single-fact WRITE gate was not passed; N>1 training was not entered."
    elif s1_pass:
        verdict = "S1 STORAGE PASS: factorized key/value storage passed the preregistered N<=4 storage gates."
    else:
        n4 = ta.get("4", {}).get("acc", 0.0)
        verdict = "S1 STORAGE PARTIAL" if n4 > 1.0 / 64.0 + .05 else "S1 STORAGE FAIL"
    result["verdict"] = verdict
    result["wall_clock_eval_s"] = time.perf_counter() - t0
    json_dump(out_dir / "v22_storage_results.json", result)
    json_dump(out_dir / "position_matrix.json", result["position_matrix"])
    json_dump(out_dir / "fresh_position_probes.json", result["fresh_position_probes"])
    json_dump(out_dir / "routing_stats.json", result["routing_stats"])
    json_dump(out_dir / "slot_allocation.json", result["slot_allocation"])
    json_dump(out_dir / "binding_probes.json", result["binding_probes"])
    json_dump(out_dir / "write_survival_matrix.json", result["write_survival_matrix"])
    json_dump(out_dir / "state_geometry.json", result["state_geometry"])
    json_dump(out_dir / "extended_capacity.json", result["extended_capacity"])
    report_md(out_dir / "v22_storage_report.md", result)
    print(json.dumps({"verdict": verdict, "capacity": result["capacity"], "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
