import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from a1_v22_s0_closure_common import LinearProbe, accuracy, load_representations, train_classifier
from a1_v22_storage_common import DEFAULT_CACHE, FactCache, get_device, seed_all
from a1_v23_r2_common import (DEFAULT_OUT, CAPACITIES, collect_states, eval_capacity, json_dump,
                              load_checkpoint, position_matrix, rows_for)

ROOT = Path(r"D:\training")
CLOSURE_OUT = ROOT / "checkpoints" / "clr_v22_s0_closure"


def fresh_probe(xtr, ytr, xv, yv, xA, yA, xB, yB, device, seed=42):
    seed_all(seed)
    m = LinearProbe(xtr.shape[1])
    m, info = train_classifier(m, xtr, ytr, xv, yv, device, lr=1e-3, max_epochs=100, patience=10, seed=42)
    return {"train_acc": accuracy(m, xtr, ytr, device), "val_acc": info["best_val_acc"],
            "testA_acc": accuracy(m, xA, yA, device), "testB_acc": accuracy(m, xB, yB, device),
            "epochs": info["epochs"], "seed": seed}


def local_probe_matrix(core, cache, device):
    reps = {}; labels = {}
    for split in ("train", "val", "testA", "testB"):
        reps[split], labels[split] = collect_states(core, cache, split, 4)
    result = {}
    for slot in range(4):
        result[f"slot{slot}"] = fresh_probe(
            reps["train"][:, slot, :], labels["train"][:, slot],
            reps["val"][:, slot, :], labels["val"][:, slot],
            reps["testA"][:, slot, :], labels["testA"][:, slot],
            reps["testB"][:, slot, :], labels["testB"][:, slot], device)
    return result, reps, labels


def purity_matrix(reps, labels, device):
    out = {"testA": {}, "testB": {}}
    for src in range(4):
        for target in range(4):
            key = f"slot{src}_to_city{target}"
            r = fresh_probe(reps["train"][:, src, :], labels["train"][:, target],
                            reps["val"][:, src, :], labels["val"][:, target],
                            reps["testA"][:, src, :], labels["testA"][:, target],
                            reps["testB"][:, src, :], labels["testB"][:, target], device)
            out["testA"][key] = r["testA_acc"]
            out["testB"][key] = r["testB_acc"]
    for split in ("testA", "testB"):
        mat = [[out[split][f"slot{i}_to_city{j}"] for j in range(4)] for i in range(4)]
        out[split + "_matrix"] = mat
        out[split + "_diagonal_mean"] = float(np.mean([mat[i][i] for i in range(4)]))
        out[split + "_off_diagonal_mean"] = float(np.mean([mat[i][j] for i in range(4) for j in range(4) if i != j]))
    return out


def preservation(core, cache):
    core.eval(); device = next(core.parameters()).device
    result = {}
    for split in ("train", "val", "testA", "testB"):
        rows = rows_for(cache, split, 4); max_by_write = []
        for i in range(0, len(rows), 128):
            facts, _, _, _ = cache.batch(split, rows[i:i + 128], 4, device)
            _, states = core.forward_facts(facts, slot_indices=[0, 1, 2, 3], keep_states=True)
            for t in range(4):
                other = [j for j in range(4) if j != t]
                max_by_write.append(float((states[t + 1][:, other] - states[t][:, other]).abs().max().item()))
        result[split] = {"max_abs_untouched_slot_drift": max(max_by_write or [0.0]),
                         "all_exact_zero": bool(max(max_by_write or [0.0]) == 0.0)}
    return result


def gradient_interference(core, heads, cache, device):
    core.eval(); heads.eval(); rows = rows_for(cache, "train", 4)[:128]
    facts, labels, _, _ = cache.batch("train", rows, 4, device)
    vectors = {}; norms = {}
    for p in range(4):
        core.zero_grad(set_to_none=True); heads.zero_grad(set_to_none=True)
        state, _ = core.forward_facts(facts, slot_indices=[0, 1, 2, 3])
        loss = F.cross_entropy(heads[p](state[:, p, :]).float(), labels[:, p])
        loss.backward()
        vec = torch.cat([core.content_read.weight.grad.flatten(), core.shared_gru.weight_ih.grad.flatten()]).detach()
        vectors[p] = vec; norms[p] = float(vec.norm().item())
    cos = {}
    for i in range(4):
        for j in range(i + 1, 4):
            cos[f"pos{i}_pos{j}"] = float(F.cosine_similarity(vectors[i][None], vectors[j][None]).item())
    return {"batch": len(rows), "gradient_norm": norms, "pairwise_gradient_cosine": cos,
            "note": "Diagnostic only; no optimizer step and no architecture/optimizer change."}


@torch.no_grad()
def overload_eval(core, heads, cache, split, cap):
    core.eval(); heads.eval(); device = next(core.parameters()).device
    rows = rows_for(cache, split, cap); good = total = 0
    for i in range(0, len(rows), 128):
        facts, labels, pos, _ = cache.batch(split, rows[i:i + 128], cap, device)
        state, _ = core.forward_facts(facts, slot_indices=[p % 4 for p in range(cap)])
        p = pos
        slot_pos = p % 4
        pred = torch.empty_like(p)
        for q in range(4):
            mask = slot_pos == q
            if mask.any():
                pred[mask] = heads[q](state[mask, q, :]).argmax(1)
        y = labels[torch.arange(len(p), device=device), p]
        good += int((pred == y).sum()); total += len(p)
    return {"acc": good / max(total, 1), "correct": good, "total": total}


def main():
    out = Path(os.environ.get("V23_R2_OUT_DIR", DEFAULT_OUT)); out.mkdir(parents=True, exist_ok=True)
    device = get_device(); log = json.loads((out / "train_log.json").read_text(encoding="utf-8"))
    o1_ran = bool(log.get("o1_ran", False))
    if not o1_ran:
        result = {"verdict": "O0 FAIL", "o0": log.get("o0"), "o0_gate": log.get("o0_gate"), "o1_ran": False,
                  "skipped": "O1 diagnostics skipped because O0 gate failed"}
        json_dump(out / "v23_r2_results.json", result)
        json_dump(out / "position_matrix.json", {"skipped": "O0 gate failed"})
        json_dump(out / "fresh_local_probes.json", {"skipped": "O0 gate failed"})
        json_dump(out / "slot_purity.json", {"skipped": "O0 gate failed"})
        json_dump(out / "gradient_interference.json", {"skipped": "O0 gate failed"})
        json_dump(out / "overload_capacity.json", {"skipped": "O0 gate failed"})
        (out / "v23_r2_report.md").write_text("# CLR-v2.3-OA-r2\n\nO0 FAIL; stopped before O1.", encoding="utf-8")
        print(json.dumps(result, indent=2)); return

    core, heads, ck = load_checkpoint(out / "best_model.pt", device, n_heads=4)
    cache = FactCache(os.environ.get("V22_CACHE_DIR", DEFAULT_CACHE))
    capacity = {split: {str(c): eval_capacity(core, heads, cache, split, c) for c in CAPACITIES}
                for split in ("train", "val", "testA", "testB")}
    pos = {split: {str(c): position_matrix(core, heads, cache, split, c) for c in CAPACITIES}
           for split in ("testA", "testB")}
    json_dump(out / "position_matrix.json", pos)
    fresh, reps, labels = local_probe_matrix(core, cache, device)
    purity = purity_matrix(reps, labels, device)
    preserve = preservation(core, cache)
    grad = gradient_interference(core, heads, cache, device)
    json_dump(out / "fresh_local_probes.json", fresh)
    json_dump(out / "slot_purity.json", purity)
    json_dump(out / "gradient_interference.json", grad)
    overload = {"skipped": "O1 did not pass"}
    gates = {
        "testA": {"N1": capacity["testA"]["1"]["acc"] >= .80, "N2": capacity["testA"]["2"]["acc"] >= .78,
                  "N4": capacity["testA"]["4"]["acc"] >= .75},
        "testB": {"N1": capacity["testB"]["1"]["acc"] >= .78, "N2": capacity["testB"]["2"]["acc"] >= .75,
                  "N4": capacity["testB"]["4"]["acc"] >= .70},
        "N4_all_positions_ge_68": bool(min(pos["testA"]["4"][i]["acc"] for i in range(4)) >= .68 and
                                         min(pos["testB"]["4"][i]["acc"] for i in range(4)) >= .68),
    }
    o1_pass = bool(all(gates[split].values()) for split in ("testA", "testB")) and gates["N4_all_positions_ge_68"]
    if o1_pass:
        overload = {split: {str(c): overload_eval(core, heads, cache, split, c) for c in (8, 16)}
                    for split in ("testA", "testB")}
    json_dump(out / "overload_capacity.json", overload)
    result = {"verdict": "O1 STORAGE PASS" if o1_pass else "O1 FAIL", "o0": log.get("o0"),
              "o0_gate": log.get("o0_gate"), "o1": log.get("o1"), "capacity": capacity,
              "position_matrix": pos, "gates": gates, "fresh_local_probes": fresh, "slot_purity": purity,
              "numerical_preservation": preserve, "gradient_interference": grad,
              "overload_capacity": overload, "o1_ran": True, "checkpoint_step": ck.get("step"),
              "interpretation": ""}
    if o1_pass:
        result["interpretation"] = "Oracle-local storage passed N<=4; overload measured only after pass."
    elif all(capacity["testA"][str(c)]["acc"] > .015625 and capacity["testB"][str(c)]["acc"] > .015625 for c in CAPACITIES):
        result["interpretation"] = "O0 passed but O1 did not pass; inspect local probes and shared-parameter gradient interference."
    else:
        result["interpretation"] = "O1 failed with one or more capacities near chance."
    json_dump(out / "v23_r2_results.json", result)
    report(out, result)
    print(json.dumps({"verdict": result["verdict"], "capacity": capacity, "gates": gates,
                      "preservation": preserve, "fresh": fresh, "purity": purity}, indent=2, default=str))


def report(out, r):
    cap = r["capacity"]; gates = r["gates"]; fresh = r["fresh_local_probes"]; purity = r["slot_purity"]
    lines = ["# CLR-v2.3-OA-r2 — Canonical Oracle Allocation Retest", "", f"## Verdict: **{r['verdict']}**", "",
             "No learned router, query retrieval, flatten readout, trainable init_s, or A2 was used.", "",
             "## Capacity curve", "", "| Split | N=1 | N=2 | N=4 |", "|---|---:|---:|---:|"]
    for s in ("testA", "testB"):
        lines.append(f"| {s} | {cap[s]['1']['acc']:.3f} | {cap[s]['2']['acc']:.3f} | {cap[s]['4']['acc']:.3f} |")
    lines += ["", "## Position matrix", "", "### TestA"]
    for c in (1, 2, 4): lines.append(f"- N={c}: " + ", ".join(f"pos{i}={x['acc']:.3f}" for i,x in enumerate(r.get("position_matrix", {}).get("testA", {}).get(str(c), []))))
    lines += ["", "### TestB"]
    for c in (1, 2, 4): lines.append(f"- N={c}: " + ", ".join(f"pos{i}={x['acc']:.3f}" for i,x in enumerate(r.get("position_matrix", {}).get("testB", {}).get(str(c), []))))
    lines += ["", "## O1 gate", "", json.dumps(gates, indent=2), "", "## Fresh local probes", ""]
    for k,v in fresh.items(): lines.append(f"- {k}: TestA={v['testA_acc']:.3f}, TestB={v['testB_acc']:.3f}")
    lines += ["", "## Slot purity", "", json.dumps(purity, indent=2), "", "## Numerical preservation", "", json.dumps(r["numerical_preservation"], indent=2),
              "", "## Shared-parameter gradient interference", "", json.dumps(r["gradient_interference"], indent=2),
              "", "## Interpretation", "", r["interpretation"], "", "## Stop", "", "Completed CLR-v2.3-OA-r2; no next architecture was started."]
    (out / "v23_r2_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
