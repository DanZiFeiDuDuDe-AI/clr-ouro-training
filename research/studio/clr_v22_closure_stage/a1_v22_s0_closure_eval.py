import json
import time
from pathlib import Path

from a1_v22_s0_closure_common import closure_paths, get_device, geometry_for, load_representations, probe_one


def prepare(data):
    for rep in data["splits"].values():
        rep["key_candidate"] = rep["key_candidate"][:, 0]
        rep["value_candidate"] = rep["value_candidate"][:, 0]
        rep["fulls"] = rep["state"].reshape(len(rep["state"]), -1)
        rep["force_slot0_full"] = rep["force_slot0"].reshape(len(rep["force_slot0"]), -1)
        rep["force_uniform_full"] = rep["force_uniform"].reshape(len(rep["force_uniform"]), -1)
        rep["bypass_full"] = rep["bypass"].reshape(len(rep["bypass"]), -1)
        rep["zero_init_full"] = rep["zero_init"].reshape(len(rep["zero_init"]), -1)
        rep["random_init_full"] = rep["random_init"].reshape(len(rep["random_init"]), -1)
    return data


def main():
    out = closure_paths(); data = prepare(load_representations(out / "closure_representations.npz"))
    train = json.loads((out / "closure_probe_train_results.json").read_text(encoding="utf-8"))
    device = get_device(); t0 = time.perf_counter()
    # All ablation probes are fresh Linear(256,64), using the same train/val/TestA/TestB protocol.
    ab = {}
    for name in ("fulls", "force_slot0_full", "force_uniform_full", "bypass_full", "zero_init_full", "random_init_full"):
        ab[name] = probe_one(data, name, 256, device=device)
    routing = {k: ab[k] for k in ("fulls", "force_slot0_full", "force_uniform_full")}
    bypass = {k: ab[k] for k in ("bypass_full", "fulls")}
    init_controls = {k: probe_one(data, k, 256, device=device)
                     for k in ("fulls", "zero_init_full", "random_init_full")}
    (out / "routing_ablation.json").write_text(json.dumps(routing, indent=2), encoding="utf-8")
    (out / "gru_bypass.json").write_text(json.dumps(bypass, indent=2), encoding="utf-8")
    (out / "init_state.json").write_text(json.dumps(init_controls, indent=2), encoding="utf-8")

    geo = {}
    tr = data["splits"]["train"]
    for name in ("H", "value_read", "value_candidate", "slot0", "fulls"):
        geo[name] = geometry_for(tr[name].reshape(len(tr[name]), -1), tr["labels"])
    (out / "geometry.json").write_text(json.dumps(geo, indent=2), encoding="utf-8")

    testA = {k: v.get("testA_acc") for k, v in train["probes"].items()}
    testB = {k: v.get("testB_acc") for k, v in train["probes"].items()}
    result = {
        "protocol": {"checkpoint_frozen": True, "main_parameters_updated": False, "ouro_rerun": False,
                     "probe": "AdamW lr=1e-3 weight_decay=0 batch=128 max_epochs=100 patience=10 seed=42"},
        "v22_oracle_head": {"testA": 0.4175, "testB": 0.3750},
        "reported_ls0_distributed_reference": {"testA": 0.91, "testB": 0.90},
        "probe_results": train["probes"], "kv_factorization": train["kv"],
        "bottleneck_baselines": train["bottlenecks"], "single_slot": train["single_slot"],
        "routing_ablation": routing, "gru_bypass": bypass, "init_state": init_controls, "geometry": geo,
        "testA_summary": testA, "testB_summary": testB,
        "wall_clock_eval_s": time.perf_counter() - t0,
    }
    # Pre-registered interpretation rules, applied only to the measured closure numbers.
    b32 = train["bottlenecks"]["32"]
    vr = train["probes"]["value_read"]
    vc = train["probes"]["value_candidate"]
    ss = train["single_slot"]
    result["interpretation"] = {
        "q1_2048_to_32_sufficient": bool(b32["testA_acc"] >= 0.85 and b32["testB_acc"] >= 0.85),
        "q2_valueread_lost_information": bool(vr["testA_acc"] < 0.60 or vr["testB_acc"] < 0.60),
        "q3_valuegru_further_loss": bool(vc["testA_acc"] + 0.05 < vr["testA_acc"] or vc["testB_acc"] + 0.05 < vr["testB_acc"]),
        "q4_routing_main_at_n1": bool(abs(routing["fulls"]["testA_acc"] - routing["force_slot0_full"]["testA_acc"]) > 0.10),
        "q5_fixed_single_slot_recovers_ls0": bool(ss["testA_acc"] >= 0.85 and ss["testB_acc"] >= 0.85),
        "q6_ls0_distributed_dependency": "inferred only if SS-64 is low while the reported LS0 distributed reference is high",
        "q7_primary_failure": "closure measurement; see bottleneck, ValueRead, ValueGRU, routing, and SS-64 rows",
        "q8_next_direction": "diagnostic recommendation only; no architecture change implemented",
    }
    (out / "closure_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report(out, result)
    print(json.dumps(result, indent=2))


def report(out, r):
    lines = ["# CLR-v2.2-S0 Closure — Locate the Single-Fact Write Collapse", "",
             "All v2.2-S0 parameters were frozen. No Ouro rerun, no S1, no architecture change.", "",
             "## Verdict context", "",
             "The v2.2-S0 oracle head was 41.75% on TestA and 37.50% on TestB; the original LS0 reference was about 91%/90%.", "",
             "## Main representation probes", "",
             "| Representation | TestA | TestB |", "|---|---:|---:|"]
    for name, m in r["probe_results"].items():
        lines.append("| %s | %.3f | %.3f |" % (name, m["testA_acc"], m["testB_acc"]))
    lines += ["| H direct (reported LS0 reference) | 0.910 | 0.900 |", ""]
    lines += ["## Bottleneck baselines", "", "| H -> d -> city | TestA | TestB |", "|---|---:|---:|"]
    for d, m in r["bottleneck_baselines"].items(): lines.append("| %s | %.3f | %.3f |" % (d, m["testA_acc"], m["testB_acc"]))
    lines += ["", "## Single-slot baseline", "", "| Baseline | TestA | TestB |", "|---|---:|---:|",
              "| SS-64 fixed slot0 | %.3f | %.3f |" % (r["single_slot"]["testA_acc"], r["single_slot"]["testB_acc"]), ""]
    lines += ["## Key/value probes", "", "```json", json.dumps(r["kv_factorization"], indent=2), "```", ""]
    lines += ["## Routing and GRU bypass", "", "```json", json.dumps({"routing": r["routing_ablation"], "gru_bypass": r["gru_bypass"]}, indent=2), "```", ""]
    lines += ["## Init-state controls", "", "```json", json.dumps(r["init_state"], indent=2), "```", ""]
    lines += ["## Geometry", "", "```json", json.dumps(r["geometry"], indent=2), "```", ""]
    lines += ["## Q1-Q8", ""]
    for k, v in r["interpretation"].items(): lines.append("- %s: %s" % (k, v))
    lines += ["", "## Decision boundary", "",
              "This is a closure diagnosis only. It does not authorize v2.3, S1, query retrieval, or A2.", "",
              "## Stop", "", "Stopped after CLR-v2.2-S0 Closure."]
    (out / "closure_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__": main()
