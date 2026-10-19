#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_seqstate_s0 / analyze_results.py — merge per-run JSONs into the final
S0 report tables (params, k-curve, early readout, interventions, dynamics,
collapse diagnostics). CPU-only; run after the pipeline finishes:

    py analyze_results.py [--runs-dir runs] [--tag s0]
"""
import argparse
import glob
import json
import os

CONFIGS = [
    ("vector 1x64", "vector", 1, 64),
    ("sequence 2x32", "sequence", 2, 32),
    ("sequence 4x16", "sequence", 4, 16),
    ("sequence 8x8", "sequence", 8, 8),
]


def load(path):
    try:
        return json.load(open(path))
    except Exception as e:
        return {"_error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--tag", default="s0")
    args = ap.parse_args()

    rows = []
    for label, st, K, d in CONFIGS:
        d = os.path.join(args.runs_dir, f"{args.tag}_{st}_{K}x{d}_s42")
        res = load(os.path.join(d, "results.json"))
        ev = load(os.path.join(d, "eval_s0.json"))
        dyn = load(os.path.join(d, "dynamics_s0.json"))
        rows.append((label, d, res, ev, dyn))

    print("# S0 screening — merged results\n")
    print("## 1. params + final k=4 accuracy\n")
    print("| config | trainable | backbone frozen | ceiling | with_S t2p | without_S | delta | best_step |")
    print("|:--|--:|--:|--:|--:|--:|--:|--:|")
    for label, d, res, ev, dyn in rows:
        c = res.get("config", {})
        w = res.get("final_with", {}).get("t2p", {})
        o = res.get("final_without", {}).get("t2p", {})
        n = w.get("total", 0)
        print(f"| {label} | {res.get('trainable_params', '?'):,} | "
              f"{res.get('backbone_frozen_params', '?')} | "
              f"{res.get('ceiling', '?')} | {w.get('correct','?')}/{n} "
              f"({w.get('acc', '?')}) | {o.get('correct','?')}/{o.get('total','?')} "
              f"| {res.get('eval_log',[{}])[-1].get('delta_t2p', '?')} | "
              f"{res.get('eval_log', [{}])[-1].get('step', '?')} |")

    print("\n## 2. k-curve (eval-time max_ut, t2p)\n")
    print("| config | params | k1 | k2 | k3 | k4 |")
    print("|:--|--:|--:|--:|--:|--:|")
    for label, d, res, ev, dyn in rows:
        kc = ev.get("k_curve", {})
        cells = []
        for k in range(1, 5):
            v = kc.get(f"k{k}", {})
            cells.append(f"{v.get('t2p_counts', '?')} = {v.get('t2p_acc', '?')}")
        print(f"| {label} | {res.get('trainable_params', '?'):,} | "
              + " | ".join(cells) + " |")

    print("\n### k-curve W95 + without_S floor\n")
    for label, d, res, ev, dyn in rows:
        kc = ev.get("k_curve", {})
        parts = []
        for k in range(1, 5):
            v = kc.get(f"k{k}", {})
            parts.append(f"k{k}: {v.get('t2p_counts','?')} "
                         f"W95{v.get('t2p_wilson95','?')} "
                         f"(without {v.get('without_t2p_counts','?')})")
        print(f"- **{label}**  " + ";  ".join(parts))

    print("\n## 3. early readout (full 4-loop evolution, t2p per loop)\n")
    for label, d, res, ev, dyn in rows:
        er = ev.get("early_readout", {})
        parts = [f"loop{i+1}: {er.get(f'loop{i+1}', {}).get('t2p_counts', '?')}"
                 for i in range(4)]
        print(f"- **{label}**  " + ",  ".join(parts))

    print("\n## 4. interventions (best/any sequence config)\n")
    for label, d, res, ev, dyn in rows:
        if label == "vector 1x64":
            continue
        inv = ev.get("interventions", {})
        if not inv or "_error" in inv or "note" in inv:
            print(f"- **{label}**: {inv.get('note', 'no data')}")
            continue
        order = ["normal", "zero_slots", "random_slots", "static_trained"]
        order += [f"permute_{i}" for i in (1, 2, 3)]
        order += [f"single_slot_{j}" for j in range(1, 5)]
        order += [f"dropout_p{p}_s{s}" for p in (0.25, 0.5, 0.75) for s in (1, 2, 3)]
        print(f"- **{label}**  perms used: {inv.get('_perms', [])}")
        for name in order:
            v = inv.get(name)
            if v:
                print(f"    {name:18s} t2p={v['t2p_counts']} = {v['t2p_acc']:.4f} "
                      f"W95{v['t2p_wilson95']}")

    print("\n## 5. dynamics / collapse diagnostics\n")
    for label, d, res, ev, dyn in rows:
        if "_error" in dyn:
            print(f"- **{label}**: no dynamics data")
            continue
        print(f"- **{label}** (K={dyn.get('K')}, d={dyn.get('d')})")
        sn = dyn.get("per_slot_norm", {})
        print("    per-slot norm (loop1..4):")
        for i in range(4):
            cell = sn.get(f"loop{i+1}", {})
            vals = ", ".join(f"s{j+1}={cell.get(f'slot{j+1}', {}).get('mean', '?')}"
                             for j in range(dyn.get("K", 1)))
            print(f"      loop{i+1}: {vals}")
        er_ = dyn.get("effective_rank", {})
        print("    effective rank (participation ratio): " +
              ", ".join(f"loop{i+1}={er_.get(f'loop{i+1}', {}).get('participation_ratio', '?')}"
                        for i in range(4)))
        pc = dyn.get("pooled_cross_traj_cos", {})
        print("    pooled cross-traj cos: " +
              ", ".join(f"loop{i+1}={pc.get(f'loop{i+1}', {}).get('mean', '?')}"
                        for i in range(4)))
        sm = dyn.get("slot_cos_matrix", {})
        if sm:
            print("    mean pairwise slot cos: " +
                  ", ".join(f"loop{i+1}={sm.get(f'loop{i+1}', {}).get('mean_pairwise', '?')}"
                            for i in range(4)))
        xc = dyn.get("cross_traj_cos_slot", {})
        if xc:
            print("    cross-trajectory cos per slot (loop1..4):")
            for i in range(4):
                cell = xc.get(f"loop{i+1}", {})
                vals = ", ".join(f"s{j+1}={cell.get(f'slot{j+1}', {}).get('mean', '?')}"
                                 for j in range(dyn.get("K", 1)))
                print(f"      loop{i+1}: {vals}")

    print("\n## 6. run metadata\n")
    for label, d, res, ev, dyn in rows:
        print(f"- {label}: wall={res.get('wall_clock_s')}s, "
              f"vram_peak={res.get('vram_peak_gb')}GB, "
              f"ckpt_sha256={res.get('ckpt_sha256')}")


if __name__ == "__main__":
    main()
