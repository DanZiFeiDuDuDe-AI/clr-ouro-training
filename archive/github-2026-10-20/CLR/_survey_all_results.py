#!/usr/bin/env python3
"""Survey all existing results in checkpoints/ and 02_results/."""
import json, os, re

WORKSPACE = "H:/CLR"
CKPT = f"{WORKSPACE}/checkpoints"
OUT = f"{WORKSPACE}/02_results"

print("=" * 70)
print("EXISTING RESULT SURVEY")
print("=" * 70)

# 1. All results.json in checkpoints/
print("\n--- checkpoints/*/results.json ---")
found = []
for dname in sorted(os.listdir(CKPT)):
    rpath = os.path.join(CKPT, dname, "results.json")
    if not os.path.exists(rpath):
        continue
    with open(rpath) as f:
        try:
            d = json.load(f)
        except:
            print(f"  {dname}: invalid JSON")
            continue
    phase = "STATIC" if d.get("phase_b_static_s") else "evolved"
    bd = d.get("best_delta_t2p", "?")
    if bd != "?":
        bd = f"{bd*100:+.1f}pp"
    fd = d.get("final_delta_t2p", "?")
    if fd != "?":
        fd = f"{fd*100:+.1f}pp"
    cl = d.get("bayes_ceiling_t2p", "?")
    if cl != "?":
        cl = f"{cl*100:.1f}%"
    ds = str(d.get("config", {}).get("DS", "?"))
    model = d.get("model", "?")
    steps = str(d.get("config", {}).get("steps", "?"))
    print(f"  {dname:<35s} DS={ds:<4s} {phase:<7s} steps={steps:<4s} "
          f"ceiling={cl:<7s} bestΔ={bd:<8s} finalΔ={fd:<8s}")
    
    # Check eval_log for per-step deltas
    elog = d.get("eval_log", [])
    delta_str = ""
    if elog and isinstance(elog, list) and len(elog) > 0:
        try:
            parts = []
            for e in elog:
                if isinstance(e, dict) and "delta" in e and isinstance(e["delta"], dict):
                    d_val = e["delta"].get("t2p")
                    if d_val is not None:
                        parts.append(f"s{e.get('step','?')}={d_val*100:+.1f}pp")
            delta_str = " | ".join(parts)
        except:
            pass
    if delta_str:
        print(f"  {'':35s} deltas: {delta_str}")
    found.append(dname)

print(f"\nTotal: {len(found)} results.json files found")

# 2. Check 02_results/
print("\n--- 02_results/ ---")
for fname in sorted(os.listdir(OUT)):
    fpath = os.path.join(OUT, fname)
    if os.path.isfile(fpath):
        sz = os.path.getsize(fpath)
        print(f"  {fname:<35s} {sz:,} bytes")
    elif os.path.isdir(fpath):
        print(f"  {fname:<35s} (dir)")

# 3. Phase-C ready? check max_turns in data
print("\n--- Data generator depths ---")
# This is script-level, let me check clutrr_data.py
from clutrr_data import generate_dataset, bayes_ceiling, RELATIONS
import random
for depth in [(1,), (1,2), (1,2,3), (2,3,4)]:
    rng = random.Random(42)
    data = generate_dataset(50, depth, None, rng)
    max_t = max(len(t["turns"]) for t in data)
    print(f"  generate_dataset sample (depth={depth}): max turns = {max_t}")

print("\n--- Summary of remaining gaps ---")
print("""
Gap 1: Phase C Lifetime (k=1,2,4,8) 
    → Need depth-limited data generator
Gap 2: Phase D Capacity (DS=16,32,128) 
    → DS=64 already done; need sweep for rest
Gap 3: Phase E Probe Analysis (S_t trajectory tracking)
    → Training log has |S| tracking, just need post-hoc visualization
Gap 4: Static S in other seeds (currently only seed 42)
    → To match evolved multi-seed robustness
""")
