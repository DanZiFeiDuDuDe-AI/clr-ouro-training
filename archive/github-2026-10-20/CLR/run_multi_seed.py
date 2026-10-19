#!/usr/bin/env python3
"""
CLR Phase 1 v2 — Multi-seed runner for CLUTRR Local (diagnostic, no LoRA)
Seeds: 42, 1103, 130, 520

Runs sequentially on GTX 1650, each ~45 min → ~3h total.
Saves per-seed results in H:\CLR\02_results\seed_<N>/
"""
import subprocess, sys, os, json, time, shutil

WORKSPACE = "H:/CLR"
SEEDS = [42, 1103, 130, 520]
SCRIPT = f"{WORKSPACE}/local_train_clutrr_v2.py"
OUT_BASE = f"{WORKSPACE}/02_results"
TRAINING_LOG = f"{WORKSPACE}/training.log"

base_env = os.environ.copy()
base_env["PYTHONUNBUFFERED"] = "1"
# Remove smoke if set
base_env.pop("CLR_SMOKE", None)

results_summary = []

for seed in SEEDS:
    tag = f"seed_{seed}"
    out_dir = f"{OUT_BASE}/{tag}"
    ckpt_dir = f"{WORKSPACE}/checkpoints/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Modify the script to use this seed — simplest: pass as env var
    env = base_env.copy()
    env["CLR_SEED"] = str(seed)
    env["CLR_CKPT_DIR"] = ckpt_dir

    print(f"\n{'='*60}")
    print(f"  Running seed={seed}  (output → {out_dir})")
    print(f"{'='*60}\n", flush=True)

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, SCRIPT],
        capture_output=True, text=True, env=env, timeout=7200
    )
    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.1f}s", flush=True)

    # Save stdout/stderr
    with open(f"{out_dir}/stdout.log", "w") as f:
        f.write(proc.stdout)
    with open(f"{out_dir}/stderr.log", "w") as f:
        f.write(proc.stderr)

    # Parse results.json from ckpt_dir (default OUT is in the script)
    source_result = f"{WORKSPACE}/checkpoints/local_clutrr_v2_nolora/results.json"
    if os.path.exists(source_result):
        with open(source_result) as f:
            data = json.load(f)
        data["seed"] = seed
        data["elapsed_sec"] = elapsed
        with open(f"{out_dir}/results.json", "w") as f:
            json.dump(data, f, indent=2)
        delta = data.get("best_delta_t2p", "N/A")
        print(f"  best_delta_t2p = {delta}", flush=True)
        if isinstance(delta, (int, float)):
            results_summary.append({"seed": seed, "best_delta": delta,
                                    "final_delta": data.get("final_delta_t2p", None),
                                    "final_with": data.get("final_with_S_t2p", None),
                                    "final_without": data.get("final_without_S_t2p", None),
                                    "elapsed_min": elapsed / 60})
    else:
        print(f"  WARNING: results.json not found at {source_result}", flush=True)
        results_summary.append({"seed": seed, "error": "no results.json"})

# Summary
print(f"\n{'='*60}")
print("  MULTI-SEED RESULTS SUMMARY")
print(f"{'='*60}")
all_deltas = []
for r in results_summary:
    d = r.get("best_delta")
    s = r["seed"]
    print(f"  seed {s:4d}: best_delta_t2p={d}")
    if isinstance(d, (int, float)):
        all_deltas.append(d)

if all_deltas:
    import statistics
    mean = statistics.mean(all_deltas)
    std = statistics.stdev(all_deltas) if len(all_deltas) > 1 else 0.0
    print(f"\n  Mean Δt2p = {mean:.4f} ± {std:.4f}  ({len(all_deltas)} seeds)")
    print(f"  Min = {min(all_deltas):.4f}, Max = {max(all_deltas):.4f}")

with open(f"{OUT_BASE}/multi_seed_summary.json", "w") as f:
    json.dump({"seeds": results_summary,
               "mean_delta": statistics.mean(all_deltas) if all_deltas else None,
               "std_delta": statistics.stdev(all_deltas) if len(all_deltas) > 1 else None,
               "n_seeds": len(all_deltas)}, f, indent=2)

print(f"\nSummary saved to {OUT_BASE}/multi_seed_summary.json")
print("Done.", flush=True)
