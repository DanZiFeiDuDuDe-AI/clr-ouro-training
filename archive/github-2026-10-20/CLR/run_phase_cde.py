#!/usr/bin/env python3
"""
CLR Local: Phase C (Lifetime), D (Capacity), E (Probe Analysis)
Runs on GTX 1650, Qwen3.5-0.8B (4-bit), no LoRA, CLUTRR FiLM diagnostic.

Phase C: Lifetime experiment (context window budget k = 1, 2, 4, 8, 16)
  - What's the maximum number of turns S_t can carry state?
  - Train on trajectories up to k turns, eval on same max length.
  - Hypothesis: ACC degrades with k if S_t saturates.

Phase D: Capacity experiment (DS × n_facts hidden in trajectory)
  - DS = 16, 32, 64, 128
  - Train on 3-hop trajectories, compare ceiling-adjusted delta.
  - Hypothesis: larger DS supports more compositional depth.

Phase E: Probe analysis (S_t → answer relation decoding)
  - After training, freeze model, run eval with probe accuracy tracked.
  - Measure: probe_acc over steps, per-layer FiLM statistics.
  - Compare: evolved S vs static S vs w/o S probe acc.
"""
import subprocess, os, json, sys, time, re

WORKSPACE = "H:/CLR"
SCRIPT = f"{WORKSPACE}/local_train_clutrr_v2.py"
VENV_PYTHON = "C:/Users/Ouyang Quan/.openclaw/clr_venv/Scripts/python.exe"
OUT_BASE = f"{WORKSPACE}/02_results"
LOG = f"{WORKSPACE}/phase_cde.log"

base_env = os.environ.copy()
for k in ["PYTHONUNBUFFERED", "CLR_SMOKE"]:
    base_env.pop(k, None)
base_env["PYTHONUNBUFFERED"] = "1"
base_env["CLR_SMOKE"] = ""  # not smoke

def run_experiment(tag, extra_env, timeout_sec=3600):
    """Run one training job and return results path."""
    env = base_env.copy()
    env.update(extra_env)
    out_dir = f"{OUT_BASE}/{tag}"
    ckpt_dir = f"{WORKSPACE}/checkpoints/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    env["CLR_CKPT_DIR"] = ckpt_dir
    
    print(f"\n{'='*60}", flush=True)
    print(f"  [{tag}] Starting...", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    
    # Create a wrapper script that captures env
    wrapper = f"""
import os, sys
os.chdir(r'{WORKSPACE}')
os.environ.update({json.dumps(env)})
sys.argv = [r'{SCRIPT}']
exec(open(r'{SCRIPT}').read())
"""
    wrapper_file = f"{out_dir}/_wrapper.py"
    with open(wrapper_file, "w") as f:
        f.write(wrapper)
    
    proc = subprocess.run(
        [VENV_PYTHON, wrapper_file],
        capture_output=True, text=True, env=env, timeout=timeout_sec
    )
    elapsed = time.time() - t0
    
    with open(f"{out_dir}/stdout.log", "w") as f:
        f.write(proc.stdout)
    with open(f"{out_dir}/stderr.log", "w") as f:
        f.write(proc.stderr)
    
    print(f"  [{tag}] Done in {elapsed:.0f}s", flush=True)
    return f"{ckpt_dir}/results.json"

def extract_results(results_path):
    """Pull key metrics from results.json."""
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        d = json.load(f)
    return {
        "best_delta_t2p": d.get("best_delta_t2p"),
        "final_delta_t2p": d.get("final_delta_t2p"),
        "final_with_S_t2p": d.get("final_with_S_t2p"),
        "final_without_S_t2p": d.get("final_without_S_t2p"),
        "ceiling_t2p": d.get("bayes_ceiling_t2p"),
        "eval_log": d["eval_log"] if "eval_log" in d else [],
    }

def extract_from_log(log_path):
    """Extract per-step metric from training log."""
    if not os.path.exists(log_path):
        return {}
    with open(log_path) as f:
        lines = f.readlines()
    s_norms = []
    for line in lines:
        m = re.search(r'\|S\|=([\d.]+)', line)
        if m:
            s_norms.append(float(m.group(1)))
    return {"s_norm_track": s_norms}

all_results = {}

# ============================================================
# Phase C: Lifetime (k = 1, 2, 4, 8, 16)
# ============================================================
print("\n" + "="*60)
print("  PHASE C: LIFETIME EXPERIMENT")
print("="*60)

LIFETIME_K = [1, 2, 4, 8]
# k=16 would need longer training to converge given the ceiling;
# we use seed=42 for this comparison

# We need to modify the dataset generation in the script for variable depths.
# Current script uses (1,2,3) with fixed distribution.
# For lifetime, generate_dataset's depth limits can be set via env:
# CLR_MAX_DEPTH = k → generate only up to k-hop trajectories.

for k in LIFETIME_K:
    tag = f"phase_c_lifetime_k{k}_s42"
    # Strategy: restrict trajectory depth to k
    # We'll need to set CLR_DATA_PARAMS -- but script doesn't support it.
    # Alternative: run different seed with different depth settings via data generation config.
    # For now, we run standard 2-3 hop (same as evolved, which already shows k≈3 behavior).
    # Proper lifetime experiment needs depth-controlled data generator.
    pass

print("\n[Phase C] Lifetime experiment needs depth-controlled data generator. Skipping for now.")
print("[Phase C] Will design generate_dataset(k) variant first.")

# ============================================================
# Phase D: Capacity (DS = 16, 32, 64, 128)
# ============================================================
print("\n" + "="*60)
print("  PHASE D: CAPACITY EXPERIMENT (DS sweep)")
print("="*60)

DS_VALUES = [16, 32, 64, 128]
for ds in DS_VALUES:
    # Already have DS=64 from seed_42 results
    if ds == 64:
        try:
            r = extract_results(f"{WORKSPACE}/checkpoints/local_clutrr_v2_nolora/results.json")
            if r:
                all_results[f"DS={ds}"] = r
                print(f"  [DS={ds}] Already done: best Δ={r['best_delta_t2p']:.3f}")
        except:
            pass
        continue
    
    tag = f"phase_d_ds{ds}_s42"
    results_path = f"{WORKSPACE}/checkpoints/{tag}/results.json"
    
    if os.path.exists(results_path):
        r = extract_results(results_path)
        if r:
            all_results[f"DS={ds}"] = r
            print(f"  [DS={ds}] Already done, skip (best Δ={r['best_delta_t2p']:.3f})")
            continue
    
    extra = {
        "CLR_DS": str(ds),
        "CLR_SEED": "42",
        "CLR_OUT_SUFFIX": tag,
    }
    rp = run_experiment(tag, extra, timeout_sec=7200)
    r = extract_results(rp)
    if r:
        all_results[f"DS={ds}"] = r
        print(f"  [DS={ds}] best Δ={r['best_delta_t2p']:.3f}")

# ============================================================
# Phase E: Probe Analysis (measure probe accuracy + S_t stats)
# ============================================================
print("\n" + "="*60)
print("  PHASE E: PROBE ANALYSIS (stats tracking)")
print("="*60)

# The training script already logs |S_norm| and probe_acc.
# We just need to extract and visualize from existing runs.

# Check evolved runs
evolved_seeds = {
    42: f"{WORKSPACE}/checkpoints/local_clutrr_v2_nolora",
    1103: f"{WORKSPACE}/checkpoints/seed_1103",
    130: f"{WORKSPACE}/checkpoints/seed_130",
    520: f"{WORKSPACE}/checkpoints/seed_520",
}

print("\n  Extracting probe + |S| from evolved runs...")
probe_summary = {}
for sd, p in evolved_seeds.items():
    train_log = f"{p}/training.log"
    stats = extract_from_log(train_log)
    probe_summary[sd] = stats
    s_norms = stats.get("s_norm_track", [])
    if s_norms:
        print(f"  seed {sd}: |S| final={s_norms[-1]:.4f}, "
              f"min={min(s_norms):.4f}, max={max(s_norms):.4f}, "
              f"n={len(s_norms)}")

# Also get static S stats
static_log = f"{WORKSPACE}/checkpoints/phaseb_static_s42/training.log"
static_stats = extract_from_log(static_log)
print(f"  static S: |S| track = {static_stats.get('s_norm_track', [])[:10]}")

# And qwen2.5-0.5b run
qwen_log = f"{WORKSPACE}/checkpoints/qwen2.5-0.5b_evolved_s42/training.log"
qwen_stats = extract_from_log(qwen_log)
print(f"  Qwen2.5-0.5B evolved: |S| final={qwen_stats.get('s_norm_track', ['?'])[-1] if qwen_stats.get('s_norm_track') else 'N/A'}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("  PHASE C/D/E SUMMARY")
print("="*60)
print(f"{'Experiment':<30s} {'Best Δ':>8s} {'Final Δ':>8s} {'Ceiling':>8s}")
print("-" * 56)

for name, r in sorted(all_results.items()):
    bd = f"{r['best_delta_t2p']*100:+.1f}pp" if r.get('best_delta_t2p') is not None else "N/A"
    fd = f"{r['final_delta_t2p']*100:+.1f}pp" if r.get('final_delta_t2p') is not None else "N/A"
    cl = f"{r['ceiling_t2p']*100:.1f}%" if r.get('ceiling_t2p') is not None else "N/A"
    print(f"{name:<30s} {bd:>8s} {fd:>8s} {cl:>8s}")

print(f"\nProbe tracking summarized above.")

# Save consolidated results
consolidated = {
    "phase_d_capacity": all_results,
    "probe_summary": {str(k): v for k, v in probe_summary.items()},
    "static_s_norm": static_stats,
    "qwen25_s_norm": qwen_stats,
}
with open(f"{OUT_BASE}/phase_cde_results.json", "w") as f:
    json.dump(consolidated, f, indent=2)

print(f"\nConsolidated results saved to {OUT_BASE}/phase_cde_results.json")
print("Done.", flush=True)
