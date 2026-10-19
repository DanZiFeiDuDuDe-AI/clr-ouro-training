#!/usr/bin/env python3
"""Phase D: Capacity sweep — DS = 16, 32, 128 (DS=64 already exists)."""
import subprocess, os, json, sys, time, re

WORKSPACE = "H:/CLR"
SCRIPT = f"{WORKSPACE}/local_train_clutrr_v2.py"
VENV_PYTHON = "C:/Users/Ouyang Quan/.openclaw/clr_venv/Scripts/python.exe"
OUT_BASE = f"{WORKSPACE}/02_results"

base_env = os.environ.copy()
for k in ["PYTHONUNBUFFERED", "CLR_SMOKE"]:
    base_env.pop(k, None)
base_env["PYTHONUNBUFFERED"] = "1"
base_env["CLR_SMOKE"] = ""

def run_experiment(tag, extra_env, timeout_sec=7200):
    env = base_env.copy()
    env.update(extra_env)
    out_dir = f"{OUT_BASE}/{tag}"
    ckpt_dir = f"{WORKSPACE}/checkpoints/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    env["CLR_CKPT_DIR"] = ckpt_dir
    
    wrapper = f"""
import os, sys
os.chdir(r'{WORKSPACE}')
os.environ.update({json.dumps(env)})
sys.argv = [r'{SCRIPT}']
exec(open(r'{SCRIPT}').read())
"""
    wrapper_file = f"{out_dir}/_wrapper.py"
    with open(wrapper_file, "w", encoding="utf-8") as f:
        f.write(wrapper)
    
    log_path = f"{out_dir}/run.log"
    t0 = time.time()
    print(f"\n[{tag}] Starting...", flush=True)
    
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            [VENV_PYTHON, wrapper_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, text=True, bufsize=1
        )
        for line in proc.stdout:
            print(line, end='', flush=True)
            logf.write(line)
        proc.wait()
    
    elapsed = time.time() - t0
    print(f"\n[{tag}] Done in {elapsed:.0f}s", flush=True)
    return f"{ckpt_dir}/results.json", elapsed

def extract_results(results_path):
    if not os.path.exists(results_path):
        print(f"  [WARN] Results not found: {results_path}")
        return None
    with open(results_path) as f:
        d = json.load(f)
    return {
        "best_delta_t2p": d.get("best_delta_t2p"),
        "final_delta_t2p": d.get("final_delta_t2p"),
        "final_with_S_t2p": d.get("final_with_S_t2p"),
        "final_without_S_t2p": d.get("final_without_S_t2p"),
        "ceiling_t2p": d.get("bayes_ceiling_t2p"),
        "model": d.get("model"),
    }

# DS = 16, 32, 128 (seed 42, 400 steps)
# Already existing: DS=64 (seed 42), and sweep_DS128/* (4 steps, not 400)

# First check if full 400-step DS=128 already exists
# (from sweep directory, all are smoke=4 steps not 400)
DS_CONFIGS = [
    (16, "phase_d_ds16_s42"),
    (32, "phase_d_ds32_s42"),
    (128, "phase_d_ds128_s42"),
]

results = {}

for ds, tag in DS_CONFIGS:
    ckpt_dir = f"{WORKSPACE}/checkpoints/{tag}"
    results_path = f"{ckpt_dir}/results.json"
    
    if os.path.exists(results_path):
        r = extract_results(results_path)
        if r and r["best_delta_t2p"] is not None:
            bd = r["best_delta_t2p"]
            if bd > 0 and bd < 1.0:  # plausible range (not smoke numbers)
                print(f"[{tag}] Already done: best Δ={bd*100:+.1f}pp", flush=True)
                results[ds] = r
                continue
    
    # Run full 400-step training
    extra = {
        "CLR_DS": str(ds),
        "CLR_SEED": "42",
        "CLR_OUT_SUFFIX": tag,
    }
    rp, elapsed = run_experiment(tag, extra)
    r = extract_results(rp)
    if r:
        results[ds] = r
        print(f"[{tag}] best Δ={r['best_delta_t2p']*100:+.1f}pp in {elapsed/60:.1f}min", flush=True)

# DS=64 already exists
rp = f"{WORKSPACE}/checkpoints/local_clutrr_v2_nolora/results.json"
r = extract_results(rp)
if r:
    results[64] = r

# Summary
print("\n" + "=" * 60)
print("PHASE D: CAPACITY SWEEP SUMMARY")
print("=" * 60)
print(f"{'DS':>6s} {'Best Δ':>8s} {'Final Δ':>8s} {'Ceiling':>8s} {'w/S final':>8s} {'w/o S final':>8s}")
print("-" * 56)
for ds in sorted(results.keys()):
    r = results[ds]
    if r:
        bd = f"{r['best_delta_t2p']*100:+.1f}pp" if r.get('best_delta_t2p') is not None else "N/A"
        fd = f"{r['final_delta_t2p']*100:+.1f}pp" if r.get('final_delta_t2p') is not None else "N/A"
        cl = f"{r['ceiling_t2p']*100:.1f}%" if r.get('ceiling_t2p') is not None else "N/A"
        ws = f"{r['final_with_S_t2p']*100:.1f}%" if r.get('final_with_S_t2p') is not None else "N/A"
        wo = f"{r['final_without_S_t2p']*100:.1f}%" if r.get('final_without_S_t2p') is not None else "N/A"
        print(f"{ds:6d} {bd:>8s} {fd:>8s} {cl:>8s} {ws:>8s} {wo:>8s}")

with open(f"{OUT_BASE}/phase_d_results.json", "w") as f:
    json.dump({"DS": {str(k): v for k, v in results.items()}}, f, indent=2)

print(f"\nSaved: {OUT_BASE}/phase_d_results.json")
print("Phase D complete.", flush=True)
