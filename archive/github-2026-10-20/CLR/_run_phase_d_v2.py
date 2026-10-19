#!/usr/bin/env python3
"""Phase D: Capacity sweep v2 — subprocess direct call, no wrapper encoding issues."""
import subprocess, os, json, sys, time, re, signal

WORKSPACE = "H:/CLR"
SCRIPT = f"{WORKSPACE}/local_train_clutrr_v2.py"
VENV_PYTHON = "C:/Users/Ouyang Quan/.openclaw/clr_venv/Scripts/python.exe"
OUT_BASE = f"{WORKSPACE}/02_results"

base_env = os.environ.copy()
base_env["PYTHONUNBUFFERED"] = "1"
# Keep all PATH and system env vars; only add/unset what we need

DS_CONFIGS = [
    (16, "phase_d_ds16_s42"),
    # (32, "phase_d_ds32_s42"),
    # (128, "phase_d_ds128_s42"),
]

def run_full(tag, extra_env):
    env = base_env.copy()
    env["CLR_SMOKE"] = ""
    env.update(extra_env)
    env["CLR_OUT_SUFFIX"] = tag
    
    ckpt_dir = f"{WORKSPACE}/checkpoints/{tag}"
    
    print(f"\n[Phase D {tag}] Starting...", flush=True)
    t0 = time.time()
    
    log_path = f"{OUT_BASE}/{tag}/run.log"
    os.makedirs(f"{OUT_BASE}/{tag}", exist_ok=True)
    
    proc = subprocess.Popen(
        [VENV_PYTHON, SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    
    with open(log_path, "w", encoding="utf-8") as logf:
        for line in proc.stdout:
            print(line, end='', flush=True)
            logf.write(line)
    
    proc.wait()
    elapsed = time.time() - t0
    print(f"[Phase D {tag}] Done in {elapsed/60:.1f} min", flush=True)
    
    rp = f"{ckpt_dir}/results.json"
    if os.path.exists(rp):
        with open(rp) as f:
            return json.load(f), elapsed
    return None, elapsed

results = {}
for ds, tag in DS_CONFIGS:
    ckpt_dir = f"{WORKSPACE}/checkpoints/{tag}"
    rp = f"{ckpt_dir}/results.json"
    
    # Check if already done (400 steps, positive delta)
    if os.path.exists(rp):
        with open(rp) as f:
            d = json.load(f)
        bd = d.get("best_delta_t2p")
        steps = d.get("config", {}).get("steps", 0)
        if bd is not None and bd > 0 and steps >= 400:
            print(f"[{tag}] Already done: best Δ={bd*100:+.1f}pp, steps={steps}")
            results[ds] = d
            continue
    
    r, elapsed = run_full(tag, {
        "CLR_DS": str(ds),
        "CLR_SEED": "42",
    })
    if r:
        results[ds] = r

# DS=64
rp = f"{WORKSPACE}/checkpoints/local_clutrr_v2_nolora/results.json"
if os.path.exists(rp):
    with open(rp) as f:
        results[64] = json.load(f)

print("\n" + "=" * 60)
print("PHASE D: CAPACITY SWEEP SUMMARY")
print("=" * 60)
print(f"{'DS':>6s} {'Best Δ':>10s} {'Final Δ':>10s} {'Ceiling':>10s} {'w/S':>8s} {'w/o S':>8s}")
print("-" * 52)
for ds in sorted(results.keys()):
    r = results[ds]
    bd = f"{r['best_delta_t2p']*100:+.1f}pp" if r.get('best_delta_t2p') is not None else "N/A"
    fd = f"{r['final_delta_t2p']*100:+.1f}pp" if r.get('final_delta_t2p') is not None else "N/A"
    cl = f"{r['bayes_ceiling_t2p']*100:.1f}%" if r.get('bayes_ceiling_t2p') is not None else "N/A"
    ws = f"{r['final_with_S_t2p']*100:.1f}%" if r.get('final_with_S_t2p') is not None else "N/A"
    wo = f"{r['final_without_S_t2p']*100:.1f}%" if r.get('final_without_S_t2p') is not None else "N/A"
    print(f"{ds:6d} {bd:>10s} {fd:>10s} {cl:>10s} {ws:>8s} {wo:>8s}")

with open(f"{OUT_BASE}/phase_d_results.json", "w") as f:
    json.dump({"DS": {str(k): v for k, v in results.items()}}, f, indent=2)
print(f"\nSaved: {OUT_BASE}/phase_d_results.json")
