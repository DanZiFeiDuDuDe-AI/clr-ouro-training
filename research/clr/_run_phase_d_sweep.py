#!/usr/bin/env python3
"""
Phase D sequential runner: DS=32 → 128 → 256 (and 8 if fast enough)
Fable 5 ordered: fill capacity curve shape first with seed 42.
"""
import subprocess, os, json, time, sys

WORKSPACE = "H:/CLR"
SCRIPT = f"{WORKSPACE}/local_train_clutrr_v2.py"
VENV = "C:/Users/Ouyang Quan/.openclaw/clr_venv/Scripts/python.exe"
OUT = f"{WORKSPACE}/02_results"

DS_QUEUE = [32, 128, 256]  # 16 and 64 already done

base_env = os.environ.copy()
base_env["PYTHONUNBUFFERED"] = "1"

def run_one(ds, tag):
    env = base_env.copy()
    env["CLR_DS"] = str(ds)
    env["CLR_SEED"] = "42"
    env["CLR_OUT_SUFFIX"] = tag
    env["CLR_SMOKE"] = ""
    
    os.makedirs(f"{OUT}/{tag}", exist_ok=True)
    log_path = f"{OUT}/{tag}/run.log"
    
    print(f"\n[{tag}] DS={ds} starting...", flush=True)
    t0 = time.time()
    
    proc = subprocess.Popen(
        [VENV, SCRIPT],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, text=True, bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    
    with open(log_path, "w", encoding="utf-8") as logf:
        step_count = 0
        for line in proc.stdout:
            print(line, end='', flush=True)
            logf.write(line)
            # Track progress roughly
            if 'step ' in line and '/400 |' in line:
                step_count += 1
                if step_count % 50 == 0:
                    print(f"  [{tag}] progress: {step_count} evals logged", flush=True)
    
    proc.wait()
    elapsed = time.time() - t0
    print(f"[{tag}] DS={ds} done in {elapsed/60:.1f} min", flush=True)
    
    rp = f"{WORKSPACE}/checkpoints/{tag}/results.json"
    if os.path.exists(rp):
        with open(rp) as f:
            r = json.load(f)
        bd = r.get("best_delta_t2p", 0) * 100
        fd = r.get("final_delta_t2p", 0) * 100
        ws = r.get("final_with_S_t2p", 0) * 100
        wo = r.get("final_without_S_t2p", 0) * 100
        cl = r.get("bayes_ceiling_t2p", 0) * 100
        print(f"  RESULT: DS={ds} | Best Δ={bd:+7.2f}pp | Final Δ={fd:+7.2f}pp | Ceil={cl:5.1f}% | w/S={ws:5.1f}% | w/o S={wo:5.1f}%")
        return r
    else:
        print(f"  [WARN] DS={ds}: no results.json found at {rp}")
        return None

results = {}
for ds, tag in [(32, "phase_d_ds32_s42"), (128, "phase_d_ds128_s42"), (256, "phase_d_ds256_s42")]:
    rp = f"{WORKSPACE}/checkpoints/{tag}/results.json"
    if os.path.exists(rp):
        with open(rp) as f:
            r = json.load(f)
        bd = r.get("best_delta_t2p", 0)
        if bd is not None and bd > 0.01:
            print(f"[{tag}] Already done (best Δ={bd*100:+.1f}pp), skip.", flush=True)
            results[ds] = r
            continue
    r = run_one(ds, tag)
    if r:
        results[ds] = r

# Add existing DS=16 and 64
for ds, tag in [(16, "phase_d_ds16_s42"), (64, "local_clutrr_v2_nolora")]:
    rp = f"{WORKSPACE}/checkpoints/{tag}/results.json"
    if os.path.exists(rp):
        with open(rp) as f:
            r = json.load(f)
        results[ds] = r

# Summary
print("\n" + "=" * 65)
print("PHASE D: CAPACITY SWEEP — FINAL SUMMARY")
print("=" * 65)
print(f"{'DS':>5s} {'Best Δ':>9s} {'Final Δ':>9s} {'Ceiling':>9s} {'w/S':>8s} {'w/o S':>8s}")
print("-" * 52)
for ds in sorted(results.keys(), key=lambda x: int(x)):
    r = results[ds]
    if r:
        bd = f"{r['best_delta_t2p']*100:+.1f}pp" if r.get('best_delta_t2p') is not None else "N/A"
        fd = f"{r['final_delta_t2p']*100:+.1f}pp" if r.get('final_delta_t2p') is not None else "N/A"
        cl = f"{r['bayes_ceiling_t2p']*100:.1f}%" if r.get('bayes_ceiling_t2p') is not None else "N/A"
        ws = f"{r['final_with_S_t2p']*100:.1f}%" if r.get('final_with_S_t2p') is not None else "N/A"
        wo = f"{r['final_without_S_t2p']*100:.1f}%" if r.get('final_without_S_t2p') is not None else "N/A"
        print(f"{ds:5s} {bd:>9s} {fd:>9s} {cl:>9s} {ws:>8s} {wo:>8s}")

# Save consolidated
with open(f"{OUT}/phase_d_results.json", "w") as f:
    json.dump({"DS": {str(k): v for k, v in results.items()}}, f, indent=2)
print(f"\nSaved: {OUT}/phase_d_results.json")
print("\nPhase D complete. Fable 5: your capacity curve is ready.", flush=True)
