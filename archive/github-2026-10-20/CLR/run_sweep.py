#!/usr/bin/env python3
"""
CLR Param Sweep Runner
Controls DS and FiLM layers via env vars.
Usage:
    python run_sweep.py           # run full matrix
    python run_sweep.py --smoke   # run 1-step smoketest per config

Env vars read by local_train_clutrr_v2.py:
    CLR_SEED, CLR_DATA_SEED, CLR_DS, CLR_FILM_LAYERS, CLR_OUT_SUFFIX
"""
import os, sys, json, time, subprocess, argparse
from pathlib import Path

WORKSPACE = "H:\\CLR"
TRAIN_SCRIPT = f"{WORKSPACE}\\local_train_clutrr_v2.py"

# ── Sweep Matrix ──
DS_VALUES = [64, 128, 256, 512]
FILM_LAYERS_CONFIGS = {
    "tail3":    str([19, 21, 23]),       # last 3 full-attn (Qwen3.5-0.8B: 24 layers)
    "tail6":    str([3, 7, 11, 15, 19, 23]),  # current default
    "all24":    str(list(range(24))),     # all layers
    "sparse12": str([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]),
}
SEEDS = [42, 130]

BASE_RESULTS_DIR = f"{WORKSPACE}\\checkpoints"

def run_config(ds, film_key, film_layers_list, seed, smoke=False):
    suffix = f"DS{ds}_{film_key}_s{seed}"
    out_dir = f"{BASE_RESULTS_DIR}\\sweep_{suffix}"
    
    env = os.environ.copy()
    env["CLR_SEED"] = str(seed)
    env["CLR_DATA_SEED"] = str(seed)
    env["CLR_DS"] = str(ds)
    env["CLR_FILM_LAYERS"] = film_layers_list
    env["CLR_OUT_SUFFIX"] = f"sweep_{suffix}"
    if smoke:
        env["CLR_SMOKE"] = "1"
    
    print(f"\n{'='*60}")
    print(f"  DS={ds} | FiLM={film_key} ({film_layers_list}) | seed={seed} {'| SMOKE' if smoke else ''}")
    print(f"  OUT: {out_dir}")
    print(f"{'='*60}")
    
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-X", "utf8", TRAIN_SCRIPT],
        env=env, capture_output=True, text=True
    )
    elapsed = (time.time() - t0) / 60
    
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}, {elapsed:.1f}min)")
        print(result.stderr[-1000:])
        return {"status": "failed", "rc": result.returncode}
    
    print(f"  OK ({elapsed:.1f}min)")
    return {"status": "ok", "elapsed_min": round(elapsed, 1)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Smoketest only")
    p.add_argument("--ds", type=int, nargs="+", default=None, help="Override DS values")
    p.add_argument("--film", choices=list(FILM_LAYERS_CONFIGS.keys()), nargs="+", default=None, help="Override FiLM configs")
    p.add_argument("--seed", type=int, nargs="+", default=None, help="Override seeds")
    return p.parse_args()


def main():
    args = parse_args()
    smoke = args.smoke
    
    ds_values = args.ds if args.ds else DS_VALUES
    film_configs = {k: FILM_LAYERS_CONFIGS[k] for k in (args.film if args.film else FILM_LAYERS_CONFIGS)}
    seeds = args.seed if args.seed else SEEDS
    
    print(f"CLR Sweep Matrix")
    print(f"  DS: {ds_values}")
    print(f"  FiLM: {list(film_configs.keys())}")
    print(f"  Seeds: {seeds}")
    total = len(ds_values) * len(film_configs) * len(seeds)
    print(f"  Total configs: {total} ({'SMOKE' if smoke else 'FULL'})")
    print()
    
    results = []
    for ds in ds_values:
        for film_key, film_list in film_configs.items():
            for seed in seeds:
                r = run_config(ds, film_key, film_list, seed, smoke=smoke)
                r["ds"] = ds
                r["film_key"] = film_key
                r["seed"] = seed
                results.append(r)
                
                if smoke:
                    print("  SMOKE OK" if r["status"] == "ok" else f"  SMOKE FAIL ({r.get('rc')})")
                else:
                    print(f"  {'OK' if r['status'] == 'ok' else 'FAIL'}")
    
    # summary
    print(f"\n{'='*60}")
    print(f" Sweep {'SMOKE' if smoke else 'FULL'} Complete")
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] != "ok")
    print(f"  OK: {ok}/{total} | FAIL: {fail}/{total}")
    
    # save results
    out_path = f"{WORKSPACE}\\02_results\\sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
