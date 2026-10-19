# -*- coding: utf-8 -*-
"""Run m1-reset training for a given seed in a console-attached subprocess
(CUDA-safe), capture logs. Usage: python run_m1_seed_fg.py 43"""
import subprocess, os, sys, time
seed = sys.argv[1]
PY = r"C:\Users\AI Training\AppData\Local\Programs\Python\Python313\python.exe"
out_dir = "D:/training/checkpoints/p0i_pilot_m1_reset_s%s" % seed
env = dict(os.environ)
env.update({
    "P0I_INIT": "m1",
    "P0I_MODE": "reset",
    "P0I_STEPS": "600",
    "P0I_LR": "3e-4",
    "P0I_SEED": seed,
    "P0I_OUT_DIR": out_dir,
})
fo = open("D:/training/logs_m1_s%s_fg_out.txt" % seed, "w", encoding="utf-8")
fe = open("D:/training/logs_m1_s%s_fg_err.txt" % seed, "w", encoding="utf-8")
t0 = time.time()
p = subprocess.Popen([PY, "-u", "p0i_pilot_train.py"], cwd="D:/training",
                     env=env, stdout=fo, stderr=fe, text=False)
print("TRAIN_PID=%d seed=%s out=%s" % (p.pid, seed, out_dir), flush=True)
try:
    rc = p.wait(timeout=3600)
    print("TRAIN_RC=%d elapsed=%.0fs" % (rc, time.time() - t0), flush=True)
except subprocess.TimeoutExpired:
    print("WRAPPER_TIMEOUT (training continues as pid %d)" % p.pid, flush=True)
