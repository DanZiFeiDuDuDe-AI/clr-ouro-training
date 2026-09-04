# -*- coding: utf-8 -*-
"""Foreground training runner: run persist training in a console-attached
subprocess (CUDA-safe unlike detached/hidden), capture full logs."""
import subprocess, os, sys, time
PY = r"C:\Users\AI Training\AppData\Local\Programs\Python\Python313\python.exe"
env = dict(os.environ)
env.update({
    "P0I_INIT": "m1",
    "P0I_MODE": "persist",
    "P0I_STEPS": "600",
    "P0I_LR": "3e-4",
    "P0I_OUT_DIR": "D:/training/checkpoints/p0i_pilot_m1_persist_s42",
})
fo = open("D:/training/logs_persist_fg_out.txt", "w", encoding="utf-8")
fe = open("D:/training/logs_persist_fg_err.txt", "w", encoding="utf-8")
t0 = time.time()
p = subprocess.Popen([PY, "-u", "p0i_pilot_train.py"], cwd="D:/training",
                     env=env, stdout=fo, stderr=fe, text=False)
print("TRAIN_PID=%d" % p.pid, flush=True)
try:
    rc = p.wait(timeout=3600)
    print("TRAIN_RC=%d elapsed=%.0fs" % (rc, time.time() - t0), flush=True)
except subprocess.TimeoutExpired:
    print("WRAPPER_TIMEOUT (training continues as pid %d)" % p.pid, flush=True)
    # keep child alive; wrapper exits
