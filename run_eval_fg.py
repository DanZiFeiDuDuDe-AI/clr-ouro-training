# -*- coding: utf-8 -*-
"""Run P0I eval in a console-attached subprocess (CUDA-safe), capture logs."""
import subprocess, os, time
PY = r"C:\Users\AI Training\AppData\Local\Programs\Python\Python313\python.exe"
fo = open("D:/training/logs_eval_fg_out.txt", "w", encoding="utf-8")
fe = open("D:/training/logs_eval_fg_err.txt", "w", encoding="utf-8")
t0 = time.time()
p = subprocess.Popen([PY, "-u", "p0i_pilot_eval.py"], cwd="D:/training",
                     env=dict(os.environ), stdout=fo, stderr=fe, text=False)
print("EVAL_PID=%d" % p.pid, flush=True)
try:
    rc = p.wait(timeout=3600)
    print("EVAL_RC=%d elapsed=%.0fs" % (rc, time.time() - t0), flush=True)
except subprocess.TimeoutExpired:
    print("WRAPPER_TIMEOUT (eval continues as pid %d)" % p.pid, flush=True)
