# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
log = open("D:/training/logs_load_test.txt", "w", encoding="utf-8")
def p(*a):
    msg = " ".join(map(str, a))
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()
p("START")
import torch
p("torch", torch.__version__, "cuda", torch.cuda.is_available())
p("mem", torch.cuda.mem_get_info())
from clr_ouro import load_ouro
p("imported clr_ouro")
m, t = load_ouro()
p("load_ouro OK")
del m
torch.cuda.empty_cache()
p("DONE")
