#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eval a halt checkpoint: t2p with/without S + mean expected exit."""
import os, sys, json, math, torch
from pathlib import Path
sys.path.insert(0, r"D:/LoopLM")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
from clutrr_data import generate_dataset, bayes_ceiling
from clr_ouro import load_ouro, inject_film, turn_forward_v1, PlaaCore, HaltGate, DS, FILM_LAYERS

CKPT = os.environ.get("CLR_EVAL_CKPT", r"D:/LoopLM/checkpoints/clr_ouro_v1_halt_s42_cw0p3_rep_ho/best_model.pt")
N = int(os.environ.get("CLR_EVAL_N", "150"))
SEED = int(os.environ.get("CLR_SEED", "42"))
MAX_TOKENS = 128
REPLACE = os.environ.get("CLR_HALT_REPLACE", "1") == "1"
OUT = os.environ.get("CLR_EVAL_OUT", "")

Z = 1.959963984540054
def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + Z*Z/n
    c = (p + Z*Z/(2*n))/d
    h = (Z/d)*math.sqrt(p*(1-p)/n + Z*Z/(4*n*n))
    return (max(0.0, c-h), min(1.0, c+h))

def answer_token_id(tok, answer):
    ids = tok(answer, add_special_tokens=False)["input_ids"]
    if isinstance(ids[0], list): ids = ids[0]
    return ids[0]

@torch.no_grad()
def eval_halt(model, tok, trajs, film_layers, plaa_core, halt_gate, ablate_s=False):
    correct=[0,0,0,0]; total=[0,0,0,0]; exit_sum=0.0; exit_n=0
    gate_hist=[]
    for traj in trajs:
        S = plaa_core.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to("cuda")
            logits, S_new, _, aux = turn_forward_v1(
                model, inp, film_layers, plaa_core, S,
                ablate_s=ablate_s, halt_gate=None if ablate_s else halt_gate,
                use_weighted_exit=True)
            tgt = answer_token_id(tok, turn["answer"])
            ok = int(logits.argmax().item() == tgt)
            S = S_new
            total[3]+=1; correct[3]+=ok
            if ti==0: total[0]+=1; correct[0]+=ok
            else: total[1]+=1; correct[1]+=ok
            if ti==len(traj["turns"])-1: total[2]+=1; correct[2]+=ok
            exit_sum += float(aux["expected_exit_step"]); exit_n += 1
            gate_hist.append([float(x) for x in aux["gate_means"]])
            if exit_n % 40 == 0:
                print("  progress turns=%d t2p_running=%.3f Eexit=%.3f"%(exit_n, correct[1]/max(total[1],1), exit_sum/exit_n), flush=True)
        torch.cuda.empty_cache()
    keys=["t1","t2p","last","all"]
    res={k:{"correct":correct[i],"total":total[i],"acc":correct[i]/max(total[i],1)} for i,k in enumerate(keys)}
    res["mean_expected_exit"]=exit_sum/max(exit_n,1)
    # mean gate sigmoid per loop
    if gate_hist:
        nloop=len(gate_hist[0]); means=[]
        for j in range(nloop):
            means.append(sum(row[j] for row in gate_hist)/len(gate_hist))
        res["mean_gate_lambda"]=means
    return res

def main():
    print("EVAL CKPT", CKPT, "N", N, "replace", REPLACE, flush=True)
    torch.manual_seed(SEED)
    eval_data = generate_dataset(N)
    print("ceiling", bayes_ceiling(eval_data), flush=True)
    model, tok = load_ouro()
    film_layers = inject_film(model)
    plaa_core = PlaaCore()
    halt_gate = HaltGate(gain_init=5.0, replace=REPLACE)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    plaa_core.load_state_dict(ckpt["plaa_core"])
    for l, sd in zip(film_layers, ckpt["scale_proj"]):
        l.scale_proj.load_state_dict(sd)
    if "halt_gate" in ckpt:
        print("halt load", halt_gate.load_state_dict(ckpt["halt_gate"], strict=False), flush=True)
    print("gain", float(halt_gate.gain.float()), "replace", halt_gate.replace, flush=True)
    w = eval_halt(model, tok, eval_data, film_layers, plaa_core, halt_gate, False)
    o = eval_halt(model, tok, eval_data, film_layers, plaa_core, halt_gate, True)
    kw,nw=w["t2p"]["correct"],w["t2p"]["total"]
    ko,no=o["t2p"]["correct"],o["t2p"]["total"]
    ci_w,ci_o=wilson(kw,nw),wilson(ko,no)
    print("with_S    %d/%d=%.4f W95[%.3f,%.3f] E[exit]=%.4f gates=%s"%(kw,nw,kw/max(nw,1),ci_w[0],ci_w[1],w["mean_expected_exit"],w.get("mean_gate_lambda")), flush=True)
    print("without_S %d/%d=%.4f W95[%.3f,%.3f] E[exit]=%.4f"%(ko,no,ko/max(no,1),ci_o[0],ci_o[1],o["mean_expected_exit"]), flush=True)
    print("delta %+.4f"%(kw/max(nw,1)-ko/max(no,1)), flush=True)
    out = OUT or str(Path(CKPT).parent / ("final_eval_n%d.json"%N))
    payload={"ckpt":CKPT,"n":N,"seed":SEED,"replace":REPLACE,"with_S":w,"without_S":o,
             "delta_t2p":kw/max(nw,1)-ko/max(no,1),"ceiling":bayes_ceiling(eval_data),
             "ckpt_step":ckpt.get("step"),"ckpt_score":ckpt.get("best_score")}
    json.dump(payload, open(out,"w"), indent=2)
    print("wrote", out, flush=True)

if __name__ == "__main__":
    from pathlib import Path
    main()
