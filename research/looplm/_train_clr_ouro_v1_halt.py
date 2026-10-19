#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLR x Ouro v1 + S->halting gate - Paper 2 Direction C prototype."""
import torch
import torch.nn.functional as F
import os, sys, json, random, time, math
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/LoopLM")
from clutrr_data import generate_dataset, bayes_ceiling
from clr_ouro import (load_ouro, inject_film, turn_forward_v1, PlaaCore,
                      HaltGate, DS, FILM_LAYERS)
SMOKE = os.environ.get("CLR_SMOKE") == "1"
TOTAL_STEPS = 4 if SMOKE else int(os.environ.get("CLR_STEPS", "400"))
EVAL_EVERY = 2 if SMOKE else int(os.environ.get("CLR_EVAL_EVERY", "50"))
TRAIN_SAMPLES = 24 if SMOKE else 600
EVAL_TRAJ_TRAIN = 6 if SMOKE else 40
EVAL_TRAJ_FINAL = 12 if SMOKE else 150
MAX_TOKENS = 128
LR = 3e-4
SEED = int(os.environ.get("CLR_SEED", "42"))
COMPUTE_W = float(os.environ.get("CLR_COMPUTE_W", "0.02"))
COMPUTE_TARGET = float(os.environ.get("CLR_COMPUTE_TARGET", "1.5"))
WARM_CKPT = os.environ.get("CLR_WARM_CKPT", "D:/LoopLM/checkpoints/clr_ouro_v1_s42/best_model.pt")
_cw_tag = ("_cw%s" % str(COMPUTE_W).replace(".", "p")) if not SMOKE else ""
_rep_tag = "_rep" if os.environ.get("CLR_HALT_REPLACE")=="1" else ""
_ho_tag = "_ho" if os.environ.get("CLR_HALT_ONLY")=="1" else ""
OUT_DIR = "D:/LoopLM/checkpoints/clr_ouro_v1_halt_s%d%s%s%s%s" % (SEED, _cw_tag, _rep_tag, _ho_tag, "_smoke" if SMOKE else "")
os.makedirs(OUT_DIR, exist_ok=True)
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
        torch.cuda.empty_cache()
    keys=["t1","t2p","last","all"]
    res={k:{"correct":correct[i],"total":total[i],"acc":correct[i]/max(total[i],1)} for i,k in enumerate(keys)}
    res["mean_expected_exit"]=exit_sum/max(exit_n,1)
    return res

def load_warm(plaa_core, film_layers, path, halt_gate=None):
    if not path or not os.path.isfile(path):
        print("  warm-start skipped (missing %s)"%path, flush=True); return
    ckpt=torch.load(path, map_location="cpu", weights_only=False)
    plaa_core.load_state_dict(ckpt["plaa_core"])
    for l, sd in zip(film_layers, ckpt["scale_proj"]):
        l.scale_proj.load_state_dict(sd)
    if halt_gate is not None and "halt_gate" in ckpt:
        missing=halt_gate.load_state_dict(ckpt["halt_gate"], strict=False); print("  halt_gate load:", missing, flush=True)
        print("  warm-start halt_gate loaded", flush=True)
    print("  warm-start from %s (step=%s, delta=%s, score=%s)"%(path, ckpt.get("step"), ckpt.get("best_delta"), ckpt.get("best_score")), flush=True)

def main():
    print("="*64, flush=True)
    print("CLR x Ouro v1+halt | seed=%d smoke=%s steps=%d compute_w=%s target=%s"%(SEED,SMOKE,TOTAL_STEPS,COMPUTE_W,COMPUTE_TARGET), flush=True)
    print("="*64, flush=True)
    torch.manual_seed(SEED); random.seed(SEED)
    train_data=generate_dataset(TRAIN_SAMPLES)
    eval_data=generate_dataset(EVAL_TRAJ_FINAL)
    print("train=%d eval=%d ceiling=%.4f"%(len(train_data),len(eval_data),bayes_ceiling(eval_data)), flush=True)
    model, tok = load_ouro()
    film_layers=inject_film(model)
    plaa_core=PlaaCore(); halt_gate=HaltGate(gain_init=float(os.environ.get("CLR_HALT_GAIN","5.0")), replace=os.environ.get("CLR_HALT_REPLACE")=="1")
    load_warm(plaa_core, film_layers, WARM_CKPT, halt_gate=halt_gate)
    HALT_ONLY = os.environ.get("CLR_HALT_ONLY") == "1"
    if HALT_ONLY:
        for p_ in plaa_core.parameters(): p_.requires_grad = False
        for l in film_layers:
            for p_ in l.scale_proj.parameters(): p_.requires_grad = False
        trainable=list(halt_gate.parameters())
        print("  HALT_ONLY=1: freezing plaa_core + scale_proj", flush=True)
    else:
        trainable=list(plaa_core.parameters())+list(halt_gate.parameters())
        for l in film_layers: trainable += list(l.scale_proj.parameters())
    n_par=sum(p.numel() for p in trainable); n_halt=sum(p.numel() for p in halt_gate.parameters())
    print("trainable: %.1fK params (halt_proj=%d)"%(n_par/1e3,n_halt), flush=True)
    assert not any(p.requires_grad for p in model.model.early_exit_gate.parameters())
    from transformers import get_scheduler
    halt_params=list(halt_gate.parameters())
    other_params=[p for p in trainable if id(p) not in set(map(id, halt_params))]
    param_groups=[]
    if other_params:
        param_groups.append({"params": other_params, "lr": LR})
    param_groups.append({"params": halt_params, "lr": float(os.environ.get("CLR_HALT_LR", str(LR*5)))})
    opt=torch.optim.AdamW(param_groups, weight_decay=0.01)
    print("  opt groups: other_lr=%s halt_lr=%s replace=%s gain_init=%s" % (
        LR, param_groups[-1]["lr"], os.environ.get("CLR_HALT_REPLACE")=="1", os.environ.get("CLR_HALT_GAIN","5.0")), flush=True)
    sched=get_scheduler("cosine", optimizer=opt, num_warmup_steps=(0 if SMOKE else 20), num_training_steps=TOTAL_STEPS)
    log=[]; step=0; best_score=-1e9; t_start=time.time()
    while step < TOTAL_STEPS:
        random.shuffle(train_data)
        for traj in train_data:
            if step >= TOTAL_STEPS: break
            S=plaa_core.init_state(1,"cuda"); b_loss=0.0; b_exit=0.0; n_turns=0
            for turn in traj["turns"]:
                inp=tok(turn["prompt"], return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to("cuda")
                logits,S_new,s_norms,aux=turn_forward_v1(model,inp,film_layers,plaa_core,S,halt_gate=halt_gate,use_weighted_exit=True)
                tgt=answer_token_id(tok, turn["answer"])
                target=torch.tensor(tgt, dtype=torch.long, device="cuda")
                gen_loss=F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                coh=(S_new.float()-S.float()).norm().mean()
                e_exit=aux["expected_exit_step"]
                pdf = aux["exit_pdf_last"]
                # continuous compute cost on expected UT index (0..3)
                compute_pen = e_exit
                # encourage probability mass on early loops (0 and 1)
                early_mass = pdf[0] + (pdf[1] if pdf.numel() > 1 else 0.0)
                loss=0.95*gen_loss + 0.05*coh + COMPUTE_W*compute_pen - COMPUTE_W*0.5*early_mass
                loss.backward(); b_loss+=float(loss); b_exit+=float(e_exit); n_turns+=1
                S=S_new.detach(); del inp, logits; torch.cuda.empty_cache()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step+=1
            if step%10==0 or SMOKE:
                el=time.time()-t_start
                print("  step %4d/%d | L=%.4f | |S|=%.3f | exit~%.2f | s_norms=%s | %.1fs/step | ETA %.0fmin"%(step,TOTAL_STEPS,b_loss,float(S.norm()),b_exit/max(n_turns,1),[round(x,2) for x in s_norms],el/step,(TOTAL_STEPS-step)*el/step/60), flush=True)
            if step%EVAL_EVERY==0 or step>=TOTAL_STEPS:
                norms=[round(float(l.scale_proj.weight.float().norm()),3) for l in film_layers]
                hw=float(halt_gate.halt_proj.weight.float().norm()); hg=float(halt_gate.gain.float().mean())
                print((chr(10)+"  eval @ step %d | scale_proj norms: %s | |halt_proj|=%.4f gain=%.3f")%(step,norms,hw,hg), flush=True)
                assert all(x>0 for x in norms)
                sub=eval_data[:EVAL_TRAJ_TRAIN]
                wres=eval_halt(model,tok,sub,film_layers,plaa_core,halt_gate,False)
                ores=eval_halt(model,tok,sub,film_layers,plaa_core,halt_gate,True)
                d=wres["t2p"]["acc"]-ores["t2p"]["acc"]
                score=d-0.15*wres["mean_expected_exit"]
                log.append({"step":step,"with_S":wres,"without_S":ores,"delta_t2p":d,"score":score,"halt_norm":hw})
                print("  t2p with=%.4f (%d/%d) without=%.4f delta=%+.4f E[exit]=%.3f score=%+.4f"%(wres["t2p"]["acc"],wres["t2p"]["correct"],wres["t2p"]["total"],ores["t2p"]["acc"],d,wres["mean_expected_exit"],score), flush=True)
                if score>best_score and step>(2 if SMOKE else 50):
                    best_score=score
                    torch.save({"step":step,"seed":SEED,"ds":DS,"film_layers":FILM_LAYERS,"variant":"v1_per_loop_halt","plaa_core":plaa_core.state_dict(),"halt_gate":halt_gate.state_dict(),"scale_proj":[l.scale_proj.state_dict() for l in film_layers],"eval_log":log,"best_score":best_score,"compute_w":COMPUTE_W,"compute_target":COMPUTE_TARGET,"warm_ckpt":WARM_CKPT}, OUT_DIR+"/best_model.pt")
                    print("  -> saved best (score=%+.4f)"%score, flush=True)
    print(chr(10)+"FINAL expanded eval:", flush=True)
    wres=eval_halt(model,tok,eval_data,film_layers,plaa_core,halt_gate,False)
    ores=eval_halt(model,tok,eval_data,film_layers,plaa_core,halt_gate,True)
    kw,nw=wres["t2p"]["correct"],wres["t2p"]["total"]
    ko,no=ores["t2p"]["correct"],ores["t2p"]["total"]
    ci_w,ci_o=wilson(kw,nw),wilson(ko,no)
    print("  with_S    %d/%d = %.4f  W95[%.3f,%.3f]"%(kw,nw,kw/max(nw,1),ci_w[0],ci_w[1]), flush=True)
    print("  without_S %d/%d = %.4f  W95[%.3f,%.3f]"%(ko,no,ko/max(no,1),ci_o[0],ci_o[1]), flush=True)
    print("  delta = %+.4f  E[exit]=%.3f"%(kw/max(nw,1)-ko/max(no,1), wres["mean_expected_exit"]), flush=True)
    json.dump({"final_with":wres,"final_without":ores,"eval_log":log,"ceiling":bayes_ceiling(eval_data),"seed":SEED,"trainable_params":n_par,"halt_params":n_halt,"protocol":"v1 per-loop S + S-conditioned halt gate","compute_w":COMPUTE_W,"compute_target":COMPUTE_TARGET,"warm_ckpt":WARM_CKPT,"max_tokens":MAX_TOKENS,"halt_only":os.environ.get("CLR_HALT_ONLY")=="1"}, open(OUT_DIR+"/results.json","w"), indent=2)
    print(chr(10)+"DONE. Results in %s/"%OUT_DIR, flush=True)

if __name__ == "__main__":
    main()
