#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Old CLR v1 control: persist / reset / random state eval on same protocol."""
import torch
import os, sys, json, random, math

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"

from clutrr_data import generate_dataset, bayes_ceiling
from clr_ouro import (load_ouro, inject_film, turn_forward_v1, PlaaCore,
                      DS, FILM_LAYERS)

SEED = int(os.environ.get("CLR_SEED", "42"))
EVAL_TRAJ_FINAL = int(os.environ.get("CLR_EVAL_TRAJ", "150"))
MAX_TOKENS = 128
Z = 1.959963984540054

def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + Z*Z/n
    c = (p + Z*Z/(2*n))/d
    h = (Z/d)*math.sqrt(p*(1-p)/n + Z*Z/(4*n*n))
    return (max(0.0, c-h), min(1.0, c+h))

@torch.no_grad()
def eval_v1_cond(model, tok, trajs, film_layers, plaa_core, cond="persist"):
    correct = [0,0,0,0]; total = [0,0,0,0]
    norm_acc = 0.0; norm_cnt = 0
    for traj in trajs:
        S = plaa_core.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            if cond == "reset" and ti > 0:
                S = plaa_core.init_state(1, "cuda")
            elif cond == "random":
                S = torch.randn(1, DS, device="cuda", dtype=torch.bfloat16) * 1.0
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            logits, S_new, _ = turn_forward_v1(model, inp, film_layers,
                                               plaa_core, S)
            ans = tok(turn["answer"], add_special_tokens=False)
            tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
            ok = int(logits.argmax().item() == tgt)
            S = S_new
            norm_acc += float(S_new.float().norm()); norm_cnt += 1
            total[3] += 1; correct[3] += ok
            if ti == 0: total[0] += 1; correct[0] += ok
            else:       total[1] += 1; correct[1] += ok
            if ti == len(traj["turns"])-1: total[2] += 1; correct[2] += ok
        torch.cuda.empty_cache()
    keys = ["t1","t2p","last","all"]
    out = {k: {"correct": correct[i], "total": total[i],
               "acc": correct[i]/max(total[i],1)} for i,k in enumerate(keys)}
    out["mean_state_norm"] = norm_acc / max(norm_cnt, 1)
    return out

def main():
    torch.manual_seed(SEED); random.seed(SEED)
    ck = torch.load(r"D:/training/checkpoints/clr_ouro_v1_s42/best_model.pt",
                    map_location="cpu", weights_only=False)
    print("old v1 ckpt | step:", ck.get("step"), "| ds:", DS, "| film:", FILM_LAYERS)
    eval_data = generate_dataset(EVAL_TRAJ_FINAL)
    print(f"eval trajs={len(eval_data)} ceiling={bayes_ceiling(eval_data):.4f}", flush=True)

    model, tok = load_ouro()
    film_layers = inject_film(model)
    core = PlaaCore(d_model=2048, d_state=DS)
    core.load_state_dict(ck["plaa_core"])
    core.eval().cuda()
    for l, sd in zip(film_layers, ck["scale_proj"]):
        l.scale_proj.load_state_dict({k: torch.tensor(v) for k, v in sd.items()})
        l.scale_proj.eval()
    model.eval().cuda()

    final = {}
    for cond in ["persist", "reset", "random"]:
        r = eval_v1_cond(model, tok, eval_data, film_layers, core, cond)
        final[cond] = r
        k, n = r["t2p"]["correct"], r["t2p"]["total"]
        ci = wilson(k, n)
        print(f"  {cond:8s} t2p {k}/{n} = {k/max(n,1):.4f}  "
              f"W95[{ci[0]:.3f},{ci[1]:.3f}]  |S|={r['mean_state_norm']:.3f}", flush=True)
    summary = {"final": final, "ceiling": bayes_ceiling(eval_data)}
    json.dump(summary, open("D:/training/checkpoints/clr_ouro_v1_s42/cond_eval.json", "w"), indent=2)
    print("saved cond_eval.json")

if __name__ == "__main__":
    main()
