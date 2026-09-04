#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent final eval for CLR-v2: load best_model.pt, run persist/reset/random."""
import torch, torch.nn.functional as F
import os, sys, json, random, math

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"

from clutrr_data import generate_dataset, bayes_ceiling
from clr_v2 import (CLRv2, inject_film_v2, turn_forward_v2, load_ouro,
                    FILM_LAYERS, LATENT_SLOTS, LATENT_DIM, LATENT_STEPS)

CKPT = os.environ.get("CLR2_CKPT", "D:/training/checkpoints/clr_v2_s42/best_model.pt")
SEED = int(os.environ.get("CLR2_SEED", "42"))
EVAL_TRAJ_FINAL = int(os.environ.get("CLR2_EVAL_TRAJ", "150"))
MAX_TOKENS = 128
Z = 1.959963984540054

def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + Z*Z/n
    c = (p + Z*Z/(2*n))/d
    h = (Z/d)*math.sqrt(p*(1-p)/n + Z*Z/(4*n*n))
    return (max(0.0, c-h), min(1.0, c+h))

@torch.no_grad()
def eval_v2(model, tok, trajs, film_layers, clrv2, cond="persist",
            latent_steps=1, random_scale=1.0):
    correct = [0,0,0,0]; total = [0,0,0,0]
    norm_acc = 0.0; norm_cnt = 0; loop_norm_acc = 0.0; loop_norm_cnt = 0
    for traj in trajs:
        S = clrv2.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            if cond == "reset" and ti > 0:
                S = clrv2.init_state(1, "cuda")
            elif cond == "random":
                S = torch.randn(1, LATENT_SLOTS, LATENT_DIM,
                                device="cuda", dtype=torch.bfloat16) * random_scale
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            logits, S_new, s_norms = turn_forward_v2(
                model, inp, film_layers, clrv2, S, latent_steps=latent_steps)
            ans = tok(turn["answer"], add_special_tokens=False)
            tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
            ok = int(logits.argmax().item() == tgt)
            S = S_new
            norm_acc += float(S_new.float().norm()); norm_cnt += 1
            loop_norm_acc += sum(s_norms); loop_norm_cnt += len(s_norms)
            total[3] += 1; correct[3] += ok
            if ti == 0: total[0] += 1; correct[0] += ok
            else:       total[1] += 1; correct[1] += ok
            if ti == len(traj["turns"])-1: total[2] += 1; correct[2] += ok
        torch.cuda.empty_cache()
    keys = ["t1","t2p","last","all"]
    out = {k: {"correct": correct[i], "total": total[i],
               "acc": correct[i]/max(total[i],1)} for i,k in enumerate(keys)}
    out["mean_state_norm"] = norm_acc / max(norm_cnt, 1)
    out["mean_loop_norm"] = loop_norm_acc / max(loop_norm_cnt, 1)
    return out

def main():
    torch.manual_seed(SEED); random.seed(SEED)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("ckpt:", os.path.basename(CKPT), "| saved step:", ck.get("step"),
          "| best_delta:", ck.get("best_delta"), "| latent_steps:", ck.get("latent_steps"))
    latent_steps = int(ck.get("latent_steps", 1))
    eval_data = generate_dataset(EVAL_TRAJ_FINAL)
    print(f"eval trajs={len(eval_data)} ceiling={bayes_ceiling(eval_data):.4f}", flush=True)

    model, tok = load_ouro()
    film_layers = inject_film_v2(model)
    clrv2 = CLRv2(latent_steps=latent_steps)
    clrv2.load_state_dict(ck["clrv2"])
    clrv2.eval().cuda()

    final = {}
    for cond in ["persist", "reset", "random"]:
        r = eval_v2(model, tok, eval_data, film_layers, clrv2, cond,
                    latent_steps=latent_steps)
        final[cond] = r
        k, n = r["t2p"]["correct"], r["t2p"]["total"]
        ci = wilson(k, n)
        print(f"  {cond:8s} t2p {k}/{n} = {k/max(n,1):.4f}  "
              f"W95[{ci[0]:.3f},{ci[1]:.3f}]  |S|={r['mean_state_norm']:.3f} "
              f"loop|S|={r['mean_loop_norm']:.3f}", flush=True)
    summary = {"final": final, "ceiling": bayes_ceiling(eval_data), "ckpt": CKPT}
    out_path = os.path.join(os.path.dirname(CKPT), "final_eval.json")
    json.dump(summary, open(out_path, "w"), indent=2)
    print("saved:", out_path)

if __name__ == "__main__":
    main()
