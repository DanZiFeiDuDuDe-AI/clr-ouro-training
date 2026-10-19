#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLR × Ouro v1 — per-LOOP state update (M3 granularity ablation).

Difference vs v0: within each turn, S is updated by the GRU after EVERY
UT loop (4 updates/turn instead of 1), and the FiLM of loop k+1 sees the
state updated from loop k's hidden. Because the loop stack shares weights,
S is the only signal that can make iteration k differ from iteration k+1
(the loop-symmetry-breaking argument). S still carries across turns and
remains the only cross-turn channel.

Training uses a single grad pass per turn (the per-loop GRU updates are part
of the graph, so the GRU receives gradient from every loop). Eval protocol:
prediction from the same forward that evolves S within the turn; without_S
injects zeros at every loop.

Smoke mode:  $env:CLR_SMOKE="1"
"""
import torch
import torch.nn.functional as F
import os, sys, json, random, time, math

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/LoopLM")

from clutrr_data import generate_dataset, bayes_ceiling
from clr_ouro import (load_ouro, inject_film, turn_forward_v1, PlaaCore,
                      DS, FILM_LAYERS)

SMOKE = os.environ.get("CLR_SMOKE") == "1"
TOTAL_STEPS = 4 if SMOKE else 400
EVAL_EVERY = 2 if SMOKE else 100
TRAIN_SAMPLES = 24 if SMOKE else 600
EVAL_TRAJ_TRAIN = 6 if SMOKE else 40
EVAL_TRAJ_FINAL = 12 if SMOKE else 150
MAX_TOKENS = 128
LR = 3e-4
SEED = int(os.environ.get("CLR_SEED", "42"))
OUT_DIR = "D:/LoopLM/checkpoints/clr_ouro_v1_s%d%s" % (SEED, "_smoke" if SMOKE else "")
os.makedirs(OUT_DIR, exist_ok=True)

Z = 1.959963984540054
def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + Z*Z/n
    c = (p + Z*Z/(2*n))/d
    h = (Z/d)*math.sqrt(p*(1-p)/n + Z*Z/(4*n*n))
    return (max(0.0, c-h), min(1.0, c+h))

@torch.no_grad()
def eval_v1(model, tok, trajs, film_layers, plaa_core, ablate_s=False):
    correct = [0,0,0,0]; total = [0,0,0,0]
    for traj in trajs:
        S = plaa_core.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            logits, S_new, _ = turn_forward_v1(model, inp, film_layers,
                                               plaa_core, S, ablate_s=ablate_s)
            ans = tok(turn["answer"], add_special_tokens=False)
            tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
            ok = int(logits.argmax().item() == tgt)
            S = S_new
            total[3] += 1; correct[3] += ok
            if ti == 0: total[0] += 1; correct[0] += ok
            else:       total[1] += 1; correct[1] += ok
            if ti == len(traj["turns"])-1: total[2] += 1; correct[2] += ok
        torch.cuda.empty_cache()
    keys = ["t1","t2p","last","all"]
    return {k: {"correct": correct[i], "total": total[i],
                "acc": correct[i]/max(total[i],1)} for i,k in enumerate(keys)}

def main():
    print("="*64, flush=True)
    print(f"CLR x Ouro v1 (per-LOOP S update) | seed={SEED} smoke={SMOKE} "
          f"steps={TOTAL_STEPS} DS={DS} FiLM={FILM_LAYERS}", flush=True)
    print("="*64, flush=True)
    torch.manual_seed(SEED); random.seed(SEED)

    train_data = generate_dataset(TRAIN_SAMPLES)
    eval_data = generate_dataset(EVAL_TRAJ_FINAL)
    print(f"train={len(train_data)} eval={len(eval_data)} "
          f"ceiling_eval={bayes_ceiling(eval_data):.4f}", flush=True)

    model, tok = load_ouro()
    film_layers = inject_film(model)
    plaa_core = PlaaCore()
    trainable = list(plaa_core.parameters())
    for l in film_layers:
        trainable += list(l.scale_proj.parameters())
    n_par = sum(p.numel() for p in trainable)
    print(f"trainable: {n_par/1e3:.1f}K params", flush=True)

    from transformers import get_scheduler
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=20,
                          num_training_steps=TOTAL_STEPS)

    log = []; step = 0; best_delta = -1e9
    t_start = time.time()
    while step < TOTAL_STEPS:
        random.shuffle(train_data)
        for traj in train_data:
            if step >= TOTAL_STEPS: break
            S = plaa_core.init_state(1, "cuda")
            b_loss = 0.0
            for turn in traj["turns"]:
                inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                          max_length=MAX_TOKENS).to("cuda")
                # single grad pass: per-loop GRU updates are in the graph
                logits, S_new, s_norms = turn_forward_v1(
                    model, inp, film_layers, plaa_core, S)
                ans = tok(turn["answer"], add_special_tokens=False)
                tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
                target = torch.tensor(tgt, dtype=torch.long, device="cuda")
                gen_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                coh = (S_new.float() - S.float()).norm().mean()
                loss = 0.95 * gen_loss + 0.05 * coh
                loss.backward()
                b_loss += float(loss)
                S = S_new.detach()
                del inp, logits
                torch.cuda.empty_cache()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0 or SMOKE:
                el = time.time() - t_start
                print(f"  step {step:4d}/{TOTAL_STEPS} | L={b_loss:.4f} | "
                      f"|S|={float(S.norm()):.3f} | s_norms={[round(x,2) for x in s_norms]} | "
                      f"{el/step:.1f}s/step | ETA {(TOTAL_STEPS-step)*el/step/60:.0f}min",
                      flush=True)
            if step % EVAL_EVERY == 0 or step >= TOTAL_STEPS:
                norms = [round(float(l.scale_proj.weight.float().norm()), 3)
                         for l in film_layers]
                print(f"\n  eval @ step {step} | scale_proj norms: {norms}", flush=True)
                assert all(x > 0 for x in norms)
                sub = eval_data[:EVAL_TRAJ_TRAIN]
                w = eval_v1(model, tok, sub, film_layers, plaa_core, False)
                o = eval_v1(model, tok, sub, film_layers, plaa_core, True)
                d = w["t2p"]["acc"] - o["t2p"]["acc"]
                log.append({"step": step, "with_S": w, "without_S": o, "delta_t2p": d})
                print(f"  t2p with={w['t2p']['acc']:.4f} "
                      f"({w['t2p']['correct']}/{w['t2p']['total']}) "
                      f"without={o['t2p']['acc']:.4f} delta={d:+.4f}\n", flush=True)
                if d > best_delta and step > (2 if SMOKE else 50):
                    best_delta = d
                    torch.save({
                        "step": step, "seed": SEED, "ds": DS,
                        "film_layers": FILM_LAYERS, "variant": "v1_per_loop",
                        "plaa_core": plaa_core.state_dict(),
                        "scale_proj": [l.scale_proj.state_dict() for l in film_layers],
                        "eval_log": log, "best_delta": best_delta,
                    }, f"{OUT_DIR}/best_model.pt")
                    print(f"  -> saved best (delta={d:+.4f})", flush=True)

    print("\nFINAL expanded eval:", flush=True)
    w = eval_v1(model, tok, eval_data, film_layers, plaa_core, False)
    o = eval_v1(model, tok, eval_data, film_layers, plaa_core, True)
    kw, nw = w["t2p"]["correct"], w["t2p"]["total"]
    ko, no = o["t2p"]["correct"], o["t2p"]["total"]
    ci_w, ci_o = wilson(kw, nw), wilson(ko, no)
    print(f"  with_S    {kw}/{nw} = {kw/max(nw,1):.4f}  W95[{ci_w[0]:.3f},{ci_w[1]:.3f}]", flush=True)
    print(f"  without_S {ko}/{no} = {ko/max(no,1):.4f}  W95[{ci_o[0]:.3f},{ci_o[1]:.3f}]", flush=True)
    print(f"  delta = {kw/max(nw,1)-ko/max(no,1):+.4f}", flush=True)
    json.dump({"final_with": w, "final_without": o, "eval_log": log,
               "ceiling": bayes_ceiling(eval_data), "seed": SEED,
               "trainable_params": n_par, "protocol": "v1 per-loop S update"},
              open(f"{OUT_DIR}/results.json", "w"), indent=2)
    print(f"\nDONE. Results in {OUT_DIR}/", flush=True)

if __name__ == "__main__":
    main()
