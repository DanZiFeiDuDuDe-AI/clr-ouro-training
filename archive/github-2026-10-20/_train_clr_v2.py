#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLR-v2 pilot training — Phase 1: reproduce the old CLR cross-turn state effect.

Protocol (mirrors _train_clr_ouro_v1.py so results are comparable):
  - Task: CLUTRR multi-turn trajectories (clutrr_data.py, unchanged)
  - Backbone: Ouro-1.4B frozen (4-bit NF4), only CLR-v2 + adapters trainable
  - Training: 400 steps, lr 3e-4 cosine w/ 20 warmup, loss = 0.95*gen + 0.05*coh
  - S updated once per UT loop (v1-parity granularity), latent_steps configurable
  - Eval: next-token argmax over 150 trajectories, three state conditions:
      persist : S carried across turns (normal)
      reset   : S re-initialized at every turn boundary
      random  : S replaced by a random (wrong) state at every turn boundary
  - Records: accuracy, loss, state norm, latent update steps, param count,
    peak VRAM, wall-clock time.

Smoke:  $env:CLR2_SMOKE="1"
"""
import torch
import torch.nn.functional as F
import os, sys, json, random, time, math

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"

from clutrr_data import generate_dataset, bayes_ceiling
from clr_v2 import (CLRv2, inject_film_v2, turn_forward_v2, load_ouro,
                    FILM_LAYERS, LATENT_SLOTS, LATENT_DIM, LATENT_STEPS)

SMOKE = os.environ.get("CLR2_SMOKE") == "1"
TOTAL_STEPS = int(os.environ.get("CLR2_STEPS", "4" if SMOKE else "400"))
EVAL_EVERY = 2 if SMOKE else 100
TRAIN_SAMPLES = 24 if SMOKE else 600
EVAL_TRAJ_TRAIN = 6 if SMOKE else 40
EVAL_TRAJ_FINAL = 12 if SMOKE else 150
MAX_TOKENS = 128
LR = 3e-4
SEED = int(os.environ.get("CLR2_SEED", "42"))
LATENT_STEPS_TRAIN = int(os.environ.get("CLR2_LATENT_STEPS", "1"))
OUT_DIR = "D:/training/checkpoints/clr_v2_s%d%s%s" % (
    SEED, "_smoke" if SMOKE else "", os.environ.get("CLR2_TAG", ""))
os.makedirs(OUT_DIR, exist_ok=True)

Z = 1.959963984540054
def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + Z*Z/n
    c = (p + Z*Z/(2*n))/d
    h = (Z/d)*math.sqrt(p*(1-p)/n + Z*Z/(4*n*n))
    return (max(0.0, c-h), min(1.0, c+h))

@torch.no_grad()
def eval_v2(model, tok, trajs, film_layers, clrv2, cond="persist",
            latent_steps=LATENT_STEPS_TRAIN, random_scale=1.0):
    """cond: persist | reset | random (state handling at turn boundaries).
    Returns {t1,t2p,last,all} + mean state norm + mean per-loop norms."""
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
                model, inp, film_layers, clrv2, S,
                latent_steps=latent_steps, ablate_s=False)
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
    print("="*70, flush=True)
    print(f"CLR-v2 pilot | seed={SEED} smoke={SMOKE} steps={TOTAL_STEPS} "
          f"slots={LATENT_SLOTS} dim={LATENT_DIM} latent_steps={LATENT_STEPS_TRAIN} "
          f"FiLM={FILM_LAYERS}", flush=True)
    print("="*70, flush=True)
    torch.manual_seed(SEED); random.seed(SEED)

    train_data = generate_dataset(TRAIN_SAMPLES)
    eval_data = generate_dataset(EVAL_TRAJ_FINAL)
    print(f"train={len(train_data)} eval={len(eval_data)} "
          f"ceiling_eval={bayes_ceiling(eval_data):.4f}", flush=True)

    model, tok = load_ouro()
    film_layers = inject_film_v2(model)
    clrv2 = CLRv2(latent_steps=LATENT_STEPS_TRAIN)
    trainable = list(clrv2.parameters())
    n_par = sum(p.numel() for p in trainable)
    print(f"trainable: {n_par/1e3:.1f}K params "
          f"(read {sum(p.numel() for p in clrv2.read.parameters())/1e3:.1f}K, "
          f"core {sum(p.numel() for p in clrv2.core.parameters())/1e3:.1f}K, "
          f"write {sum(p.numel() for p in clrv2.write.parameters())/1e3:.1f}K)", flush=True)

    from transformers import get_scheduler
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=20,
                          num_training_steps=TOTAL_STEPS)

    log = []; step = 0; best_delta = -1e9
    t_start = time.time(); peak_vram = 0.0
    while step < TOTAL_STEPS:
        random.shuffle(train_data)
        for traj in train_data:
            if step >= TOTAL_STEPS: break
            S = clrv2.init_state(1, "cuda")
            b_loss = 0.0
            for turn in traj["turns"]:
                inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                          max_length=MAX_TOKENS).to("cuda")
                logits, S_new, s_norms = turn_forward_v2(
                    model, inp, film_layers, clrv2, S,
                    latent_steps=LATENT_STEPS_TRAIN)
                ans = tok(turn["answer"], add_special_tokens=False)
                tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
                target = torch.tensor(tgt, dtype=torch.long, device="cuda")
                gen_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
                coh = (S_new.float() - S.float()).norm().mean()
                loss = 0.95 * gen_loss + 0.05 * coh
                loss.backward()
                b_loss += float(loss.detach())
                S = S_new.detach()
                del inp, logits
                torch.cuda.empty_cache()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / 1e9)
            if step % 10 == 0 or SMOKE:
                el = time.time() - t_start
                print(f"  step {step:4d}/{TOTAL_STEPS} | L={b_loss:.4f} | "
                      f"|S|={float(S.norm()):.3f} | s_norms={[round(x,2) for x in s_norms]} | "
                      f"{el/step:.1f}s/step | VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB | "
                      f"ETA {(TOTAL_STEPS-step)*el/step/60:.0f}min", flush=True)
            if step % EVAL_EVERY == 0 or step >= TOTAL_STEPS:
                print(f"\n  eval @ step {step} | latent_steps={LATENT_STEPS_TRAIN}", flush=True)
                sub = eval_data[:EVAL_TRAJ_TRAIN]
                r_p = eval_v2(model, tok, sub, film_layers, clrv2, "persist")
                r_r = eval_v2(model, tok, sub, film_layers, clrv2, "reset")
                r_x = eval_v2(model, tok, sub, film_layers, clrv2, "random")
                d = r_p["t2p"]["acc"] - r_r["t2p"]["acc"]
                log.append({"step": step, "persist": r_p, "reset": r_r,
                            "random": r_x, "delta_persist_reset": d})
                print(f"  t2p persist={r_p['t2p']['acc']:.4f} "
                      f"({r_p['t2p']['correct']}/{r_p['t2p']['total']}) "
                      f"reset={r_r['t2p']['acc']:.4f} "
                      f"random={r_x['t2p']['acc']:.4f} delta(p-r)={d:+.4f}\n", flush=True)
                if d > best_delta and step > (2 if SMOKE else 50):
                    best_delta = d
                    torch.save({
                        "step": step, "seed": SEED, "slots": LATENT_SLOTS,
                        "latent_dim": LATENT_DIM, "latent_steps": LATENT_STEPS_TRAIN,
                        "film_layers": FILM_LAYERS, "variant": "clr_v2",
                        "clrv2": clrv2.state_dict(), "eval_log": log,
                        "best_delta": best_delta, "trainable_params": n_par,
                    }, f"{OUT_DIR}/best_model.pt")
                    print(f"  -> saved best (delta={d:+.4f})", flush=True)

    print("\nFINAL expanded eval (150-traj protocol):", flush=True)
    final = {}
    for cond in ["persist", "reset", "random"]:
        r = eval_v2(model, tok, eval_data, film_layers, clrv2, cond)
        final[cond] = r
        k, n = r["t2p"]["correct"], r["t2p"]["total"]
        ci = wilson(k, n)
        print(f"  {cond:8s} t2p {k}/{n} = {k/max(n,1):.4f}  "
              f"W95[{ci[0]:.3f},{ci[1]:.3f}]  |S|={r['mean_state_norm']:.3f} "
              f"loop|S|={r['mean_loop_norm']:.3f}", flush=True)

    el_total = time.time() - t_start
    summary = {
        "final": {c: final[c] for c in final},
        "ceiling": bayes_ceiling(eval_data),
        "seed": SEED, "protocol": "clr_v2 persist/reset/random",
        "trainable_params": n_par,
        "latent_slots": LATENT_SLOTS, "latent_dim": LATENT_DIM,
        "latent_steps_train": LATENT_STEPS_TRAIN,
        "peak_vram_gb": peak_vram,
        "wall_clock_s": el_total, "steps": TOTAL_STEPS,
        "s_per_step": el_total / max(step, 1),
        "eval_log": log,
    }
    json.dump(summary, open(f"{OUT_DIR}/results.json", "w"), indent=2)
    print(f"\nDONE. Results in {OUT_DIR}/  "
          f"(params={n_par/1e3:.1f}K, peak_vram={peak_vram:.2f}GB, "
          f"wall={el_total/60:.1f}min)", flush=True)

if __name__ == "__main__":
    main()
