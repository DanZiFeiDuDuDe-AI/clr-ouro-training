#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_seqstate_s0 / train.py — Sequence-State S0 screening trainer.

Question: with total latent capacity matched at 64 scalars, does replacing
the single GRU latent vector with a fixed-length latent sequence change
task performance or dynamical behaviour?

Configs (one seed, all with 64 latent scalars):
    1 x 64   vector baseline  (v1 per-loop protocol; --state-type vector
                               --seed 42 reproduces the v1 baseline numbers)
    2 x 32, 4 x 16, 8 x 8     sequence-state (tiny transformer updater)

Everything else is pinned to the v1 baseline:
    same data stream (seed), 400 steps of batch-1 trajectories, same
    LR/warmup/weight-decay/loss mix, same FiLM layers {3,7,11,15,19,23},
    same frozen 4-bit Ouro backbone, same eval protocol (per-loop state
    evolution, with_S vs without_S argmax readout).

Examples:
    python train.py --state-type vector   --num-slots 1 --slot-dim 64 --seed 42
    python train.py --state-type sequence --num-slots 4 --slot-dim 16 --seed 42
    python train.py --state-type sequence --num-slots 8 --slot-dim 8  --smoke
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

os.environ.setdefault("PYTHONUNBUFFERED", "1")
ROOT = "D:/LoopLM"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clutrr_data import generate_dataset, bayes_ceiling          # noqa: E402
from clr_ouro import (load_ouro, inject_film, turn_forward_v1,   # noqa: E402
                      FILM_LAYERS, DS)
from state_modules import VectorStateModule, SequenceStateModule  # noqa: E402

Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = (Z / d) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def build_state_module(args):
    if args.state_type == "vector":
        sm = VectorStateModule()
    elif args.state_type == "sequence":
        sm = SequenceStateModule(n_slots=args.num_slots, d_slot=args.slot_dim,
                                 n_layers=args.updater_layers,
                                 pool=args.pool_type)
    else:
        raise ValueError(args.state_type)
    return sm


def param_counts(state_module, film_layers, model):
    trainable = (list(state_module.parameters())
                 + [p for l in film_layers for p in l.scale_proj.parameters()])
    n_tr = sum(p.numel() for p in trainable)
    n_fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return n_tr, n_fr, trainable


def print_preamble(args, out_dir, n_tr, n_fr, ckpt_id, protocol):
    print("=" * 72, flush=True)
    print(f"[S0 Sequence-State screening] run={out_dir}", flush=True)
    print(f"  state type      : {args.state_type}", flush=True)
    print(f"  num slots       : {args.num_slots}", flush=True)
    print(f"  slot dim        : {args.slot_dim}   (latent capacity "
          f"{args.num_slots * args.slot_dim} == {DS})", flush=True)
    print(f"  updater layers  : {args.updater_layers}   pool: {args.pool_type}",
          flush=True)
    print(f"  trainable params: {n_tr:,} "
          f"({n_tr/1e3:.1f}K)", flush=True)
    print(f"  backbone frozen : {n_fr:,} params (4-bit Ouro-1.4B, "
          f"requires_grad=False)", flush=True)
    print(f"  dataset split   : train={args.train_samples} traj, "
          f"eval={args.eval_traj_final} traj (replayed RNG stream, seed "
          f"{args.seed})", flush=True)
    print(f"  seed            : {args.seed}", flush=True)
    print(f"  steps           : {args.steps}  LR={args.lr}  "
          f"warmup={args.warmup}", flush=True)
    print(f"  FiLM layers     : {FILM_LAYERS}", flush=True)
    print(f"  eval protocol   : {protocol}", flush=True)
    print(f"  checkpoint id   : {ckpt_id}", flush=True)
    print("=" * 72, flush=True)


@torch.no_grad()
def eval_k(model, tok, trajs, film_layers, state_module, k, max_tokens=128,
           ablate_s=False, no_update=False, intervene=None):
    """Eval under max_ut=k with the v1 per-loop protocol (t1/t2p/last/all).

    Returns the bucket dict in the v1 trainer format.
    """
    correct = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    for traj in trajs:
        S = state_module.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=max_tokens).to("cuda")
            logits, S_new, _ = turn_forward_v1(
                model, inp, film_layers, None, S, ablate_s=ablate_s,
                state_module=state_module, max_ut=k, intervene=intervene,
                no_update=no_update)
            ans = tok(turn["answer"], add_special_tokens=False)
            tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
            ok = int(logits.argmax().item() == tgt)
            S = S_new
            total[3] += 1
            correct[3] += ok
            if ti == 0:
                total[0] += 1
                correct[0] += ok
            else:
                total[1] += 1
                correct[1] += ok
            if ti == len(traj["turns"]) - 1:
                total[2] += 1
                correct[2] += ok
        torch.cuda.empty_cache()
    keys = ["t1", "t2p", "last", "all"]
    return {k_: {"correct": correct[i], "total": total[i],
                 "acc": correct[i] / max(total[i], 1)}
            for i, k_ in enumerate(keys)}


_ARGS = None


def args_max_tokens():
    return _ARGS.max_tokens if _ARGS is not None else 128


def main():
    global _ARGS
    ap = argparse.ArgumentParser(description="S0 sequence-state screening")
    ap.add_argument("--state-type", choices=["vector", "sequence"],
                    default="vector")
    ap.add_argument("--num-slots", type=int, default=4)
    ap.add_argument("--slot-dim", type=int, default=16)
    ap.add_argument("--updater-layers", type=int, default=1)
    ap.add_argument("--pool-type", choices=["flatten", "attention"],
                    default="flatten")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="s0")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0)      # 0 -> default
    ap.add_argument("--eval-every", type=int, default=0)
    args = ap.parse_args()
    _ARGS = args

    smoke = args.smoke
    args.steps = args.steps or (4 if smoke else 400)
    args.eval_every = args.eval_every or (2 if smoke else 100)
    args.train_samples = 24 if smoke else 600
    args.eval_traj_train = 6 if smoke else 40
    args.eval_traj_final = 12 if smoke else 150
    args.max_tokens = 128
    args.lr = 3e-4
    args.warmup = 20
    args.weight_decay = 0.01

    cfg_tag = "vector" if args.state_type == "vector" else "sequence"
    out_dir = os.path.join(args.out_root,
                           f"{args.tag}_{cfg_tag}_{args.num_slots}x"
                           f"{args.slot_dim}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_data = generate_dataset(args.train_samples)
    eval_data = generate_dataset(args.eval_traj_final)
    ceiling = bayes_ceiling(eval_data)

    model, tok = load_ouro()
    film_layers = inject_film(model)
    state_module = build_state_module(args)
    n_tr, n_fr, trainable = param_counts(state_module, film_layers, model)
    ckpt_id = out_dir
    protocol = ("v1 per-loop state update; argmax next-token readout; "
                "with_S vs without_S (literal-zero modulation); t1/t2p/last/all")
    print_preamble(args, out_dir, n_tr, n_fr, ckpt_id, protocol)

    cfg = vars(args)
    cfg.update({"train_samples": args.train_samples,
                "eval_traj_train": args.eval_traj_train,
                "eval_traj_final": args.eval_traj_final,
                "max_tokens": args.max_tokens, "lr": args.lr,
                "warmup": args.warmup, "weight_decay": args.weight_decay,
                "smoke": smoke, "n_trainable": n_tr, "n_backbone_frozen": n_fr,
                "protocol": protocol, "ceiling": ceiling})
    json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=2)

    from transformers import get_scheduler
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=args.warmup,
                          num_training_steps=args.steps)

    log = []
    step = 0
    best_delta = -1e9
    t_start = time.time()
    while step < args.steps:
        random.shuffle(train_data)
        for traj in train_data:
            if step >= args.steps:
                break
            S = state_module.init_state(1, "cuda")
            b_loss = 0.0
            for turn in traj["turns"]:
                inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                          max_length=args.max_tokens).to("cuda")
                logits, S_new, s_norms = turn_forward_v1(
                    model, inp, film_layers, None, S,
                    state_module=state_module)
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
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0 or smoke:
                el = time.time() - t_start
                print(f"  step {step:4d}/{args.steps} | L={b_loss:.4f} | "
                      f"|Z|={float(S.norm()):.3f} | "
                      f"z_norms={[round(x, 2) for x in s_norms]} | "
                      f"{el/step:.1f}s/step | ETA "
                      f"{(args.steps-step)*el/step/60:.0f}min",
                      flush=True)
            if step % args.eval_every == 0 or step >= args.steps:
                norms = [round(float(l.scale_proj.weight.float().norm()), 3)
                         for l in film_layers]
                print(f"\n  eval @ step {step} | scale_proj norms: {norms}",
                      flush=True)
                assert all(x > 0 for x in norms)
                sub = eval_data[:args.eval_traj_train]
                w = eval_k(model, tok, sub, film_layers, state_module,
                           k=4, ablate_s=False)
                o = eval_k(model, tok, sub, film_layers, state_module,
                           k=4, ablate_s=True)
                d = w["t2p"]["acc"] - o["t2p"]["acc"]
                log.append({"step": step, "with_S": w, "without_S": o,
                            "delta_t2p": d})
                json.dump({"eval_log": log}, open(f"{out_dir}/eval_log.json", "w"),
                          indent=2)
                print(f"  t2p with={w['t2p']['acc']:.4f} "
                      f"({w['t2p']['correct']}/{w['t2p']['total']}) "
                      f"without={o['t2p']['acc']:.4f} delta={d:+.4f}\n",
                      flush=True)
                if d > best_delta and step > (2 if smoke else 50):
                    best_delta = d
                    ckpt = {
                        "step": step, "seed": args.seed, "ds": DS,
                        "film_layers": FILM_LAYERS, "variant": "s0_seqstate",
                        "state_type": args.state_type,
                        "num_slots": args.num_slots, "slot_dim": args.slot_dim,
                        "state_module": state_module.state_dict(),
                        "plaa_core": state_module.state_dict(),
                        "scale_proj": [l.scale_proj.state_dict()
                                       for l in film_layers],
                        "eval_log": log, "best_delta": best_delta,
                        "config": cfg,
                    }
                    path = f"{out_dir}/best_model.pt"
                    torch.save(ckpt, path)
                    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
                    print(f"  -> saved best (delta={d:+.4f}) sha256={h}",
                          flush=True)

    print("\nFINAL expanded eval:", flush=True)
    w = eval_k(model, tok, eval_data, film_layers, state_module, k=4,
               ablate_s=False)
    o = eval_k(model, tok, eval_data, film_layers, state_module, k=4,
               ablate_s=True)
    kw, nw = w["t2p"]["correct"], w["t2p"]["total"]
    ko, no = o["t2p"]["correct"], o["t2p"]["total"]
    ci_w, ci_o = wilson(kw, nw), wilson(ko, no)
    print(f"  with_S    {kw}/{nw} = {kw/max(nw,1):.4f}  "
          f"W95[{ci_w[0]:.3f},{ci_w[1]:.3f}]", flush=True)
    print(f"  without_S {ko}/{no} = {ko/max(no,1):.4f}  "
          f"W95[{ci_o[0]:.3f},{ci_o[1]:.3f}]", flush=True)
    print(f"  delta = {kw/max(nw,1)-ko/max(no,1):+.4f}", flush=True)

    # ensure the best checkpoint also reflects the final eval log
    best_path = f"{out_dir}/best_model.pt"
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        ckpt["eval_log"] = log
        torch.save(ckpt, best_path)
        h = hashlib.sha256(open(best_path, "rb").read()).hexdigest()[:16]
    else:
        h = None

    results = {
        "final_with": w, "final_without": o, "eval_log": log,
        "ceiling": ceiling, "seed": args.seed,
        "trainable_params": n_tr, "backbone_frozen_params": n_fr,
        "protocol": protocol, "state_type": args.state_type,
        "num_slots": args.num_slots, "slot_dim": args.slot_dim,
        "ckpt_sha256": h, "vram_peak_gb":
            round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "wall_clock_s": round(time.time() - t_start, 1),
    }
    json.dump(results, open(f"{out_dir}/results.json", "w"), indent=2)
    print(f"\nDONE. Results in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
