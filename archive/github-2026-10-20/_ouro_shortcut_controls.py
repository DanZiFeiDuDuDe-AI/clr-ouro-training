#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ouro shortcut-control suite: entity-shuffle (P0), memoryless-argmax (P1),
distance-stratification (P2) -- ported from the 0.8B protocols
(D:\\CLR\\phase_b4_entity_ablation_v2.py, phase_b3_distance_analysis.py,
phase_b2_interventions.py) to close the gap flagged in paper1.tex's
Discussion/Limitations: these controls existed for 0.8B but not for Ouro,
the backbone carrying the paper's external-validity claim.

Protocol matched to the existing Ouro headline numbers: same checkpoint
(clr_ouro_v0_s42/best_model.pt), same train(600)/eval(150) generation
stream (torch.manual_seed(42); random.seed(42)), same no-space
candidate-scoring readout (the corrected protocol from candidate_eval_v2).

Per user instruction: report whatever pattern emerges, do not optimize for
a clean result. All three conditions run on ONE shared eval set with
per-item tracking so distance and memoryless-argmax-wrong analyses reuse
the same baseline pass.
"""
import torch
import os, sys, json, random, re, math
from collections import Counter

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/LoopLM")

from clutrr_data import generate_dataset, bayes_ceiling, RELATIONS
from clr_ouro import load_ouro, inject_film, set_state, turn_forward_v0, PlaaCore

CKPT = "D:/LoopLM/checkpoints/clr_ouro_v0_s42/best_model.pt"
OUT = "D:/LoopLM/checkpoints/clr_ouro_v0_s42/shortcut_controls.json"
N_TRAJ = 150
MAX_TOKENS = 128

Z = 1.959963984540054
def wilson(k, n):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + Z*Z/n
    c = (p + Z*Z/(2*n))/d
    h = (Z/d)*math.sqrt(p*(1-p)/n + Z*Z/(4*n*n))
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))

def boot_diff(k1, n1, k2, n2, B=20000, seed=20260712):
    rng = random.Random(seed)
    p1, p2 = k1/n1, k2/n2
    diffs = []
    for _ in range(B):
        a = sum(1 for _ in range(n1) if rng.random() < p1) / n1
        b = sum(1 for _ in range(n2) if rng.random() < p2) / n2
        diffs.append(a - b)
    diffs.sort()
    return (round(diffs[int(0.025*B)], 4), round(diffs[int(0.975*B)], 4))

@torch.no_grad()
def score_candidates(model, film_layers, prompt_ids, s_vec, cand_ids, pad_id):
    n = len(cand_ids)
    P = len(prompt_ids)
    L = P + max(len(c) for c in cand_ids)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for i, c in enumerate(cand_ids):
        seq = prompt_ids + c
        input_ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
    input_ids, attn = input_ids.cuda(), attn.cuda()
    set_state(film_layers, s_vec)
    base_out, _, _ = model.model(input_ids=input_ids, attention_mask=attn,
                                 use_cache=False)
    logprobs = torch.log_softmax(model.lm_head(base_out.last_hidden_state).float(),
                                 dim=-1)
    best, best_i = -1e30, 0
    for i, c in enumerate(cand_ids):
        lp = sum(logprobs[i, P - 1 + j, t].item() for j, t in enumerate(c)) / len(c)
        if lp > best:
            best, best_i = lp, i
    return best_i

@torch.no_grad()
def run_with_tracking(model, tok, trajs, film_layers, plaa_core, cand_ids, pad_id):
    """with_S pass; returns per-item detail records for reuse across analyses."""
    details = []
    for traj in trajs:
        S = plaa_core.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            prompt_ids = tok(turn["prompt"], truncation=True,
                             max_length=MAX_TOKENS)["input_ids"]
            pred = score_candidates(model, film_layers, prompt_ids, S,
                                    cand_ids, pad_id)
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            _, _, h_last = turn_forward_v0(model, inp, film_layers, S)
            S = plaa_core(S, h_last.unsqueeze(0))
            details.append({"ti": ti, "n_turns": len(traj["turns"]),
                            "atomic": turn["atomic"], "answer": turn["answer"],
                            "pred": RELATIONS[pred]})
        torch.cuda.empty_cache()
    return details

def t2p_from_details(details):
    t2p = [d for d in details if d["ti"] >= 1]
    correct = sum(1 for d in t2p if d["pred"] == d["answer"])
    return correct, len(t2p)

# ---- P0: entity shuffle ----
NAME_RE = re.compile(r'\b[A-Z][a-z]+\b')
SKIP_WORDS = {"Fact", "Question", "What", "Is", "The", "To", "Answer", "A", "An"}

def extract_entity_names(trajs):
    names = set()
    for traj in trajs:
        for turn in traj["turns"]:
            for m in NAME_RE.finditer(turn["prompt"]):
                w = m.group()
                if w not in SKIP_WORDS and len(w) > 1:
                    names.add(w)
    return sorted(names)

def shuffle_entities(trajs, e_map):
    out = []
    for traj in trajs:
        new_turns = []
        for turn in traj["turns"]:
            p = turn["prompt"]
            for old, new in sorted(e_map.items(), key=lambda x: -len(x[0])):
                p = p.replace(old, new)
            new_turns.append({"prompt": p, "answer": turn["answer"],
                              "atomic": turn["atomic"]})
        out.append({"turns": new_turns})
    return out

def main():
    print("="*64, flush=True)
    print("Ouro shortcut-control suite (entity-shuffle / memoryless / distance)",
          flush=True)
    print("="*64, flush=True)

    torch.manual_seed(42); random.seed(42)
    train_data = generate_dataset(600)
    eval_data = generate_dataset(N_TRAJ)
    ceiling = bayes_ceiling(eval_data)
    print(f"train=600 eval={len(eval_data)} ceiling(t2p)={ceiling:.4f}", flush=True)

    model, tok = load_ouro()
    film_layers = inject_film(model)
    plaa_core = PlaaCore()
    try:
        ckpt = torch.load(CKPT, map_location="cuda", weights_only=True)
    except Exception:
        ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    plaa_core.load_state_dict(ckpt["plaa_core"])
    for l, sd in zip(film_layers, ckpt["scale_proj"]):
        l.scale_proj.load_state_dict(sd)
    norms = [round(float(l.scale_proj.weight.float().norm()), 3) for l in film_layers]
    print(f"ckpt step {ckpt.get('step')} | norms {norms}", flush=True)
    assert all(x > 0.1 for x in norms), "zero injection weights"

    cand_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad_id = tok.pad_token_id

    # ---- baseline pass (feeds P1 + P2) ----
    print("\n[1/2] Baseline with_S pass (n=%d trajectories)..." % len(eval_data),
          flush=True)
    base_details = run_with_tracking(model, tok, eval_data, film_layers, plaa_core,
                                     cand_ids, pad_id)
    bk, bn = t2p_from_details(base_details)
    ci_base = wilson(bk, bn)
    print(f"  baseline t2p = {bk}/{bn} = {bk/bn:.4f} W95{ci_base}", flush=True)

    # ---- P1: memoryless-argmax ----
    by_atomic, glob = {}, Counter()
    for traj in train_data:
        for ti, turn in enumerate(traj["turns"]):
            if ti < 1:
                continue
            by_atomic.setdefault(turn["atomic"], Counter())[turn["answer"]] += 1
            glob[turn["answer"]] += 1
    fallback = glob.most_common(1)[0][0]
    mode_by_atomic = {a: c.most_common(1)[0][0] for a, c in by_atomic.items()}

    t2p_details = [d for d in base_details if d["ti"] >= 1]
    argmax_hits = agree = hits_wrong = n_wrong = 0
    for d in t2p_details:
        m = mode_by_atomic.get(d["atomic"], fallback)
        argmax_hits += (m == d["answer"])
        agree += (d["pred"] == m)
        if m != d["answer"]:
            n_wrong += 1
            hits_wrong += (d["pred"] == d["answer"])
    n_t2p = len(t2p_details)
    argmax_acc = argmax_hits / n_t2p
    agreement = agree / n_t2p
    acc_on_wrong = (hits_wrong / n_wrong) if n_wrong else None
    print(f"\n[P1] memoryless-argmax: argmax_acc={argmax_acc:.3f} "
          f"agreement={agreement:.3f} acc_on_argmax_wrong="
          f"{acc_on_wrong:.3f} (n={n_wrong}/{n_t2p})" if acc_on_wrong is not None
          else f"\n[P1] no argmax-wrong cases found", flush=True)

    # ---- P2: distance stratification ----
    dist_groups = {}
    for d in t2p_details:
        g = dist_groups.setdefault(d["ti"], {"total": 0, "correct": 0})
        g["total"] += 1
        g["correct"] += int(d["pred"] == d["answer"])
    print("\n[P2] accuracy by distance:", flush=True)
    for dist in sorted(dist_groups):
        g = dist_groups[dist]
        print(f"  dist={dist}: {g['correct']}/{g['total']} = "
              f"{g['correct']/g['total']:.3f}", flush=True)

    wrong_dist_groups = {}
    for d in t2p_details:
        m = mode_by_atomic.get(d["atomic"], fallback)
        if m != d["answer"]:
            g = wrong_dist_groups.setdefault(d["ti"], {"total": 0, "correct": 0})
            g["total"] += 1
            g["correct"] += int(d["pred"] == d["answer"])

    # ---- P0: entity shuffle ----
    print("\n[2/2] Entity-shuffle pass...", flush=True)
    entities = extract_entity_names(eval_data)
    print(f"  entities found: {len(entities)} -> {entities[:10]}", flush=True)
    shuffled_names = list(entities)
    random.Random(999).shuffle(shuffled_names)
    e_map = dict(zip(entities, shuffled_names))
    shuf_data = shuffle_entities(eval_data, e_map)
    shuf_details = run_with_tracking(model, tok, shuf_data, film_layers, plaa_core,
                                     cand_ids, pad_id)
    sk, sn = t2p_from_details(shuf_details)
    ci_shuf = wilson(sk, sn)
    drop = bk/bn - sk/sn
    ci_drop = boot_diff(bk, bn, sk, sn)
    print(f"  shuffled t2p = {sk}/{sn} = {sk/sn:.4f} W95{ci_shuf}", flush=True)
    print(f"  drop (baseline - shuffled) = {drop:+.4f}  boot95 {ci_drop}", flush=True)

    if drop > 0.15:
        verdict = "entity_memorization"
    elif drop > 0.05:
        verdict = "partial"
    else:
        verdict = "generalized"
    print(f"  verdict: {verdict}", flush=True)

    results = {
        "ckpt": CKPT, "ckpt_step": int(ckpt.get("step", -1)),
        "n_traj": len(eval_data), "ceiling_t2p": ceiling,
        "baseline": {"t2p_correct": bk, "t2p_total": bn,
                    "t2p": round(bk/bn, 4), "t2p_wilson95": list(ci_base)},
        "P0_entity_shuffle": {
            "n_entities": len(entities), "sample_entities": entities[:10],
            "shuffled_t2p_correct": sk, "shuffled_t2p_total": sn,
            "shuffled_t2p": round(sk/sn, 4), "shuffled_t2p_wilson95": list(ci_shuf),
            "drop": round(drop, 4), "drop_boot95": list(ci_drop),
            "verdict": verdict,
        },
        "P1_memoryless_argmax": {
            "argmax_acc": round(argmax_acc, 4), "agreement": round(agreement, 4),
            "acc_on_argmax_wrong": round(acc_on_wrong, 4) if acc_on_wrong is not None else None,
            "n_argmax_wrong": n_wrong, "n_t2p": n_t2p,
        },
        "P2_distance": {
            "by_distance": {str(k): v for k, v in dist_groups.items()},
            "acc_on_wrong_by_distance": {str(k): v for k, v in wrong_dist_groups.items()},
        },
    }
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nDONE -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
