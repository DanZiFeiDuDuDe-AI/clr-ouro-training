#!/usr/bin/env python3
"""
Phase B3 — distance-stratified analysis on full-condition predictions.
Runs only seed 42, DS=64, saves per-turn details for distance breakdown.
"""
import os, json, random, sys
from collections import Counter
import torch

import local_train_clutrr_v2 as T
from clutrr_data import RELATIONS, generate_dataset, bayes_ceiling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

WORKSPACE = T.WORKSPACE
MODEL_PATH = T.MODEL_PATH
FILM_LAYERS = T.FULL_ATTN_LAYERS

CKPT = "local_clutrr_v2_nolora"
CKPT_SEED = 42

device = "cuda"

print("Loading model...", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
    quantization_config=bnb, device_map="auto", trust_remote_code=True,
    torch_dtype=torch.bfloat16, attn_implementation="sdpa")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token if tok.eos_token else "<|endoftext|>"
for p in model.parameters():
    p.requires_grad = False
DM = model.config.hidden_size

layers = model.model.layers
film_layers = []
for idx in FILM_LAYERS:
    layers[idx] = T.FiLMWrapper(layers[idx], DM)
    film_layers.append(layers[idx])
plaa_core = T.PlaaCore(DM).cuda()
model.eval()

# Load checkpoint
ckpt_dir = f"{WORKSPACE}/checkpoints/{CKPT}"
ck = torch.load(f"{ckpt_dir}/best_model.pt", map_location=device, weights_only=False)
for idx, fl in zip(FILM_LAYERS, film_layers):
    fl.scale_proj.load_state_dict(ck["scale_proj"][idx])
plaa_core.load_state_dict(ck["plaa_core"])

def run_and_track(model, tok, film_layers, plaa_core, trajs, cand_ids, pad_id):
    """Run full condition with per-turn detail tracking."""
    details = []
    for traj in trajs:
        S = plaa_core.init_state(1, device)
        for ti, turn in enumerate(traj["turns"]):
            prompt_ids = tok(turn["prompt"], truncation=True, max_length=T.MAX_TOKENS)["input_ids"]
            for l in film_layers:
                l._s = S
            pin = torch.tensor([prompt_ids], device=device)
            hs = model(input_ids=pin, attention_mask=torch.ones_like(pin),
                       output_hidden_states=True).hidden_states[-1]
            S_next = plaa_core(S, hs[0, -1].unsqueeze(0))
            pred = T.score_candidates(model, film_layers, prompt_ids, S_next, cand_ids, pad_id)
            details.append({
                "ti": ti, "n_turns": len(traj["turns"]),
                "atomic": turn["atomic"],
                "answer": turn["answer"],
                "pred": RELATIONS[pred],
                "prompt": turn["prompt"],
            })
            S = S_next
        torch.cuda.empty_cache()
    return details

# Data
rng = random.Random(CKPT_SEED + 1000)
eval_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng)
cand_ids = [tok(" " + r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
pad_id = tok.pad_token_id

print("Running full condition with per-turn tracking...", flush=True)
details = run_and_track(model, tok, film_layers, plaa_core, eval_data, cand_ids, pad_id)

# Distance-stratified analysis
t2p = [d for d in details if d["ti"] >= 1]

# Group by distance from first turn
groups = {}
for d in t2p:
    dist = d["ti"]  # distance from turn 0 (ti=1 = 1 hop away, ti=2 = 2 hops away...)
    g = groups.setdefault(dist, {"total": 0, "correct": 0})
    g["total"] += 1
    g["correct"] += int(d["pred"] == d["answer"])

print("\n=== Accuracy by distance (full condition, seed 42) ===", flush=True)
print(f"{'dist':>5s} {'correct/total':>15s} {'acc':>8s}")
for dist in sorted(groups.keys()):
    g = groups[dist]
    acc = g["correct"] / g["total"]
    print(f"{dist:5d} {g['correct']:>5d}/{g['total']:<6d}  {acc:7.3f}")

# Now the key: acc_on_argmax_wrong by distance
rng_train = random.Random(CKPT_SEED)
train_data = generate_dataset(600, (1, 2, 3), (0.2, 0.5, 0.3), rng_train)
by_atomic = {}
glob = Counter()
for traj in train_data:
    for ti, turn in enumerate(traj["turns"]):
        if ti < 1:
            continue
        by_atomic.setdefault(turn["atomic"], Counter())[turn["answer"]] += 1
        glob[turn["answer"]] += 1
fallback = glob.most_common(1)[0][0]
mode_by_atomic = {a: c.most_common(1)[0][0] for a, c in by_atomic.items()}

wrong_groups = {}
for d in t2p:
    m = mode_by_atomic.get(d["atomic"], fallback)
    if m != d["answer"]:
        dist = d["ti"]
        g = wrong_groups.setdefault(dist, {"total": 0, "correct": 0})
        g["total"] += 1
        g["correct"] += int(d["pred"] == d["answer"])

print("\n=== acc_on_argmax_wrong by distance ===", flush=True)
if wrong_groups:
    print(f"{'dist':>5s} {'correct/wrong_total':>20s} {'acc_on_wrong':>15s}")
    for dist in sorted(wrong_groups.keys()):
        g = wrong_groups[dist]
        acc = g["correct"] / g["total"]
        print(f"{dist:5d} {g['correct']:>5d}/{g['total']:<12d}  {acc:7.3f}")
else:
    print("No argmax_wrong cases found", flush=True)

# Relation-type breakdown
print("\n=== Accuracy by relation ===", flush=True)
rel_acc = {}
for d in t2p:
    ans = d["answer"]
    rel_acc.setdefault(ans, {"total": 0, "correct": 0})["total"] += 1
    rel_acc[ans]["correct"] += int(d["pred"] == ans)
for r in sorted(rel_acc.keys()):
    g = rel_acc[r]
    acc = g["correct"] / g["total"]
    print(f"  {r:>6s}: {g['correct']:>3d}/{g['total']:<3d} = {acc:.3f}")

# Save all details
output = {
    "checkpoint": CKPT,
    "ceiling": bayes_ceiling(eval_data, min_turn=1),
    "accuracy_by_distance": {str(k): v for k, v in groups.items()},
    "acc_on_wrong_by_distance": {str(k): v for k, v in wrong_groups.items()},
    "accuracy_by_relation": {k: v for k, v in rel_acc.items()},
    "n_t2p_total": len(t2p),
}
with open(f"{WORKSPACE}/02_results/phase_b3_distance_analysis.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: {WORKSPACE}/02_results/phase_b3_distance_analysis.json", flush=True)
print("Done.", flush=True)
