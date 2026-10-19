#!/usr/bin/env python3
"""
Phase 5B — Causal Intervention on S_t's hop-phase.
Surgery: at test time, replace S_t with S from a different hop index.
If S_t truly drives reasoning state (not just correlates with it),
misaligning the hop phase should degrade performance predictably.

Conditions (all on seed 42, DS=64, Qwen3.5-0.8B):
A. Normal full chain (baseline)
B. Step 2: replace S with S from step 1 of SAME trajectory (hop backward)
C. Step 1: replace S with S from step 2 of SAME trajectory (hop forward)
D. Step 2: replace S with S from step 2 of DIFFERENT trajectory (same phase, different content)
E. Step 2: replace S with Gaussian noise (same norm as real S)
"""
import os, json, random
import torch
import torch.nn.functional as F
import numpy as np

import local_train_clutrr_v2 as T
from clutrr_data import RELATIONS, generate_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

WORKSPACE = T.WORKSPACE
MODEL_PATH = T.MODEL_PATH
FILM_LAYERS = T.FULL_ATTN_LAYERS
CKPT = "local_clutrr_v2_nolora"
CKPT_SEED = 42
device = "cuda"

# 1️⃣ Load model
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

ck = torch.load(f"{WORKSPACE}/checkpoints/{CKPT}/best_model.pt",
    map_location=device, weights_only=False)
for idx, fl in zip(FILM_LAYERS, film_layers):
    fl.scale_proj.load_state_dict(ck["scale_proj"][idx])
plaa_core.load_state_dict(ck["plaa_core"])
cand_ids = [tok(" " + r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
pad_id = tok.pad_token_id

# 2️⃣ Collect S_t trajectories (full chain)
print("Collecting S_t trajectories...", flush=True)
rng_eval = random.Random(CKPT_SEED + 1000)
eval_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_eval)

# Also collect a separate set for 'different trajectory' replacement
rng_other = random.Random(9999)
other_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_other)

def collect_trajectories(data):
    """Run model on data and collect ALL S_t per trajectory."""
    trajectories = []
    for traj_idx, traj in enumerate(data):
        S = plaa_core.init_state(1, device)
        states = [S[0].detach().cpu().float().numpy().copy()]
        turn_info = []
        for ti, turn in enumerate(traj["turns"]):
            prompt_ids = tok(turn["prompt"], truncation=True, max_length=T.MAX_TOKENS)["input_ids"]
            for l in film_layers:
                l._s = S
            pin = torch.tensor([prompt_ids], device=device)
            hs = model(input_ids=pin, attention_mask=torch.ones_like(pin),
                       output_hidden_states=True).hidden_states[-1]
            S_next = plaa_core(S, hs[0, -1].unsqueeze(0))
            states.append(S_next[0].detach().cpu().float().numpy().copy())
            turn_info.append({
                "answer": turn["answer"],
                "prompt": turn["prompt"],
            })
            S = S_next
        torch.cuda.empty_cache()
        trajectories.append({"states": states, "turns": turn_info})
    return trajectories

print("  Collecting eval trajectories...", flush=True)
eval_trajs = collect_trajectories(eval_data)
print("  Collecting reference trajectories...", flush=True)
other_trajs = collect_trajectories(other_data)

print(f"  Collected {len(eval_trajs)} eval trajectories", flush=True)

# 3️⃣ Eval functions for each condition
@torch.no_grad()
def evaluate_with_surgery(data, surgery_fn, label):
    """
    Run eval with a surgery function applied to S at each turn.
    surgery_fn(S_current, ti, traj_idx, trajectories) -> S_modified
    """
    correct_t1 = correct_t2p = total_t1 = total_t2p = 0
    all_preds = []
    
    for traj_idx, traj in enumerate(data):
        traj_states = [s.copy() for s in eval_trajs[traj_idx]["states"]]
        S = torch.from_numpy(traj_states[0]).unsqueeze(0).to(device, dtype=torch.bfloat16)
        
        for ti, turn in enumerate(traj["turns"]):
            # Apply surgery BEFORE this turn
            S_modified = surgery_fn(S, ti, traj_idx)
            
            prompt_ids = tok(turn["prompt"], truncation=True, max_length=T.MAX_TOKENS)["input_ids"]
            for l in film_layers:
                l._s = S_modified
            pin = torch.tensor([prompt_ids], device=device)
            hs = model(input_ids=pin, attention_mask=torch.ones_like(pin),
                       output_hidden_states=True).hidden_states[-1]
            S_next = plaa_core(S_modified, hs[0, -1].unsqueeze(0))
            
            # Predict using S_next (standard)
            pred = T.score_candidates(model, film_layers, prompt_ids, S_next, cand_ids, pad_id)
            ok = int(RELATIONS[pred] == turn["answer"])
            
            if ti == 0:
                correct_t1 += ok; total_t1 += 1
            else:
                correct_t2p += ok; total_t2p += 1
            all_preds.append({"ti": ti, "pred": RELATIONS[pred], "ans": turn["answer"], "ok": ok})
            
            S = S_modified  # Use modified S for next step
            
        torch.cuda.empty_cache()
    
    t1 = correct_t1 / total_t1 if total_t1 else 0
    t2p = correct_t2p / total_t2p if total_t2p else 0
    all_acc = (correct_t1 + correct_t2p) / (total_t1 + total_t2p)
    print(f"  {label:>35s}: t1={t1:.3f} t2p={t2p:.3f} all={all_acc:.3f} ({correct_t2p}/{total_t2p})", flush=True)
    return {"t1": t1, "t2p": t2p, "all": all_acc, "correct_t2p": correct_t2p, "total_t2p": total_t2p}

# Condition A: Normal (no surgery)
def no_surgery(S, ti, traj_idx):
    return S

# Condition B: Step 2 → replace with Step 1 S (hop backward)
def hop_backward(S, ti, traj_idx):
    if ti == 1:  # turn 1 → swap S with turn 0's state
        s0 = torch.from_numpy(eval_trajs[traj_idx]["states"][0]).unsqueeze(0).to(device, dtype=torch.bfloat16)
        return s0
    return S

# Condition C: Step 1 → replace with Step 2 S (hop forward)
def hop_forward(S, ti, traj_idx):
    if ti == 0:  # turn 0 → swap S with turn 1's state (if available)
        states = eval_trajs[traj_idx]["states"]
        if len(states) > 1:
            s1 = torch.from_numpy(states[1]).unsqueeze(0).to(device, dtype=torch.bfloat16)
            return s1
    return S

# Condition D: Step 2 → replace with different trajectory's Step 2 S
def same_phase_different_content(S, ti, traj_idx):
    if ti == 1:  # turn 1 → swap with another trajectory's turn 1 S
        # Pick a different trajectory
        other_idx = (traj_idx + 37) % len(other_trajs)  # assured different
        s_other = torch.from_numpy(other_trajs[other_idx]["states"][1]).unsqueeze(0).to(device, dtype=torch.bfloat16)
        return s_other
    return S

# Condition E: Step 2 → add Gaussian noise (matched norm)
def add_noise(S, ti, traj_idx):
    if ti == 1:
        noise = torch.randn_like(S) * 0.1  # scaled to typical |S| norms we observed (~0.1)
        return S + noise
    return S

# Condition F: Step 2 → replace with ZERO vector (like ablation)
def zero_surgery(S, ti, traj_idx):
    if ti == 1:
        return torch.zeros_like(S)
    return S

data = eval_data

print("\n=== Causal Intervention Results ===", flush=True)
results = {}
for name, fn in [
    ("A: Normal (full chain)", no_surgery),
    ("B: Hop backward (ti=1←S0)", hop_backward),
    ("C: Hop forward (ti=0←S1)", hop_forward),
    ("D: Diff traj S2 (control)", same_phase_different_content),
    ("E: Noise (control)", add_noise),
    ("F: Zero S (ablation)", zero_surgery),
]:
    r = evaluate_with_surgery(data, fn, name)
    results[name] = r

# Summary table
print("\n" + "=" * 65, flush=True)
print("CAUSAL INTERVENTION SUMMARY", flush=True)
print("=" * 65, flush=True)
print(f"{'Condition':<35s} {'t2p Acc':>9s} {'Δ from A':>9s} {'Δdrop':>9s}", flush=True)
print("-" * 65, flush=True)
baseline = results.get("A: Normal (full chain)", {}).get("t2p", 0)
for name, r in results.items():
    t2p = r["t2p"]
    delta = t2p - baseline
    alabel = "baseline" if name.startswith("A") else ""
    print(f"{name:<35s} {t2p:>9.3f} {delta:+9.3f}", flush=True)

# Save
output = {
    "checkpoint": CKPT,
    "conditions": {k: v for k, v in results.items()},
    "n_trajs": len(eval_trajs),
}
with open(f"{WORKSPACE}/02_results/phase_5b_causal_intervention.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {WORKSPACE}/02_results/phase_5b_causal_intervention.json", flush=True)
print("Done.", flush=True)
