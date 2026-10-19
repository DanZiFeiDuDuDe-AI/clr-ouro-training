#!/usr/bin/env python3
"""
Phase 5B v2 — Causal Intervention on S_t.
Surgery applied DIRECTLY to FiLM layer's _s at evaluation, not to GRU input.
This ensures the intervention is actually seen by the model during inference.

Conditions:
A. Normal full chain (baseline)
B. ti=1: replace FiLM _s with ti=0's S from same traj (hop backward)
C. ti=0: replace FiLM _s with ti=1's S from same traj (hop forward)
D. ti=1: replace FiLM _s with ti=1's S from DIFFERENT traj (phase control)
E. ti=1: add Gaussian noise to FiLM _s (magnitude control)
F. ti=1: set FiLM _s to zero (ablation, should match w/o S)
"""
import os, json, random
import torch
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
N_REL = 14

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

# 2️⃣ Collect trajectories for reference
print("Collecting trajectories...", flush=True)
rng_eval = random.Random(CKPT_SEED + 1000)
eval_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_eval)
rng_other = random.Random(9999)
other_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_other)

def collect_states_mapping(data):
    """Return list: per traj, per turn, the FiLM _s (S_next after GRU), in numpy."""
    all_states = []
    for traj in data:
        S = plaa_core.init_state(1, device)
        traj_S = []
        for ti, turn in enumerate(traj["turns"]):
            prompt_ids = tok(turn["prompt"], truncation=True, max_length=T.MAX_TOKENS)["input_ids"]
            for l in film_layers:
                l._s = S
            pin = torch.tensor([prompt_ids], device=device)
            hs = model(input_ids=pin, attention_mask=torch.ones_like(pin),
                       output_hidden_states=True).hidden_states[-1]
            S_next = plaa_core(S, hs[0, -1].unsqueeze(0))
            traj_S.append(S_next[0].detach().cpu().float().numpy().copy())
            S = S_next
        all_states.append(traj_S)
        torch.cuda.empty_cache()
    return all_states

eval_states = collect_states_mapping(eval_data)
other_states = collect_states_mapping(other_data)
print(f"  Collected {len(eval_states)} eval trajectories", flush=True)

# 3️⃣ Evaluate with direct FiLM _s intervention
def evaluate_with_film_intervention(data, intervention_fn, label):
    """
    intervention_fn(S_film, ti, traj_idx) -> S_film_modified
    Applied RIGHT BEFORE the model forward pass, after FiLM _s is set.
    """
    correct_t1 = correct_t2p = total_t1 = total_t2p = 0
    
    for traj_idx, traj in enumerate(data):
        S = plaa_core.init_state(1, device)
        
        for ti, turn in enumerate(traj["turns"]):
            prompt_ids = tok(turn["prompt"], truncation=True, max_length=T.MAX_TOKENS)["input_ids"]
            
            # Set FiLM _s, THEN apply intervention
            for l in film_layers:
                l._s = S
            intervention_fn(S, ti, traj_idx, traj)
            
            pin = torch.tensor([prompt_ids], device=device)
            hs = model(input_ids=pin, attention_mask=torch.ones_like(pin),
                       output_hidden_states=True).hidden_states[-1]
            S_next = plaa_core(S, hs[0, -1].unsqueeze(0))
            
            # Predict using standard score_candidates (uses post-GRU S_next)
            pred = T.score_candidates(model, film_layers, prompt_ids, S_next, cand_ids, pad_id)
            ok = int(RELATIONS[pred] == turn["answer"])
            
            if ti == 0:
                correct_t1 += ok; total_t1 += 1
            else:
                correct_t2p += ok; total_t2p += 1
            
            S = S_next  # S_next becomes the new S for next turn
        torch.cuda.empty_cache()
    
    t1 = correct_t1 / total_t1 if total_t1 else 0
    t2p = correct_t2p / total_t2p if total_t2p else 0
    all_acc = (correct_t1 + correct_t2p) / (total_t1 + total_t2p)
    print(f"  {label:>35s}: t1={t1:.3f} t2p={t2p:.3f} all={all_acc:.3f} ({correct_t2p}/{total_t2p})", flush=True)
    return {"t1": t1, "t2p": t2p, "all": all_acc}

# ---- Interventions ----
def no_intervention(S, ti, traj_idx, traj):
    pass  # FiLM _s is already set to S

def hop_backward(S, ti, traj_idx, traj):
    if ti == 1:  # second turn → use S from turn 0
        s0_np = eval_states[traj_idx][0]
        s0 = torch.from_numpy(s0_np).unsqueeze(0).to(device, dtype=torch.bfloat16)
        for l in film_layers:
            l._s = s0

def hop_forward(S, ti, traj_idx, traj):
    if ti == 0:  # first turn → use S from turn 1
        states = eval_states[traj_idx]
        if len(states) > 1:
            s1_np = states[1]
            s1 = torch.from_numpy(s1_np).unsqueeze(0).to(device, dtype=torch.bfloat16)
            for l in film_layers:
                l._s = s1

def diff_traj_control(S, ti, traj_idx, traj):
    if ti == 1:
        other_idx = (traj_idx + 37) % len(other_states)
        s_other_np = other_states[other_idx][1]
        s_other = torch.from_numpy(s_other_np).unsqueeze(0).to(device, dtype=torch.bfloat16)
        for l in film_layers:
            l._s = s_other

def add_noise(S, ti, traj_idx, traj):
    if ti == 1:
        noise = torch.randn_like(S) * 0.1
        s_noisy = S + noise
        for l in film_layers:
            l._s = s_noisy

def zero_ablation(S, ti, traj_idx, traj):
    if ti == 1:
        for l in film_layers:
            l._s = torch.zeros_like(S)

# 4️⃣ Run
data = eval_data
print("\n=== Causal Intervention via FiLM _s Surgery ===", flush=True)
results = {}
for name, fn in [
    ("A: Normal (baseline)", no_intervention),
    ("B: Hop backward (S0→ti=1)", hop_backward),
    ("C: Hop forward (S1→ti=0)", hop_forward),
    ("D: Diff traj S (control)", diff_traj_control),
    ("E: Noise (control)", add_noise),
    ("F: Zero S (ablation)", zero_ablation),
]:
    r = evaluate_with_film_intervention(data, fn, name)
    results[name] = r

# Summary
baseline = results.get("A: Normal (baseline)", {}).get("t2p", 0)
print("\n" + "=" * 65, flush=True)
print("SUMMARY: CAUSAL INTERVENTION (FiLM surgery)", flush=True)
print("=" * 65, flush=True)
print(f"{'Condition':<35s} {'t2p':>7s} {'Δ':>7s} {'Δpp':>7s}", flush=True)
print("-" * 65, flush=True)
for name, r in results.items():
    t2p = r["t2p"]
    delta = t2p - baseline
    print(f"{name:<35s} {t2p:>7.3f} {delta:+7.3f} {delta*100:+6.1f}pp", flush=True)

# Save
output = {
    "checkpoint": CKPT,
    "conditions": results,
    "n_trajs": len(eval_states),
}
with open(f"{WORKSPACE}/02_results/phase_5b_causal_intervention_v2.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: v2 JSON", flush=True)
print("Done.", flush=True)
