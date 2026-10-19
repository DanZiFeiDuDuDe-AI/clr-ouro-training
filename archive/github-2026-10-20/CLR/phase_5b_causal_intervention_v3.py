#!/usr/bin/env python3
"""
Phase 5B v3 — Causal Intervention via surgery_fn hook.
Uses official evaluate() path with traj_idx now passed to surgery_fn.
"""
import os, json, random, time
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

# 1️⃣ Model
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

# 2️⃣ Data
rng_eval = random.Random(CKPT_SEED + 1000)
eval_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_eval)
rng_other = random.Random(9999)
other_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_other)

# 3️⃣ Collect reference states
def collect_full_states(data):
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

print("Collecting eval states...", flush=True)
eval_states = collect_full_states(eval_data)
print("Collecting other states...", flush=True)
other_states = collect_full_states(other_data)

# 4️⃣ Surgery fns: surgery_fn(S, ti, traj_idx, film_layers) -> S_modified or None
# Pre-allocate tensors
backward_s = [torch.from_numpy(eval_states[t][0]).unsqueeze(0).to(device, dtype=torch.bfloat16)
              for t in range(len(eval_states))]
forward_s = [torch.from_numpy(eval_states[t][1]).unsqueeze(0).to(device, dtype=torch.bfloat16)
             if len(eval_states[t]) > 1 else None for t in range(len(eval_states))]
diff_s = [None] * len(eval_states)
for t in range(len(eval_states)):
    oi = (t + 37) % len(other_states)
    diff_s[t] = torch.from_numpy(other_states[oi][1]).unsqueeze(0).to(device, dtype=torch.bfloat16)

def no_surgery(S, ti, traj_idx, film_layers):
    return S

def hop_backward(S, ti, traj_idx, film_layers):
    if ti == 1:
        return backward_s[traj_idx]
    return S

def hop_forward(S, ti, traj_idx, film_layers):
    if ti == 0:
        return forward_s[traj_idx] if forward_s[traj_idx] is not None else S
    return S

def diff_control(S, ti, traj_idx, film_layers):
    if ti == 1:
        return diff_s[traj_idx]
    return S

def add_noise(S, ti, traj_idx, film_layers):
    if ti == 1:
        return S + torch.randn_like(S) * 0.1
    return S

def zero_ablation(S, ti, traj_idx, film_layers):
    if ti == 1:
        return torch.zeros_like(S)
    return S

# 5️⃣ Run
print("\n=== Causal Intervention (FiLM surgery via evaluate hook v3) ===", flush=True)
results = {}
for name, fn in [
    ("A: Normal (baseline)", no_surgery),
    ("B: Hop backward (S0→ti=1)", hop_backward),
    ("C: Hop forward (S1→ti=0)", hop_forward),
    ("D: Diff traj S (control)", diff_control),
    ("E: Noise (control)", add_noise),
    ("F: Zero S (ablation)", zero_ablation),
]:
    t0 = time.time()
    r = T.evaluate(model, tok, eval_data, film_layers, plaa_core, cand_ids,
                   surgery_fn=fn)
    elapsed = time.time() - t0
    results[name] = r
    t2p = r["t2p"]
    t1 = r.get("t1", 0)
    n_t1 = r.get("raw", {}).get("t1", (0,0))
    n_t2 = r.get("raw", {}).get("t2p", (0,0))
    print(f"  {name:<35s}: t1={t1:.3f} t2p={t2p:.3f} in {elapsed:.0f}s", flush=True)

# Summary
baseline = results["A: Normal (baseline)"]["t2p"]
print("\n" + "=" * 70, flush=True)
print("CAUSAL INTERVENTION — FINAL SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"{'Condition':<35s} {'t2p':>7s} {'t1':>7s} {'Δ':>8s} {'Δpp':>7s}", flush=True)
print("-" * 70, flush=True)
print(f"{'A: Normal (baseline)':<35s} {baseline:>7.3f}   (reference)", flush=True)
for name, r in results.items():
    t2p = r["t2p"]
    t1 = r.get("t1", 0)
    delta = t2p - baseline
    print(f"{name:<35s} {t2p:>7.3f} {t1:>7.3f} {delta:+8.3f} {delta*100:+6.1f}pp", flush=True)

output = {
    "checkpoint": CKPT,
    "conditions": {k: {"t2p": v["t2p"], "t1": v.get("t1", 0)} for k, v in results.items()},
    "baseline_t2p": baseline,
}
with open(f"{WORKSPACE}/02_results/phase_5b_causal_intervention_v3.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: phase_5b_causal_intervention_v3.json", flush=True)
print("Done.", flush=True)
