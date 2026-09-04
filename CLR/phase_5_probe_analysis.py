#!/usr/bin/env python3
"""
Phase 5 — Information Probe on S_t.
Run linear probes on the persistent latent state to decode what
compositional information it encodes.

Probe targets:
1. hop_idx — which turn in the trajectory (0, 1, 2)
2. composition_depth — how many atomic facts have been composed (1, 2, 3)
3. target_answer — the final relation class (14-way)
4. current_relation — the atomic relation in this turn (14-way)
5. composed_answer — the intermediate answer for this turn (14-way)

Requires: trained checkpoint + gathering S_t at each turn.
"""
import os, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import local_train_clutrr_v2 as T
from clutrr_data import RELATIONS, generate_dataset, bayes_ceiling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

WORKSPACE = T.WORKSPACE
MODEL_PATH = T.MODEL_PATH
FILM_LAYERS = T.FULL_ATTN_LAYERS
CKPT = "local_clutrr_v2_nolora"  # seed 42, DS=64, best evolved
CKPT_SEED = 42
device = "cuda"

N_RELATIONS = len(RELATIONS)

# 1️⃣ Load model + FiLM + PlaaCore
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
DM = model.config.hidden_size  # 896 for Qwen3.5-0.8B

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

# 2️⃣ Collect S_t samples with labels
print("Collecting S_t samples...", flush=True)

rng_train = random.Random(CKPT_SEED)
train_data = generate_dataset(600, (1, 2, 3), (0.2, 0.5, 0.3), rng_train)
rng_eval = random.Random(CKPT_SEED + 1000)
eval_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_eval)

def collect_states(data, label_set):
    """Run model and collect (S_t, label_dict) pairs for each turn."""
    all_S = []  # list of (DS,)-sized numpy arrays
    all_labels = {
        "hop_idx": [],
        "composition_depth": [],
        "target_answer": [],
        "current_relation": [],
        "composed_answer": [],
    }
    
    for traj in data:
        S = plaa_core.init_state(1, device)
        final_answer = traj["turns"][-1]["answer"]
        target_id = RELATIONS.index(final_answer)
        depth_so_far = 0
        
        for ti, turn in enumerate(traj["turns"]):
            prompt_ids = tok(turn["prompt"], truncation=True, max_length=T.MAX_TOKENS)["input_ids"]
            for l in film_layers:
                l._s = S
            pin = torch.tensor([prompt_ids], device=device)
            hs = model(input_ids=pin, attention_mask=torch.ones_like(pin),
                       output_hidden_states=True).hidden_states[-1]
            S_next = plaa_core(S, hs[0, -1].unsqueeze(0))
            
            # Record the S BEFORE this turn's update as the state for this turn
            all_S.append(S[0].detach().cpu().float().numpy())
            
            # Current turn's atomic relation
            curr_rel = turn["answer"]
            curr_rel_id = RELATIONS.index(curr_rel)
            
            # Composition depth: how many atomic facts have been composed so far
            depth_so_far += 1 if ti >= 1 else 0  # t1 doesn't require composition
            
            all_labels["hop_idx"].append(ti)
            all_labels["composition_depth"].append(depth_so_far)
            all_labels["target_answer"].append(target_id)
            all_labels["current_relation"].append(curr_rel_id)
            all_labels["composed_answer"].append(curr_rel_id)  # intermediate answer
            
            S = S_next
        torch.cuda.empty_cache()
    
    S_array = np.array(all_S)
    labels = {k: np.array(v) for k, v in all_labels.items()}
    print(f"  Collected {len(S_array)} S_t samples, dim={S_array.shape[1]}", flush=True)
    
    # Print label distributions
    for k, v in labels.items():
        unique, counts = np.unique(v, return_counts=True)
        print(f"  {k}: {dict(zip(unique, counts))}", flush=True)
    
    return S_array, labels

# Split train/eval for probe training
n_train = 500
train_S, train_labels = collect_states(train_data[:n_train], "train")
eval_S, eval_labels = collect_states(eval_data, "eval")

# 3️⃣ Train linear probes
print("\n=== Training Linear Probes ===", flush=True)
probe_results = {}

class LinearProbe(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_classes)
    
    def forward(self, x):
        return self.fc(x)

def train_probe(X_train, y_train, X_eval, y_eval, n_classes, label_name):
    """Train a linear probe and return eval accuracy."""
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_eval_t = torch.FloatTensor(X_eval)
    y_eval_t = torch.LongTensor(y_eval)
    
    probe = LinearProbe(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    
    # Quick training
    best_acc = 0.0
    for epoch in range(500):
        probe.train()
        opt.zero_grad()
        logits = probe(X_train_t.to(device))
        loss = F.cross_entropy(logits, y_train_t.to(device))
        loss.backward()
        opt.step()
        
        if epoch % 100 == 0 or epoch == 499:
            probe.eval()
            with torch.no_grad():
                train_acc = (probe(X_train_t.to(device)).argmax(-1) == y_train_t.to(device)).float().mean().item()
                eval_acc = (probe(X_eval_t.to(device)).argmax(-1) == y_eval_t.to(device)).float().mean().item()
            if eval_acc > best_acc:
                best_acc = eval_acc
            # Early stopping
            if epoch > 50 and best_acc > 0.95:
                break
    
    # Majority baseline
    counts = np.bincount(y_train)
    majority_acc = counts.max() / counts.sum()
    
    print(f"  {label_name:>22s}: train_acc={train_acc:.3f} eval_acc={eval_acc:.3f} best={best_acc:.3f} "
          f"majority={majority_acc:.3f} Δ={best_acc - majority_acc:+.3f} (n_classes={n_classes})",
          flush=True)
    
    return {
        "train_acc": train_acc,
        "eval_acc": eval_acc,
        "best_eval_acc": best_acc,
        "majority_baseline": majority_acc,
        "delta_over_baseline": best_acc - majority_acc,
        "n_classes": n_classes,
    }

for name, key, n_classes in [
    ("hop_idx", "hop_idx", 3),           # 0, 1, 2
    ("composition_depth", "composition_depth", 4),  # 0, 1, 2, 3
    ("target_answer", "target_answer", N_RELATIONS),
    ("current_relation", "current_relation", N_RELATIONS),
    ("composed_answer", "composed_answer", N_RELATIONS),
]:
    key = name  # same name as key
    if name == "hop_idx":
        k = "hop_idx"
    elif name == "composition_depth":
        k = "composition_depth"
    elif name == "target_answer":
        k = "target_answer"
    else:
        k = name
    
    result = train_probe(
        train_S, train_labels[k], eval_S, eval_labels[k],
        n_classes, name
    )
    probe_results[name] = result

# 4️⃣ Summary
print("\n" + "=" * 60, flush=True)
print("PROBE ANALYSIS SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"{'Probe Target':<22s} {'Eval Acc':>9s} {'Majority':>9s} {'Δ':>9s} {'Significant?':>12s}", flush=True)
print("-" * 60, flush=True)
for name, r in probe_results.items():
    significant = "YES" if r["delta_over_baseline"] > 0.10 else ("Maybe" if r["delta_over_baseline"] > 0.05 else "NO")
    print(f"{name:<22s} {r['best_eval_acc']:>9.3f} {r['majority_baseline']:>9.3f} "
          f"{r['delta_over_baseline']:+9.3f} {significant:>12s}", flush=True)

# Save
output = {
    "checkpoint": CKPT,
    "probe_results": probe_results,
    "n_train_samples": len(train_S),
    "n_eval_samples": len(eval_S),
    "s_dim": train_S.shape[1],
}
with open(f"{WORKSPACE}/02_results/phase_5_probe_analysis.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {WORKSPACE}/02_results/phase_5_probe_analysis.json", flush=True)
print("Done.", flush=True)
