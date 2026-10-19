#!/usr/bin/env python3
"""
Phase B4 — Entity Ablation.
Shuffle entity names in CLUTRR eval instances to test whether S_t
is overfitting to training entities or learning relational structure.

Design:
- Take seed 42 checkpoint (DS=64)
- Eval on normal data (baseline)
- Eval on same prompts with entity names randomly permuted
- Compare: if acc collapses → entity memorization; if stable → generalization
"""
import os, json, random, re
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
DM = model.config.hidden_size

layers = model.model.layers
film_layers = []
for idx in FILM_LAYERS:
    layers[idx] = T.FiLMWrapper(layers[idx], DM)
    film_layers.append(layers[idx])
plaa_core = T.PlaaCore(DM).cuda()
model.eval()

ckpt_dir = f"{WORKSPACE}/checkpoints/{CKPT}"
ck = torch.load(f"{ckpt_dir}/best_model.pt", map_location=device, weights_only=False)
for idx, fl in zip(FILM_LAYERS, film_layers):
    fl.scale_proj.load_state_dict(ck["scale_proj"][idx])
plaa_core.load_state_dict(ck["plaa_core"])

cand_ids = [tok(" " + r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
pad_id = tok.pad_token_id

# 2️⃣ Generate eval data + extract entity sets
rng = random.Random(CKPT_SEED + 1000)
eval_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng)
ceiling = bayes_ceiling(eval_data, min_turn=1)

# Collect all entity names from training data (for consistent permutation)
rng_train = random.Random(CKPT_SEED)
train_data = generate_dataset(600, (1, 2, 3), (0.2, 0.5, 0.3), rng_train)

def extract_entities(trajs):
    """Extract all unique entity names from trajectory data."""
    entities = set()
    for traj in trajs:
        for turn in traj["turns"]:
            # CLUTRR prompt format: "Aunt of Bob is Alice. What is the relation between Alice and Charlie?"
            prompt = turn["prompt"]
            for atom in turn.get("atomic", "").split(", "):
                if atom:
                    parts = atom.replace(".", "").split()
                    if len(parts) >= 3:
                        # e.g., "brother(Debbie, Freddy)"
                        # Extract names inside parentheses
                        inner = atom.split("(")
                        if len(inner) > 1:
                            names = inner[1].replace(")", "").replace(".", "").split(", ")
                            entities.update(n.strip() for n in names)
    return entities

orig_entities = extract_entities(train_data)
shuffled_entities = list(orig_entities)
rng_shuffle = random.Random(999)  # fixed seed for reproducibility
rng_shuffle.shuffle(shuffled_entities)

entity_map = dict(zip(sorted(orig_entities), shuffled_entities))

print(f"Entities found: {len(orig_entities)}", flush=True)
print(f"Sample mapping: {dict(list(entity_map.items())[:5])}", flush=True)

def apply_entity_permutation(turn, e_map):
    """Replace entity names in prompt using the permutation."""
    new_prompt = turn["prompt"]
    # Sort by length descending to avoid partial replacements
    for orig, repl in sorted(e_map.items(), key=lambda x: -len(x[0])):
        new_prompt = new_prompt.replace(orig, repl)
    # Also update atomic fields
    new_atomic = []
    for atom in turn.get("atomic", "").split(", "):
        if not atom:
            continue
        new_a = atom
        for orig, repl in sorted(e_map.items(), key=lambda x: -len(x[0])):
            new_a = new_a.replace(orig, repl)
        new_atomic.append(new_a)
    return new_prompt, ", ".join(new_atomic)

# 3️⃣ eval functions
@torch.no_grad()
def evaluate(data, label=""):
    """Evaluate with full S_t chain, return t2p accuracy."""
    correct_t1 = total_t1 = 0
    correct_t2p = total_t2p = 0
    details = []
    for traj in data:
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
            
            is_correct = int(RELATIONS[pred] == turn["answer"])
            if ti == 0:
                correct_t1 += is_correct
                total_t1 += 1
            else:
                correct_t2p += is_correct
                total_t2p += 1
            details.append({"ti": ti, "correct": is_correct, "pred": RELATIONS[pred], "ans": turn["answer"]})
            S = S_next
        torch.cuda.empty_cache()
    t1 = correct_t1 / total_t1 if total_t1 > 0 else 0
    t2p = correct_t2p / total_t2p if total_t2p > 0 else 0
    across = (correct_t1 + correct_t2p) / (total_t1 + total_t2p)
    print(f"  {label}: t1={t1:.3f} ({correct_t1}/{total_t1})"
          f"  t2p={t2p:.3f} ({correct_t2p}/{total_t2p})"
          f"  all={across:.3f}", flush=True)
    return {"t1": t1, "t2p": t2p, "all": across,
            "details": [(d["ti"], d["correct"], d["pred"], d["ans"]) for d in details]}

# 4️⃣ Run both conditions
norm_data = eval_data
print("\n--- Normal ---", flush=True)
baseline = evaluate(norm_data, "normal")

# Shuffled: apply entity permutation to eval data
shuf_data = []
for traj in eval_data:
    new_turns = []
    for turn in traj["turns"]:
        new_prompt, new_atomic = apply_entity_permutation(turn, entity_map)
        new_turns.append({
            "prompt": new_prompt,
            "answer": turn["answer"],
            "atomic": new_atomic,
            "turn_idx": turn.get("turn_idx")
        })
    shuf_data.append({"turns": new_turns})

print("\n--- Entity Shuffled ---", flush=True)
shuffled = evaluate(shuf_data, "shuffled")

# 5️⃣ Report
print("\n" + "=" * 50, flush=True)
print("ENTITY ABLATION RESULTS", flush=True)
print("=" * 50, flush=True)
print(f"{'Condition':<20s} {'t1':>8s} {'t2p':>8s} {'all':>8s}", flush=True)
print("-" * 48, flush=True)
print(f"{'Normal':<20s} {baseline['t1']:>8.3f} {baseline['t2p']:>8.3f} {baseline['all']:>8.3f}", flush=True)
print(f"{'Entity Shuffled':<20s} {shuffled['t1']:>8.3f} {shuffled['t2p']:>8.3f} {shuffled['all']:>8.3f}", flush=True)
print(f"{'Δ':<20s} {shuffled['t1']-baseline['t1']:+8.3f} {shuffled['t2p']-baseline['t2p']:+8.3f} {shuffled['all']-baseline['all']:+8.3f}", flush=True)

# If t2p drops significantly (e.g. >0.15), model is memorizing entities
drop = baseline["t2p"] - shuffled["t2p"]
threshold = 0.15
print(f"\nVerdict: t2p drop = {drop:.3f}", flush=True)
if drop > threshold:
    print("→ ENTITY MEMORIZATION: Model relies on entity names. Confidence in memory mechanism: LOW.", flush=True)
elif drop > 0.05:
    print("→ PARTIAL GENERALIZATION: Some entity memorization but relational structure is learned.", flush=True)
else:
    print("→ GENERALIZED: Model learned relational structure, not entity names. Confidence in memory mechanism: HIGH.", flush=True)

# Save
output = {
    "checkpoint": CKPT,
    "baseline": baseline,
    "entity_shuffled": shuffled,
    "delta_t2p": shuffled["t2p"] - baseline["t2p"],
    "n_entities": len(orig_entities),
    "verdict": "generalized" if drop <= 0.05 else ("partial" if drop <= threshold else "memorized"),
}
with open(f"{WORKSPACE}/02_results/phase_b4_entity_ablation.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {WORKSPACE}/02_results/phase_b4_entity_ablation.json", flush=True)
print("Done.", flush=True)
