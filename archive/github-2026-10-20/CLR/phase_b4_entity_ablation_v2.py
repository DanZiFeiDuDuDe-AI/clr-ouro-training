#!/usr/bin/env python3
"""
Phase B4 v2 — Entity Ablation (correct format for CLUTRR text data).
Entity names in CLUTRR are common English names (George, Chris, Beth, etc.)
Shuffle them and check if acc drops.
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

ck = torch.load(f"{WORKSPACE}/checkpoints/{CKPT}/best_model.pt", map_location=device, weights_only=False)
for idx, fl in zip(FILM_LAYERS, film_layers):
    fl.scale_proj.load_state_dict(ck["scale_proj"][idx])
plaa_core.load_state_dict(ck["plaa_core"])
cand_ids = [tok(" " + r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
pad_id = tok.pad_token_id

# 2️⃣ Extract entities from text prompts
def extract_entity_names(trajs):
    """Extract all proper-name entities from CLUTRR text prompts."""
    names = set()
    name_pattern = re.compile(r'\b[A-Z][a-z]+\b')
    skip_words = {"Fact", "Question", "What", "Is", "The", "To", "Answer", "A", "An"}
    for traj in trajs:
        for turn in traj["turns"]:
            for m in name_pattern.finditer(turn["prompt"]):
                word = m.group()
                if word not in skip_words and len(word) > 1:
                    names.add(word)
    return sorted(names)

rng = random.Random(CKPT_SEED)
train_data = generate_dataset(600, (1, 2, 3), (0.2, 0.5, 0.3), rng)
entities = extract_entity_names(train_data)
print(f"Entities found: {len(entities)}", flush=True)
print(f"Sample: {entities[:15]}", flush=True)

# Shuffle mapping
shuffled = list(entities)
random.Random(999).shuffle(shuffled)
e_map = dict(zip(entities, shuffled))

# 3️⃣ Eval function
@torch.no_grad()
def evaluate(data, label=""):
    correct_t1 = correct_t2p = total_t1 = total_t2p = 0
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
            ok = int(RELATIONS[pred] == turn["answer"])
            if ti == 0:
                correct_t1 += ok; total_t1 += 1
            else:
                correct_t2p += ok; total_t2p += 1
            S = S_next
        torch.cuda.empty_cache()
    t1 = correct_t1 / total_t1 if total_t1 else 0
    t2p = correct_t2p / total_t2p if total_t2p else 0
    all_acc = (correct_t1 + correct_t2p) / (total_t1 + total_t2p)
    print(f"  {label}: t1={t1:.3f} t2p={t2p:.3f} all={all_acc:.3f}", flush=True)
    return {"t1": t1, "t2p": t2p, "all": all_acc}

# 4️⃣ Normal eval
rng_eval = random.Random(CKPT_SEED + 1000)
eval_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_eval)
print("\n--- Normal ---", flush=True)
baseline = evaluate(eval_data, "normal")

# 5️⃣ Entity-shuffled eval
def shuffle_entities(trajs, e_map):
    shuffled_data = []
    for traj in trajs:
        new_turns = []
        for turn in traj["turns"]:
            new_prompt = turn["prompt"]
            for old, new in sorted(e_map.items(), key=lambda x: -len(x[0])):
                new_prompt = new_prompt.replace(old, new)
            new_turns.append({"prompt": new_prompt, "answer": turn["answer"]})
        shuffled_data.append({"turns": new_turns})
    return shuffled_data

shuf_data = shuffle_entities(eval_data, e_map)
print("\n--- Entity Shuffled ---", flush=True)
shuffled = evaluate(shuf_data, "shuffled")

# 6️⃣ Report
drop = baseline["t2p"] - shuffled["t2p"]
print("\n" + "=" * 50, flush=True)
print("ENTITY ABLATION RESULTS", flush=True)
print("=" * 50, flush=True)
print(f"  Normal:     t2p={baseline['t2p']:.3f} (n_entities={len(entities)})", flush=True)
print(f"  Shuffled:   t2p={shuffled['t2p']:.3f}", flush=True)
print(f"  Δ t2p:      {drop:+.3f}", flush=True)
print()

if drop > 0.15:
    print("→ ENTITY MEMORIZATION: acc drops significantly when names change.", flush=True)
elif drop > 0.05:
    print("→ PARTIAL: some entity dependence but relational structure partially learned.", flush=True)
else:
    print("→ GENERALIZED: model learned RELATIONAL structure, not entity names. MEMORY CONFIRMED.", flush=True)

output = {
    "checkpoint": CKPT,
    "n_entities": len(entities),
    "sample_entities": entities[:10],
    "baseline": baseline,
    "entity_shuffled": shuffled,
    "delta_t2p": drop,
    "verdict": "generalized" if drop <= 0.05 else ("partial" if drop <= 0.15 else "entity_memorization"),
}
with open(f"{WORKSPACE}/02_results/phase_b4_entity_ablation_v2.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: phase_b4_entity_ablation_v2.json", flush=True)
print("Done.", flush=True)
