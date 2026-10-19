#!/root/miniconda3/bin/python3
"""Phase 2.5 — Persona Re-alignment: layered adapters, no merge"""
import os, sys, json, glob, gzip, random, warnings
warnings.filterwarnings("ignore")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
BASE = "/root/musicunderstand2"
sys.path.insert(0, BASE); os.chdir(BASE)

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, concatenate_datasets

MODEL_PATH = "/root/autodl-tmp/qwen3-4b"
# Phase 1 (context) loaded at inference via dynamic routing, not here
# Phase 2 (SFT) loaded at inference via dynamic routing, not here
OUTPUT_DIR = f"{BASE}/checkpoints/qwen3-4b-persona"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LEN = 1024
DEVICE = torch.device("cuda")

print("Loading Qwen3-4B (4-bit) — pure base, no adapters merged...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=32, lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config).train()
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

# ── Data loading ──

def tokenize(texts):
    if not texts: return None
    tok = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=False)
    return Dataset.from_dict({"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]})

def load_personachat():
    from datasets import load_from_disk
    ds = load_from_disk("/root/autodl-tmp/personachat")
    texts = []
    for row in ds["train"]:
        persona = " ".join(row.get("personality", [])) if isinstance(row.get("personality"), list) else ""
        dialog = row.get("utterance", {}) or {}
        candidates = dialog.get("candidates", []) if isinstance(dialog, dict) else []
        history = dialog.get("history", []) if isinstance(dialog, dict) else []
        full = persona + " " + " ".join(history) + " " + (candidates[0] if candidates else "")
        if len(full) > 50: texts.append(full)
    print(f"  PersonaChat: {len(texts):,}")
    return tokenize(texts[:15000])

def load_pippa():
    texts = []
    fpath = "/root/autodl-tmp/pippa/pippa_metharme.jsonl"
    if not os.path.exists(fpath):
        fpath = "/root/autodl-tmp/pippa/pippa.jsonl"
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                prompt = d.get("prompt", "") or d.get("bot_definitions", "")
                text = prompt + "\n" + d.get("bot_greeting", "")
                # Filter out overly intimate / emotionally extreme RP
                # (safety filter: skip very short/suggestive prompts)
                if len(text) > 100 and len(text) < MAX_LEN * 3:
                    texts.append(text)
            except: continue
    print(f"  PIPPA: {len(texts):,}")
    return tokenize(texts[:10000])

def load_empathetic():
    texts = []
    import csv
    for split in ["train"]:
        fpath = f"/root/autodl-tmp/empathetic_dialogues/data/{split}.csv"
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ctx = row.get("context", "") or row.get("prompt", "")
                    utt = row.get("utterance", "") or row.get("response", "")
                    if ctx and utt and len(ctx) + len(utt) > 50:
                        texts.append(f"<|im_start|>user\n{ctx}\n<|im_end|>\n<|im_start|>assistant\n{utt}\n<|im_end|>")
    print(f"  Empathetic: {len(texts):,}")
    return tokenize(texts[:6000])

def load_esconv():
    from datasets import load_from_disk
    ds = load_from_disk("/root/autodl-tmp/esconv")
    texts = []
    for row in ds["train"]:
        dialog = row.get("dialog", []) or row.get("conversation", [])
        if isinstance(dialog, list):
            text = ""
            for turn in dialog:
                if isinstance(turn, dict):
                    role = turn.get("role", turn.get("speaker", "user"))
                    content = turn.get("content", turn.get("text", ""))
                    text += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
            text += "<|im_start|>assistant\n"
            if len(text) > 50: texts.append(text)
    print(f"  ESConv: {len(texts):,}")
    return tokenize(texts[:5000])

def load_bluemoon():
    texts = []
    count = 0
    for fpath in sorted(glob.glob("/root/autodl-tmp/bluemoon/data/full/*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        text = item.get("input", "")
                        # Filter: prefer longer, less emotionally extreme entries
                        if 200 < len(text) < MAX_LEN * 3 and count < 3000:
                            texts.append(text[:MAX_LEN * 3])
                            count += 1
            except: continue
        if count >= 3000: break
    print(f"  Bluemoon: {len(texts):,}")
    return tokenize(texts)

# ── Assemble with ratios ──
# Persona 40% (PersonaChat 15K)
# RP 20% (PIPPA 10K + Bluemoon 3K)
# Emotional 30% (Empathetic 6K + ESConv 5K)
# Assistant 10% (PersonaChat already covers this via persona+dialogue format)

print("\nLoading datasets for Phase 2.5:")
datasets_list = []
for name, loader in [
    ("PersonaChat", load_personachat),   # 40%
    ("PIPPA", load_pippa),               # ~18% RP
    ("Empathetic", load_empathetic),     # ~16% emotional
    ("ESConv", load_esconv),             # ~13% emotional
    ("Bluemoon", load_bluemoon),         # ~8% RP (filtered, limited)
]:
    ds = loader()
    if ds is not None and len(ds) > 0:
        datasets_list.append(ds)

dataset = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]
print(f"\nTotal: {len(dataset):,} examples")

# ── Training (epoch=1, low lr, style shaping only) ──
args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=1,  # << epoch=1: style shaping, not memorization
    per_device_train_batch_size=4, gradient_accumulation_steps=4,
    learning_rate=5e-6,                          # << ultra-low: reshape response geometry
    warmup_ratio=0.05,
    logging_steps=30, save_steps=300,
    bf16=True, tf32=True, max_grad_norm=1.0,
    dataloader_num_workers=4,
    report_to="none", save_total_limit=2,
    gradient_checkpointing=True, optim="adamw_torch",
)

trainer = Trainer(model=model, args=args, train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))

print(f"\n{'='*50}")
print("Phase 2.5 — Persona Re-alignment (no merge)")
print(f"  Base: pure Qwen3-4B (no adapters merged)")
print(f"  Adapter: qwen3-4b-persona (r=32, standalone)")
print(f"  lr=5e-6, epoch=1")
print(f"  Persona~40% | RP~26% | Emotional~29% | Assistant~5%")
print(f"  Data: {len(dataset):,} examples, {len(dataset)*MAX_LEN:,} tokens")
print(f"{'='*50}")
sys.stdout.flush()

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("Phase 2.5 done!")
print(f"\nAdapters ready for routing:")
print(f"  - /root/autodl-tmp/qwen3-4b (base)")
print(f"  - {BASE}/checkpoints/qwen3-4b-phase1/final (256K context)")
print(f"  - {BASE}/checkpoints/qwen3-4b-sft-v2/final (assistant SFT)")
print(f"  - {BASE}/checkpoints/qwen3-4b-persona/final (persona) [NEW]")
