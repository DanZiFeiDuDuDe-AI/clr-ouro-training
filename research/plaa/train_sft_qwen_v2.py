#!/root/miniconda3/bin/python3
"""Phase 2 — SFT 课程学习: UltraChat→OASST1→HH-RLHF→Dolly→Empathetic"""
import os, sys, json, glob, gzip, warnings
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
DEVICE = torch.device("cuda")

print("Loading Qwen3-4B (4-bit)...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config,
    device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=32, lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config).train()
print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

# === 数据加载 ===
MAX_LEN = 2048
def tokenize(texts):
    if not texts: return None
    tok = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=False)
    return Dataset.from_dict({"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]})

def load_ultrachat():
    import pyarrow.parquet as pq
    texts = []
    for fpath in sorted(glob.glob("/root/autodl-tmp/ultrachat/data/train_sft*.parquet"))[:5]:
        table = pq.read_table(fpath)
        print(f"    {fpath.split('/')[-1]}: {len(table)} rows", flush=True)
        for i in range(len(table)):
            conv = table.column("messages")[i].as_py() if "messages" in table.column_names else None
            if not conv and "conversation" in table.column_names:
                conv = table.column("conversation")[i].as_py()
            if isinstance(conv, list):
                text = ""
                for msg in conv:
                    role = msg.get("role", msg.get("from", "user"))
                    content = msg.get("content", msg.get("value", ""))
                    text += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
                text += "<|im_start|>assistant\n"
                if len(text) > 50: texts.append(text)
        if len(texts) > 50000: break
    print(f"  UltraChat: {len(texts):,}")
    return tokenize(texts)

def load_oasst1():
    texts = []
    for fpath in sorted(glob.glob("/root/autodl-tmp/oasst1/*.jsonl.gz")):
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    text = d.get("text", str(d))
                    if len(text) > 50: texts.append(f"<|im_start|>user\n{text[:2000]}\n<|im_end|>\n<|im_start|>assistant\n")
                except: continue
    print(f"  OASST1: {len(texts):,}")
    return tokenize(texts[:50000])  # Limit to 50K

def load_hh_rlhf():
    texts = []
    for sub in ["harmless-base", "helpful-base", "helpful-online", "helpful-rejection-sampled"]:
        for fpath in sorted(glob.glob(f"/root/autodl-tmp/hh-rlhf/{sub}/*.jsonl")):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        chosen = d.get("chosen", "")
                        if len(chosen) > 50: texts.append(chosen[:MAX_LEN*4])
                    except: continue
    print(f"  HH-RLHF: {len(texts):,}")
    return tokenize(texts)

def load_dolly():
    texts = []
    p = "/root/autodl-tmp/dolly/databricks-dolly-15k.jsonl"
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    text = f"<|im_start|>user\n{d.get('instruction','')}\n{d.get('context','')}\n<|im_end|>\n<|im_start|>assistant\n{d.get('response','')}\n<|im_end|>"
                    if len(text) > 50: texts.append(text)
                except: continue
    print(f"  Dolly: {len(texts):,}")
    return tokenize(texts)

def load_empathetic():
    texts = []
    for fpath in sorted(glob.glob("/root/autodl-tmp/empathetic_dialogues/data/*.json"))[:5]:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    text = f"<|im_start|>user\n{d.get('prompt',d.get('context',''))}\n<|im_end|>\n<|im_start|>assistant\n{d.get('response',d.get('utterance',''))}\n<|im_end|>"
                    if len(text) > 50: texts.append(text)
                except: continue
    print(f"  Empathetic: {len(texts):,}")
    return tokenize(texts[:10000])

# === 课程顺序加载 ===
print("\nLoading datasets in curriculum order:")
datasets = []
for name, loader in [
    ("UltraChat", load_ultrachat),
    ("OASST1", load_oasst1),
    ("HH-RLHF", load_hh_rlhf),
    ("Dolly", load_dolly),
    ("Empathetic", load_empathetic),
]:
    ds = loader()
    if ds is not None and len(ds) > 0:
        datasets.append(ds)

if datasets:
    dataset = concatenate_datasets(datasets)
else:
    print("ERROR: No datasets loaded!")
    sys.exit(1)

print(f"\nTotal dataset: {len(dataset):,} examples")

OUTPUT_DIR = f"{BASE}/checkpoints/qwen3-4b-sft-v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
args = TrainingArguments(
    output_dir=OUTPUT_DIR, num_train_epochs=1,
    per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=1e-4, warmup_ratio=0.05,
    logging_steps=50, save_steps=2000,
    bf16=True, tf32=True, max_grad_norm=1.0,
    dataloader_num_workers=4, dataloader_pin_memory=True,
    report_to="none", save_total_limit=3,
    gradient_checkpointing=True, optim="adamw_torch",
)

trainer = Trainer(model=model, args=args, train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))

print(f"\nStarting SFT (YaRN, r=32, 5 ep)")
print(f"Curriculum: UltraChat({len(datasets[0])}) + OASST1 + HH-RLHF + Dolly + Empathetic")
sys.stdout.flush()
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("Done!")
