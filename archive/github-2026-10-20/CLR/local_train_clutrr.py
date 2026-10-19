#!/usr/bin/env python3
"""
CLR Phase 2.2 — CLUTRR Local Edition (v2, bug-fixed)
GTX 1650 (4GB VRAM) × Qwen3.5-0.8B (4-bit)

FiLM injected into 6 full_attention layers only (not linear_attn).
DS=64, gradient checkpointing, LoRA r=16.

Fixes applied (per Fable 5 evaluation 2026-07-08):
  [v1] generate_trajectory: random.choice(COMP_TABLE.keys()) returned tuples, not strings
       → replaced by import from clutrr_data.py (verified generator with independent solver)
  [v2] coh loss weight: 0.7 → 0.05 (was overwhelming task loss)
  [v3] eval protocol: forward with S_next instead of pre-update S
  [v4] checkpoint: now saves scale_proj params and LoRA adapters
  [v5] requires_grad: re-enabled after get_peft_model()
"""

import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, re, random

for k in ['HTTP_PROXY','HTTPS_PROXY']:
    if k not in os.environ:
        os.environ[k] = 'http://127.0.0.1:7897'
os.environ["PYTHONUNBUFFERED"] = "1"

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKSPACE)

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, 
    AutoConfig, get_scheduler
)
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType

# ── Fix v1: use verified data generator ──
from clutrr_data import generate_trajectory, generate_dataset, bayes_ceiling

# ── Config ──
MODEL_PATH = f"{WORKSPACE}/models/Qwen3.5-0.8B"
DS = 64          # Latent dimension
DM = 1024        # Qwen3.5-0.8B hidden size

# Only inject into full_attention layers (indices: 3,7,11,15,19,23)
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23]

MAX_TOKENS = 128
BATCH = 1
TOTAL_STEPS = 300
EVAL_SAMPLES = 24
EVAL_EVERY = 50
LR = 3e-4
LORA_R = 16
LORA_ALPHA = 32
# Fix v2: reduce coh weight from 0.7 to 0.05
COH_WEIGHT = 0.05
OUT_DIR = f"{WORKSPACE}/checkpoints/local_clutrr_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ── FiLM on Qwen2 DecoderLayer (full_attention) ──

class FiLMWrapper(nn.Module):
    """Wrap a Qwen3.5 full_attention decoder layer with multiplicative FiLM."""
    def __init__(self, bl):
        super().__init__()
        self.bl = bl
        self.alpha = 0.1
        self._s = None
        self.scale_proj = nn.Linear(DS, DM, dtype=torch.bfloat16, device='cuda')
        nn.init.normal_(self.scale_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.scale_proj.bias)
    
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        o = self.bl(
            hidden_states, position_embeddings,
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, **kwargs
        )
        h = o[0] if isinstance(o, (tuple, list)) else o
        if self._s is not None:
            scale = torch.tanh(self.scale_proj(self._s).unsqueeze(1))
            h = h * (1 + self.alpha * scale)
        return (h,) + o[1:] if isinstance(o, (tuple, list)) else h

class PlaaCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(DS, DS, dtype=torch.bfloat16)
        self.init_s = nn.Parameter(torch.zeros(1, DS, dtype=torch.bfloat16))
    
    def forward(self, S, h_proj):
        return self.gru(h_proj.to(torch.bfloat16), S)
    
    def init_state(self, B, device):
        base = self.init_s.expand(B, -1)
        return (base + torch.randn_like(base)*0.01).contiguous().to(device)

# ── Dataset (adapted for new generator format) ──

class CLUTRRTrajectoryDataset(Dataset):
    def __init__(self, data):
        self.trajs = data
        print(f"Dataset: {len(self.trajs)} trajectories", flush=True)
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, idx):
        # Fix v1: new generator uses "prompt" key instead of "question"
        turns = self.trajs[idx]["turns"]
        adapted = []
        for t in turns:
            adapted.append({
                "question": t["prompt"],  # map prompt → question
                "answer": t["answer"]
            })
        return adapted

def collate_fn(batch):
    max_t = max(len(t) for t in batch)
    result = []
    for ti in range(max_t):
        prompts, answers = [], []
        for t in batch:
            if ti < len(t):
                prompts.append(t[ti]["question"])
                answers.append(t[ti]["answer"])
        result.append({"prompts": prompts, "answers": answers})
    return result

# ── Evaluation ──

@torch.no_grad()
def evaluate(model, tok, dl, film_layers, plaa_core, ablate_s=False):
    correct_all = total_all = 0
    correct_last = total_last = 0
    
    for batch in dl:
        n_t = len(batch)
        B = len(batch[0]["prompts"])
        if B == 0: continue
        
        S = plaa_core.init_state(B, "cuda")
        S_next = S  # will be updated per-turn
        
        for ti in range(n_t):
            prompts, answers = batch[ti]["prompts"], batch[ti]["answers"]
            if not prompts: continue
            
            inp = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=MAX_TOKENS).to("cuda")
            
            # Fix v3: use S_next (post-update state) for generation, not pre-update S
            s_for_gen = torch.zeros_like(S) if ablate_s else S_next
            
            for l in film_layers:
                l._s = s_for_gen
            
            out = model(**inp, output_hidden_states=True)
            lens = inp["attention_mask"].sum(dim=1) - 1
            
            # Generation loss
            logits = out.logits[torch.arange(len(prompts)), lens, :]
            
            ans_ids = tok(answers, return_tensors="pt", padding=True,
                         truncation=True, max_length=16).to("cuda")
            pred = logits.argmax(dim=-1)
            target = ans_ids["input_ids"][:, 0]
            
            if ti == n_t - 1:
                correct_last += (pred == target).sum().item()
                total_last += len(prompts)
            correct_all += (pred == target).sum().item()
            total_all += len(prompts)
            
            # Update S → S_next for next turn
            h = out.hidden_states[-1]
            h_avg = h[torch.arange(len(prompts)), lens]
            S_next = plaa_core(S, h_avg[:, :DS])
            
            del out, logits, ans_ids, pred, target, inp
        
        torch.cuda.empty_cache()
    
    return (
        correct_all/total_all if total_all else 0,
        correct_last/total_last if total_last else 0
    )

# ── Main ──

def main():
    print("=" * 60, flush=True)
    print("CLR Phase 2.2 - CLUTRR Local Edition v2 (bug-fixed)", flush=True)
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory/1024**3
    print(f"Model: Qwen3.5-0.8B | GPU: {gpu} ({vram:.1f}GB)", flush=True)
    print(f"DS={DS} | FiLM layers={FULL_ATTN_LAYERS} | Steps={TOTAL_STEPS} | LR={LR}", flush=True)
    print(f"Fixes: coh_weight={COH_WEIGHT}, verified data generator", flush=True)
    print("=" * 60, flush=True)
    
    # Generate synthetic data (fix v1: use verified clutrr_data module)
    print("\nGenerating CLUTRR data (verified generator)...", flush=True)
    random.seed(42)
    train_data = generate_dataset(400)
    test_data = generate_dataset(EVAL_SAMPLES * 3)
    print(f"Bayes ceiling on train: {bayes_ceiling(train_data):.3f}", flush=True)
    print(f"Bayes ceiling on test: {bayes_ceiling(test_data):.3f}", flush=True)
    
    # Load model
    print(f"\nLoading Qwen3.5-0.8B (4-bit)...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
        quantization_config=bnb, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "<|endoftext|>"
    
    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False
    
    # Inject FiLM into full_attention layers only
    layers = model.model.layers
    film_layers = []
    for idx in FULL_ATTN_LAYERS:
        layers[idx] = FiLMWrapper(layers[idx])
        film_layers.append(layers[idx])
    print(f"FiLM injected into {len(film_layers)} layers", flush=True)
    
    # LoRA
    print("Adding LoRA adapters...", flush=True)
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    # Fix v5: re-enable FiLM scale_proj gradients after PEFT freeze
    for idx in FULL_ATTN_LAYERS:
        for p in layers[idx].scale_proj.parameters():
            p.requires_grad = True
    
    model.print_trainable_parameters()
    
    # PLAA core
    plaa_core = PlaaCore().cuda()
    
    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad] + list(plaa_core.parameters())
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M", flush=True)
    
    # Data
    train_ds = CLUTRRTrajectoryDataset(train_data)
    test_ds = CLUTRRTrajectoryDataset(test_data)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn, num_workers=0)
    random.seed(42)
    test_sub = torch.utils.data.Subset(test_ds, random.sample(range(len(test_ds)), min(EVAL_SAMPLES, len(test_ds))))
    test_dl = DataLoader(test_sub, batch_size=BATCH, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Scheduler
    scheduler = get_scheduler("cosine", optimizer=opt, num_warmup_steps=20, num_training_steps=TOTAL_STEPS)
    
    # Training loop
    log_f = open(f"{OUT_DIR}/training.log", "w")
    eval_log  = []
    step = 0
    best_delta = -float("inf")
    
    print(f"\nTraining: {TOTAL_STEPS} steps...", flush=True)
    model.train()
    
    while step < TOTAL_STEPS:
        for batch in train_dl:
            if step >= TOTAL_STEPS: break
            
            n_t = len(batch)
            B = len(batch[0]["prompts"])
            if B == 0: continue
            
            S = plaa_core.init_state(B, "cuda")
            b_loss = 0.0
            
            for ti in range(n_t):
                prompts, answers = batch[ti]["prompts"], batch[ti]["answers"]
                if not prompts: continue
                
                inp = tok(prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=MAX_TOKENS).to("cuda")
                
                # Forward: inject S_t, get hidden states (no grad through backbone)
                with torch.no_grad():
                    for l in film_layers:
                        l._s = S
                    out = model(**inp, output_hidden_states=True)
                    h = out.hidden_states[-1].detach()
                    lens = inp["attention_mask"].sum(dim=1) - 1
                    h_avg = h[torch.arange(len(prompts)), lens]
                
                # S_t update
                S_next = plaa_core(S, h_avg[:, :DS])
                coh = (S_next.float() - S.float()).norm(dim=1).mean()
                
                # Generation loss with S_next (fix v3: use S_next not S)
                for l in film_layers:
                    l._s = S_next
                gen_out = model(**inp, output_hidden_states=False)
                lens = inp["attention_mask"].sum(dim=1) - 1
                logits = gen_out.logits[torch.arange(len(prompts)), lens, :]
                
                ans_ids = tok(answers, return_tensors="pt", padding=True,
                            truncation=True, max_length=16).to("cuda")
                shift_logits = logits
                shift_labels = ans_ids["input_ids"][:, 0]
                
                gen_loss = F.cross_entropy(shift_logits, shift_labels)
                
                # Fix v2: coh weight 0.7 → 0.05
                loss = (1.0 - COH_WEIGHT) * gen_loss + COH_WEIGHT * coh
                loss = loss / n_t
                loss.backward()
                b_loss += loss.item()
                
                S = S_next.detach()
                del out, h, h_avg, gen_out, logits, ans_ids, inp
            
            # Step
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            scheduler.step()
            opt.zero_grad()
            step += 1
            
            if step % 10 == 0:
                print(f"  step {step:4d}/{TOTAL_STEPS} | L={b_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
                log_f.write(f"step {step} L={b_loss:.4f}\n")
                log_f.flush()
            
            if step % EVAL_EVERY == 0 or step >= TOTAL_STEPS:
                print(f"\n  Eval at step {step}...", flush=True)
                model.eval()
                torch.cuda.empty_cache()
                
                acc_w, acc_wl = evaluate(model, tok, test_dl, film_layers, plaa_core, ablate_s=False)
                torch.cuda.empty_cache()
                acc_wo, acc_wol = evaluate(model, tok, test_dl, film_layers, plaa_core, ablate_s=True)
                torch.cuda.empty_cache()
                
                d_all, d_last = acc_w - acc_wo, acc_wl - acc_wol
                
                eval_log.append({
                    "step": step, "acc_with_S": acc_w, "acc_without_S": acc_wo, "delta": d_all,
                    "acc_with_S_last": acc_wl, "acc_without_S_last": acc_wol, "delta_last": d_last,
                })
                
                print(f"  EVAL: with_S={acc_w:.4f} without_S={acc_wo:.4f} delta={d_all:+.4f}", flush=True)
                print(f"  LAST: with={acc_wl:.4f} without={acc_wol:.4f} delta_last={d_last:+.4f}", flush=True)
                log_f.write(f"EVAL s{step}: all={acc_w:.4f}/{acc_wo:.4f}d={d_all:+.4f} last={acc_wl:.4f}/{acc_wol:.4f}dl={d_last:+.4f}\n")
                log_f.flush()
                
                if d_all > best_delta and step > 50:
                    best_delta = d_all
                    # Fix v4: save complete checkpoint
                    ckpt = {
                        "step": step,
                        "plaa_core": plaa_core.state_dict(),
                        "scale_proj": [l.scale_proj.state_dict() for l in film_layers],
                        "lora": model.state_dict(),
                        "eval_log": eval_log,
                    }
                    torch.save(ckpt, f"{OUT_DIR}/best_model.pt")
                    print(f"  -> saved complete checkpoint (delta={d_all:+.4f})", flush=True)
                
                model.train()
    
    # Final
    print(f"\n{'='*60}", flush=True)
    print("Results", flush=True)
    print(f"  Final eval — with_S={eval_log[-1]['acc_with_S']:.4f}" if eval_log else "  No eval log", flush=True)
    print(f"  Delta: {eval_log[-1]['delta']:+.4f}" if eval_log else "", flush=True)
    print(f"  Best delta: {best_delta:+.4f}", flush=True)
    print(f"  Bayes ceiling tells accuracy achievable without cross-turn memory", flush=True)
    print(f"\nAll checkpoints in {OUT_DIR}/", flush=True)
    log_f.close()
    
    # Save final results
    results = {
        "final_delta": eval_log[-1]["delta"] if eval_log else None,
        "best_delta": best_delta,
        "test_bayes_ceiling": bayes_ceiling(test_data),
    }
    json.dump(results, open(f"{OUT_DIR}/results.json", "w"), indent=2)
    print(f"Results saved to {OUT_DIR}/results.json", flush=True)

if __name__ == "__main__":
    main()
