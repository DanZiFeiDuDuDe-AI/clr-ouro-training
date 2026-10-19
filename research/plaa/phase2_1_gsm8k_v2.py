#!/root/miniconda3/bin/python3
"""
PLAA Phase 2.1 — GSM8K Multi-turn Reasoning with S_t (FIXED)

Core question: Can S_t carry cross-turn reasoning state when
conversation history is truncated?

Fixes applied:
1. EVAL: use attention_mask for proper position indexing (not hardcoded -1)
2. EVAL: measure full answer token accuracy + PPL, not just first-token argmax
3. TRAIN: FiLM scale_proj now in optimizer (was missing)
4. EVAL: FiLM layers._s set during eval forward (was only set in S-update pass)
"""

import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, time, re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
BASE = "/root/musicunderstand2"
sys.path.insert(0, BASE)
os.chdir(BASE)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from torch.utils.data import Dataset, DataLoader

DM, DS = 2048, 128
FILM_START, FILM_END = 27, 40
MAX_TOKENS = 128
BATCH = 1
STEPS = 500
INJECT_LAYER = 8
MODEL_PATH = "/root/autodl-tmp/models/Qwen3.6-35B-A3B"
DATA_PATH = "/root/autodl-tmp/plaa_data"
OUT_DIR = f"{BASE}/checkpoints/plaa_phase2_gsm8k_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Blackwell fix ──
_cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
_cfg._experts_implementation = "eager"
if hasattr(_cfg, 'text_config'): _cfg.text_config._experts_implementation = "eager"
import transformers.modeling_utils as mu
mu.PreTrainedModel._grouped_mm_can_dispatch = lambda self: (_ for _ in ()).throw(ValueError())

# ── Modules ──
class S_tInjectionGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(DS, DM, bias=False, dtype=torch.bfloat16).cuda()
        nn.init.zeros_(self.gate.weight)
        self._S = None
    def forward(self, S):
        self._S = S.to(torch.bfloat16)
        return self.gate(self._S)

class PlaaCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(DS, DS, dtype=torch.bfloat16).cuda()
        self.init_s = nn.Parameter(torch.zeros(1, DS, dtype=torch.bfloat16, device='cuda'))
    def forward(self, S, hp): return self.gru(hp.to(torch.bfloat16), S)
    def init_state(self, B):
        return (self.init_s.expand(B, -1) + torch.randn_like(self.init_s.expand(B, -1))*0.01).contiguous()

class FiLMLayer(nn.Module):
    def __init__(self, bl):
        super().__init__()
        self.bl = bl; self.scale_proj = nn.Linear(DS, DM, dtype=torch.bfloat16).cuda()
        nn.init.zeros_(self.scale_proj.weight); nn.init.zeros_(self.scale_proj.bias)
        self.alpha = 0.1; self._s = None
    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        o = self.bl(hidden_states, position_embeddings,
                    attention_mask=attention_mask, position_ids=position_ids,
                    past_key_values=past_key_values, **kwargs)
        h = o if isinstance(o, torch.Tensor) else o[0]
        if self._s is not None:
            s = torch.tanh(self.scale_proj(self._s).unsqueeze(1))
            h = h * (1 + self.alpha * s)
        return h

# ── GSM8K Dataset ──
class GSM8KTrajectoryDataset(Dataset):
    def __init__(self, path, tokenizer, max_tokens=MAX_TOKENS):
        self.trajs = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                turns = item["turns"]
                parsed = []
                for turn in turns:
                    q_match = re.search(r'<turn_\d+_question>(.*?)</turn_\d+_question>', turn)
                    a_match = re.search(r'<turn_\d+_answer>(.*?)</turn_\d+_answer>', turn)
                    if q_match and a_match:
                        q = q_match.group(1).strip()
                        a = a_match.group(1).strip()
                        parsed.append({"question": q, "answer": a})
                if len(parsed) >= 2:
                    self.trajs.append(parsed)
        print(f"GSM8K dataset: {len(self.trajs)} trajectories")

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

def collate_gsm8k(batch):
    """Convert batch of trajectories to per-turn format."""
    max_turns = max(len(t) for t in batch)
    result = []
    for ti in range(max_turns):
        prompts, answers = [], []
        for traj in batch:
            if ti < len(traj):
                prompts.append(traj[ti]["question"])
                answers.append(traj[ti]["answer"])
        result.append({"prompts": prompts, "answers": answers})
    return result


# ── Fixed Evaluation ──
def evaluate_ppl_and_acc(model, tok, dl, layers, plaa_core, gate, ablate_s=False):
    """
    Measure PPL and token accuracy with/without S_t.
    
    Uses attention_mask for proper position indexing.
    Measures per-token accuracy across the full answer, not just first token.
    Sets FiLM _s during eval forward so FiLM is active.
    """
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    
    for batch in dl:
        n_turns = len(batch)
        S = plaa_core.init_state(len(batch[0]["prompts"]))
        
        for ti in range(n_turns):
            prompts = batch[ti]["prompts"]
            answers = batch[ti]["answers"]
            if not prompts:
                continue
            
            # Tokenize prompt only
            inp = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_TOKENS).to('cuda')
            
            # Tokenize full prompt + answer for loss computation
            full_texts = [p + " " + a for p, a in zip(prompts, answers)]
            full = tok(full_texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=MAX_TOKENS + 64).to('cuda')
            
            # Set S_t
            if ablate_s:
                gate._S = torch.zeros_like(S)
            else:
                gate._S = S
            
            # Set FiLM _s for eval forward (FIX 4)
            for i in range(FILM_START, FILM_END): layers[i]._s = S
        
            # Forward pass on full sequence
            with torch.no_grad():
                out = model(**full, output_hidden_states=True, return_dict=True)
                logits = out.logits
                
                # --- Metric 1: PPL on answer portion ---
                # Find where answer starts (after prompt length)
                prompt_lens = inp["attention_mask"].sum(dim=1)  # [B]
                for b_idx in range(len(prompts)):
                    plen = prompt_lens[b_idx].item()
                    # The answer starts at position plen in the full sequence
                    # shift: logits[t] predicts token[t+1]
                    # So answer tokens start at logits[plen-1] predicting answer[0]
                    ans_logits = logits[b_idx, plen-1:-1, :].float()  # (ans_len, vocab)
                    ans_labels = full["input_ids"][b_idx, plen:].contiguous()
                    
                    # Remove padding tokens from loss
                    non_pad = ans_labels != 0
                    ans_logits = ans_logits[non_pad]
                    ans_labels = ans_labels[non_pad]
                    
                    if ans_labels.numel() == 0:
                        continue
                    
                    loss = F.cross_entropy(ans_logits, ans_labels)
                    total_loss += loss.item() * ans_labels.numel()
                    total_tokens += ans_labels.numel()
                    
                    # Token accuracy
                    preds = ans_logits.argmax(dim=-1)
                    correct_tokens += (preds == ans_labels).sum().item()
            
            # Update S for next turn (using prompt-only forward to get h_avg)
            with torch.no_grad():
                for i in range(FILM_START, FILM_END): layers[i]._s = S
                out_p = model(**inp, output_hidden_states=True, return_dict=True)
                hidden = out_p.hidden_states[-1]
                lens = inp["attention_mask"].sum(dim=1) - 1
                h_avg = hidden[torch.arange(len(prompts)), lens]
                S = plaa_core(S, h_avg[:, :DS].to(torch.bfloat16))
    
    avg_ppl = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
    token_acc = correct_tokens / max(total_tokens, 1)
    return avg_ppl, token_acc


# ── Build Model ──
print("\nLoading Qwen3.6-35B-A3B (4-bit)...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=_cfg, quantization_config=bnb,
    device_map={"": "cuda:0"}, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
for p in model.parameters(): p.requires_grad = False

layers = model.model.layers
for i in range(FILM_START, FILM_END): layers[i] = FiLMLayer(layers[i])
plaa_core = PlaaCore()
gate = S_tInjectionGate()

# Injection hook
_inner = model
if 'PeftModel' in type(_inner).__name__:
    _inner = _inner.base_model.model
inj_layer = _inner.model.layers[INJECT_LAYER]
def make_hook(gate_mod):
    def hook(mod, inp, out):
        if gate_mod._S is None: return out
        inj = gate_mod(gate_mod._S).unsqueeze(1)
        if isinstance(out, tuple): return (out[0] + inj,) + out[1:]
        return out + inj
    return hook
hook_handle = inj_layer.register_forward_hook(make_hook(gate))

# ── Data ──
train_ds = GSM8KTrajectoryDataset(f"{DATA_PATH}/gsm8k_train.jsonl", tok)
test_ds = GSM8KTrajectoryDataset(f"{DATA_PATH}/gsm8k_test.jsonl", tok)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_gsm8k, num_workers=0)
import random
random.seed(42)
test_subset = random.sample(range(len(test_ds)), min(50, len(test_ds)))
test_sub = torch.utils.data.Subset(test_ds, test_subset)
test_dl = DataLoader(test_sub, batch_size=BATCH, shuffle=False, collate_fn=collate_gsm8k, num_workers=0)

# ── Training ──
# FIX 3: include FiLM scale_proj params in optimizer
film_params = []
for i in range(FILM_START, FILM_END):
    film_params.extend(layers[i].scale_proj.parameters())

opt = torch.optim.AdamW(
    list(gate.parameters()) + list(plaa_core.parameters()) + film_params,
    lr=1e-4, weight_decay=0.01
)
trainable_count = sum(p.numel() for p in opt.param_groups[0]['params'])
print(f"Trainable: {trainable_count/1e6:.2f}M  (gate + plaa_core + {len(film_params)} FiLM params)")

log_f = open(f"{OUT_DIR}/training.log", "w")
step = 0
eval_log = []

while step < STEPS:
    for batch in train_dl:
        if step >= STEPS: break
        n_turns = len(batch)
        S = plaa_core.init_state(len(batch[0]["prompts"]))
        batch_loss = 0.0

        for ti in range(n_turns):
            prompts = batch[ti]["prompts"]
            if not prompts: continue
            inp = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                     max_length=MAX_TOKENS).to('cuda')

            gate(S)  # sets gate._S
            for i in range(FILM_START, FILM_END): layers[i]._s = S

            out = model(**inp, output_hidden_states=True, return_dict=True)
            logits = out.logits; hidden = out.hidden_states[-1]
            lens = inp["attention_mask"].sum(dim=1) - 1
            h_avg = hidden[torch.arange(len(prompts)), lens]
            S_next = plaa_core(S, h_avg[:, :DS].to(torch.bfloat16))

            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = inp["input_ids"][:, 1:].contiguous()
            lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=0)
            coh_loss = (S_next.float() - S.float()).norm(dim=1).mean()
            loss = 0.5 * lm_loss + 0.5 * coh_loss

            loss.backward()
            batch_loss += loss.item()
            S = S_next.detach()

        torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
        opt.step(); opt.zero_grad()
        step += 1

        if step % 100 == 0:
            gn = gate.gate.weight.norm().item()
            # Check FiLM gradient flow
            film_grad_norm = sum(p.grad.norm().item() for p in film_params if p.grad is not None)
            print(f"  step {step:4d} | L={batch_loss:.2f} | ||W_gate||={gn:.4f} | ||FiLM_grad||={film_grad_norm:.4f}")
            log_f.write(f"step {step} L={batch_loss:.2f} W_gate={gn:.4f} FiLM_grad={film_grad_norm:.4f}\n")

            # Evaluate (FIX 1 + 2: proper metric with attention_mask indexing)
            ppl_with, acc_with = evaluate_ppl_and_acc(model, tok, test_dl, layers, plaa_core, gate, ablate_s=False)
            ppl_without, acc_without = evaluate_ppl_and_acc(model, tok, test_dl, layers, plaa_core, gate, ablate_s=True)
            eval_log.append({
                "step": step,
                "ppl_with_S": ppl_with,
                "ppl_without_S": ppl_without,
                "ppl_delta": ppl_without - ppl_with,
                "acc_with_S": acc_with,
                "acc_without_S": acc_without,
                "acc_delta": acc_with - acc_without
            })
            print(f"  EVAL: PPL with_S={ppl_with:.2f} without_S={ppl_without:.2f} Δ={ppl_without-ppl_with:+.2f}")
            print(f"        ACC with_S={acc_with:.4f} without_S={acc_without:.4f} Δ={acc_with-acc_without:+.4f}")
            log_f.write(f"EVAL step {step}: ppl_with={ppl_with:.2f} ppl_without={ppl_without:.2f} delta={ppl_without-ppl_with:+.2f}\n")
            log_f.write(f"EVAL step {step}: acc_with={acc_with:.4f} acc_without={acc_without:.4f} delta={acc_with-acc_without:+.4f}\n")
            log_f.flush()

# ── Final ──
print(f"\n{'='*60}")
print("Phase 2.1 — GSM8K Results (FIXED)")
print(f"{'='*60}")
for e in eval_log:
    print(f"  step {e['step']:4d}: PPL Δ={e['ppl_delta']:+.2f}  ACC Δ={e['acc_delta']:+.4f}")

with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump({"eval_log": eval_log}, f, indent=2)

hook_handle.remove()
log_f.close()
print(f"\nDone. Results: {OUT_DIR}/")
