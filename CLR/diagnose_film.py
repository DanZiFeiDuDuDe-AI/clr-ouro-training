#!/usr/bin/env python3
"""
CLR Phase 1 — 诊断 + 修复脚本
验证 FiLM scale_proj 的梯度路径并修复 Phase 1 核心问题
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, json, random

for k in ['HTTP_PROXY','HTTPS_PROXY']:
    if k not in os.environ:
        os.environ[k] = 'http://127.0.0.1:7897'

WORKSPACE = r'C:\Users\Ouyang Quan\.openclaw\workspace\local_clr'
os.chdir(WORKSPACE)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

MODEL_PATH = f"{WORKSPACE}/models/Qwen3.5-0.8B"
DS, DM = 64, 1024
FULL_ATTN = [3, 7, 11, 15, 19, 23]

print("=" * 60)
print("DIAGNOSTIC: FiLM scale_proj gradient path")
print("=" * 60)

# ── Load model ──
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
    device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
for p in model.parameters(): p.requires_grad = False

# ── FiLM (FIXED: create scale_proj eagerly, NOT inside no_grad) ──
class FiLMWrapper(nn.Module):
    def __init__(self, bl):
        super().__init__()
        self.bl = bl
        self.alpha = 0.1
        self._s = None
        # FIX 1: Eager creation on cuda, small random init to break symmetry
        self.scale_proj = nn.Linear(DS, DM, dtype=torch.bfloat16, device='cuda')
        nn.init.normal_(self.scale_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.scale_proj.bias)
    
    def forward(self, hidden_states, position_embeddings, **kw):
        o = self.bl(hidden_states, position_embeddings, **kw)
        h = o[0] if isinstance(o, (tuple,list)) else o
        if self._s is not None:
            scale = torch.tanh(self.scale_proj(self._s).unsqueeze(1))
            h = h * (1 + self.alpha * scale)
        return (h,) + o[1:] if isinstance(o, (tuple,list)) else h

# Inject FiLM
layers = model.model.layers
for idx in FULL_ATTN:
    layers[idx] = FiLMWrapper(layers[idx])

# FIX 2: Verify parameters are accessible and require grad
for idx in FULL_ATTN:
    w = layers[idx].scale_proj
    print(f"L{idx}: scale_proj.weight.requires_grad={w.weight.requires_grad}, device={w.weight.device}")
    assert w.weight.requires_grad, f"L{idx} scale_proj weight not requiring grad!"

# ── LoRA ──
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM))

# Check that scale_proj params are still in the merged model
film_params_found = []
for name, p in model.named_parameters():
    if 'scale_proj' in name:
        film_params_found.append((name, p.requires_grad, p.device))
print(f"\nFiLM params in PEFT model: {len(film_params_found)}")
if len(film_params_found) == 0:
    print("⚠️  CRITICAL: No scale_proj params found in model.parameters() after PEFT!")
else:
    for n, rg, d in film_params_found:
        print(f"  {n}: requires_grad={rg}, device={d}")

# ── Quick gradient flow test ──
print("\n--- Gradient Flow Test ---")
inp = tok("Alex is the father of Ben. What is Alex to Ben?", return_tensors="pt").to("cuda")

S = torch.randn(1, DS, dtype=torch.bfloat16, device='cuda', requires_grad=True)
for l in FULL_ATTN: layers[l]._s = S

out = model(**inp, output_hidden_states=False)
loss = out.logits[0,-1].mean()
loss.backward()

# Check scale_proj gradients
grads_ok = 0
for idx in FULL_ATTN:
    g = layers[idx].scale_proj.weight.grad
    if g is not None:
        gn = g.norm().item()
        if gn > 0:
            grads_ok += 1
        print(f"  L{idx} scale_proj.weight.grad norm: {gn:.6f}")

print(f"\nFiLM layers with non-zero gradients: {grads_ok}/{len(FULL_ATTN)}")
if grads_ok == 0:
    print("❌ CRITICAL: Scale_proj getting ZERO gradients from generation loss!")
    print("Root cause confirmed: lazy init + no_grad or gradient path broken.")
else:
    print("✅ FiLM gradient path alive!")

del model, out
torch.cuda.empty_cache()
print("\nDiagnostic complete.")
