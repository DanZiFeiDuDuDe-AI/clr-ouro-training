---
language: en
license: mit
tags:
- qwen3-4b
- peft
- film
- latent-modulation
- plaa
base_model: Qwen/Qwen3-4B
---

# PLAA - Multiplicative FiLM

Multiplicative FiLM modulation for Qwen3-4B. Per-layer feature amplitude gating via hs x (1 + alpha*tanh(W_l * S_t)).

## Architecture

Injects modulation layers into Qwen3-4B layers 16-28. Each layer has an independent scale projection from the latent state S_t.

```
hs = hs * (1 + alpha * tanh(W_l * S_t))
```

## Contents

- `adapter_model.safetensors` — PEFT LoRA adapter (Phase 2.5 persona alignment)
- `plaa_full.pt` — PlaaCore GRU + FiLM scale_proj weights
- `modeling_plaa.py` — FiLMLayer + PlaaCore definition
- `config.json` — PEFT adapter config

## Loading

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load base model
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B",
    quantization_config=bnb, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.bfloat16)

# Inject FiLM layers
from modeling_plaa import FiLMLayer, PlaaCore
for i in range(16, 29):
    base.model.layers[i] = FiLMLayer(base.model.layers[i])

# Load PEFT adapter
peft = PeftModel.from_pretrained(base, "./", adapter_name="plaa")
peft.set_adapter("plaa")

# Load FiLM weights
import torch
ckpt = torch.load("./plaa_full.pt")
plaa_core = PlaaCore()
plaa_core.load_state_dict(ckpt["plaa_core"])
for i in range(16, 29):
    peft.base_model.model.model.layers[i].scale_proj.load_state_dict(
        ckpt["scale_proj"][i])
    peft.base_model.model.model.layers[i].cuda()

# Inference
S = plaa_core.init_state(1)
for i in range(16, 29):
    peft.base_model.model.model.layers[i]._s = S
inp = tokenizer(["Hello"], return_tensors="pt").to("cuda")
out = peft.generate(**inp, max_new_tokens=50)
```

## Results

| Condition | Pure LM Loss |
|-----------|:---:|
| Vanilla Qwen3-4B | 3.53 |
| Trained mFiLM | 2.70 |
| FiLM removed | 2.74 |
| State frozen | 2.70 |

Causal ablation Δ = 0.044. See paper for details.
