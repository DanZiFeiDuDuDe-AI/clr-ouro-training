"""Probe: does gamma receive gradient through 4-bit frozen backbone?"""
import os, sys
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
import torch
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention, RetentionAttention

model, tok = load_ouro()
conv = convert_ouro_to_retention(model, [3, 7, 11, 15, 19, 23])
ret0 = conv[0][0]
print("gamma requires_grad:", ret0.gamma_logit.requires_grad)
print("gamma grad before:", ret0.gamma_logit.grad)

inp = tok("A is mother of B.", return_tensors="pt").to("cuda")
base_out, _, _ = model.model(input_ids=inp["input_ids"],
                             attention_mask=inp["attention_mask"], use_cache=False)
h = base_out.last_hidden_state
pos = int(inp["attention_mask"].sum(dim=1)[0].item()) - 1
logits = model.lm_head(h[:, pos])[0]
tgt = torch.tensor([100], device="cuda")
loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), tgt)
print("loss:", loss.item())
loss.backward()
print("gamma grad after:", ret0.gamma_logit.grad)
gnz = ret0.gamma_logit.grad is not None and ret0.gamma_logit.grad.abs().sum().item() > 0
print("gamma grad nonzero:", gnz)

# Also test retention alone (no backbone) to confirm gamma gets grad
print("--- retention standalone grad test ---")
r = RetentionAttention(0, 16, 128).to("cuda").to(torch.bfloat16)
r.gamma_logit.requires_grad_(True)
q = torch.randn(1, 16, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=False) * 0.1
k = torch.randn(1, 16, 8, 128, device="cuda", dtype=torch.bfloat16) * 0.1
v = torch.randn(1, 16, 8, 128, device="cuda", dtype=torch.bfloat16) * 0.1
out, _ = r.recurrent_forward(q, k, v)
fake = torch.randn_like(out)
loss2 = ((out - fake) ** 2).mean()
loss2.backward()
print("standalone gamma grad:", r.gamma_logit.grad)
print("standalone gamma grad nonzero:", r.gamma_logit.grad.abs().sum().item() > 0)
