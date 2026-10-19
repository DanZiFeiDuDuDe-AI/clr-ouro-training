# -*- coding: utf-8 -*-
"""Diagnose CLR-v2: does the FiLM injection actually affect logits?
Compare logits of the SAME turn-2 prompt under different state vectors."""
import torch, os, sys
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")
from clr_v2 import (CLRv2, inject_film_v2, turn_forward_v2, load_ouro,
                    LATENT_SLOTS, LATENT_DIM)
from clutrr_data import generate_dataset

ckpt = torch.load("D:/training/checkpoints/clr_v2_s42/best_model.pt",
                  map_location="cpu", weights_only=False)
print("ckpt step", ckpt["step"], "delta", ckpt["best_delta"])

model, tok = load_ouro()
film_layers = inject_film_v2(model)
clrv2 = CLRv2(latent_steps=ckpt["latent_steps"])
clrv2.load_state_dict(ckpt["clrv2"])
clrv2.eval()

traj = generate_dataset(12)[3]  # a 3-turn trajectory
print("turns:", [len(t["turns"]) for t in generate_dataset(12)])

# run turn 0 to get evolved state
S0 = clrv2.init_state(1, "cuda")
inp0 = tok(traj["turns"][0]["prompt"], return_tensors="pt",
           truncation=True, max_length=128).to("cuda")
logits0, S_evolved, norms0 = turn_forward_v2(
    model, inp0, film_layers, clrv2, S0, latent_steps=1)
print("turn0 logits argmax:", int(logits0.argmax()),
      "| evolved S norm:", float(S_evolved.float().norm()))
S_init = clrv2.init_state(1, "cuda")
S_rand = torch.randn(1, LATENT_SLOTS, LATENT_DIM, device="cuda",
                     dtype=torch.bfloat16)

inp1 = tok(traj["turns"][1]["prompt"], return_tensors="pt",
           truncation=True, max_length=128).to("cuda")

def logits_with(S, tag):
    L, _, _ = turn_forward_v2(model, inp1, film_layers, clrv2, S,
                              latent_steps=1)
    top = torch.topk(L, 5)
    print(f"[{tag}] top5 ids={top.indices.tolist()} vals={[round(float(v),3) for v in top.values]}")
    return L

Le = logits_with(S_evolved, "evolved")
Li = logits_with(S_init, "init   ")
Lr = logits_with(S_rand, "random ")
print("max|Le-Li| =", float((Le - Li).abs().max()))
print("max|Le-Lr| =", float((Le - Lr).abs().max()))
print("argmax equal evolved/init:", int(Le.argmax()) == int(Li.argmax()),
      "evolved/random:", int(Le.argmax()) == int(Lr.argmax()))

# WriteAdapter scale stats
with torch.no_grad():
    for tag, S in [("evolved", S_evolved), ("init", S_init), ("random", S_rand)]:
        sc = clrv2.write(S)
        print(f"scale[{tag}] mean={float(sc.float().mean()):.5f} "
              f"std={float(sc.float().std()):.5f} absmax={float(sc.float().abs().max()):.5f}")

# FiLM wrapper called check
calls = [0] * len(film_layers)
def make_counter(i):
    def wrap(mod, inp, *a, **kw):
        calls[i] += 1
        return mod.layer(inp, *a, **kw)
    return wrap
for i, fl in enumerate(film_layers):
    fl.forward = make_counter(fl)
_, _, _ = turn_forward_v2(model, inp1, film_layers, clrv2, S_evolved, latent_steps=1)
print("FiLM layer call counts:", calls)
