# -*- coding: utf-8 -*-
"""Compare trained scale magnitudes: old CLR v1 (scale_proj) vs CLR-v2 (WriteAdapter)."""
import torch, os, sys
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

# old v1
ck1 = torch.load("D:/training/checkpoints/clr_ouro_v1_s42/best_model.pt",
                 map_location="cpu", weights_only=False)
sdp = ck1["plaa_core"]["scale_proj.weight"] if "plaa_core" in ck1 else \
      [k for k in ck1 if "scale_proj" in k]
print("old ckpt keys:", list(ck1.keys())[:12])
if "plaa_core" in ck1:
    w = ck1["plaa_core"]["scale_proj.weight"].float()
    b = ck1["plaa_core"].get("scale_proj.bias", torch.zeros(1)).float()
    print(f"OLD v1 scale_proj: W mean={w.mean():.6f} std={w.std():.6f} absmax={w.abs().max():.6f} "
          f"bias mean={b.mean():.6f} absmax={b.abs().max():.6f}")
    # simulate scale on random-ish S of norm ~7 (from s_dynamics)
    for sn in [1.0, 7.0]:
        S = torch.randn(1, 64) * (sn / 8.0)
        sc = 0.1 * torch.tanh(S @ w.t() + b)
        print(f"  sim S norm={sn}: scale mean={sc.mean():.5f} std={sc.std():.5f} absmax={sc.abs().max():.5f}")

# v2
ck2 = torch.load("D:/training/checkpoints/clr_v2_s42/best_model.pt",
                 map_location="cpu", weights_only=False)
w2 = ck2["clrv2"]["write.proj.weight"].float()
b2 = ck2["clrv2"]["write.proj.bias"].float()
print(f"NEW v2 WriteAdapter: W mean={w2.mean():.6f} std={w2.std():.6f} absmax={w2.abs().max():.6f} "
      f"bias mean={b2.mean():.6f} absmax={b2.abs().max():.6f}")
