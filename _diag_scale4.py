import torch
ck = torch.load(r"D:\training\checkpoints\clr_ouro_v1_s42\best_model.pt",
                map_location="cpu", weights_only=False)
sp = ck["scale_proj"]  # list of 6 OrderedDict
all_scales = []
for i, layer in enumerate(sp):
    w = torch.tensor(layer["weight"]).float()  # [2048, 64]
    b = torch.tensor(layer["bias"]).float() if "bias" in layer else torch.zeros(w.shape[0])
    for sn in [7.0]:
        S = torch.randn(1, 64) * (sn / 8.0)
        sc = 0.1 * torch.tanh(S @ w.t() + b)
        all_scales.append(sc)
        print(f"layer{i} sim S norm={sn}: scale mean={sc.mean():.5f} std={sc.std():.5f} absmax={sc.abs().max():.5f}")
# combined stats across layers
cat = torch.cat(all_scales, dim=1)
print(f"ALL layers combined: scale std={cat.std():.5f} absmax={cat.abs().max():.5f}")
print(f"share of |scale|>0.01: {(cat.abs()>0.01).float().mean():.4f}")
print(f"share of |scale|>0.05: {(cat.abs()>0.05).float().mean():.4f}")

# v2
ck2 = torch.load(r"D:\training\checkpoints\clr_v2_s42\best_model.pt",
                 map_location="cpu", weights_only=False)
w2 = ck2["clrv2"]["write.proj.weight"].float()
b2 = ck2["clrv2"]["write.proj.bias"].float()
print(f"NEW v2 WriteAdapter: W mean={w2.mean():.6f} std={w2.std():.6f} absmax={w2.abs().max():.6f} "
      f"bias mean={b2.mean():.6f} absmax={b2.abs().max():.6f}")
