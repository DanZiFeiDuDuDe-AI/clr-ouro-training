import torch, os, sys
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")
import clr_ouro as old

ck = torch.load(r"D:\training\checkpoints\clr_ouro_v1_s42\best_model.pt",
                map_location="cpu", weights_only=False)
sc_proj = [torch.tensor(d["weight"]).float() for d in ck["scale_proj"]]
biases = [torch.tensor(d["bias"]).float() if "bias" in d else None for d in ck["scale_proj"]]
print("bias stats per layer:")
for i, (w, b) in enumerate(zip(sc_proj, biases)):
    if b is not None:
        print(f"  layer{i} bias mean={b.mean():.5f} absmax={b.abs().max():.5f}")

# simulate without_S: zeros S -> scale = 0.1*tanh(scale_proj(0)) = 0.1*tanh(bias)
print("\nwithout_S simulated scale per layer (zeros S):")
for i, (w, b) in enumerate(zip(sc_proj, biases)):
    if b is not None:
        sc = 0.1 * torch.tanh(b)
        print(f"  layer{i} scale mean={sc.mean():.5f} std={sc.std():.5f} absmax={sc.abs().max():.5f}")

# simulate with_S on real evolved states from s_dynamics norm ~7
print("\nwith_S simulated scale (S norm 7, random dir):")
for i, (w, b) in enumerate(zip(sc_proj, biases)):
    S = torch.randn(1, 64) * (7.0 / 8.0)
    sc = 0.1 * torch.tanh(S @ w.t() + (b if b is not None else 0))
    print(f"  layer{i} scale mean={sc.mean():.5f} std={sc.std():.5f} absmax={sc.abs().max():.5f}")
