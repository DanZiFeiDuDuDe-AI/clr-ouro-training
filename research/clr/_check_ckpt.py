import torch
ck = torch.load("H:/CLR/checkpoints/local_clutrr_v2_nolora/best_model.pt", map_location="cpu", weights_only=False)
sp = ck["scale_proj"]
for k in sp:
    w = sp[k]["weight"]
    print(f"Layer {k}: weight shape {w.shape}")
    break
# Also check what DM is in the ckpt config
if "config" in ck:
    print(f"Config DM: {ck['config'].get('DM', 'N/A')}")
