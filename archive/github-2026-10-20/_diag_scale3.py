import torch
ck = torch.load(r"D:\training\checkpoints\clr_ouro_v1_s42\best_model.pt",
                map_location="cpu", weights_only=False)
sp = ck["scale_proj"]
print("scale_proj type:", type(sp))
if isinstance(sp, dict):
    for k, v in sp.items():
        if hasattr(v, "shape"):
            print("  key", k, tuple(v.shape))
elif isinstance(sp, (list, tuple)):
    print("  len", len(sp))
    for i, v in enumerate(sp):
        if hasattr(v, "shape"):
            print("  [%d]" % i, tuple(v.shape))
        else:
            print("  [%d]" % i, type(v), str(v)[:80])
