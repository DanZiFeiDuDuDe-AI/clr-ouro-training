import torch
ck = torch.load(r"D:\training\checkpoints\clr_ouro_v1_s42\best_model.pt",
                map_location="cpu", weights_only=False)
print(type(ck))
print(list(ck.keys())[:20])
if isinstance(ck, dict):
    for k, v in ck.items():
        if isinstance(v, dict):
            print("sub", k, list(v.keys())[:15])
