import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from a1_v24_o2_common import O2Cache, rows_for, json_dump
from clr_v2 import CLRv2, LATENT_DIM, LATENT_SLOTS


LS1 = Path(r"D:\training\checkpoints\clr_v2_a1_ls1\best_model.pt")
OUT = Path(os.environ.get("V24_O2_OUT_DIR", r"D:\training\checkpoints\clr_v24_o2_retrieval"))
CACHE = os.environ.get("V22_CACHE_DIR", r"D:\training\checkpoints\clr_v2_a1_ls2\cache")


@torch.no_grad()
def run_split(model, head, cache, split, cap, device, batch_size=128):
    rows = rows_for(cache, split, cap); good = first = last = 0; total = 0
    for i in range(0, len(rows), batch_size):
        rb = rows[i:i + batch_size]
        facts, labels, _, _, target_y = cache.batch(split, rb, cap, device)
        state = model.init_state(len(rb), device)
        for t in range(cap):
            state = model.update(state, facts[:, t].to(device).to(torch.bfloat16), latent_steps=1)
        pred = head(state.flatten(1).to(torch.bfloat16)).argmax(1)
        good += int((pred == target_y).sum())
        first += int((pred == labels[:, 0]).sum()); last += int((pred == labels[:, cap - 1]).sum())
        total += len(rb)
    return {"n": total, "answer_acc": good / max(total, 1), "first_city_acc": first / max(total, 1),
            "last_city_acc": last / max(total, 1), "query_ignored": True,
            "note": "LS1 checkpoint used without O2 controller; target query is not supplied to the model."}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(LS1, map_location="cpu", weights_only=False)
    model = CLRv2().to(device=device, dtype=torch.bfloat16); model.load_state_dict(ck["clrv2"], strict=True); model.eval()
    head = nn.Linear(LATENT_SLOTS * LATENT_DIM, 64, dtype=torch.bfloat16, device=device); head.load_state_dict(ck["head"], strict=True); head.eval()
    cache = O2Cache(CACHE); out = {"protocol": "CLR-v2.4-O2 LS1 zero-shot multi-write baseline", "checkpoint": str(LS1), "chance": 1/64, "device": str(device), "curves": {}}
    for split in ("testA", "testB"):
        out["curves"][split] = {str(c): run_split(model, head, cache, split, c, device) for c in (1, 2, 4, 8, 16)}
    OUT.mkdir(parents=True, exist_ok=True); json_dump(OUT / "zero_shot_multiwrite.json", out); print(json.dumps(out, indent=2))


if __name__ == "__main__": main()
