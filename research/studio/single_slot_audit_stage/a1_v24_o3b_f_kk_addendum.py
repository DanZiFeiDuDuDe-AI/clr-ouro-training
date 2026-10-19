# -*- coding: utf-8 -*-
"""O3b-F protocol addendum: K-K frozen-key metric diagnosis.

O3b-D/E equality is defined on two KeyEncoder outputs.  The first O3b-F
implementation also evaluated a Q-Encoder -> K cross-interface matcher for
the metric section.  This addendum keeps every representation and split
fixed, and repeats only the metric ladder with K_a as q and K_b as k.
It is offline and never reconnects to real memory writes/reads.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import a1_v24_o3b_f as base
from a1_v24_o3b import FactPool, MRCache, KeyEncoder, DIM
from a1_v24_o3b_diag import KeyBank


OUT = Path(os.environ.get("V24_O3BF_OUT", r"D:/training/checkpoints/clr_v24_o3b_f_binding_diag"))
CACHE_ROOT = Path(os.environ.get("V24_O3BF_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
O21B = Path(os.environ.get("V24_O3BF_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
SEED = int(os.environ.get("V24_O3BF_SEED", "42"))
DEVICE_NAME = os.environ.get("V24_O3BF_DEVICE", "cuda")
PAIR_TRAIN = 512 if os.environ.get("V24_O3BF_SMOKE", "0") == "1" else 6000
PAIR_EVAL = 256 if os.environ.get("V24_O3BF_SMOKE", "0") == "1" else 3000
STEPS = int(os.environ.get("V24_O3BF_MATCHER_STEPS", "800"))


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base.clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def kk_data(d):
    """Turn pair_refs' (K_a,K_b) into fit_metric's (q,k) interface."""
    out = dict(d)
    out["qa"] = d["ka"].astype(np.float32)
    out["ka"] = d["kb"].astype(np.float32)
    return out


def kk_retrieval(bank, matcher=None, max_episodes=None, mode="cosine"):
    """Nearest-key retrieval inside each len=4 episode, using K as query."""
    eps = [ep for ep in bank.pool.episodes if len(ep) == 4]
    if max_episodes:
        eps = eps[:max_episodes]
    refs = {(r["episode"], r["pos"]): r for r in bank.refs}
    total = correct = 0
    for ep in eps:
        for target0 in ep:
            target = refs[(target0["episode"], target0["pos"])]
            cand = [refs[(x["episode"], x["pos"])] for x in ep]
            q = bank.K[target["bank_idx"]]
            k = np.stack([bank.K[x["bank_idx"]] for x in cand])
            if matcher is None:
                scores = k @ q if mode == "cosine" else -np.linalg.norm(k - q[None, :], axis=1)
            else:
                with torch.no_grad():
                    qq = base.tensor(np.repeat(q[None, :], len(k), axis=0), matcher._device)
                    kk = base.tensor(k, matcher._device)
                    scores = matcher(qq, kk).cpu().numpy()
            correct += int(int(np.argmax(scores)) == target["pos"])
            total += 1
    return {"n": total, "accuracy": float(correct / max(total, 1))}


def main():
    base.seed_all(SEED + 991)
    start = time.perf_counter()
    device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    # Reuse the exact cache and frozen O2.1b KeyEncoder used by O3b-F.
    cache = MRCache(CACHE_ROOT)
    pools = {s: FactPool(cache, s) for s in ("train", "val", "testA", "testB")}
    kc = torch.load(O21B, map_location="cpu", weights_only=False)
    key = KeyEncoder(DIM).to(device)
    key.load_state_dict(kc["key"], strict=True)
    key.eval()
    for p in key.parameters():
        p.requires_grad_(False)
    banks = {s: KeyBank(pools[s], key, device) for s in pools}
    sets = {
        "train": base.pair_refs(banks["train"], PAIR_TRAIN, SEED + 1),
        "val": base.pair_refs(banks["val"], PAIR_EVAL, SEED + 2),
        "testA": base.pair_refs(banks["testA"], PAIR_EVAL, SEED + 3),
        "testB": base.pair_refs(banks["testB"], PAIR_EVAL, SEED + 4),
    }
    kksets = {s: kk_data(d) for s, d in sets.items()}
    val, ta, tb = kksets["val"], kksets["testA"], kksets["testB"]
    raw = {s: (d["qa"] * d["ka"]).sum(1) for s, d in kksets.items()}
    euc = {s: -np.linalg.norm(d["qa"] - d["ka"], axis=1) for s, d in kksets.items()}
    results = {
        "protocol": "O3b-D/E exact K-K equality; qa=K_a, ka=K_b; same splits and pair seeds as O3b-F",
        "cosine": base.pair_metric_table(val, ta, tb, "K-K cosine", raw["val"], raw["testA"], raw["testB"]),
        "euclidean": base.pair_metric_table(val, ta, tb, "K-K negative euclidean", euc["val"], euc["testA"], euc["testB"]),
        "learned": {},
    }
    metric_models = {}
    # fit_metric reads base.SMOKE/EVAL_EVERY/MATCHER_STEPS globals.
    base.SMOKE = False
    base.MATCHER_STEPS = STEPS
    base.EVAL_EVERY = int(os.environ.get("V24_O3BF_EVAL_EVERY", "200"))
    for kind in ("full_bilinear", "lowrank16", "lowrank32", "lowrank64", "symmetric_bilinear", "interaction_mlp"):
        model, rec, _ = base.fit_metric(kksets["train"], val, ta, tb, kind, device, STEPS)
        model._device = device
        metric_models[kind] = model
        results["learned"][kind] = rec
    results["retrieval"] = {"cosine": {}, "euclidean": {}}
    for s in ("testA", "testB"):
        results["retrieval"]["cosine"][s] = kk_retrieval(banks[s])
        results["retrieval"]["euclidean"][s] = kk_retrieval(banks[s], None, mode="euclidean")
    for kind, model in metric_models.items():
        results["retrieval"][kind] = {s: kk_retrieval(banks[s], model) for s in ("testA", "testB")}
    results["best_bilinear"] = max(
        (k for k in results["learned"] if k.startswith("full_bilinear") or k.startswith("lowrank") or k.startswith("symmetric")),
        key=lambda k: results["learned"][k]["testB"]["f1"],
    )
    results["best_metric"] = max(results["learned"], key=lambda k: results["learned"][k]["testB"]["f1"])
    results["wall_clock_s"] = time.perf_counter() - start
    dump(OUT / "kk_metric_addendum.json", results)
    print(json.dumps({
        "protocol": results["protocol"],
        "cosine_testA_f1": results["cosine"]["testA"]["f1"],
        "cosine_testB_f1": results["cosine"]["testB"]["f1"],
        "best_bilinear": results["best_bilinear"],
        "best_bilinear_testA_f1": results["learned"][results["best_bilinear"]]["testA"]["f1"],
        "best_bilinear_testB_f1": results["learned"][results["best_bilinear"]]["testB"]["f1"],
        "best_metric": results["best_metric"],
        "best_metric_testB_f1": results["learned"][results["best_metric"]]["testB"]["f1"],
        "wall_clock_s": results["wall_clock_s"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
