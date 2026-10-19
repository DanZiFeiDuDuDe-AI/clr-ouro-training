"""Offline supplementary Q->K error audit for completed O3b-G.

This does not train or alter any model. The preregistered O3b-G gate remains
the K-K protocol; this file only supplies non-degenerate error categories using
the established O3b-F query-to-key retrieval audit.
"""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from a1_v24_o3b import FactPool, MRCache, KeyEncoder, QueryEncoder, DIM
from a1_v24_o3b_diag import KeyBank
from a1_v24_o3b_g import BindingKey


OUT = Path(os.environ.get("V24_O3BG_OUT", r"D:/training/checkpoints/clr_v24_o3b_g_binding_key"))
CACHE_ROOT = Path(os.environ.get("V24_O3BG_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
O21B = Path(os.environ.get("V24_O3BG_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
O21A_MR = Path(os.environ.get("V24_O3BG_O21A_MR", r"D:/training/checkpoints/clr_v24_o21a_mr/best_model.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANTS = ["G1_hadamard", "G2_residual_l0p5", "G2_residual_l1", "G2_residual_l2", "G3_projected_tensor", "G4_learned_concat"]


def encode(model, x):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 2048):
            out.append(model(torch.from_numpy(np.asarray(x[i:i + 2048], dtype=np.float32)).to(DEVICE))[0].cpu().numpy())
    return np.concatenate(out, axis=0)


def qstats(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return {"n": 0, "mean": None, "std": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "negative_rate": None}
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()), "p10": float(np.quantile(x, .1)), "p25": float(np.quantile(x, .25)), "p50": float(np.quantile(x, .5)), "p75": float(np.quantile(x, .75)), "p90": float(np.quantile(x, .9)), "negative_rate": float(np.mean(x < 0))}


def audit(bank, qarr, karr):
    refs = {(r["episode"], r["pos"]): r for r in bank.refs}
    eps = [ep for ep in bank.pool.episodes if len(ep) == 4]
    errors, groups, margins, p2m, p3m = [], Counter(), [], [], []
    total = correct = 0
    for ep in eps:
        for target0 in ep:
            target = refs[(target0["episode"], target0["pos"])]
            cand = [refs[(x["episode"], x["pos"])] for x in ep]
            q = qarr[target["bank_idx"]]; ks = karr[[x["bank_idx"] for x in cand]]; scores = ks @ q
            j = int(np.argmax(scores)); ok = j == target["pos"]; total += 1; correct += int(ok)
            wrong = [i for i in range(len(cand)) if i != target["pos"]]
            margins.append(float(scores[target["pos"]] - max(scores[wrong])))
            p2 = [i for i, x in enumerate(cand) if i != target["pos"] and x["person"] == target["person"] and x["relation"] != target["relation"]]
            p3 = [i for i, x in enumerate(cand) if i != target["pos"] and x["person"] != target["person"] and x["relation"] == target["relation"]]
            if p2: p2m.append(float(scores[target["pos"]] - max(scores[p2])))
            if p3: p3m.append(float(scores[target["pos"]] - max(scores[p3])))
            if not ok:
                pred = cand[j]; same_e = pred["person"] == target["person"]; same_r = pred["relation"] == target["relation"]
                kind = "E+_R-" if same_e and not same_r else ("E-_R+" if (not same_e and same_r) else "E-_R-")
                groups[kind] += 1
                if len(errors) < 250:
                    errors.append({"kind": kind, "true_pos": int(target["pos"]), "pred_pos": int(j), "true_person": target["person"], "true_relation": int(target["relation"]), "pred_person": pred["person"], "pred_relation": int(pred["relation"]), "margin": margins[-1]})
    return {"n": total, "correct": correct, "accuracy": float(correct / max(total, 1)), "error_counts": dict(groups), "error_fractions": {k: float(v / max(sum(groups.values()), 1)) for k, v in groups.items()}, "errors": errors, "margin": {"all_negatives": qstats(margins), "same_entity_wrong_relation": qstats(p2m), "wrong_entity_same_relation": qstats(p3m)}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = MRCache(CACHE_ROOT); pools = {s: FactPool(cache, s) for s in ("testA", "testB")}
    kc = torch.load(O21B, map_location="cpu", weights_only=False); base = KeyEncoder(DIM).to(DEVICE); base.load_state_dict(kc["key"], strict=True); base.eval()
    qc = torch.load(O21A_MR, map_location="cpu", weights_only=False); query = QueryEncoder(DIM).to(DEVICE); query.load_state_dict(qc["query"], strict=True); query.eval()
    for mod in (base, query):
        for p in mod.parameters(): p.requires_grad_(False)
    banks = {s: KeyBank(pools[s], base, DEVICE) for s in pools}
    qbanks = {}
    with torch.no_grad():
        for s, bank in banks.items():
            qparts = []
            for i in range(0, len(bank.refs), 256):
                hs = np.stack([bank.pool.query_hidden(r) for r in bank.refs[i:i + 256]]).astype(np.float32)
                qparts.append(query(torch.from_numpy(hs).to(DEVICE)).cpu().numpy().astype(np.float32))
            qbanks[s] = np.concatenate(qparts, axis=0)
    out = {}
    for name in VARIANTS:
        ck = torch.load(OUT / "checkpoints" / f"{name}_best.pt", map_location=DEVICE, weights_only=False)
        nentity = int(ck["model"]["entity_classifier.weight"].shape[0])
        model = BindingKey(name, nentity).to(DEVICE); model.load_state_dict(ck["model"], strict=True); model.eval()
        out[name] = {}
        for s, bank in banks.items():
            q = encode(model, qbanks[s]); k = encode(model, bank.K); out[name][s] = audit(bank, q, k)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    (OUT / "cross_interface_retrieval.json").write_text(json.dumps({"protocol": "supplementary established O3b-F Q->K retrieval; not the O3b-G K-K gate", "variants": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({n: {s: {"accuracy": v[s]["accuracy"], "errors": v[s]["error_counts"]} for s in v} for n, v in out.items()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
