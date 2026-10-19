# -*- coding: utf-8 -*-
"""CLR-v2.4 O3b-J: KeyEncoder projection bottleneck localization.

Diagnostic-only.  Reuses the O3b-H token cache and the frozen O2.1b KeyEncoder;
it never reconnects the real memory path and never changes or trains production
components.  The only trained objects are temporary linear/MLP probes and
pair judges used for measurement.
"""
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v24_o3b import MRCache, KeyEncoder, DIM


ROOT = Path(r"D:/training")
OUT = Path(os.environ.get("V24_O3BJ_OUT", r"D:/training/checkpoints/clr_v24_o3b_j_projection_localization"))
CACHE_ROOT = Path(os.environ.get("V24_O3BJ_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
HOUT = Path(os.environ.get("V24_O3BJ_HOUT", r"D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag"))
O21B = Path(os.environ.get("V24_O3BJ_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
SEED = int(os.environ.get("V24_O3BJ_SEED", "42"))
DEVICE_NAME = os.environ.get("V24_O3BJ_DEVICE", "cuda")
PROBE_STEPS = int(os.environ.get("V24_O3BJ_PROBE_STEPS", "180"))
PAIR_STEPS = int(os.environ.get("V24_O3BJ_PAIR_STEPS", "140"))
PAIR_N = int(os.environ.get("V24_O3BJ_PAIR_N", "1800"))
SPECTRUM_N = int(os.environ.get("V24_O3BJ_SPECTRUM_N", "1024"))

SPLITS = ("train", "val", "testA", "testB")
NODES = ("Z0", "Z1", "Z2", "Z3", "Z4")
NODE_OP = {
    "Z0": "KeyEncoder input carrier (2048)",
    "Z1": "Linear(2048 -> 256), pre-activation",
    "Z2": "GELU(256)",
    "Z3": "Linear(256 -> 64), pre-normalization",
    "Z4": "L2 normalization over 64 dimensions, final K",
}


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def pct(x):
    return "n/a" if x is None else f"{100.0 * float(x):.2f}%"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def build_infos(cache):
    pmap = {p: i for i, p in enumerate(cache.train_persons)}
    infos = {}
    for split in SPLITS:
        data = cache.data[split]
        meta = cache.meta[split]
        offsets = data["fact_offsets"]
        refs = []
        for ep, m in enumerate(meta):
            cap = int(m["capacity"])
            for pos in range(cap):
                refs.append({
                    "split": split, "episode": ep, "pos": pos,
                    "flat": int(offsets[ep]) + pos,
                    "person": str(m["persons"][pos]),
                    "relation": int(m["relation_ids"][pos]),
                })
        H = data["fact_H"].astype(np.float32)
        QH = data["fact_query_H"].astype(np.float32)
        ent = np.asarray([pmap.get(r["person"], -1) for r in refs], dtype=np.int64)
        rel = np.asarray([r["relation"] for r in refs], dtype=np.int64)
        persons = np.asarray([r["person"] for r in refs], dtype=object)
        episode = np.asarray([r["episode"] for r in refs], dtype=np.int64)
        pos = np.asarray([r["pos"] for r in refs], dtype=np.int64)
        infos[split] = {"refs": refs, "H": H, "QH": QH, "entity": ent,
                        "relation": rel, "persons": persons,
                        "episode": episode, "pos": pos}
    return infos, pmap


def load_token_arrays():
    p = HOUT / "token_layer_features.npz"
    if not p.exists():
        raise FileNotFoundError(f"required reused O3b-H cache missing: {p}")
    out = {}
    with np.load(p, allow_pickle=False) as z:
        for split in SPLITS:
            out[split] = {
                "entity_mention_mean": z[f"fact_{split}_entity_mention_mean"].astype(np.float32),
                "layer_23_mention_mean": z[f"fact_{split}_layer_23_mention_mean"].astype(np.float32),
            }
    return out, p


def aligned_carriers(infos, token):
    out = {}
    for name, alpha in (("J-I0", None), ("J-I2a", 1.0), ("J-I2b", 2.0)):
        out[name] = {}
        for split in SPLITS:
            raw = infos[split]["H"]
            # Keep the three carriers row-aligned.  O3b-H only extracted the
            # bounded mention cache for the first 4096 train facts; all
            # validation/test facts are covered.  Restricting I0 to the same
            # rows makes every later pair set and control exactly comparable.
            mention = token[split]["entity_mention_mean"]
            n = min(len(raw), len(mention))
            base = raw[:n]
            arr = base if alpha is None else base + np.float32(alpha) * mention[:n]
            out[name][split] = arr.astype(np.float32, copy=False)
    return out


@torch.no_grad()
def key_nodes(key, x, device, batch=512):
    out = {n: [] for n in NODES}
    key.eval()
    for st in range(0, len(x), batch):
        z0 = torch.from_numpy(np.asarray(x[st:st + batch], dtype=np.float32)).to(device)
        z1 = key.net[0](z0)
        z2 = key.net[1](z1)
        z3 = key.net[2](z2)
        z4 = F.normalize(z3, dim=-1)
        for name, value in (("Z0", z0), ("Z1", z1), ("Z2", z2), ("Z3", z3), ("Z4", z4)):
            out[name].append(value.float().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def known_labels(y):
    return np.asarray(y, dtype=np.int64) >= 0


def auc_score(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Mann-Whitney formulation with average ranks for ties.
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    rp = ranks[y == 1].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_threshold(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    if len(score) == 0:
        return 0.0
    vals = np.unique(score)
    if len(vals) > 300:
        vals = np.quantile(vals, np.linspace(0, 1, 301))
    candidates = np.r_[vals, vals.min() - 1e-8, vals.max() + 1e-8]
    best = (0.0, float(candidates[0]))
    for th in candidates:
        p = score >= th
        tp = float(np.sum(p & (y == 1))); fp = float(np.sum(p & (y == 0)))
        fn = float(np.sum((~p) & (y == 1)))
        f1 = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, float(th))
    return best[1]


def binary_metrics(y, score, threshold=None):
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    if threshold is None:
        threshold = best_threshold(y, score)
    pred = score >= threshold
    tp = float(np.sum(pred & (y == 1))); tn = float(np.sum((~pred) & (y == 0)))
    fp = float(np.sum(pred & (y == 0))); fn = float(np.sum((~pred) & (y == 1)))
    precision = tp / max(tp + fp, 1.0); recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    tpr = tp / max(tp + fn, 1.0); tnr = tn / max(tn + fp, 1.0)
    return {"threshold": float(threshold), "accuracy": float((tp + tn) / max(len(y), 1)),
            "balanced_accuracy": float(0.5 * (tpr + tnr)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1), "auroc": auc_score(y, score),
            "n": int(len(y)), "positive": int(np.sum(y == 1)), "negative": int(np.sum(y == 0))}


def multiclass_fit(train_x, train_y, evals, nclass, device, steps=PROBE_STEPS, seed=SEED, mlp=False):
    x = np.asarray(train_x, dtype=np.float32); y = np.asarray(train_y, dtype=np.int64)
    mask = y >= 0; x = x[mask]; y = y[mask]
    torch.manual_seed(seed)
    if mlp:
        model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(), nn.Linear(64, nclass)).to(device)
    else:
        model = nn.Linear(x.shape[1], nclass).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    X = torch.from_numpy(x).to(device); Y = torch.from_numpy(y).long().to(device)
    rng = np.random.default_rng(seed + 913); bs = min(1024, len(Y))
    model.train()
    for _ in range(max(1, steps)):
        ix = rng.integers(0, len(Y), size=bs)
        loss = F.cross_entropy(model(X[ix]), Y[ix])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); result = {"train_n": int(len(Y)), "nclass": int(nclass), "mlp": bool(mlp)}
    with torch.no_grad():
        for split, (xx, yy) in evals.items():
            yy = np.asarray(yy, dtype=np.int64); m = yy >= 0
            rec = {"known_fraction": float(m.mean()) if len(m) else 0.0, "n_known": int(m.sum())}
            if m.any():
                pred = model(torch.from_numpy(np.asarray(xx, dtype=np.float32)[m]).to(device)).argmax(1).cpu().numpy()
                rec["accuracy"] = float(np.mean(pred == yy[m]))
            else:
                rec["accuracy"] = None
            result[split] = rec
    del model, X, Y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def group_masks(info):
    by_entity = defaultdict(set); by_relation = defaultdict(set)
    for p, r in zip(info["persons"], info["relation"]):
        by_entity[str(p)].add(int(r)); by_relation[int(r)].add(str(p))
    same_entity = np.asarray([len(by_entity[str(p)]) >= 2 for p in info["persons"]], dtype=bool)
    same_relation = np.asarray([len(by_relation[int(r)]) >= 2 for r in info["relation"]], dtype=bool)
    return same_entity, same_relation


def joint_labels(infos, pmap):
    train_keys = sorted({f"{p}\t{int(r)}" for p, r in zip(infos["train"]["persons"], infos["train"]["relation"])})
    jmap = {k: i for i, k in enumerate(train_keys)}
    out = {}
    for split in SPLITS:
        out[split] = np.asarray([jmap.get(f"{p}\t{int(r)}", -1) for p, r in zip(infos[split]["persons"], infos[split]["relation"])], dtype=np.int64)
    return out, jmap


def probe_node(carrier, node, reps, infos, pmap, jlabels, device):
    xtr = reps["train"][node]
    ent_evals = {s: (reps[s][node], infos[s]["entity"][:len(reps[s][node])]) for s in SPLITS}
    rel_evals = {s: (reps[s][node], infos[s]["relation"][:len(reps[s][node])]) for s in SPLITS}
    joint_evals = {s: (reps[s][node], jlabels[s][:len(reps[s][node])]) for s in SPLITS}
    se_train, sr_train = group_masks(infos["train"])
    ent = multiclass_fit(xtr, infos["train"]["entity"][:len(xtr)], ent_evals, len(pmap), device, seed=SEED)
    rel = multiclass_fit(xtr, infos["train"]["relation"][:len(xtr)], rel_evals, 4, device, seed=SEED + 1)
    joint = multiclass_fit(xtr, jlabels["train"][:len(xtr)], joint_evals, len(set(jlabels["train"][jlabels["train"] >= 0])), device, seed=SEED + 2)
    cond_rel_evals = {}; cond_ent_evals = {}
    for s in SPLITS:
        n = len(reps[s][node]); se, sr = group_masks(infos[s])
        cond_rel_evals[s] = (reps[s][node][se[:n]], infos[s]["relation"][:n][se[:n]])
        cond_ent_evals[s] = (reps[s][node][sr[:n]], infos[s]["entity"][:n][sr[:n]])
    cond_rel = multiclass_fit(xtr[se_train[:len(xtr)]], infos["train"]["relation"][:len(xtr)][se_train[:len(xtr)]], cond_rel_evals, 4, device, seed=SEED + 3)
    cond_ent = multiclass_fit(xtr[sr_train[:len(xtr)]], infos["train"]["entity"][:len(xtr)][sr_train[:len(xtr)]], cond_ent_evals, len(pmap), device, seed=SEED + 4)
    return {"carrier": carrier, "node": node, "entity": ent, "relation": rel, "joint": joint,
            "entity_conditioned_relation": cond_rel, "relation_conditioned_entity": cond_ent}


def pair_groups(info):
    by_key = defaultdict(list); by_entity = defaultdict(list); by_rel = defaultdict(list)
    for i, (p, r) in enumerate(zip(info["persons"], info["relation"])):
        by_key[(str(p), int(r))].append(i); by_entity[str(p)].append(i); by_rel[int(r)].append(i)
    return by_key, by_entity, by_rel


def make_address_pairs(info, per_type, seed):
    rng = np.random.default_rng(seed); by_key, by_ent, by_rel = pair_groups(info); n = len(info["persons"])
    types = {"P0_same_entity_same_relation": [], "P1_diff_entity_same_relation": [],
             "P2_same_entity_diff_relation": [], "P3_diff_entity_diff_relation": []}
    def add_from(candidates, name):
        candidates = [list(x) for x in candidates if len(x) >= 2]
        if not candidates: return
        for _ in range(per_type):
            g = candidates[int(rng.integers(0, len(candidates)))]
            a, b = rng.choice(g, 2, replace=False); types[name].append((int(a), int(b)))
    add_from(by_key.values(), "P0_same_entity_same_relation")
    for _ in range(per_type):
        for _try in range(100):
            a = int(rng.integers(0, n)); r = int(info["relation"][a]); cand = [i for i in by_rel[r] if str(info["persons"][i]) != str(info["persons"][a])]
            if cand:
                types["P1_diff_entity_same_relation"].append((a, int(cand[int(rng.integers(0, len(cand)))]))); break
    for _ in range(per_type):
        for _try in range(100):
            a = int(rng.integers(0, n)); p = str(info["persons"][a]); cand = [i for i in by_ent[p] if int(info["relation"][i]) != int(info["relation"][a])]
            if cand:
                types["P2_same_entity_diff_relation"].append((a, int(cand[int(rng.integers(0, len(cand)))]))); break
    for _ in range(per_type):
        for _try in range(100):
            a = int(rng.integers(0, n)); b = int(rng.integers(0, n))
            if str(info["persons"][a]) != str(info["persons"][b]) and int(info["relation"][a]) != int(info["relation"][b]):
                types["P3_diff_entity_diff_relation"].append((a, b)); break
    return {k: np.asarray(v, dtype=np.int64).reshape(-1, 2) for k, v in types.items()}


def cosine(a, b):
    aa = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    bb = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
    return np.sum(aa * bb, axis=1)


def euclid(a, b):
    return np.linalg.norm(a - b, axis=1)


def overlap(a, b, bins=30):
    if len(a) == 0 or len(b) == 0: return None
    lo = float(min(np.min(a), np.min(b))); hi = float(max(np.max(a), np.max(b)))
    if hi <= lo: return 1.0
    ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    return float(np.minimum(ha, hb).sum() / max(np.maximum(ha, hb).sum(), 1e-12))


def pair_geometry(X_by_split, pairsets, node):
    """Raw cosine/Euclidean geometry for one node across the fixed splits."""
    rows = []; compact = {split: {} for split in ("val", "testA", "testB")}
    for split in ("val", "testA", "testB"):
        X = X_by_split[split]
        for typ, ab in pairsets[split].items():
            a, b = X[ab[:, 0]], X[ab[:, 1]]
            cs = cosine(a, b); ds = euclid(a, b)
            rows.append({"node": node, "split": split, "pair_type": typ, "n": len(ab),
                         "cos_mean": float(np.mean(cs)) if len(cs) else None, "cos_std": float(np.std(cs)) if len(cs) else None,
                         "cos_median": float(np.median(cs)) if len(cs) else None,
                         "cos_p05": float(np.quantile(cs, .05)) if len(cs) else None, "cos_p95": float(np.quantile(cs, .95)) if len(cs) else None,
                         "euclid_mean": float(np.mean(ds)) if len(ds) else None, "euclid_std": float(np.std(ds)) if len(ds) else None,
                         "euclid_median": float(np.median(ds)) if len(ds) else None,
                         "euclid_p05": float(np.quantile(ds, .05)) if len(ds) else None, "euclid_p95": float(np.quantile(ds, .95)) if len(ds) else None})
            compact[split][typ] = (cs, ds)
    for split in ("testA", "testB"):
        for neg in ("P1_diff_entity_same_relation", "P2_same_entity_diff_relation"):
            p0 = compact[split].get("P0_same_entity_same_relation", (np.array([]), np.array([])))[0]
            pn = compact[split].get(neg, (np.array([]), np.array([])))[0]
            if len(p0) and len(pn):
                y = np.r_[np.ones(len(p0), dtype=np.int64), np.zeros(len(pn), dtype=np.int64)]
                sc = np.r_[p0, pn]
                valp0 = compact["val"].get("P0_same_entity_same_relation", (np.array([]),))[0]
                valpn = compact["val"].get(neg, (np.array([]),))[0]
                th = best_threshold(np.r_[np.ones(len(valp0), dtype=np.int64), np.zeros(len(valpn), dtype=np.int64)], np.r_[valp0, valpn]) if len(valp0) and len(valpn) else best_threshold(y, sc)
                m = binary_metrics(y, sc, th)
                rows.append({"node": node, "split": split, "pair_type": f"P0_vs_{neg}", **m,
                             "histogram_overlap": overlap(p0, pn)})
    return rows


def pair_features(a, b):
    return np.concatenate([a * b, np.abs(a - b)], axis=1).astype(np.float32)


def pair_judge(train_reps, val_reps, test_reps, pairsets, device, mlp=False):
    results = {}
    for neg in ("P1_diff_entity_same_relation", "P2_same_entity_diff_relation"):
        def make(rep, split):
            p0 = pairsets[split]["P0_same_entity_same_relation"]; pn = pairsets[split][neg]
            a0, b0 = rep[p0[:, 0]], rep[p0[:, 1]]; an, bn = rep[pn[:, 0]], rep[pn[:, 1]]
            return np.r_[pair_features(a0, b0), pair_features(an, bn)], np.r_[np.ones(len(p0), dtype=np.int64), np.zeros(len(pn), dtype=np.int64)]
        Xt, yt = make(train_reps, "train"); Xv, yv = make(val_reps, "val")
        torch.manual_seed(SEED + (23 if mlp else 17)); model = nn.Sequential(nn.Linear(Xt.shape[1], 64), nn.GELU(), nn.Linear(64, 1)).to(device) if mlp else nn.Linear(Xt.shape[1], 1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01); Xtt=torch.from_numpy(Xt).to(device); ytt=torch.from_numpy(yt.astype(np.float32)).to(device); rng=np.random.default_rng(SEED+len(neg)+(9 if mlp else 0)); bs=min(1024,len(yt))
        model.train()
        for _ in range(PAIR_STEPS):
            ix=rng.integers(0,len(yt),size=bs); loss=F.binary_cross_entropy_with_logits(model(Xtt[ix]).squeeze(1),ytt[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        def score(X):
            with torch.no_grad(): return model(torch.from_numpy(X).to(device)).squeeze(1).cpu().numpy()
        th=best_threshold(yv,score(Xv)); out={}
        for split,rep in (("train",train_reps),("val",val_reps),("testA",test_reps["testA"]),("testB",test_reps["testB"])):
            X,y=make(rep,split); out[split]=binary_metrics(y,score(X),th)
        results[neg]=out
        del model, Xtt, ytt
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results


def rankdata(x):
    order=np.argsort(x,kind="mergesort"); r=np.empty(len(x),dtype=np.float64); i=0
    while i<len(order):
        j=i+1
        while j<len(order) and x[order[j]]==x[order[i]]: j+=1
        r[order[i:j]]=0.5*(i+1+j); i=j
    return r


def rank_corr(a,b):
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12: return None
    return float(np.corrcoef(rankdata(np.asarray(a)),rankdata(np.asarray(b)))[0,1])


def spectrum_metrics(X, device, sample=SPECTRUM_N):
    if len(X)>sample:
        idx=np.linspace(0,len(X)-1,sample,dtype=np.int64); X=X[idx]
    X=np.asarray(X,dtype=np.float32); Xc=X-X.mean(0,keepdims=True)
    with torch.no_grad():
        s=torch.linalg.svdvals(torch.from_numpy(Xc).to(device)).float().cpu().numpy()
    eig=(s*s)/max(len(X)-1,1); eig=eig[eig>max(float(eig.max())*1e-10 if len(eig) else 0.0,1e-12)]
    p=eig/max(float(eig.sum()),1e-12); eff=float(np.exp(-np.sum(p*np.log(np.maximum(p,1e-30))))) if len(p) else 0.0; pr=float((eig.sum()**2)/max(np.sum(eig**2),1e-12)) if len(eig) else 0.0
    return {"n":int(len(X)),"dim":int(X.shape[1]),"effective_rank":eff,"participation_ratio":pr,
            "top1_explained":float(eig[0]/max(eig.sum(),1e-12)) if len(eig) else 0.0,
            "top10_explained":float(eig[:10].sum()/max(eig.sum(),1e-12)) if len(eig) else 0.0,
            "top50_explained":float(eig[:50].sum()/max(eig.sum(),1e-12)) if len(eig) else 0.0,
            "singular_values":s[:128].tolist(),"eigenvalues":eig[:128].tolist()}


def random_basis(indim, outdim, seed):
    rng=np.random.default_rng(seed); A=rng.normal(size=(indim,outdim)).astype(np.float64); Q,_=np.linalg.qr(A,mode="reduced"); return Q.astype(np.float32)


def pca_basis(X, outdims):
    n=min(len(X),SPECTRUM_N); idx=np.linspace(0,len(X)-1,n,dtype=np.int64); Xc=X[idx].astype(np.float32)-X[idx].astype(np.float32).mean(0,keepdims=True); _,_,vt=np.linalg.svd(Xc,full_matrices=False); return X[idx].mean(0).astype(np.float32),{d:vt[:d].astype(np.float32) for d in outdims}


def control_metrics(name, reps, infos, pmap, jlabels, pairsets, device):
    node="control"; xtr=reps["train"]; ent_e={s:(reps[s],infos[s]["entity"][:len(reps[s])]) for s in SPLITS}; rel_e={s:(reps[s],infos[s]["relation"][:len(reps[s])]) for s in SPLITS}; joint_e={s:(reps[s],jlabels[s][:len(reps[s])]) for s in SPLITS}
    ent=multiclass_fit(xtr,infos["train"]["entity"][:len(xtr)],ent_e,len(pmap),device,seed=SEED+11); rel=multiclass_fit(xtr,infos["train"]["relation"][:len(xtr)],rel_e,4,device,seed=SEED+12); joint=multiclass_fit(xtr,jlabels["train"][:len(xtr)],joint_e,len(set(jlabels["train"][jlabels["train"]>=0])),device,seed=SEED+13)
    gr,_=pair_geometry({"control":reps["testA"]},{"testA":pairsets["testA"],"testB":pairsets["testB"],"val":pairsets["val"]}) if False else (None,None)
    def raw_f1(split, neg):
        p0=pairsets[split]["P0_same_entity_same_relation"];pn=pairsets[split][neg]; sc=np.r_[cosine(reps[split][p0[:,0]],reps[split][p0[:,1]]),cosine(reps[split][pn[:,0]],reps[split][pn[:,1]])]; y=np.r_[np.ones(len(p0),dtype=np.int64),np.zeros(len(pn),dtype=np.int64)]
        vp0=pairsets["val"]["P0_same_entity_same_relation"];vpn=pairsets["val"][neg]; vs=np.r_[cosine(reps["val"][vp0[:,0]],reps["val"][vp0[:,1]]),cosine(reps["val"][vpn[:,0]],reps["val"][vpn[:,1]])];vy=np.r_[np.ones(len(vp0),dtype=np.int64),np.zeros(len(vpn),dtype=np.int64)]
        return binary_metrics(y,sc,best_threshold(vy,vs))
    return {"control":name,"entity_testA":ent["testA"]["accuracy"],"relation_testA":rel["testA"]["accuracy"],"joint_testA":joint["testA"]["accuracy"],"entity_testB_known_fraction":ent["testB"]["known_fraction"],"p0_p1_f1_testA":raw_f1("testA","P1_diff_entity_same_relation")["f1"],"p0_p2_f1_testA":raw_f1("testA","P2_same_entity_diff_relation")["f1"],"effective_rank":spectrum_metrics(reps["train"],device)["effective_rank"]}


def operation_rows(probe_rows):
    out=[]
    by=defaultdict(dict)
    for r in probe_rows: by[r["carrier"]][r["node"]]=r
    for carrier, rows in by.items():
        for a,b in zip(NODES[:-1],NODES[1:]):
            for metric in ("entity_testA","relation_testA","joint_testA"):
                drop=None if rows[a].get(metric) is None or rows[b].get(metric) is None else float(rows[b][metric]-rows[a][metric])
                out.append({"carrier":carrier,"node_before":a,"node_after":b,"operation":NODE_OP[b],"metric":metric,"drop":drop,"major_information_loss":bool(drop is not None and drop<=-0.10),"critical_information_loss":bool(drop is not None and drop<=-0.25)})
    return out


def make_plots(probe_rows, geo_rows, spec_rows, control_rows):
    figdir=OUT/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as ex:
        # The remote training environment intentionally has no plotting
        # dependency.  Emit real, data-backed SVGs with the standard library.
        def svg_line(path,title,series,ymin=0.0,ymax=1.0,log=False):
            W,H,L,R,T,B=900,540,80,30,55,70; pw,ph=W-L-R,H-T-B
            def yy(v):
                if v is None or not np.isfinite(v): return None
                q=(math.log(max(v,1e-12))-math.log(max(ymin,1e-12)))/(math.log(max(ymax,1e-12))-math.log(max(ymin,1e-12))) if log else (v-ymin)/max(ymax-ymin,1e-12)
                return T+ph*(1-q)
            colors=["#2563eb","#dc2626","#059669","#9333ea","#ea580c","#0891b2"]
            parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="{W/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="#333"/><line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="#333"/>']
            for si,(label,vals) in enumerate(series):
                pts=[]
                for i,v in enumerate(vals):
                    y=yy(v)
                    if y is not None: pts.append(f"{L+(pw*i/max(len(vals)-1,1)):.1f},{y:.1f}")
                if pts: parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[si%len(colors)]}" stroke-width="2"/>')
                for i,v in enumerate(vals):
                    y=yy(v)
                    if y is not None: parts.append(f'<circle cx="{L+(pw*i/max(len(vals)-1,1)):.1f}" cy="{y:.1f}" r="3" fill="{colors[si%len(colors)]}"/>')
                parts.append(f'<text x="{L+pw-5}" y="{T+15+18*si}" text-anchor="end" font-family="Arial" font-size="12" fill="{colors[si%len(colors)]}">{label}</text>')
            for i,n in enumerate(NODES): parts.append(f'<text x="{L+(pw*i/4):.1f}" y="{T+ph+22}" text-anchor="middle" font-family="Arial" font-size="12">{n}</text>')
            parts.append(f'<text x="15" y="{T+ph/2}" transform="rotate(-90 15 {T+ph/2})" text-anchor="middle" font-family="Arial" font-size="12">value</text></svg>')
            path.write_text("".join(parts),encoding="utf-8")
        def svg_bar(path,title,labels,vals):
            W,H,L,R,T,B=1000,560,70,25,55,100; pw,ph=W-L-R,H-T-B; vmax=max(max(vals or [1]),1e-6); parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="{W/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text><line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="#333"/><line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="#333"/>']
            bw=pw/max(len(vals),1)*.72
            for i,(lab,v) in enumerate(zip(labels,vals)):
                h=ph*max(float(v),0)/vmax; x=L+pw*(i+.14)/max(len(vals),1); y=T+ph-h; parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="#2563eb"/><text x="{x+bw/2:.1f}" y="{T+ph+18}" transform="rotate(35 {x+bw/2:.1f} {T+ph+18})" text-anchor="start" font-family="Arial" font-size="11">{lab}</text>')
            parts.append('</svg>'); path.write_text("".join(parts),encoding="utf-8")
        def probe_series(metric):
            return [(c,[next((r.get(metric) for r in probe_rows if r["carrier"]==c and r["node"]==n),None) for n in NODES]) for c in sorted(set(r["carrier"] for r in probe_rows))]
        svg_line(figdir/"01_entity_probe_vs_depth.svg","Entity probe vs KeyEncoder depth",probe_series("entity_testA"))
        svg_line(figdir/"02_relation_probe_vs_depth.svg","Relation probe vs KeyEncoder depth",probe_series("relation_testA"))
        svg_line(figdir/"03_joint_address_probe_vs_depth.svg","Joint address probe vs KeyEncoder depth",probe_series("joint_testA"))
        def geo_series(contrast):
            return [(c,[next((r.get("f1") for r in geo_rows if r.get("carrier")==c and r.get("node")==n and r.get("split")=="testA" and r.get("pair_type")==contrast),None) for n in NODES]) for c in sorted(set(r.get("carrier") for r in geo_rows))]
        svg_line(figdir/"04_P0_P1_f1.svg","P0/P1 discrimination vs depth",geo_series("P0_vs_P1_diff_entity_same_relation"))
        svg_line(figdir/"05_P0_P2_f1.svg","P0/P2 discrimination vs depth",geo_series("P0_vs_P2_same_entity_diff_relation"))
        rankseries=[(c,[next((r["effective_rank"] for r in spec_rows if r["carrier"]==c and r["node"]==n),None) for n in NODES]) for c in sorted(set(r["carrier"] for r in spec_rows))]
        svg_line(figdir/"06_effective_rank.svg","Effective rank vs depth",rankseries,ymin=0,ymax=max([v for _,vs in rankseries for v in vs if v is not None]+[1])*1.05)
        cr=[r for r in control_rows if r["carrier"]=="J-I2b"]; svg_bar(figdir/"07_learned_vs_random_projection.svg","Learned vs random/PCA projection: I2b entity probe",[r["control"] for r in cr],[r["entity_testA"] or 0 for r in cr])
        ss=[]
        for n in ("Z0","Z1","Z3","Z4"):
            r=next((r for r in spec_rows if r["carrier"]=="J-I2b" and r["node"]==n),None)
            if r: ss.append((n,json.loads(r["singular_values_json"])))
        svg_line(figdir/"08_singular_eigen_spectrum.svg","Singular spectrum at key I2b nodes",[(n,v[:5]+[None]*(5-len(v[:5]))) for n,v in ss],ymin=1e-4,ymax=max([max(v[:5]) for _,v in ss]+[1]),log=True)
        (figdir/"PLOTS_UNAVAILABLE.txt").write_text(f"matplotlib unavailable ({ex}); standard-library SVG fallback generated.",encoding="utf-8"); return
    def line(metric,filename,title):
        plt.figure(figsize=(8,5))
        for c in sorted(set(r["carrier"] for r in probe_rows)):
            rr=[r for r in probe_rows if r["carrier"]==c]; plt.plot(NODES,[r.get(metric) or np.nan for r in rr],marker="o",label=c)
        plt.xticks(range(5),NODES); plt.ylim(0,1.02); plt.grid(alpha=.25); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(figdir/filename,dpi=150); plt.close()
    line("entity_testA","01_entity_probe_vs_depth.png","Entity probe vs KeyEncoder depth")
    line("relation_testA","02_relation_probe_vs_depth.png","Relation probe vs KeyEncoder depth")
    line("joint_testA","03_joint_address_probe_vs_depth.png","Joint address probe vs KeyEncoder depth")
    def geo_line(contrast,filename,title):
        plt.figure(figsize=(8,5))
        for c in sorted(set(r["carrier"] for r in geo_rows if r.get("pair_type")==contrast and r.get("split")=="testA")):
            rr=[r for r in geo_rows if r.get("carrier")==c and r.get("pair_type")==contrast and r.get("split")=="testA"]; plt.plot(NODES,[r.get("f1") if r.get("f1") is not None else np.nan for r in rr],marker="o",label=c)
        plt.xticks(range(5),NODES); plt.ylim(0,1.02); plt.grid(alpha=.25); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(figdir/filename,dpi=150); plt.close()
    geo_line("P0_vs_P1_diff_entity_same_relation","04_P0_P1_f1.png","P0/P1 discrimination vs depth")
    geo_line("P0_vs_P2_same_entity_diff_relation","05_P0_P2_f1.png","P0/P2 discrimination vs depth")
    plt.figure(figsize=(8,5))
    for c in sorted(set(r["carrier"] for r in spec_rows)):
        rr=[r for r in spec_rows if r["carrier"]==c and r["split"]=="train"]; rr=sorted(rr,key=lambda x:NODES.index(x["node"])); plt.plot(NODES,[r["effective_rank"] for r in rr],marker="o",label=c)
    plt.xticks(range(5),NODES); plt.grid(alpha=.25); plt.title("Effective rank vs depth"); plt.legend(); plt.tight_layout(); plt.savefig(figdir/"06_effective_rank.png",dpi=150); plt.close()
    crows=[r for r in control_rows if r["carrier"]=="J-I2b"]
    plt.figure(figsize=(8,5)); names=[r["control"] for r in crows]; vals=[r["entity_testA"] or 0 for r in crows]; plt.bar(range(len(names)),vals); plt.xticks(range(len(names)),names,rotation=45,ha="right"); plt.ylim(0,1.02); plt.grid(axis="y",alpha=.25); plt.title("Learned vs random/PCA projection: I2b entity probe"); plt.tight_layout(); plt.savefig(figdir/"07_learned_vs_random_projection.png",dpi=150); plt.close()
    plt.figure(figsize=(8,5)); key=[r for r in spec_rows if r["carrier"]=="J-I2b" and r["node"] in ("Z0","Z1","Z3","Z4") and r["split"]=="train"]
    for r in key:
        vals=json.loads(r["singular_values_json"]); plt.plot(range(1,min(65,len(vals))+1),vals[:64],label=r["node"])
    plt.yscale("log"); plt.grid(alpha=.25); plt.title("Singular spectrum at key I2b nodes"); plt.xlabel("singular value index"); plt.legend(); plt.tight_layout(); plt.savefig(figdir/"08_singular_eigen_spectrum.png",dpi=150); plt.close()


def main():
    seed_all(); t0=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if DEVICE_NAME=="cuda" and torch.cuda.is_available() else "cpu")
    cache=MRCache(CACHE_ROOT); infos,pmap=build_infos(cache); token,token_path=load_token_arrays(); carriers=aligned_carriers(infos,token); jlabels,jmap=joint_labels(infos,pmap)
    ck=torch.load(O21B,map_location="cpu",weights_only=False); key=KeyEncoder(DIM).to(device); key.load_state_dict(ck["key"],strict=True); key.eval()
    for p in key.parameters(): p.requires_grad_(False)
    all_reps={}; probe_rows=[]; probe_json={}
    for carrier in carriers:
        all_reps[carrier]={}; probe_json[carrier]={}
        for split in SPLITS:
            all_reps[carrier][split]=key_nodes(key,carriers[carrier][split],device)
        for node in NODES:
            rec=probe_node(carrier,node,{s:all_reps[carrier][s] for s in SPLITS},infos,pmap,jlabels,device); probe_json[carrier][node]=rec
            row={"carrier":carrier,"node":node,"dim":all_reps[carrier]["train"][node].shape[1]}
            for kind in ("entity","relation","joint","entity_conditioned_relation","relation_conditioned_entity"):
                for split in SPLITS:
                    row[f"{kind}_{split}"]=rec[kind][split]["accuracy"]; row[f"{kind}_{split}_known_fraction"]=rec[kind][split]["known_fraction"]
            probe_rows.append(row)
    write_csv(OUT/"layerwise_probe.csv",probe_rows); dump(OUT/"layerwise_probe.json",probe_json)
    # Fixed address pair sets are shared across all representations for each split.
    pair_infos={}
    for s in SPLITS:
        n=min(len(carriers[c][s]) for c in carriers)
        pair_infos[s]={"persons":infos[s]["persons"][:n],"relation":infos[s]["relation"][:n]}
    pairsets={s:make_address_pairs(pair_infos[s],max(100,PAIR_N//4),SEED+100+i) for i,s in enumerate(SPLITS)}
    pair_counts={s:{k:int(len(v)) for k,v in pairsets[s].items()} for s in SPLITS}; dump(OUT/"pair_sets.json",pair_counts)
    geo_rows=[]; geo_json={}; judge_rows=[]
    for carrier in carriers:
        # Recompute full split geometry with carrier labels in flat CSV.
        for n in NODES:
            rr=pair_geometry({s:all_reps[carrier][s][n] for s in ("val","testA","testB")},{s:pairsets[s] for s in ("val","testA","testB")},n)
            for x in rr:
                x["carrier"]=carrier; geo_rows.append(x)
            tr=all_reps[carrier]["train"][n]; va=all_reps[carrier]["val"][n];
            judges=pair_judge(tr,va,{"testA":all_reps[carrier]["testA"][n],"testB":all_reps[carrier]["testB"][n]},pairsets,device,mlp=False)
            mlps=pair_judge(tr,va,{"testA":all_reps[carrier]["testA"][n],"testB":all_reps[carrier]["testB"][n]},pairsets,device,mlp=True)
            for neg in judges:
                for split in ("testA","testB"):
                    judge_rows.append({"carrier":carrier,"node":n,"judge":"linear","contrast":neg,"split":split,"f1":judges[neg][split]["f1"],"auroc":judges[neg][split]["auroc"],"balanced_accuracy":judges[neg][split]["balanced_accuracy"]})
                    judge_rows.append({"carrier":carrier,"node":n,"judge":"pair_mlp","contrast":neg,"split":split,"f1":mlps[neg][split]["f1"],"auroc":mlps[neg][split]["auroc"],"balanced_accuracy":mlps[neg][split]["balanced_accuracy"]})
    # Add carrier to raw geometry rows and derive per-node summary rows.
    for x in geo_rows:
        if "f1" not in x and x.get("pair_type","").startswith("P0_vs_"): x["f1"]=None
    write_csv(OUT/"layerwise_geometry.csv",geo_rows); write_csv(OUT/"matcher_controls.csv",judge_rows)
    # Operation loss events and cliffs.
    ops=operation_rows(probe_rows); write_csv(OUT/"operation_events.csv",ops)
    cliffs={}
    for carrier in carriers:
        rr={r["node"]:r for r in probe_rows if r["carrier"]==carrier}; cliffs[carrier]={}
        for metric,label in (("entity_testA","Entity cliff"),("relation_testA","Relation cliff"),("joint_testA","Joint-binding cliff")):
            vals=[rr[n].get(metric) for n in NODES]; drops=[None if vals[i] is None or vals[i+1] is None else vals[i+1]-vals[i] for i in range(4)]; i=int(np.nanargmin([d if d is not None else 0 for d in drops])); cliffs[carrier][label]={"node_before":NODES[i],"node_after":NODES[i+1],"operation":NODE_OP[NODES[i+1]],"drop":drops[i]}
    # Geometry cliffs from raw P0/P2 F1.
    for carrier in carriers:
        vals=[]
        for n in NODES:
            row=next((r for r in geo_rows if r.get("carrier")==carrier and r.get("node")==n and r.get("split")=="testA" and r.get("pair_type")=="P0_vs_P2_same_entity_diff_relation"),None); vals.append(None if row is None else row.get("f1"))
        drops=[None if vals[i] is None or vals[i+1] is None else vals[i+1]-vals[i] for i in range(4)]; i=int(np.nanargmin([d if d is not None else 0 for d in drops])); cliffs[carrier]["Geometry cliff"]={"node_before":NODES[i],"node_after":NODES[i+1],"operation":NODE_OP[NODES[i+1]],"drop":drops[i]}
    dump(OUT/"cliffs.json",cliffs)
    # Rank/spectrum and distance preservation.
    spec_rows=[]; dist_rows=[]
    for carrier in carriers:
        for node in NODES:
            sm=spectrum_metrics(all_reps[carrier]["train"][node],device); spec_rows.append({"carrier":carrier,"split":"train","node":node,**{k:(json.dumps(v) if k in ("singular_values","eigenvalues") else v) for k,v in sm.items()},"singular_values_json":json.dumps(sm["singular_values"]),"eigenvalues_json":json.dumps(sm["eigenvalues"])})
        # Pairwise distance preservation against Z0 using fixed P0/P1/P2/P3 pairs.
        for split in ("testA","testB"):
            for typ,ab in pairsets[split].items():
                base_a=all_reps[carrier][split]["Z0"][ab[:,0]]; base_b=all_reps[carrier][split]["Z0"][ab[:,1]]; bcos=1-cosine(base_a,base_b); beuc=euclid(base_a,base_b)
                for node in NODES:
                    a=all_reps[carrier][split][node][ab[:,0]]; b=all_reps[carrier][split][node][ab[:,1]]
                    dist_rows.append({"carrier":carrier,"split":split,"pair_type":typ,"node":node,"cosine_rank_correlation":rank_corr(bcos,1-cosine(a,b)),"euclidean_rank_correlation":rank_corr(beuc,euclid(a,b)),"n":len(ab)})
    write_csv(OUT/"spectrum_metrics.csv",spec_rows); write_csv(OUT/"distance_preservation.csv",dist_rows)
    # Projection controls: fixed random orthogonal and label-free PCA controls.
    control_rows=[]
    for ci,carrier in enumerate(carriers):
        z0={s:all_reps[carrier][s]["Z0"] for s in SPLITS}; mean,pbas=pca_basis(z0["train"],(64,256)); controls={"identity":z0}
        for d in (256,64):
            R=random_basis(z0["train"].shape[1],d,SEED+700+ci+d); controls[f"random_{d}"]={s:z0[s]@R for s in SPLITS}; controls[f"random_{d}_l2"]={s:F.normalize(torch.from_numpy(z0[s]@R),dim=-1).numpy() for s in SPLITS}; controls[f"pca_{d}"]={s:(z0[s]-mean)@pbas[d].T for s in SPLITS}; controls[f"pca_{d}_l2"]={s:F.normalize(torch.from_numpy((z0[s]-mean)@pbas[d].T),dim=-1).numpy() for s in SPLITS}
        for name,reps in controls.items():
            if name=="identity": continue
            row=control_metrics(name,reps,infos,pmap,jlabels,pairsets,device); row["carrier"]=carrier; row["dim"]=int(reps["train"].shape[1]); control_rows.append(row)
    write_csv(OUT/"projection_controls.csv",control_rows)
    # Architecture description is explicit and node-addressable.
    arch=["CLR-v2.4 O3b-J — frozen O2.1b KeyEncoder computation graph","","Z0 = KeyEncoder input carrier: float32 vector, shape [B, 2048]","Z1 = net[0]: Linear(2048 -> 256), affine learned projection, shape [B, 256]","Z2 = net[1]: GELU activation, shape [B, 256]","Z3 = net[2]: Linear(256 -> 64), affine learned dimensionality reduction, shape [B, 64]","Z4 = F.normalize(Z3, dim=-1): L2 normalization over the final feature dimension, shape [B, 64], final K","","No residual branch; no LayerNorm/RMSNorm; no pooling; no head merge; no mean/sum; no reshape; no learned temperature or scale; no additional normalization.","Weights are loaded from O2.1b b1_best.pt and frozen. Only diagnostic probes/judges are trained."]
    (OUT/"keyencoder_architecture.txt").write_text("\n".join(arch)+"\n",encoding="utf-8")
    # Diagnostics verdict and report.
    ib=next(r for r in probe_rows if r["carrier"]=="J-I2b" and r["node"]=="Z4"); i2b=cliffs["J-I2b"]
    ctrl=[r for r in control_rows if r["carrier"]=="J-I2b"]; learned_ent=ib["entity_testA"]; random_ent=max([r["entity_testA"] for r in ctrl if r["control"].startswith("random_64") and not r["control"].endswith("_l2")]+[0]); pca_ent=max([r["entity_testA"] for r in ctrl if r["control"].startswith("pca_64") and not r["control"].endswith("_l2")]+[0]); z0rank=next(r["effective_rank"] for r in spec_rows if r["carrier"]=="J-I2b" and r["node"]=="Z0"); z4rank=next(r["effective_rank"] for r in spec_rows if r["carrier"]=="J-I2b" and r["node"]=="Z4")
    if max(random_ent,pca_ent)-learned_ent>=.15: primary="J-B LEARNED PROJECTION-INDUCED INVARIANCE"
    elif z0rank>0 and z4rank/z0rank<.55: primary="J-A DIMENSIONAL COMPRESSION BOTTLENECK"
    elif ib["entity_testA"] is not None and ib["relation_testA"] is not None and ib["joint_testA"] is not None and ib["joint_testA"]<min(ib["entity_testA"],ib["relation_testA"])-.15: primary="J-E JOINT BINDING LOSS"
    else: primary="J-F MULTIPLE BOTTLENECKS"
    secondary=[]
    if i2b.get("Entity cliff",{}).get("drop",0) is not None and i2b["Entity cliff"]["drop"]<=-.10: secondary.append("projection operation cliff")
    if ib["entity_testA"] is not None and ib["entity_testA"]<.50: secondary.append("entity representation information loss")
    if not secondary: secondary.append("none independently established")
    def cliff_text(x): return f"{x['node_before']} -> {x['node_after']}\noperation = {x['operation']}\ndrop = {100*x['drop']:.2f}pp" if x.get("drop") is not None else "not measurable"
    lines=["# CLR v2.4 — O3b-J KeyEncoder Projection Bottleneck Localization","","## O3b-J Verdict","",f"`O3b-J: {primary}`","",f"Secondary:\n`{'; '.join(secondary)}`","","### Critical localization","",f"Entity cliff (J-I2b):\n{cliff_text(i2b['Entity cliff'])}","",f"Relation cliff (J-I2b):\n{cliff_text(i2b['Relation cliff'])}","",f"Joint-binding cliff (J-I2b):\n{cliff_text(i2b['Joint-binding cliff'])}","",f"Geometry cliff (J-I2b, P0/P2):\n{cliff_text(i2b['Geometry cliff'])}","","### Controls","",f"Learned projection:\nI2b Z4 entity={pct(learned_ent)}, relation={pct(ib['relation_testA'])}, joint={pct(ib['joint_testA'])} on TestA.","",f"Random projection:\nBest fixed random 64-D entity probe={pct(random_ent)} on I2b TestA.","",f"PCA/truncation:\nBest label-free PCA 64-D entity probe={pct(pca_ent)} on I2b TestA.","",f"Rank:\nI2b effective rank Z0={z0rank:.2f}, Z4={z4rank:.2f}.","","### Answer to core question","",f"For the strongest upstream carrier (J-I2b), the input starts with entity and relation information supplied by O3b-I, but the layerwise probes localize the largest loss to {i2b['Entity cliff']['node_before']} -> {i2b['Entity cliff']['node_after']}, specifically {i2b['Entity cliff']['operation']}. The random/PCA controls quantify whether the loss is explained by dimension reduction alone or is selective to the learned mapping; these results are recorded in `projection_controls.csv`. Joint and conditional probes distinguish independent entity/relation readability from true binding, while P0/P2 geometry and the frozen matcher controls separate representation loss from metric mismatch. No production memory path, Ouro/backbone, KeyEncoder, architecture, or addressing model was changed.","","### Reconnect allowed","","No","","### O3c allowed","","No","","### Repeated V update contamination","","Unresolved; not touched.","","### Recommended next experiment","","Only after review: use the single largest diagnosed operation as the target of a narrowly scoped frozen-representation ablation/control, without reconnecting addressing or modifying production architecture in this run.",""]
    (OUT/"v24_o3b_j_projection_localization_report.md").write_text("\n".join(lines),encoding="utf-8")
    dump(OUT/"config.json",{"phase":"CLR-v2.4-O3b-J","seed":SEED,"device":str(device),"cache":str(CACHE_ROOT),"reused_token_cache":str(token_path),"source_key_checkpoint":str(O21B),"carriers":list(carriers),"nodes":list(NODES),"keyencoder_frozen":True,"ouro_rerun":False,"reconnect":False,"o3c":False,"architecture_changed":False,"repeated_v_update_contamination":"unresolved; not touched","wall_clock_s":time.perf_counter()-t0})
    dump(OUT/"gate_evaluation.json",{"baseline_reused":True,"diagnostic_only":True,"reconnect":False,"o3c":False,"architecture_changed":False,"repeated_v_update_contamination":"unresolved; not touched","stop_rule":"O3b-J complete; stop after report"})
    make_plots(probe_rows,geo_rows,spec_rows,control_rows)
    print(json.dumps({"out":str(OUT),"report":str(OUT/"v24_o3b_j_projection_localization_report.md"),"primary":primary,"secondary":secondary,"wall_clock_s":time.perf_counter()-t0},ensure_ascii=False))


if __name__ == "__main__":
    main()
