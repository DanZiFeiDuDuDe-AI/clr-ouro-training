# -*- coding: utf-8 -*-
"""CLR-v2.4 O3b-H: Entity Carrier / Input Sufficiency Diagnosis.

Offline diagnosis only.  All backbone/Ouro, KeyEncoder, QueryEncoder, value,
GRU, allocator, storage and downstream modules are frozen.  This script trains
only small diagnostic probes and optionally extracts a bounded sample of
token/layer representations from the already-frozen Ouro model.  Nothing is
reconnected to a real memory write/read path and no new architecture is
designed.
"""
import copy
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import a1_v24_o3b_f as fdiag
from a1_v24_o3b import FactPool, MRCache, KeyEncoder, QueryEncoder, DIM
from a1_v24_o3b_g import BindingKey


ROOT = Path(r"D:/training")
OUT = Path(os.environ.get("V24_O3BH_OUT", r"D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag"))
CACHE_ROOT = Path(os.environ.get("V24_O3BH_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
O21B = Path(os.environ.get("V24_O3BH_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
O21A_MR = Path(os.environ.get("V24_O3BH_O21A_MR", r"D:/training/checkpoints/clr_v24_o21a_mr/best_model.pt"))
O3BG_OUT = Path(os.environ.get("V24_O3BH_O3BG_OUT", r"D:/training/checkpoints/clr_v24_o3b_g_binding_key"))
SEED = int(os.environ.get("V24_O3BH_SEED", "42"))
DEVICE_NAME = os.environ.get("V24_O3BH_DEVICE", "cuda")
SMOKE = os.environ.get("V24_O3BH_SMOKE", "0") == "1" or "--smoke" in os.sys.argv
PROBE_STEPS = int(os.environ.get("V24_O3BH_PROBE_STEPS", "320" if not SMOKE else "30"))
PAIR_STEPS = int(os.environ.get("V24_O3BH_PAIR_STEPS", "350" if not SMOKE else "30"))
PAIR_N = int(os.environ.get("V24_O3BH_PAIR_N", "3500" if not SMOKE else "300"))
TOKEN_FACTS_PER_SPLIT = int(os.environ.get("V24_O3BH_TOKEN_FACTS", "4096" if not SMOKE else "64"))
TOKEN_OTHER_PER_SPLIT = int(os.environ.get("V24_O3BH_TOKEN_OTHER", "1024" if not SMOKE else "64"))
TOKEN_BATCH = int(os.environ.get("V24_O3BH_TOKEN_BATCH", "8"))


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
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def pct(x):
    return "n/a" if x is None else f"{100.0 * float(x):.2f}%"


def tensor(x, device):
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def qstats(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            "p10": float(np.quantile(x, .10)), "p25": float(np.quantile(x, .25)),
            "p50": float(np.quantile(x, .50)), "p75": float(np.quantile(x, .75)),
            "p90": float(np.quantile(x, .90))}


def macro_f1(y, p, nclass):
    vals = []
    for c in range(nclass):
        tp = int(((y == c) & (p == c)).sum())
        fp = int(((y != c) & (p == c)).sum())
        fn = int(((y == c) & (p != c)).sum())
        vals.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(vals))


def multiclass_stats(y, logits, nclass):
    y = np.asarray(y, dtype=np.int64)
    logits = np.asarray(logits)
    p = logits.argmax(1)
    topk = min(5, nclass)
    top = np.argpartition(-logits, topk - 1, axis=1)[:, :topk]
    return {"n": int(len(y)), "accuracy": float(np.mean(p == y)) if len(y) else None,
            "top1": float(np.mean(p == y)) if len(y) else None,
            "top5": float(np.mean(np.any(top == y[:, None], axis=1))) if len(y) else None,
            "macro_f1": macro_f1(y, p, nclass) if len(y) else None}


def roc_auc(y, score):
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum()); neg = len(y) - pos
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def pr_auc(y, score):
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    if pos == 0:
        return None
    order = np.argsort(-score, kind="mergesort"); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    precision = tp / np.maximum(tp + fp, 1); recall = tp / pos
    return float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))


def best_threshold(y, score):
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    vals = np.unique(score)
    if len(vals) > 500:
        vals = np.quantile(vals, np.linspace(0.0, 1.0, 501))
    best = (-1.0, 0.5)
    for t in vals:
        p = score >= t
        tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
        fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1]


def binary_metrics(y, score, threshold):
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    p = (score >= threshold).astype(np.int64)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return {"n": int(len(y)), "accuracy": float((p == y).mean()),
            "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)), "auroc": roc_auc(y, score),
            "pr_auc": pr_auc(y, score), "threshold": float(threshold),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def overlap(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if not len(a) or not len(b):
        return None
    lo = min(float(a.min()), float(b.min())); hi = max(float(a.max()), float(b.max()))
    if hi <= lo:
        return 1.0
    h1, _ = np.histogram(a, bins=np.linspace(lo, hi, 101)); h2, _ = np.histogram(b, bins=np.linspace(lo, hi, 101))
    h1 = h1 / max(h1.sum(), 1); h2 = h2 / max(h2.sum(), 1)
    return float(np.minimum(h1, h2).sum())


def build_infos(cache):
    infos = {}
    train_persons = list(cache.train_persons)
    pmap = {p: i for i, p in enumerate(train_persons)}
    for split in ("train", "val", "testA", "testB"):
        pool = FactPool(cache, split)
        refs = [r for ep in pool.episodes for r in ep]
        H = np.stack([pool.hidden(r) for r in refs]).astype(np.float32)
        QH = np.stack([pool.query_hidden(r) for r in refs]).astype(np.float32)
        ent = np.asarray([pmap.get(r["person"], -1) for r in refs], dtype=np.int64)
        rel = np.asarray([int(r["relation"]) for r in refs], dtype=np.int64)
        eps = np.asarray([int(r["episode"]) for r in refs], dtype=np.int64)
        pos = np.asarray([int(r["pos"]) for r in refs], dtype=np.int64)
        persons = np.asarray([r["person"] for r in refs], dtype=object)
        infos[split] = {"pool": pool, "refs": refs, "H": H, "QH": QH, "entity": ent, "relation": rel,
                        "episode": eps, "pos": pos, "persons": persons}
    return infos, pmap


@torch.no_grad()
def base_features(base_key, x, device, batch=512):
    out = {"projection_linear_pre_activation": [], "projection_activation": [],
           "projection_final_pre_norm": [], "final_normalized_K": []}
    base_key.eval()
    for st in range(0, len(x), batch):
        z = tensor(x[st:st + batch], device)
        a = base_key.net[0](z)
        b = base_key.net[1](a)
        c = base_key.net[2](b)
        k = F.normalize(c, dim=-1)
        for name, v in (("projection_linear_pre_activation", a), ("projection_activation", b),
                        ("projection_final_pre_norm", c), ("final_normalized_K", k)):
            out[name].append(v.float().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


@torch.no_grad()
def apply_query(query, x, device, batch=512):
    out = []
    query.eval()
    for st in range(0, len(x), batch):
        out.append(query(tensor(x[st:st + batch], device)).float().cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def apply_g3(g3, x, device, batch=512):
    out = {"G3_E": [], "G3_R": [], "G3_K": []}
    g3.eval()
    for st in range(0, len(x), batch):
        k, e, r = g3(tensor(x[st:st + batch], device))
        for n, v in (("G3_E", e), ("G3_R", r), ("G3_K", k)):
            out[n].append(v.float().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def train_probe(xtr, ytr, evals, nclass, device, shallow=False, steps=PROBE_STEPS, seed=SEED):
    m = ytr >= 0
    xtr = np.asarray(xtr, dtype=np.float32)[m]; ytr = np.asarray(ytr, dtype=np.int64)[m]
    torch.manual_seed(seed)
    if shallow:
        model = nn.Sequential(nn.Linear(xtr.shape[1], 64), nn.GELU(), nn.Linear(64, nclass)).to(device)
    else:
        model = nn.Linear(xtr.shape[1], nclass).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    X = tensor(xtr, device); Y = torch.from_numpy(ytr).long().to(device)
    bs = min(1024, len(Y)); rng = np.random.default_rng(seed + 913)
    model.train()
    for _ in range(max(1, steps)):
        ix = rng.integers(0, len(Y), size=bs)
        loss = F.cross_entropy(model(X[ix]), Y[ix])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); result = {"params": sum(p.numel() for p in model.parameters()), "shallow": shallow,
                             "train_n": int(len(Y)), "nclass": int(nclass)}
    for name, (xx, yy) in evals.items():
        mm = np.asarray(yy) >= 0
        if not mm.any():
            result[name] = {"accuracy": None, "known_fraction": float(mm.mean()), "n_known": 0}
            continue
        with torch.no_grad(): logits = model(tensor(np.asarray(xx)[mm], device)).cpu().numpy()
        result[name] = {**multiclass_stats(np.asarray(yy)[mm], logits, nclass),
                        "known_fraction": float(mm.mean()), "n_known": int(mm.sum())}
    return model, result


def probe_with_seeds(xtr, ytr, evals, nclass, device, shallow, name, seeds):
    records = []
    for sd in seeds:
        _, rec = train_probe(xtr, ytr, evals, nclass, device, shallow=shallow, seed=sd)
        records.append(rec)
    def agg(split, key="accuracy"):
        vals = [r[split].get(key) for r in records if r.get(split, {}).get(key) is not None]
        if not vals: return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals), "values": vals}
    return {"name": name, "shallow": shallow, "seeds": seeds,
            "seed_records": records, "aggregate": {s: agg(s) for s in evals}}


def make_pair_set(info, n, seed):
    rng = np.random.default_rng(seed); by = defaultdict(list)
    for i, p in enumerate(info["persons"]): by[str(p)].append(i)
    people = [p for p, ids in by.items() if len(ids) >= 2]
    pos = []; neg = []
    for _ in range(n // 2):
        p = people[int(rng.integers(0, len(people)))]; ids = by[p]
        a, b = rng.choice(ids, size=2, replace=False); pos.append((int(a), int(b), 1))
    for _ in range(n - len(pos)):
        a = int(rng.integers(0, len(info["persons"])))
        for _try in range(50):
            b = int(rng.integers(0, len(info["persons"])))
            if str(info["persons"][a]) != str(info["persons"][b]): break
        neg.append((a, b, 0))
    out = pos + neg; rng.shuffle(out)
    return {"a": np.asarray([x[0] for x in out], dtype=np.int64), "b": np.asarray([x[1] for x in out], dtype=np.int64),
            "y": np.asarray([x[2] for x in out], dtype=np.int64)}


def pair_features(X, pairs):
    a = X[pairs["a"]]; b = X[pairs["b"]]
    return np.concatenate([a * b, np.abs(a - b)], axis=1).astype(np.float32)


def pair_probe(rep_arrays, pairsets, device, name):
    Xtr = pair_features(rep_arrays["train"], pairsets["train"]); Y = pairsets["train"]["y"]
    Xv = pair_features(rep_arrays["val"], pairsets["val"]); yv = pairsets["val"]["y"]
    model = nn.Linear(Xtr.shape[1], 1).to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    Xt = tensor(Xtr, device); yt = torch.from_numpy(Y.astype(np.float32)).to(device); rng = np.random.default_rng(SEED + len(name))
    bs = min(1024, len(Y)); model.train()
    for _ in range(PAIR_STEPS):
        ix = rng.integers(0, len(Y), size=bs); lg = model(Xt[ix]).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(lg, yt[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    def scores(X):
        with torch.no_grad(): return model(tensor(X, device)).squeeze(1).cpu().numpy()
    val_score = scores(Xv); th = best_threshold(yv, val_score)
    out = {"name": name, "params": sum(p.numel() for p in model.parameters()), "threshold_val": th}
    for split in ("train", "val", "testA", "testB"):
        sc = scores(pair_features(rep_arrays[split], pairsets[split]))
        out[split] = binary_metrics(pairsets[split]["y"], sc, th)
    return out


def geometry_from_pairs(rep_arrays, pairsets):
    out = {}
    for split in ("testA", "testB"):
        pairs = pairsets[split]; a = rep_arrays[split][pairs["a"]]; b = rep_arrays[split][pairs["b"]]
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-8)
        sc = (a * b).sum(1); pos = sc[pairs["y"] == 1]; neg = sc[pairs["y"] == 0]
        out[split] = {"same_entity": qstats(pos), "different_entity": qstats(neg),
                      "cosine_overlap": overlap(pos, neg)}
    return out


def entity_labels(info): return info["entity"].copy()


def run_stage_probes(stage_arrays, infos, pmap, device):
    reps = sorted(set().union(*(v.keys() for v in stage_arrays.values())))
    output = {}
    seeds3 = [SEED, SEED + 1, SEED + 2]
    for rep in reps:
        xtr = stage_arrays["train"][rep]; evals_ent = {s: (stage_arrays[s][rep], entity_labels(infos[s])) for s in infos}
        evals_rel = {s: (stage_arrays[s][rep], infos[s]["relation"]) for s in infos}
        ent = probe_with_seeds(xtr, infos["train"]["entity"], evals_ent, len(pmap), device,
                               shallow=False, name=f"{rep}_entity_linear", seeds=seeds3 if rep in ("H_input", "final_normalized_K", "G3_E", "G3_K") else [SEED])
        rel = probe_with_seeds(xtr, infos["train"]["relation"], evals_rel, 4, device,
                               shallow=False, name=f"{rep}_relation_linear", seeds=[SEED])
        output[rep] = {"entity_linear": ent, "relation_linear": rel}
        if rep in ("H_input", "final_normalized_K", "G3_E", "G3_R", "G3_K"):
            output[rep]["entity_mlp"] = probe_with_seeds(xtr, infos["train"]["entity"], evals_ent, len(pmap), device,
                                                          shallow=True, name=f"{rep}_entity_mlp", seeds=[SEED])
            output[rep]["relation_mlp"] = probe_with_seeds(xtr, infos["train"]["relation"], evals_rel, 4, device,
                                                            shallow=True, name=f"{rep}_relation_mlp", seeds=[SEED])
    return output


def load_base_models(device):
    kc = torch.load(O21B, map_location="cpu", weights_only=False)
    key = KeyEncoder(DIM).to(device); key.load_state_dict(kc["key"], strict=True); key.eval()
    qc = torch.load(O21A_MR, map_location="cpu", weights_only=False)
    query = QueryEncoder(DIM).to(device); query.load_state_dict(qc["query"], strict=True); query.eval()
    for mod in (key, query):
        for p in mod.parameters(): p.requires_grad_(False)
    return key, query


def load_g3(device, nentity):
    ck_path = O3BG_OUT / "checkpoints" / "G3_projected_tensor_best.pt"
    model = BindingKey("G3_projected_tensor", nentity).to(device)
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


def h0_reproduction():
    prior = {}
    bp = O3BG_OUT / "baseline_reproduction.json"; vr = O3BG_OUT / "variant_results.json"; pr = O3BG_OUT / "probe_results.json"
    if bp.exists(): prior["baseline"] = json.loads(bp.read_text(encoding="utf-8"))
    if vr.exists(): prior["variants"] = json.loads(vr.read_text(encoding="utf-8"))
    if pr.exists(): prior["probes"] = json.loads(pr.read_text(encoding="utf-8"))
    b = prior.get("baseline", {}); g3 = prior.get("variants", {}).get("G3_projected_tensor", {})
    baseline_ok = bool(b and abs(b.get("testA", {}).get("f1", -1) - .8843853820598006) <= .02 and
                       abs(b.get("testB", {}).get("f1", -1) - .8655913978494623) <= .02)
    out = {"source": str(O3BG_OUT), "reuse_only": True, "no_prior_training": True,
           "baseline_reproduced": baseline_ok,
           "baseline": {k: b.get(k) for k in ("testA", "testB")},
           "g3_best_validation_variant": {k: g3.get(k) for k in ("testA", "testB", "same_entity_relation", "entity_fixed_relation")},
           "prior_probe_artifact_available": "probes" in prior,
           "protocol": "O3b-G frozen K-K addendum artifacts; exact split/checkpoint reused"}
    return out, prior


def relation_anchor(rel):
    return {0: ["当前城市", "住在", "居住", "住"], 1: ["出生城市", "出生地", "出生"],
            2: ["偏好城市", "偏爱的城市", "最喜欢", "喜欢"], 3: ["工作城市", "工作地点", "工作"]}.get(int(rel), [])


def find_subseq(seq, sub):
    if not sub or len(sub) > len(seq): return None
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub: return (i, i + len(sub))
    return None


def detailed_ouro_batch(model, tok, texts, persons, relations, device="cuda"):
    """Return local/final token features and selected final-UT layer features."""
    from transformers.masking_utils import create_causal_mask
    import inspect as _inspect
    m = model.model; id_rows = []
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"][: 1023]
        id_rows.append(ids + [tok.eos_token_id])
    B = len(id_rows); T = max(len(x) for x in id_rows)
    inp_ids = torch.zeros((B, T), dtype=torch.long, device=device); am = torch.zeros((B, T), dtype=torch.long, device=device)
    for i, ids in enumerate(id_rows):
        inp_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device); am[i, :len(ids)] = 1
    inputs_embeds = m.embed_tokens(inp_ids); cache_position = torch.arange(T, device=device); position_ids = cache_position.unsqueeze(0)
    params = set(_inspect.signature(create_causal_mask).parameters)
    kw = dict(config=m.config, attention_mask=am, past_key_values=None, position_ids=position_ids)
    if "input_embeds" in params and "inputs_embeds" not in params:
        kw["input_embeds"] = inputs_embeds
        if "cache_position" in params: kw["cache_position"] = cache_position
    else: kw["inputs_embeds"] = inputs_embeds
    causal_mask = create_causal_mask(**kw); position_embeddings = m.rotary_emb(inputs_embeds, position_ids); hidden = inputs_embeds
    layer_count = len(m.layers); wanted = sorted(set([0, max(0, layer_count // 4), max(0, layer_count // 2), max(0, (3 * layer_count) // 4), layer_count - 1]))
    captures = {}
    for current_ut in range(m.total_ut_steps):
        for li, layer in enumerate(m.layers[:m.config.num_hidden_layers]):
            hidden = layer(hidden, attention_mask=causal_mask, position_ids=position_ids, past_key_value=None,
                           use_cache=False, cache_position=cache_position, position_embeddings=position_embeddings,
                           current_ut=current_ut)
            if isinstance(hidden, tuple): hidden = hidden[0]
            if current_ut == m.total_ut_steps - 1 and li in wanted:
                captures[li] = hidden.detach().float().cpu().numpy()
        hidden = m.norm(hidden)
    seq = hidden.detach().float().cpu().numpy()
    all_layers = {f"layer_{li}_mention_mean": [] for li in wanted}
    reps = {"entity_mention_mean": [], "entity_mention_max": [], "relation_span_mean": [],
            "final_token": [], "prev_token": [], "next_token": [], "global_mean": []}
    for i, (ids, person, rel) in enumerate(zip(id_rows, persons, relations)):
        pids = tok(str(person), add_special_tokens=False)["input_ids"]
        ps = find_subseq(ids, pids) or (0, min(len(pids), max(1, len(ids) - 1)))
        rs = None
        for anchor in relation_anchor(rel):
            rs = find_subseq(ids, tok(anchor, add_special_tokens=False)["input_ids"])
            if rs: break
        if rs is None: rs = (ps[1], max(ps[1] + 1, len(ids) - 1))
        body = np.arange(0, max(1, len(ids) - 1), dtype=np.int64)
        mention = np.arange(ps[0], max(ps[0] + 1, ps[1]), dtype=np.int64)
        relation = np.arange(rs[0], max(rs[0] + 1, rs[1]), dtype=np.int64)
        mention = mention[mention < len(ids) - 1]; relation = relation[relation < len(ids) - 1]
        if not len(mention): mention = np.asarray([0], dtype=np.int64)
        if not len(relation): relation = np.asarray([min(1, len(ids) - 1)], dtype=np.int64)
        def mean(ix): return seq[i, ix].mean(0)
        def mx(ix): return seq[i, ix].max(0)
        reps["entity_mention_mean"].append(mean(mention)); reps["entity_mention_max"].append(mx(mention)); reps["relation_span_mean"].append(mean(relation))
        reps["final_token"].append(seq[i, len(ids) - 1]); reps["prev_token"].append(seq[i, max(0, mention[0] - 1)]); reps["next_token"].append(seq[i, min(len(ids) - 2, mention[-1] + 1)])
        reps["global_mean"].append(mean(body))
        for li in wanted: all_layers[f"layer_{li}_mention_mean"].append(captures[li][i, mention].mean(0))
    for d in (reps, all_layers):
        for k in d: d[k] = np.stack(d[k]).astype(np.float32)
    return reps, all_layers, {"layer_count": layer_count, "wanted_layers": wanted, "max_tokens": 1024}


def select_token_records(cache, per_split, other_per_split):
    records = {"fact": {}, "query": {}}
    for split in ("train", "val", "testA", "testB"):
        metas = cache.meta[split]; facts = []
        for ei, m in enumerate(metas):
            off = int(sum(int(x["capacity"]) for x in metas[:ei]))
            for pos, text in enumerate(m["fact_texts"]):
                facts.append({"split": split, "episode": ei, "pos": pos, "flat": off + pos, "text": text,
                              "person": m["persons"][pos], "relation": int(m["relation_ids"][pos]), "entity": str(m["persons"][pos])})
        facts = facts[:per_split]
        qs = []
        for ei, m in enumerate(metas[:other_per_split]):
            qs.append({"split": split, "episode": ei, "text": m["query_text"], "person": m["query_person"],
                       "relation": int(m["query_relation_id"]), "entity": str(m["query_person"])})
        records["fact"][split] = facts; records["query"][split] = qs
    return records


def run_token_diagnostics(cache, device):
    """Bounded frozen-Ouro extraction; absence is explicitly recorded."""
    result = {"available": False, "cache_has_token_or_layer": False, "sampling": {}, "fact": {}, "query": {}, "layers": {}}
    if SMOKE and os.environ.get("V24_O3BH_SKIP_OURO", "0") == "1":
        result["reason"] = "smoke skip requested"; return result
    try:
        from clr_v2 import load_ouro
        records = select_token_records(cache, TOKEN_FACTS_PER_SPLIT, TOKEN_OTHER_PER_SPLIT)
        result["sampling"] = {"facts_per_split": TOKEN_FACTS_PER_SPLIT, "queries_per_split": TOKEN_OTHER_PER_SPLIT}
        model, tok = load_ouro(); model.eval()
        for kind in ("fact", "query"):
            for split in ("train", "val", "testA", "testB"):
                rs = records[kind][split]; result[kind][split] = {}
                for st in range(0, len(rs), TOKEN_BATCH):
                    ch = rs[st:st + TOKEN_BATCH]; texts = [x["text"] for x in ch]
                    persons = [x["person"] for x in ch]; rels = [x["relation"] for x in ch]
                    rep, layers, meta = detailed_ouro_batch(model, tok, texts, persons, rels, device=str(device))
                    for k, v in rep.items(): result[kind][split].setdefault(k, []).append(v)
                    for k, v in layers.items(): result[kind][split].setdefault(k, []).append(v)
                    result["layers"] = meta
                for k in list(result[kind][split]): result[kind][split][k] = np.concatenate(result[kind][split][k], axis=0)
        del model, tok
        result["available"] = True
        result["feature_dims"] = {k: int(v.shape[1]) for k, v in result["fact"]["train"].items()}
        npz = {}
        for kind in ("fact", "query"):
            for split in result[kind]:
                for name, arr in result[kind][split].items(): npz[f"{kind}_{split}_{name}"] = arr.astype(np.float32)
        np.savez_compressed(OUT / "token_layer_features.npz", **npz)
        return result
    except Exception as exc:
        result["reason"] = f"frozen-Ouro token/layer extraction unavailable: {type(exc).__name__}: {exc}"
        return result


def token_probe_table(token_result, infos, pmap, device):
    if not token_result.get("available"): return {"available": False, "reason": token_result.get("reason", "unavailable")}
    out = {"available": True, "fact": {}, "query": {}}
    # Token samples are deliberately bounded and are not used to train any production module.
    for kind in ("fact", "query"):
        names = sorted(token_result[kind]["train"])
        for name in names:
            xtr = token_result[kind]["train"][name]
            ytr = infos["train"]["entity"][:len(xtr)] if kind == "fact" else np.asarray([x["entity"] for x in select_token_records(MRCache(CACHE_ROOT), TOKEN_FACTS_PER_SPLIT, TOKEN_OTHER_PER_SPLIT)["query"]["train"]])
            if kind == "fact":
                evals_ent = {s: (token_result[kind][s][name], infos[s]["entity"][:len(token_result[kind][s][name])]) for s in infos}
                evals_rel = {s: (token_result[kind][s][name], infos[s]["relation"][:len(token_result[kind][s][name])]) for s in infos}
            else:
                qrec = select_token_records(MRCache(CACHE_ROOT), TOKEN_FACTS_PER_SPLIT, TOKEN_OTHER_PER_SPLIT)["query"]
                qmap = {s: np.asarray([pmap.get(x["entity"], -1) for x in qrec[s]], dtype=np.int64) for s in qrec}
                rmap = {s: np.asarray([x["relation"] for x in qrec[s]], dtype=np.int64) for s in qrec}
                evals_ent = {s: (token_result[kind][s][name], qmap[s]) for s in qmap}; evals_rel = {s: (token_result[kind][s][name], rmap[s]) for s in qmap}
                ytr = qmap["train"]
            _, ent = train_probe(xtr, ytr, evals_ent, len(pmap), device, shallow=False, seed=SEED)
            _, rel = train_probe(xtr, ytr if False else (infos["train"]["relation"][:len(xtr)] if kind == "fact" else np.asarray([x["relation"] for x in select_token_records(MRCache(CACHE_ROOT), TOKEN_FACTS_PER_SPLIT, TOKEN_OTHER_PER_SPLIT)["query"]["train"]])), evals_rel, 4, device, shallow=False, seed=SEED + 1)
            out[kind][name] = {"entity_linear": ent, "relation_linear": rel}
    return out


def position_breakdown(model, rep_arrays, infos, pmap, device, split="testA"):
    xtr = rep_arrays["train"]; ytr = infos["train"]["entity"]
    evals = {split: (rep_arrays[split], infos[split]["entity"])}
    _, rec = train_probe(xtr, ytr, evals, len(pmap), device, shallow=False, seed=SEED)
    # Re-run predictions to associate errors with episode positions.
    probe = nn.Linear(xtr.shape[1], len(pmap)).to(device); opt = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=.01)
    X = tensor(xtr, device); Y = torch.from_numpy(ytr).long().to(device); rng = np.random.default_rng(SEED + 889)
    for _ in range(PROBE_STEPS):
        ix = rng.integers(0, len(Y), size=min(1024, len(Y))); loss = F.cross_entropy(probe(X[ix]), Y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad(): pred = probe(tensor(rep_arrays[split], device)).argmax(1).cpu().numpy()
    info = infos[split]; counts = Counter(); member_errors = Counter(); total_errors = 0
    for i, (p, ep) in enumerate(zip(pred, info["episode"])):
        true = int(info["entity"][i]);
        if true < 0 or p == true: continue
        total_errors += 1; counts["errors"] += 1
        ep_people = [str(info["persons"][j]) for j in np.flatnonzero(info["episode"] == ep)]
        if str(pmap and list(pmap.keys())[int(p)] if int(p) < len(pmap) else "") in ep_people:
            member_errors["episode_member_error"] += 1
            pp = str(list(pmap.keys())[int(p)])
            if pp == ep_people[0]: member_errors["first_member"] += 1
            if pp == ep_people[-1]: member_errors["last_member"] += 1
    return {"probe": rec, "total_errors": total_errors, "member_error_counts": dict(member_errors),
            "member_error_rate": float(member_errors["episode_member_error"] / max(total_errors, 1)),
            "last_given_member_rate": float(member_errors["last_member"] / max(member_errors["episode_member_error"], 1)),
            "first_given_member_rate": float(member_errors["first_member"] / max(member_errors["episode_member_error"], 1))}


def difficulty_breakdown(rep_arrays, infos, pmap, device):
    xtr = rep_arrays["train"]; ytr = infos["train"]["entity"]
    probe = nn.Linear(xtr.shape[1], len(pmap)).to(device); opt = torch.optim.AdamW(probe.parameters(), lr=3e-4, weight_decay=.01)
    X = tensor(xtr, device); Y = torch.from_numpy(ytr).long().to(device); rng = np.random.default_rng(SEED + 991)
    for _ in range(PROBE_STEPS):
        ix = rng.integers(0, len(Y), size=min(1024, len(Y))); loss = F.cross_entropy(probe(X[ix]), Y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    out = {}
    with torch.no_grad():
        for split in ("testA", "testB"):
            p = probe(tensor(rep_arrays[split], device)).argmax(1).cpu().numpy(); info = infos[split]
            cats = defaultdict(list)
            for i, ep in enumerate(info["episode"]):
                # episode type is kept in FactPool metadata.
                kind = str(info["pool"].meta[int(ep)]["episode_type"])
                cats["episode_type:" + kind].append(int(p[i] == info["entity"][i]) if info["entity"][i] >= 0 else 0)
            for rel in range(4):
                m = info["relation"] == rel; cats[f"relation:{rel}"] = [int(x) for x in (p[m] == info["entity"][m]) if np.any(m)] if np.any(m) else []
            out[split] = {k: {"n": len(v), "accuracy": float(np.mean(v)) if v else None} for k, v in cats.items()}
    return out


def write_simple_plot(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = 900, 420; left, bottom = 210, 55; maxv = max([float(v) for _, v in rows] + [1.0])
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial;font-size:14px}.bar{fill:#3b82f6}</style>']
    bh = max(16, int((H - bottom - 20) / max(len(rows), 1) - 8))
    for i, (name, val) in enumerate(rows):
        y = 25 + i * (bh + 8); w = int((W - left - 30) * float(val) / maxv)
        svg += [f'<text x="10" y="{y + bh - 2}">{name}</text>', f'<rect class="bar" x="{left}" y="{y}" width="{w}" height="{bh}"/>', f'<text x="{left + w + 5}" y="{y + bh - 2}">{float(val):.3f}</text>']
    svg.append('</svg>'); path.write_text("".join(svg), encoding="utf-8")


def main():
    seed_all(); start = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(CACHE_ROOT); infos, pmap = build_infos(cache)
    h0, prior = h0_reproduction(); dump(OUT / "h0_baseline_reproduction.json", h0)
    if not h0["baseline_reproduced"]:
        dump(OUT / "gate_evaluation.json", {"baseline_reproduced": False, "verdict": "STOP: H0 O3b-G reproduction failed", "reconnect": False, "o3c": False})
        (OUT / "v24_o3b_h_entity_carrier_diag_report.md").write_text("# CLR v2.4 O3b-H\n\n**STOP: H0 baseline reproduction failed.**\n", encoding="utf-8")
        return
    key, query = load_base_models(device)
    for p in list(key.parameters()) + list(query.parameters()): assert not p.requires_grad
    nentity = len(pmap); g3 = load_g3(device, nentity)
    stage_arrays = {}; query_arrays = {}
    for split, info in infos.items():
        bf = base_features(key, info["H"], device); stage_arrays[split] = {"H_input": info["H"], **bf}
        stage_arrays[split].update(apply_g3(g3, bf["final_normalized_K"], device))
        query_arrays[split] = {"query_H": info["QH"], "query_Q": apply_query(query, info["QH"], device)}
    # H4/H5: raw input -> projection -> normalization -> frozen O3b-G factorized output.
    stage_probe = run_stage_probes(stage_arrays, infos, pmap, device)
    dump(OUT / "keyencoder_internal_probes.json", stage_probe)
    dump(OUT / "representation_metadata.json", {"cache": str(CACHE_ROOT), "cache_manifest": json.loads((CACHE_ROOT / "manifest.json").read_text(encoding="utf-8")),
          "raw_H_shape": {s: list(v["H"].shape) for s, v in infos.items()}, "stage_dims": {k: int(v.shape[1]) for k, v in stage_arrays["train"].items()},
          "frozen": ["Ouro/backbone", "O2.1b KeyEncoder", "O2.1a-MR QueryEncoder", "V/ValueEncoder", "GRU", "allocator", "memory write/update", "downstream answer head"],
          "diagnostic_trainable": ["linear probes", "shallow MLP probes", "small pairwise equality probes"], "git_sha": "NO_GIT_REPO"})
    # H6 query/write comparison.
    qwrite = {}
    for rep_map in (stage_arrays, query_arrays):
        for rep in set().union(*(x.keys() for x in rep_map.values())):
            if rep not in rep_map["train"]: continue
            xtr = rep_map["train"][rep]; ent_evals = {s: (rep_map[s][rep], infos[s]["entity"]) for s in infos}; rel_evals = {s: (rep_map[s][rep], infos[s]["relation"]) for s in infos}
            qwrite[rep] = {"entity": probe_with_seeds(xtr, infos["train"]["entity"], ent_evals, nentity, device, False, rep + "_entity", [SEED]),
                           "relation": probe_with_seeds(xtr, infos["train"]["relation"], rel_evals, 4, device, False, rep + "_relation", [SEED])}
    dump(OUT / "query_write_comparison.json", qwrite)
    # H2/H3: bounded frozen-Ouro local carriers. The original cache itself has no token/layer arrays.
    token_result = run_token_diagnostics(cache, device); dump(OUT / "token_span_sweep.json", {k: v for k, v in token_result.items() if k not in ("fact", "query")})
    if token_result.get("available"):
        # Probe directly against the bounded sample; label arrays are aligned by construction.
        token_table = {"available": True, "note": "bounded frozen-Ouro diagnostic sample; no production module was changed"}
        recs = select_token_records(cache, TOKEN_FACTS_PER_SPLIT, TOKEN_OTHER_PER_SPLIT)
        for kind in ("fact", "query"):
            token_table[kind] = {}
            for name in sorted(token_result[kind]["train"]):
                if kind == "fact":
                    y = {s: np.asarray([pmap.get(x["entity"], -1) for x in recs[kind][s]], dtype=np.int64) for s in recs[kind]}; yr = {s: np.asarray([x["relation"] for x in recs[kind][s]], dtype=np.int64) for s in recs[kind]}
                else:
                    y = {s: np.asarray([pmap.get(x["entity"], -1) for x in recs[kind][s]], dtype=np.int64) for s in recs[kind]}; yr = {s: np.asarray([x["relation"] for x in recs[kind][s]], dtype=np.int64) for s in recs[kind]}
                xtr = token_result[kind]["train"][name]; ent = probe_with_seeds(xtr, y["train"], {s: (token_result[kind][s][name], y[s]) for s in y}, nentity, device, False, f"token_{kind}_{name}_entity", [SEED]); rel = probe_with_seeds(xtr, yr["train"], {s: (token_result[kind][s][name], yr[s]) for s in yr}, 4, device, False, f"token_{kind}_{name}_relation", [SEED]); token_table[kind][name] = {"entity": ent, "relation": rel}
    else:
        token_table = {"available": False, "reason": token_result.get("reason", "not available")}
    dump(OUT / "token_span_probe_results.json", token_table)
    dump(OUT / "layer_sweep.json", {"available": bool(token_result.get("available")), "layer_metadata": token_result.get("layers"), "source": "bounded frozen-Ouro extraction; original cache had no layer tensors"})
    # H1/H4/H5 matrix and 3-seed key probes are contained in this compact summary.
    matrix = {}
    for rep in ("H_input", "projection_linear_pre_activation", "projection_activation", "projection_final_pre_norm", "final_normalized_K", "G3_E", "G3_R", "G3_K"):
        if rep not in stage_probe: continue
        matrix[rep] = {"entity": stage_probe[rep]["entity_linear"], "relation": stage_probe[rep]["relation_linear"]}
    matrix["query_H"] = qwrite["query_H"]; matrix["query_Q"] = qwrite["query_Q"]
    dump(OUT / "main_diagnostic_matrix.json", matrix)
    # H19/H20 open-set same-entity equality and geometry.
    pairsets = {s: make_pair_set(infos[s], PAIR_N, SEED + 100 + i) for i, s in enumerate(("train", "val", "testA", "testB"))}
    dump(OUT / "entity_equality_pair_sets.json", {s: {"n": len(v["y"]), "positive": int(v["y"].sum()), "negative": int((v["y"] == 0).sum())} for s, v in pairsets.items()})
    eq_reps = {"raw_H": {s: infos[s]["H"] for s in infos}, "query_H": {s: infos[s]["QH"] for s in infos},
               "final_K": {s: stage_arrays[s]["final_normalized_K"] for s in infos}, "G3_E": {s: stage_arrays[s]["G3_E"] for s in infos},
               "G3_R": {s: stage_arrays[s]["G3_R"] for s in infos}, "G3_K": {s: stage_arrays[s]["G3_K"] for s in infos}}
    equality = {}; geom = {}
    for rep, arrs in eq_reps.items():
        equality[rep] = pair_probe(arrs, pairsets, device, rep); geom[rep] = geometry_from_pairs(arrs, pairsets)
    dump(OUT / "entity_equality.json", equality); dump(OUT / "entity_geometry.json", geom)
    # H8/H10 with the raw H carrier: errors are associated with actual episode positions, not labels at inference.
    raw_break = position_breakdown(nn.Identity(), {s: stage_arrays[s]["H_input"] for s in infos}, infos, pmap, device)
    dump(OUT / "recency_bias.json", raw_break)
    dump(OUT / "difficulty_breakdown.json", difficulty_breakdown({s: stage_arrays[s]["H_input"] for s in infos}, infos, pmap, device))
    # H9 is explicitly diagnostic: label-based address control is not fed to KeyEncoder or inference.
    oracle = {"available": True, "inference_uses_oracle_labels": False,
              "entity_source": "ground-truth labels used only for this offline upper-bound control",
              "relation_source": "ground-truth labels used only for this offline upper-bound control",
              "joint_address_equality": "100% by construction", "interpretation": "not a learned architecture result"}
    dump(OUT / "identity_preserving_oracle_control.json", oracle)
    # A small plot makes the first-loss location auditable without treating visualization as evidence.
    bars = []
    for rep in ("H_input", "projection_linear_pre_activation", "projection_activation", "projection_final_pre_norm", "final_normalized_K", "G3_E", "G3_R", "G3_K"):
        v = stage_probe.get(rep, {}).get("entity_linear", {}).get("aggregate", {}).get("testA", {}).get("mean")
        if v is not None: bars.append((rep, v))
    write_simple_plot(OUT / "plots" / "entity_probe_stages.svg", bars)
    # Verdict logic is evidence-based and conservative; no reconnect is ever enabled here.
    h_input_ent = stage_probe["H_input"]["entity_linear"]["aggregate"]["testA"]["mean"]
    token_best = None
    if token_table.get("available"):
        vals = []
        for name, rec in token_table.get("fact", {}).items():
            v = rec["entity"]["aggregate"].get("testA", {}).get("mean")
            if v is not None: vals.append((v, name))
        if vals: token_best = max(vals)
    final_ent = stage_probe["final_normalized_K"]["entity_linear"]["aggregate"]["testA"]["mean"]
    if h_input_ent < .50 and (token_best is None or token_best[0] < .50):
        verdict = "O3b-H: UPSTREAM ENTITY INFORMATION INSUFFICIENT"
    elif token_best is not None and token_best[0] >= .80 and h_input_ent < .50:
        verdict = "O3b-H: ENTITY CARRIER EXISTS BUT CURRENT STATE SELECTION MISSES IT"
    elif h_input_ent >= .80 and final_ent < h_input_ent - .25:
        verdict = "O3b-H: ENTITY INFORMATION DESTROYED BY KEY PROJECTION"
    elif qwrite.get("query_H", {}).get("entity", {}).get("aggregate", {}).get("testA", {}).get("mean", 0) >= .80 and h_input_ent < .50:
        verdict = "O3b-H: WRITE-SIDE ENTITY CARRIER FAILURE"
    else:
        verdict = "O3b-H: MIXED ENTITY CARRIER + PROJECTION FAILURE"
    cfg = {"phase": "CLR-v2.4-O3b-H", "seed": SEED, "git_sha": "NO_GIT_REPO", "device": str(device), "cache": str(CACHE_ROOT),
           "cache_reused": True, "cache_token_layer_available": bool(token_result.get("available")), "frozen": ["Ouro/backbone", "KeyEncoder", "QueryEncoder", "V/ValueEncoder", "GRU", "allocator", "memory write/update", "downstream answer head"],
           "diagnostic_trainable": ["linear probes", "shallow MLP probes", "small equality probes"], "probe_steps": PROBE_STEPS, "pair_steps": PAIR_STEPS, "pair_n": PAIR_N,
           "token_sample": {"facts_per_split": TOKEN_FACTS_PER_SPLIT, "other_per_split": TOKEN_OTHER_PER_SPLIT}, "wall_clock_s": time.perf_counter() - start,
           "peak_vram": {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None},
           "verdict": verdict, "reconnect": False, "o3c": False, "repeated_v_update_contamination": "independent unresolved issue"}
    dump(OUT / "config.json", cfg)
    gate = {"baseline_reproduced": True, "verdict": verdict, "reconnect": False, "o3c": False,
            "h_input_entity_testA": h_input_ent, "final_K_entity_testA": final_ent, "token_best": token_best,
            "closed_world_testB_entity_classifier": "N/A; TestB is unseen-person", "testB_identity_equality_available": True,
            "protocol_impacting_issue": None}
    dump(OUT / "gate_evaluation.json", gate)
    # Report is deliberately explicit about unavailable/available carrier levels and the causal boundary.
    raw_rel = stage_probe["H_input"]["relation_linear"]["aggregate"]["testA"]["mean"]
    q_ent = qwrite["query_H"]["entity"]["aggregate"]["testA"]["mean"]
    q_rel = qwrite["query_H"]["relation"]["aggregate"]["testA"]["mean"]
    eq_raw_b = equality["raw_H"]["testB"]; eq_k_b = equality["final_K"]["testB"]
    report = ["# CLR v2.4 — O3b-H: Entity Carrier / Input Sufficiency Diagnosis", "", f"**{verdict}**", "",
              "## Executive Summary", f"H0 reused the frozen O3b-G artifacts and reproduced the baseline: TestA/TestB K-K cosine F1={pct(h0['baseline']['testA']['f1'])}/{pct(h0['baseline']['testB']['f1'])}. No backbone, KeyEncoder, QueryEncoder, value path, GRU, allocator or real memory path was changed.",
              f"The raw write-side H→entity probe on TestA is {pct(h_input_ent)}, while H→relation is {pct(raw_rel)}. Query-side raw H→entity is {pct(q_ent)} and H→relation is {pct(q_rel)}. The report separates closed-world entity classification from TestB unseen-entity equality.", "",
              "## Frozen Components", "Backbone/Ouro, O2.1b KeyEncoder, O2.1a-MR QueryEncoder, V/ValueEncoder, GRU, allocator, memory write/update, storage and downstream answer head were frozen. Diagnostic probes are not connected to inference.", "",
              "## Baseline Reproduction", f"O3b-G artifact reuse successful={h0['baseline_reproduced']}. Prior baseline F1 TestA/TestB={pct(h0['baseline']['testA']['f1'])}/{pct(h0['baseline']['testB']['f1'])}; prior G3 variant and probe artifacts are referenced from {O3BG_OUT}.", "",
              "## Main Diagnostic Matrix", "| Representation | Entity probe | Relation probe | Interpretation |", "|---|---:|---:|---|",
              f"| raw write H | {pct(h_input_ent)} | {pct(raw_rel)} | actual KeyEncoder input; TestA closed-world entity probe |",
              f"| entity token/span | {'see token_span_probe_results.json' if token_table.get('available') else 'unavailable: not cached'} | {'see token_span_probe_results.json' if token_table.get('available') else 'unavailable: not cached'} | bounded frozen-Ouro carrier search |",
              f"| final token | {'see token_span_probe_results.json' if token_table.get('available') else 'unavailable: not cached'} | {'see token_span_probe_results.json' if token_table.get('available') else 'unavailable: not cached'} | local carrier search |",
              f"| query H | {pct(q_ent)} | {pct(q_rel)} | query-side raw carrier |",
              f"| E branch / G3 | {pct(stage_probe['G3_E']['entity_linear']['aggregate']['testA']['mean'])} | {pct(stage_probe['G3_E']['relation_linear']['aggregate']['testA']['mean'])} | prior O3b-G learned branch |",
              f"| R branch / G3 | {pct(stage_probe['G3_R']['entity_linear']['aggregate']['testA']['mean'])} | {pct(stage_probe['G3_R']['relation_linear']['aggregate']['testA']['mean'])} | prior O3b-G relation branch |",
              f"| final K / O3b-G G3 | {pct(stage_probe['G3_K']['entity_linear']['aggregate']['testA']['mean'])} | {pct(stage_probe['G3_K']['relation_linear']['aggregate']['testA']['mean'])} | final learned key |", "",
              "## KeyEncoder Internal Information Flow", "Stages and per-stage entity/relation probes are in `keyencoder_internal_probes.json`; `entity_probe_stages.svg` is only a visual aid. The first major drop is determined from the stage table, not from the verdict label alone.", "",
              "## Query vs Write Side", f"Raw write H entity={pct(h_input_ent)}; raw query H entity={pct(q_ent)}; projected write K and query Q are in `query_write_comparison.json`. TestB closed-world entity classification is N/A because its persons are unseen.", "",
              "## Token / Span / Layer Availability", f"The original MR cache contains only fact/query `[2048]` final hidden vectors; no token-level or per-layer arrays. Bounded frozen-Ouro extraction available={token_table.get('available')}. Full details and any extraction reason are in `token_span_sweep.json`, `layer_sweep.json`, and `token_span_probe_results.json`.", "",
              "## TestB Same-Entity Equality", f"Raw H TestB equality={json.dumps(eq_raw_b, ensure_ascii=False)}; final-K TestB equality={json.dumps(eq_k_b, ensure_ascii=False)}. This pairwise test is the open-set identity result; no TestB entity labels were used for training or threshold selection.", "",
              "## Geometry / Position / Difficulty", "Same/different-entity cosine distributions and overlap are in `entity_geometry.json`; position/recency audit is in `recency_bias.json`; multi-relation/episode-type breakdown is in `difficulty_breakdown.json`.", "",
              "## Verdict", f"{verdict}. This is an offline carrier diagnosis only. Reconnect allowed: NO. O3c allowed: NO.", "",
              "## Repeated V Update Contamination", "Repeated V update contamination remains an independent unresolved issue.", "",
              "## Capability Boundary", "This run does not modify or validate real write/read, allocator, value update, multi-fact memory, or reasoning. No O3b-I was designed.", ""]
    (OUT / "v24_o3b_h_entity_carrier_diag_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "baseline_reproduced": True, "raw_write_H_entity": h_input_ent, "raw_write_H_relation": raw_rel, "raw_query_H_entity": q_ent, "raw_query_H_relation": q_rel, "final_K_entity": final_ent, "token_available": bool(token_result.get("available")), "testB_raw_H_equality_f1": eq_raw_b.get("f1"), "testB_final_K_equality_f1": eq_k_b.get("f1"), "reconnect": False, "o3c": False, "wall_clock_s": cfg["wall_clock_s"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
