# -*- coding: utf-8 -*-
"""CLR-v2.4-O3b-D: offline same-key equivalence diagnosis and repair.

Only a small K-K judge is trained.  The learned semantic KeyEncoder, value
path, query path, O3a allocator, and the physical slots remain frozen.  Real
write integration is attempted only when the offline judge clears the
pre-registered gates.
"""
import copy
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v24_o3b import (
    FactPool, MRCache, KeyEncoder, UpdateRouter, make_examples,
    batch_examples, load_all, router_eval, aggregate_runs, run_set,
    category_summary, DIM, SLOTS, SEED as O3B_SEED,
)
from a1_v24_o21a import QueryEncoder
from a1_v24_o3b import O21B as O21B_DEFAULT
from a1_v24_o3b import CACHE_ROOT as CACHE_DEFAULT
from a1_v24_o3b import OUT as O3B_OUT_DEFAULT
from a1_v24_o2_common import json_dump


OUT = Path(os.environ.get("V24_O3BD_OUT", r"D:/training/checkpoints/clr_v24_o3b_diag"))
CACHE_ROOT = Path(os.environ.get("V24_O3BD_CACHE", str(CACHE_DEFAULT)))
O21B = Path(os.environ.get("V24_O3BD_O21B", str(O21B_DEFAULT)))
O3B_OUT = Path(os.environ.get("V24_O3BD_O3B_OUT", str(O3B_OUT_DEFAULT)))
SEED = int(os.environ.get("V24_O3BD_SEED", "42"))
STEPS = int(os.environ.get("V24_O3BD_STEPS", "3000"))
EVAL_EVERY = int(os.environ.get("V24_O3BD_EVAL_EVERY", "200"))
SMOKE = os.environ.get("V24_O3BD_SMOKE", "0") == "1" or "--smoke" in os.sys.argv
DEVICE_NAME = os.environ.get("V24_O3BD_DEVICE", "cuda")
PAIR_N = 6000 if not SMOKE else 512
PROBE_STEPS = 600 if not SMOKE else 20


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, Path): return str(x)
    return x


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def auc_binary(y, score):
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    pos = score[y == 1]; neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return None
    # Tie-aware Mann-Whitney estimate.
    total = 0.0
    for p in pos:
        total += float((neg < p).sum()) + 0.5 * float((neg == p).sum())
    return total / float(len(pos) * len(neg))


def binary_metric(y, prob, threshold=0.5, categories=None):
    y = np.asarray(y, dtype=np.int64); prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    out = {"accuracy": float((pred == y).mean()), "balanced_accuracy": float((tpr + tnr) / 2),
           "precision": float(precision), "recall": float(recall), "f1": float(f1),
           "auroc": auc_binary(y, prob), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
           "threshold": float(threshold), "n": int(len(y))}
    if categories is not None:
        cats = np.asarray(categories)
        out["per_category_accuracy"] = {}
        for cat in sorted(set(cats.tolist())):
            m = cats == cat
            out["per_category_accuracy"][str(cat)] = {"n": int(m.sum()), "accuracy": float((pred[m] == y[m]).mean())}
    return out


def choose_threshold(y, prob):
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 197), prob])
    best = (float("-inf"), 0.5)
    for t in candidates:
        m = binary_metric(y, prob, float(t))
        key = (m["f1"], m["balanced_accuracy"], -abs(float(t) - 0.5))
        if key > (best[0], -1.0, -1.0): best = (m["f1"], float(t))
    return best[1]


def ece_brier(y, prob, bins=10):
    y = np.asarray(y, dtype=np.float64); prob = np.asarray(prob, dtype=np.float64)
    ece = 0.0; hist = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i in range(bins):
        m = (prob >= edges[i]) & ((prob < edges[i + 1]) if i < bins - 1 else (prob <= edges[i + 1]))
        if not m.any(): continue
        conf = float(prob[m].mean()); acc = float(y[m].mean()); frac = float(m.mean())
        ece += frac * abs(conf - acc); hist.append({"bin": i, "n": int(m.sum()), "confidence": conf, "positive_rate": acc})
    return {"ece": float(ece), "brier": float(np.mean((prob - y) ** 2)), "histogram": hist}


class KeyBank:
    def __init__(self, pool, key, device):
        self.pool = pool; self.refs = [r for ep in pool.episodes for r in ep]
        self.refs = [dict(r) for r in self.refs]
        for i, r in enumerate(self.refs): r["bank_idx"] = i
        self.by_key = defaultdict(list); self.by_person = defaultdict(list); self.by_relation = defaultdict(list)
        self.by_value = defaultdict(list)
        for r in self.refs:
            self.by_key[r["key"]].append(r); self.by_person[r["person"]].append(r)
            self.by_relation[r["relation"]].append(r); self.by_value[r["value"]].append(r)
        self.K = self._encode(key, device)

    def _encode(self, key, device):
        out = []
        with torch.no_grad():
            for st in range(0, len(self.refs), 256):
                hs = np.stack([self.pool.hidden(r) for r in self.refs[st:st + 256]]).astype(np.float32)
                out.append(key(torch.from_numpy(hs).to(device)).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def attrs(self, i): return self.refs[int(i)]


RELATION_NAMES = {}


def lexical_relation(a, b):
    a = str(a); b = str(b)
    if a == b: return False
    ta = set(a.replace("-", "_").split("_")); tb = set(b.replace("-", "_").split("_"))
    if ta & tb: return True
    # The MR protocol uses *_city relation families; character overlap is a
    # conservative fallback for any future paraphrase-derived cache.
    return len(set(a) & set(b)) >= max(2, min(len(a), len(b)) // 2)


def similar_entity(a, b):
    a = str(a); b = str(b)
    if a == b: return False
    if len(a) >= 2 and len(b) >= 2 and (a[:2] == b[:2] or a[-2:] == b[-2:]): return True
    common = len(set(a) & set(b))
    return common >= 2 and common >= min(len(a), len(b)) - 1


def pair_ok(a, b):
    return a["episode"] != b["episode"]


def sample_pair(bank, rng, category, tries=3000):
    refs = bank.refs
    if category == "same_key":
        groups = [g for g in bank.by_key.values() if len(g) >= 2]
        for _ in range(tries):
            if not groups: break
            g = groups[int(rng.integers(0, len(groups)))]
            a, b = g[int(rng.integers(0, len(g)))], g[int(rng.integers(0, len(g)))]
            if a["bank_idx"] != b["bank_idx"] and pair_ok(a, b): return a, b
    for _ in range(tries):
        a = refs[int(rng.integers(0, len(refs)))]; b = refs[int(rng.integers(0, len(refs)))]
        if a["bank_idx"] == b["bank_idx"] or not pair_ok(a, b): continue
        same_person = a["person"] == b["person"]; same_relation = a["relation"] == b["relation"]
        same_value = a["value"] == b["value"]; same_key = a["key"] == b["key"]
        if category == "same_entity_diff_relation" and same_person and not same_relation and not same_key: return a, b
        if category == "same_relation_diff_entity" and same_relation and not same_person: return a, b
        if category == "same_value_diff_key" and same_value and not same_key: return a, b
        if category == "fully_different" and not same_person and not same_relation and not same_value: return a, b
        if category == "lexical_hard_negative" and not same_key and not same_person and lexical_relation(a["relation_name"], b["relation_name"]): return a, b
        if category == "similar_entity_names" and not same_key and similar_entity(a["person"], b["person"]): return a, b
    # Deterministic fallback keeps the requested category visible in stats,
    # but is only reached for a genuinely sparse cache.
    for a in refs:
        for b in refs:
            if a["bank_idx"] != b["bank_idx"] and pair_ok(a, b): return a, b
    raise RuntimeError(f"no pair available for {category}")


def make_pairset(bank, count, seed):
    rng = np.random.default_rng(seed); npos = count // 2; nneg = count - npos
    neg_cats = ["same_entity_diff_relation", "same_relation_diff_entity", "same_value_diff_key", "fully_different", "lexical_hard_negative", "similar_entity_names"]
    pairs = []
    for _ in range(npos):
        a, b = sample_pair(bank, rng, "same_key"); pairs.append((a, b, 1, "P1_same_key"))
    for i in range(nneg):
        cat = neg_cats[i % len(neg_cats)]; a, b = sample_pair(bank, rng, cat)
        pairs.append((a, b, 0, {"same_entity_diff_relation": "P2_same_entity_diff_relation", "same_relation_diff_entity": "P3_same_relation_diff_entity", "same_value_diff_key": "P4_same_value_diff_key", "fully_different": "P5_fully_different", "lexical_hard_negative": "P6_lexical_hard_negative", "similar_entity_names": "P7_similar_entity_names"}[cat]))
    rng.shuffle(pairs)
    ka = np.stack([bank.K[a["bank_idx"]] for a, _, _, _ in pairs]).astype(np.float32)
    kb = np.stack([bank.K[b["bank_idx"]] for _, b, _, _ in pairs]).astype(np.float32)
    y = np.asarray([p[2] for p in pairs], dtype=np.int64); cats = np.asarray([p[3] for p in pairs])
    same_entity = np.asarray([int(a["person"] == b["person"]) for a, b, _, _ in pairs], dtype=np.int64)
    same_relation = np.asarray([int(a["relation"] == b["relation"]) for a, b, _, _ in pairs], dtype=np.int64)
    return {"ka": ka, "kb": kb, "y": y, "categories": cats, "same_entity": same_entity, "same_relation": same_relation,
            "pair_counts": Counter(cats.tolist()), "ref_pair_count": len(pairs)}


def cosine(ka, kb):
    return (ka * kb).sum(1) / (np.linalg.norm(ka, axis=1) * np.linalg.norm(kb, axis=1) + 1e-8)


def base_features(ka, kb):
    c = cosine(ka, kb)[:, None]; d = np.linalg.norm(ka - kb, axis=1)[:, None]
    ad = np.abs(ka - kb); prod = ka * kb
    return np.concatenate([c, d, ad.mean(1, keepdims=True), prod.mean(1, keepdims=True), np.linalg.norm(ka, axis=1, keepdims=True), np.linalg.norm(kb, axis=1, keepdims=True)], axis=1).astype(np.float32)


def mlp_features(ka, kb): return np.concatenate([ka, kb, np.abs(ka - kb), ka * kb], axis=1).astype(np.float32)
def sym_features(ka, kb): return np.concatenate([np.abs(ka - kb), ka * kb, (ka + kb) / 2.0], axis=1).astype(np.float32)


class TinyMLP(nn.Module):
    def __init__(self, inp, hidden=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def tensor(x, device): return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


@torch.no_grad()
def model_probs(model, x, device):
    model.eval(); out = []
    for st in range(0, len(x), 512): out.append(torch.sigmoid(model(tensor(x[st:st + 512], device))).cpu().numpy())
    return np.concatenate(out)


def train_binary(train_x, train_y, val_x, val_y, device, steps, seed, name, weight=None):
    seed_all(seed); xmean = train_x.mean(0); xstd = train_x.std(0); xstd[xstd < 1e-6] = 1.0
    tx = ((train_x - xmean) / xstd).astype(np.float32); vx = ((val_x - xmean) / xstd).astype(np.float32)
    model = TinyMLP(tx.shape[1], 64).to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(seed + 77); best = None; logs = []; t0 = time.perf_counter(); bs = 256
    ty = torch.from_numpy(train_y.astype(np.float32)).to(device)
    for step in range(1, steps + 1):
        ix = rng.integers(0, len(tx), size=min(bs, len(tx))); logits = model(tensor(tx[ix], device)); loss = F.binary_cross_entropy_with_logits(logits, ty[ix])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == steps:
            vp = model_probs(model, vx, device); th = choose_threshold(val_y, vp); vm = binary_metric(val_y, vp, th); logs.append({"step": step, "loss": float(loss.detach()), "val": vm, "threshold": th})
            if best is None or (vm["f1"], vm["balanced_accuracy"]) > (best["val"]["f1"], best["val"]["balanced_accuracy"]):
                best = {"state": copy.deepcopy(model.state_dict()), "threshold": th, "val": vm, "step": step}
    model.load_state_dict(best["state"]); record = {"name": name, "best_step": best["step"], "threshold": best["threshold"], "val": best["val"], "params": sum(p.numel() for p in model.parameters()), "wall_clock_s": time.perf_counter() - t0, "logs": logs, "normalization": {"mean": xmean, "std": xstd}}
    return model, record


def evaluate_set(model, record, data, feature_fn, device, target_name="y"):
    y = data[target_name]; x = feature_fn(data["ka"], data["kb"]); mean = np.asarray(record["normalization"]["mean"]); std = np.asarray(record["normalization"]["std"]); p = model_probs(model, ((x - mean) / std).astype(np.float32), device); return binary_metric(y, p, record["threshold"], data["categories"]), p


def baseline_result(data, score, threshold, direction="score"):
    return binary_metric(data["y"], score, threshold, data["categories"])


def fit_threshold_on_val(y, score): return choose_threshold(y, score)


def geometry(data):
    out = {}
    c = cosine(data["ka"], data["kb"]); d = np.linalg.norm(data["ka"] - data["kb"], axis=1)
    for cat in sorted(set(data["categories"].tolist())):
        m = data["categories"] == cat
        out[cat] = {"n": int(m.sum()), "cosine_mean": float(c[m].mean()), "cosine_std": float(c[m].std()), "l2_mean": float(d[m].mean()), "l2_std": float(d[m].std())}
    p1 = c[data["categories"] == "P1_same_key"]; p2 = c[data["categories"] == "P2_same_entity_diff_relation"]
    if len(p1) and len(p2):
        lo = max(float(p1.min()), float(p2.min())); hi = min(float(p1.max()), float(p2.max()))
        bins = np.linspace(min(float(c.min()), -1.0), max(float(c.max()), 1.0), 101)
        h1, _ = np.histogram(p1, bins=bins, density=True); h2, _ = np.histogram(p2, bins=bins, density=True)
        overlap = float(np.minimum(h1 / max(h1.sum(), 1e-9), h2 / max(h2.sum(), 1e-9)).sum())
        out["P1_vs_P2_overlap"] = {"range_overlap": max(0.0, hi - lo), "histogram_overlap": overlap}
    return out


def threshold_report(val, testA, testB):
    sv = cosine(val["ka"], val["kb"]); sa = cosine(testA["ka"], testA["kb"]); sb = cosine(testB["ka"], testB["kb"]); th = fit_threshold_on_val(val["y"], sv)
    return {"score": "cosine", "chosen_threshold_val": th, "val": baseline_result(val, sv, th), "testA": baseline_result(testA, sa, th), "testB": baseline_result(testB, sb, th), "best_possible_testA": {"threshold": choose_threshold(testA["y"], sa), "metrics": baseline_result(testA, sa, choose_threshold(testA["y"], sa))}, "best_possible_testB": {"threshold": choose_threshold(testB["y"], sb), "metrics": baseline_result(testB, sb, choose_threshold(testB["y"], sb))}}


def margin_data(examples, pool, key, device):
    ys = []; bests = []; seconds = []
    with torch.no_grad():
        for ex in examples:
            hs = np.stack([pool.hidden(r) for r in ex["base"]]).astype(np.float32)
            kn = key(torch.from_numpy(pool.hidden(ex["incoming"])[None]).to(device)).cpu().numpy()[0]
            kb = key(torch.from_numpy(hs).to(device)).cpu().numpy()
            s = cosine(np.repeat(kn[None], len(kb), 0), kb); ss = np.sort(s)[::-1]
            bests.append(float(ss[0])); seconds.append(float(ss[1] if len(ss) > 1 else 0.0)); ys.append(int(ex["action"]))
    return {"y": np.asarray(ys, dtype=np.int64), "best": np.asarray(bests), "second": np.asarray(seconds)}


def margin_analysis(val, testA, testB):
    out = {}
    for name, fn in (("absolute_best", lambda z: z["best"]), ("margin", lambda z: z["best"] - z["second"]), ("ratio", lambda z: z["best"] / (np.abs(z["second"]) + 1e-6))):
        sv, sa, sb = fn(val), fn(testA), fn(testB); th = choose_threshold(val["y"], sv)
        out[name] = {"threshold_val": th, "val": binary_metric(val["y"], sv, th), "testA": binary_metric(testA["y"], sa, th), "testB": binary_metric(testB["y"], sb, th)}
    # Small linear combination of absolute best and margin.
    tx = np.stack([val["best"], val["best"] - val["second"]], 1).astype(np.float32); ax = np.stack([testA["best"], testA["best"] - testA["second"]], 1).astype(np.float32); bx = np.stack([testB["best"], testB["best"] - testB["second"]], 1).astype(np.float32)
    model, rec = train_binary(tx, val["y"], tx, val["y"], torch.device("cpu"), 500 if not SMOKE else 20, SEED + 881, "absolute_plus_margin")
    def pr(x):
        mean, std = np.asarray(rec["normalization"]["mean"]), np.asarray(rec["normalization"]["std"]); return model_probs(model, ((x - mean) / std).astype(np.float32), torch.device("cpu"))
    out["absolute_plus_margin"] = {"record": rec, "val": binary_metric(val["y"], pr(tx), rec["threshold"]), "testA": binary_metric(testA["y"], pr(ax), rec["threshold"]), "testB": binary_metric(testB["y"], pr(bx), rec["threshold"])}
    return out


def train_probe(train, val, testA, testB, target_name, device):
    model, rec = train_binary(sym_features(train["ka"], train["kb"]), train[target_name], sym_features(val["ka"], val["kb"]), val[target_name], device, PROBE_STEPS, SEED + (17 if target_name == "same_entity" else 19), f"pair_probe_{target_name}")
    out = {"record": rec}
    for name, data in (("val", val), ("testA", testA), ("testB", testB)):
        out[name], _ = evaluate_set(model, rec, data, sym_features, device, target_name)
    return out


def relation_entity_probes(banks, device):
    # A light K -> relation probe directly checks whether relation survives in K.
    out = {}
    train = banks["train"]; val = banks["val"]; ta = banks["testA"]; tb = banks["testB"]
    def fit(num_classes, label_fn, name):
        seed_all(SEED + 700 + num_classes); model = nn.Linear(DIM, num_classes).to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01); rng = np.random.default_rng(SEED + 1700 + num_classes)
        # This is a diagnostic probe only.  Bound it to a fixed subset and
        # minibatches so it cannot dominate the K-K judge experiment.
        limit = min(len(train.refs), 12000 if not SMOKE else 2048); subset = rng.choice(len(train.refs), size=limit, replace=False); x = tensor(train.K[subset], device); y = torch.tensor([label_fn(train.refs[int(i)]) for i in subset], dtype=torch.long, device=device)
        probe_steps = 100 if not SMOKE else 20; bs = min(512, limit)
        for step in range(1, probe_steps + 1):
            ix = rng.integers(0, limit, size=bs); loss = F.cross_entropy(model(x[ix]), y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        def acc(bank):
            yy = np.asarray([label_fn(r) for r in bank.refs]); xx = tensor(bank.K, device); pred = model(xx).argmax(1).detach().cpu().numpy(); return float((pred == yy).mean())
        return {"params": sum(p.numel() for p in model.parameters()), "train": acc(train), "val": acc(val), "testA": acc(ta), "testB": acc(tb)}
    rel_ids = sorted({int(r["relation"]) for r in train.refs}); rel_map = {v: i for i, v in enumerate(rel_ids)}
    out["K_to_relation"] = fit(len(rel_map), lambda r: rel_map.get(int(r["relation"]), 0), "relation")
    out["relation_ids"] = rel_map
    out["entity_note"] = "Entity classification is not a valid TestB metric because TestB entity names are unseen; equality is evaluated by the pair probe instead."
    return out


def symmetry(model, record, data, feature_fn, device):
    x = feature_fn(data["ka"], data["kb"]); xr = feature_fn(data["kb"], data["ka"]); mean = np.asarray(record["normalization"]["mean"]); std = np.asarray(record["normalization"]["std"])
    p = model_probs(model, ((x - mean) / std).astype(np.float32), device); q = model_probs(model, ((xr - mean) / std).astype(np.float32), device)
    return {"disagreement_rate": float((np.abs(p - q) > 1e-6).mean()), "max_abs_probability_delta": float(np.max(np.abs(p - q))), "mean_abs_probability_delta": float(np.mean(np.abs(p - q)))}


class JudgeAdapter(nn.Module):
    """Reconnect adapter: new judge controls action; old pair target stays frozen."""
    def __init__(self, judge, rec, feature_fn, old_router, device):
        super().__init__(); self.judge = judge; self.rec = rec; self.feature_fn = feature_fn; self.old_router = old_router; self.device_ref = device
    def forward(self, k_new, existing_k, occupied):
        b = k_new.shape[0]; ka = k_new[:, None, :].expand(-1, SLOTS, -1).reshape(-1, DIM); kb = existing_k.reshape(-1, DIM)
        if self.feature_fn is sym_features: f = torch.cat([(ka - kb).abs(), ka * kb, (ka + kb) / 2.0], -1)
        elif self.feature_fn is mlp_features: f = torch.cat([ka, kb, (ka - kb).abs(), ka * kb], -1)
        else: raise RuntimeError("reconnect requires a learned pair judge, not scalar baseline")
        mean = torch.as_tensor(self.rec["normalization"]["mean"], device=k_new.device, dtype=f.dtype); std = torch.as_tensor(self.rec["normalization"]["std"], device=k_new.device, dtype=f.dtype); p_pair = torch.sigmoid(self.judge((f - mean) / std)).reshape(b, SLOTS)
        maxp = p_pair.masked_fill(~occupied.bool(), -1e4).max(1).values; p = maxp
        action_logits = torch.stack([torch.log((1 - p).clamp_min(1e-6)), torch.log(p.clamp_min(1e-6))], -1)
        with torch.no_grad():
            _, old_pair = self.old_router(k_new, existing_k, occupied)
        return action_logits, old_pair


def run_reconnect(best_name, best_model, best_record, feature_fn, pools, key, query, core, vh, fa, device):
    if best_name not in ("B3_pair_mlp", "B4_symmetric_pair_judge"): return {"status": "SKIPPED", "reason": "best judge is a scalar baseline or linear head; reconnect was not needed"}
    old_router = UpdateRouter().to(device); ck = torch.load(O3B_OUT / "router_best.pt", map_location=device, weights_only=False); old_router.load_state_dict(ck["router"], strict=True); old_router.eval()
    adapter = JudgeAdapter(best_model, best_record, feature_fn, old_router, device).to(device).eval()
    eval_n = 1000 if not SMOKE else 128; rows = {}
    for split, seedoff in (("testA", 3), ("testB", 4)):
        exs = make_examples(pools[split], eval_n, SEED + seedoff, True)
        raw = router_eval(adapter, key, exs, pools[split], device, best_record["threshold"])
        runs = aggregate_runs(run_set(exs, pools[split], key, query, core, vh, fa, adapter, best_record["threshold"], device, "learned"))
        hard = make_examples(pools[split], eval_n, SEED + 5 + (0 if split == "testA" else 1), False, True)
        rows[split] = {"detection": {k: v for k, v in raw.items() if k not in ("y", "score", "pred", "target_ok", "categories")}, "pipeline": runs, "hard_negatives": category_summary(adapter, key, hard, pools[split], device, best_record["threshold"])}
    return {"status": "RAN", "judge": best_name, "source_router": str(O3B_OUT / "router_best.pt"), "results": rows}


def main():
    seed_all(); t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(CACHE_ROOT); pools = {s: FactPool(cache, s) for s in ("train", "val", "testA", "testB")}
    kc = torch.load(O21B, map_location="cpu", weights_only=False); key = KeyEncoder(DIM).to(device); key.load_state_dict(kc["key"], strict=True); key.eval()
    for p in key.parameters(): p.requires_grad_(False)
    banks = {s: KeyBank(pools[s], key, device) for s in pools}
    sets = {"train": make_pairset(banks["train"], PAIR_N, SEED + 1), "val": make_pairset(banks["val"], PAIR_N // 2, SEED + 2), "testA": make_pairset(banks["testA"], PAIR_N // 2, SEED + 3), "testB": make_pairset(banks["testB"], PAIR_N // 2, SEED + 4)}
    stats = {s: {"n": int(len(d["y"])), "positive": int(d["y"].sum()), "negative": int((d["y"] == 0).sum()), "categories": dict(d["pair_counts"]), "unique_source_episodes": int(len({r["episode"] for r in banks[s].refs})), "unique_persons": int(len({r["person"] for r in banks[s].refs})), "unique_relations": int(len({r["relation"] for r in banks[s].refs}))} for s, d in sets.items()}
    stats["protocol"] = {"positive": "same entity and same relation", "negative_hard_ratio": "six negative categories balanced; P2 is not diluted by easy negatives", "cross_split_exact_fact_pair_leakage": False, "same_episode_pairing": False}
    dump(OUT / "pair_dataset_stats.json", stats)
    val, ta, tb = sets["val"], sets["testA"], sets["testB"]
    baselines = {}
    for name, score_fn in (("B0_raw_cosine", cosine), ("B1_negative_l2", lambda a, b: -np.linalg.norm(a - b, axis=1))):
        sv, sa, sb = score_fn(val["ka"], val["kb"]), score_fn(ta["ka"], ta["kb"]), score_fn(tb["ka"], tb["kb"]); th = choose_threshold(val["y"], sv)
        baselines[name] = {"chosen_threshold_val": th, "val": baseline_result(val, sv, th), "testA": baseline_result(ta, sa, th), "testB": baseline_result(tb, sb, th)}
        dump(OUT / ("cosine_baseline.json" if name.startswith("B0") else "distance_baseline.json"), baselines[name])
    learned = {}
    for name, feature_fn, out_file in (("B2_margin_linear", base_features, "linear_pair.json"), ("B3_pair_mlp", mlp_features, "pair_mlp.json"), ("B4_symmetric_pair_judge", sym_features, "symmetric_pair_judge.json")):
        train_x = feature_fn(sets["train"]["ka"], sets["train"]["kb"]); val_x = feature_fn(val["ka"], val["kb"])
        model, rec = train_binary(train_x, sets["train"]["y"], val_x, val["y"], device, STEPS if not SMOKE else 20, SEED + len(learned) * 11 + 30, name)
        entry = {"record": rec}
        for s, d in (("val", val), ("testA", ta), ("testB", tb)): entry[s], _ = evaluate_set(model, rec, d, feature_fn, device)
        entry["symmetry_testA"] = symmetry(model, rec, ta, feature_fn, device); entry["symmetry_testB"] = symmetry(model, rec, tb, feature_fn, device)
        learned[name] = {"model": model, "feature_fn": feature_fn, "record": rec, "metrics": entry}; dump(OUT / out_file, {k: v for k, v in entry.items() if k != "model"})
    best_name = max(learned, key=lambda n: learned[n]["metrics"]["val"]["f1"])
    # Include scalar baselines in model selection without allowing a test set to choose the threshold.
    best_val = learned[best_name]["metrics"]["val"]["f1"]; best_source = "learned:" + best_name
    for name, b in baselines.items():
        if b["val"]["f1"] > best_val: best_val = b["val"]["f1"]; best_source = "baseline:" + name
    geom = {"testA": geometry(ta), "testB": geometry(tb)}; dump(OUT / "geometry_stats.json", geom)
    th_analysis = {"testA": threshold_report(val, ta, tb)["testA"], "testB": threshold_report(val, ta, tb)["testB"], "full": threshold_report(val, ta, tb)}; dump(OUT / "threshold_analysis.json", th_analysis)
    margin_val = margin_data(make_examples(pools["val"], 1000 if not SMOKE else 128, SEED + 120, True), pools["val"], key, device); margin_a = margin_data(make_examples(pools["testA"], 1000 if not SMOKE else 128, SEED + 121, True), pools["testA"], key, device); margin_b = margin_data(make_examples(pools["testB"], 1000 if not SMOKE else 128, SEED + 122, True), pools["testB"], key, device); dump(OUT / "margin_analysis.json", margin_analysis(margin_val, margin_a, margin_b))
    probes = {"same_entity": train_probe(sets["train"], val, ta, tb, "same_entity", device), "same_relation": train_probe(sets["train"], val, ta, tb, "same_relation", device), "single_K": relation_entity_probes(banks, device)}; dump(OUT / "entity_relation_pair_probe.json", probes)
    best = learned[best_name]; best_metrics = best["metrics"]
    calibration = {"best_judge": best_name, "testA": ece_brier(ta["y"], model_probs(best["model"], ((best["feature_fn"](ta["ka"], ta["kb"]) - best["record"]["normalization"]["mean"]) / best["record"]["normalization"]["std"]).astype(np.float32), device)), "testB": ece_brier(tb["y"], model_probs(best["model"], ((best["feature_fn"](tb["ka"], tb["kb"]) - best["record"]["normalization"]["mean"]) / best["record"]["normalization"]["std"]).astype(np.float32), device))}; dump(OUT / "calibration.json", calibration)
    dump(OUT / "symmetry_results.json", {"testA": best_metrics["symmetry_testA"], "testB": best_metrics["symmetry_testB"], "best_judge": best_name})
    # Hard-negative summary for the selected judge.
    hard = {s: {c: best_metrics[s]["per_category_accuracy"].get(c) for c in sorted(set(sets[s]["categories"].tolist()))} for s in ("testA", "testB")}; dump(OUT / "hard_negative_results.json", hard)
    offline = {"best_judge": best_name, "best_source": best_source, "threshold": best["record"]["threshold"], "testA": best_metrics["testA"], "testB": best_metrics["testB"], "offline_gate": {"testA_f1_ge_95": best_metrics["testA"]["f1"] >= .95, "testB_f1_ge_95": best_metrics["testB"]["f1"] >= .95, "testA_update_recall_ge_95": best_metrics["testA"]["recall"] >= .95, "testB_update_recall_ge_95": best_metrics["testB"]["recall"] >= .95, "P2_testA_ge_95": best_metrics["testA"]["per_category_accuracy"].get("P2_same_entity_diff_relation", {}).get("accuracy", 0) >= .95, "P2_testB_ge_90": best_metrics["testB"]["per_category_accuracy"].get("P2_same_entity_diff_relation", {}).get("accuracy", 0) >= .90}}
    dump(OUT / "offline_best_judge.json", offline)
    reconnect_ok = all(offline["offline_gate"].values()) and best_name in ("B3_pair_mlp", "B4_symmetric_pair_judge")
    reconnect = {"status": "SKIPPED", "reason": "offline equivalence gate not met; per preregistration do not reconnect"}
    if reconnect_ok:
        # Load frozen query/value/allocator only on the reconnect branch.
        _, query, core, vh, fa = load_all(device)
        reconnect = run_reconnect(best_name, best["model"], best["record"], best["feature_fn"], pools, key, query, core, vh, fa, device)
    dump(OUT / "reconnect_results.json", reconnect)
    verdict = "O3b-D PASS: SAME-KEY EQUIVALENCE REPAIRED" if reconnect.get("status") == "RAN" and all(reconnect["results"][s]["detection"]["binary"]["f1_update"] >= .95 for s in ("testA", "testB")) else ("O3b-D FAIL: RELATION IDENTITY UNDER-REPRESENTED" if best_metrics["testA"]["per_category_accuracy"].get("P2_same_entity_diff_relation", {}).get("accuracy", 0) < .90 or best_metrics["testB"]["per_category_accuracy"].get("P2_same_entity_diff_relation", {}).get("accuracy", 0) < .90 else "O3b-D FAIL: K REPRESENTATION EQUALITY-POOR")
    cfg = {"phase": "CLR-v2.4-O3b-D", "seed": SEED, "device": str(device), "source_key_checkpoint": str(O21B), "source_o3b_checkpoint": str(O3B_OUT / "router_best.pt"), "cache": str(CACHE_ROOT), "frozen": ["KeyEncoder", "ValueWriter/V/GRU", "QueryEncoder", "read router", "decoder", "answer head", "O3a allocator", "Ouro", "physical slots"], "trainable": ["small K-K SameKeyJudge only"], "pair_protocol": "P1-P7 balanced; source split disjoint; cross-pair exact fact leakage false", "steps": STEPS, "best_judge": best_name, "best_step": best["record"]["best_step"], "judge_params": best["record"]["params"], "threshold": best["record"]["threshold"], "wall_clock_s": time.perf_counter() - t0, "peak_vram": {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None}}
    dump(OUT / "config.json", cfg); dump(OUT / "metrics.json", {"verdict": verdict, "offline": offline, "reconnect": reconnect, "baselines": baselines, "learned": {k: v["metrics"] for k, v in learned.items()}})
    report = ["# CLR-v2.4-O3b-D — Same-Key Equivalence Diagnosis and Repair", "", f"**{verdict}**", "", "## Scope", "Only `(K_new, K_old) -> SAME/DIFFERENT` was studied. KeyEncoder, value path, query path, O3a allocator, Ouro, and physical slots were frozen. No V write, allocation, eviction, or K retraining was performed unless the offline gate passed.", "", "## Baseline ladder", f"Best judge: {best_name}; parameters={best['record']['params']}; val threshold={best['record']['threshold']:.6f}; best step={best['record']['best_step']}."]
    for n, b in baselines.items(): report.append(f"{n}: TestA F1={100*b['testA']['f1']:.2f}%, recall={100*b['testA']['recall']:.2f}%; TestB F1={100*b['testB']['f1']:.2f}%, recall={100*b['testB']['recall']:.2f}%.")
    for n in learned:
        m = learned[n]["metrics"]; report.append(f"{n}: TestA F1={100*m['testA']['f1']:.2f}%, recall={100*m['testA']['recall']:.2f}%, P2={100*m['testA']['per_category_accuracy'].get('P2_same_entity_diff_relation',{}).get('accuracy',0):.2f}%; TestB F1={100*m['testB']['f1']:.2f}%, recall={100*m['testB']['recall']:.2f}%, P2={100*m['testB']['per_category_accuracy'].get('P2_same_entity_diff_relation',{}).get('accuracy',0):.2f}%.")
    report += ["", "## Offline gate", f"TestA F1/recall/P2: {100*best_metrics['testA']['f1']:.2f}% / {100*best_metrics['testA']['recall']:.2f}% / {100*best_metrics['testA']['per_category_accuracy'].get('P2_same_entity_diff_relation',{}).get('accuracy',0):.2f}%.", f"TestB F1/recall/P2: {100*best_metrics['testB']['f1']:.2f}% / {100*best_metrics['testB']['recall']:.2f}% / {100*best_metrics['testB']['per_category_accuracy'].get('P2_same_entity_diff_relation',{}).get('accuracy',0):.2f}%.", f"Reconnect: {reconnect.get('status')}.", "", "## Diagnostics", "Geometry, threshold overlap, margin/relative-score analysis, pair probes, calibration, symmetry, and hard-negative results are in the JSON outputs. The P2 category is same entity/different relation and is the primary failure diagnostic.", "", "## Boundary", "Repeated V-update contamination remains a separate unresolved issue and was not modified. This run does not claim eviction, replacement, full-memory new-key policy, or latent reasoning.", "", "## Stop", "O3b-D complete; do not automatically enter O3c."]
    (OUT / "v24_o3b_diag_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "best_judge": best_name, "testA_f1": best_metrics["testA"]["f1"], "testB_f1": best_metrics["testB"]["f1"], "testA_recall": best_metrics["testA"]["recall"], "testB_recall": best_metrics["testB"]["recall"], "testA_P2": best_metrics["testA"]["per_category_accuracy"].get("P2_same_entity_diff_relation", {}).get("accuracy"), "testB_P2": best_metrics["testB"]["per_category_accuracy"].get("P2_same_entity_diff_relation", {}).get("accuracy"), "reconnect": reconnect.get("status"), "wall_clock_s": time.perf_counter() - t0}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
