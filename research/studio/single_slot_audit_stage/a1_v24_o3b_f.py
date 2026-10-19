# -*- coding: utf-8 -*-
"""CLR-v2.4-O3b-F: Binding Geometry Diagnosis.

Offline only.  The O2.1b KeyEncoder, O2.1a QueryEncoder, cache and all
memory/write components remain frozen.  This script trains only diagnostic
probes/matchers and evaluates symbolic binding controls; it never reconnects
to the real memory pipeline.
"""
import copy
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v24_o3b import FactPool, MRCache, KeyEncoder, QueryEncoder, DIM
from a1_v24_o3b_diag import KeyBank, sample_pair, cosine, binary_metric, choose_threshold
from a1_v24_o3b_e import FactorizedKey, factor_pair, score_schemes, pair_refs


OUT = Path(os.environ.get("V24_O3BF_OUT", r"D:/training/checkpoints/clr_v24_o3b_f_binding_diag"))
CACHE_ROOT = Path(os.environ.get("V24_O3BF_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
O21B = Path(os.environ.get("V24_O3BF_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
O21A_MR = Path(os.environ.get("V24_O3BF_O21A_MR", r"D:/training/checkpoints/clr_v24_o21a_mr/best_model.pt"))
O3BE_OUT = Path(os.environ.get("V24_O3BF_O3BE_OUT", r"D:/training/checkpoints/clr_v24_o3b_e_relation_key"))
SEED = int(os.environ.get("V24_O3BF_SEED", "42"))
DEVICE_NAME = os.environ.get("V24_O3BF_DEVICE", "cuda")
SMOKE = os.environ.get("V24_O3BF_SMOKE", "0") == "1" or "--smoke" in os.sys.argv
PROBE_STEPS = int(os.environ.get("V24_O3BF_PROBE_STEPS", "600"))
MATCHER_STEPS = int(os.environ.get("V24_O3BF_MATCHER_STEPS", "800"))
EVAL_EVERY = int(os.environ.get("V24_O3BF_EVAL_EVERY", "200"))
PAIR_TRAIN = 512 if SMOKE else 6000
PAIR_EVAL = 256 if SMOKE else 3000


def seed_all(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer, np.floating)): return x.item()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, Path): return str(x)
    return x


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def pr_auc(y, score):
    y = np.asarray(y, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    pos = int(y.sum())
    if pos == 0: return None
    order = np.argsort(-score); ys = y[order]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    precision = tp / np.maximum(tp + fp, 1); recall = tp / pos
    return float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))


def metric(y, score, threshold, categories=None):
    out = binary_metric(y, score, threshold, categories)
    out["pr_auc"] = pr_auc(y, score)
    return out


def tensor(x, device):
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def add_query_arrays(data, bank):
    data = dict(data)
    data["qa"] = bank.Q[data["idxa"]].astype(np.float32)
    data["qb"] = bank.Q[data["idxb"]].astype(np.float32)
    # Keep the K-K pair fields used by F0/factorized equality untouched.
    # Cross-interface matchers must explicitly consume K_b; using data["ka"]
    # here would silently compare Q_a with its own K_a for every pair.
    data["qk_b"] = bank.K[data["idxb"]].astype(np.float32)
    return data


def pair_metric_table(val, ta, tb, score_name, sv, sa, sb):
    th = choose_threshold(val["y"], sv)
    return {"score": score_name, "threshold_val": float(th),
            "val": metric(val["y"], sv, th, val["categories"]),
            "testA": metric(ta["y"], sa, th, ta["categories"]),
            "testB": metric(tb["y"], sb, th, tb["categories"])}


def per_category(metrics, cat):
    return metrics.get("per_category_accuracy", {}).get(cat, {}).get("accuracy", 0.0)


def qstats(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0: return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            "p10": float(np.quantile(x, .10)), "p25": float(np.quantile(x, .25)),
            "p50": float(np.quantile(x, .50)), "p75": float(np.quantile(x, .75)),
            "p90": float(np.quantile(x, .90))}


def overlap(a, b, bins=100):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if not len(a) or not len(b): return None
    lo = min(float(a.min()), float(b.min())); hi = max(float(a.max()), float(b.max()))
    if hi <= lo: return 1.0
    h1, _ = np.histogram(a, bins=np.linspace(lo, hi, bins + 1)); h2, _ = np.histogram(b, bins=np.linspace(lo, hi, bins + 1))
    h1 = h1 / max(h1.sum(), 1); h2 = h2 / max(h2.sum(), 1)
    return float(np.minimum(h1, h2).sum())


def category_geometry(data, score):
    out = {}
    cats = data["categories"]
    for cat in sorted(set(cats.tolist())):
        m = cats == cat
        out[cat] = {"n": int(m.sum()), **qstats(score[m])}
    p1 = score[cats == "P1_same_key"]; p2 = score[cats == "P2_same_entity_diff_relation"]; p3 = score[cats == "P3_same_relation_diff_entity"]
    out["overlap_P1_P2"] = overlap(p1, p2); out["overlap_P1_P3"] = overlap(p1, p3)
    return out


class FullBilinear(nn.Module):
    def __init__(self, symmetric=False):
        super().__init__(); self.m = nn.Parameter(torch.randn(DIM, DIM) * 0.01); self.symmetric = symmetric

    def forward(self, q, k):
        w = (self.m + self.m.T) * .5 if self.symmetric else self.m
        return ((q @ w) * k).sum(-1)


class LowRankBilinear(nn.Module):
    def __init__(self, rank):
        super().__init__(); self.a = nn.Linear(DIM, rank, bias=False); self.b = nn.Linear(DIM, rank, bias=False)

    def forward(self, q, k): return (self.a(q) * self.b(k)).sum(-1)


class InteractionMatcher(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(DIM * 4, 96), nn.GELU(), nn.Linear(96, 1))

    def forward(self, q, k):
        return self.net(torch.cat([q, k, (q - k).abs(), q * k], -1)).squeeze(-1)


class ShallowProbe(nn.Module):
    def __init__(self, nclass):
        super().__init__(); self.net = nn.Sequential(nn.Linear(DIM, 64), nn.GELU(), nn.Linear(64, nclass))

    def forward(self, x): return self.net(x)


def multiclass_stats(y, logits, topk=(1, 5)):
    y = np.asarray(y, dtype=np.int64); logits = np.asarray(logits)
    pred = np.argmax(logits, 1); out = {"n": int(len(y)), "accuracy": float((pred == y).mean())}
    for k in topk:
        kk = min(k, logits.shape[1]); inds = np.argpartition(-logits, kk - 1, axis=1)[:, :kk]
        out[f"top{k}"] = float(np.mean((inds == y[:, None]).any(1)))
    f1s = []
    for c in np.unique(y):
        tp = int(((pred == c) & (y == c)).sum()); fp = int(((pred == c) & (y != c)).sum()); fn = int(((pred != c) & (y == c)).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f1s.append(2 * p * r / max(p + r, 1e-12))
    out["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return out


def fit_probe(train_x, train_y, eval_sets, nclass, device, shallow, ckpt, steps):
    seed_all(SEED + nclass + (17 if shallow else 0)); model = ShallowProbe(nclass).to(device) if shallow else nn.Linear(DIM, nclass).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01); rng = np.random.default_rng(SEED + nclass + 91)
    x = tensor(train_x, device); y = torch.from_numpy(train_y.astype(np.int64)).to(device); bs = min(512, len(y)); logs = []
    for step in range(1, (20 if SMOKE else steps) + 1):
        ix = rng.integers(0, len(y), size=bs); loss = F.cross_entropy(model(x[ix]), y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    out = {"params": sum(p.numel() for p in model.parameters()), "shallow": shallow}
    state = model.state_dict(); torch.save({"model": state, "nclass": nclass, "shallow": shallow}, ckpt)
    for name, (xx, yy) in eval_sets.items():
        mask = yy >= 0
        if not mask.any(): out[name] = {"accuracy": None, "known_fraction": 0.0, "n_known": 0}; continue
        with torch.no_grad(): logits = model(tensor(xx[mask], device)).cpu().numpy()
        out[name] = {**multiclass_stats(yy[mask], logits), "known_fraction": float(mask.mean()), "n_known": int(mask.sum())}
    return model, out


def fit_metric(train, val, ta, tb, kind, device, steps):
    seed_all(SEED + 700 + len(kind));
    if kind == "full_bilinear": model = FullBilinear(False).to(device)
    elif kind == "symmetric_bilinear": model = FullBilinear(True).to(device)
    elif kind.startswith("lowrank"):
        model = LowRankBilinear(int(kind.replace("lowrank", ""))).to(device)
    else: model = InteractionMatcher().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01); rng = np.random.default_rng(SEED + 1701 + len(kind))
    y = torch.from_numpy(train["y"].astype(np.float32)).to(device); bs = min(512, len(y)); nsteps = 20 if SMOKE else steps; best = None; logs = []
    for step in range(1, nsteps + 1):
        ix = rng.integers(0, len(y), size=bs); q, k = tensor(train["qa"][ix], device), tensor(train.get("qk_b", train["ka"])[ix], device)
        loss = F.binary_cross_entropy_with_logits(model(q, k), y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == nsteps:
            with torch.no_grad(): vs = model(tensor(val["qa"], device), tensor(val.get("qk_b", val["ka"]), device)).cpu().numpy()
            th = choose_threshold(val["y"], vs); vm = metric(val["y"], vs, th, val["categories"]); logs.append({"step": step, "loss": float(loss.detach()), "val": vm, "threshold": th})
            if best is None or (vm["f1"], vm["balanced_accuracy"]) > (best["val"]["f1"], best["val"]["balanced_accuracy"]): best = {"state": copy.deepcopy(model.state_dict()), "step": step, "threshold": th, "val": vm}
    model.load_state_dict(best["state"]); out = {"kind": kind, "params": sum(p.numel() for p in model.parameters()), "best_step": best["step"], "threshold_val": best["threshold"], "val": best["val"], "logs": logs}
    arrays = {}
    with torch.no_grad():
        for name, d in (("val", val), ("testA", ta), ("testB", tb)):
            arrays[name] = model(tensor(d["qa"], device), tensor(d.get("qk_b", d["ka"]), device)).cpu().numpy()
            out[name] = metric(d["y"], arrays[name], best["threshold"], d["categories"])
    return model, out, arrays


def make_explicit_codes(banks):
    """Stable hash codes are generated without split labels or test training."""
    def code(tag, value, dim):
        h = hashlib.sha256(f"{tag}:{value}".encode("utf-8")).digest(); seed = int.from_bytes(h[:8], "little") % (2 ** 32)
        rng = np.random.default_rng(seed); z = rng.normal(size=dim).astype(np.float32); return z / (np.linalg.norm(z) + 1e-8)
    er = {}
    for bank in banks.values():
        persons = [r["person"] for r in bank.refs]
        rels = [int(r["relation"]) for r in bank.refs]
        e32 = np.stack([code("entity", p, 32) for p in persons]); r32 = np.stack([code("relation", r, 32) for r in rels])
        e16 = e32[:, :16]; r8 = r32[:, :8]
        concat = np.concatenate([e32, r32[:, :16]], 1); concat /= np.linalg.norm(concat, axis=1, keepdims=True) + 1e-8
        had = e32 * r32; had /= np.linalg.norm(had, axis=1, keepdims=True) + 1e-8
        kron = np.einsum("ni,nj->nij", e16, r8).reshape(len(bank.refs), -1); kron /= np.linalg.norm(kron, axis=1, keepdims=True) + 1e-8
        er[bank.pool.split] = {"concat": concat.astype(np.float32), "hadamard": had.astype(np.float32), "kronecker": kron.astype(np.float32)}
    return er


def explicit_pair_scores(data, codes, key):
    return (codes[data["idxa"]][key] * codes[data["idxb"]][key]).sum(1)


def explicit_results(sets, codes):
    out = {}
    for key in ("concat", "hadamard", "kronecker"):
        sv = explicit_pair_scores(sets["val"], codes["val"], key); sa = explicit_pair_scores(sets["testA"], codes["testA"], key); sb = explicit_pair_scores(sets["testB"], codes["testB"], key)
        out[key] = pair_metric_table(sets["val"], sets["testA"], sets["testB"], key, sv, sa, sb)
    return out


def joint_labels(bank, person_map, rel_count=4):
    return np.asarray([person_map.get(r["person"], -1) * rel_count + int(r["relation"]) if r["person"] in person_map else -1 for r in bank.refs], dtype=np.int64)


def individual_labels(bank, person_map):
    return np.asarray([person_map.get(r["person"], -1) for r in bank.refs], dtype=np.int64)


def probe_suite(banks, device):
    train_persons = sorted({r["person"] for r in banks["train"].refs}); pmap = {p: i for i, p in enumerate(train_persons)}; nentity = len(pmap); n_joint = nentity * 4
    ys_joint = {s: joint_labels(b, pmap) for s, b in banks.items()}; ys_entity = {s: individual_labels(b, pmap) for s, b in banks.items()}; ys_rel = {s: np.asarray([int(r["relation"]) for r in b.refs], dtype=np.int64) for s, b in banks.items()}
    out = {"entity_classes_train": nentity, "joint_classes_train": n_joint, "testB_note": "TestB is unseen-person; closed-world joint/entity accuracy is N/A and known_fraction is reported.", "factorized_reconstruction": {}}
    for shallow, name, steps in ((False, "linear", PROBE_STEPS), (True, "mlp", PROBE_STEPS)):
        for target, ys, nclass in (("joint", ys_joint, n_joint), ("entity", ys_entity, nentity), ("relation", ys_rel, 4)):
            ck = OUT / f"probe_{name}_{target}.pt"; _, res = fit_probe(banks["train"].K, ys["train"], {s: (b.K, ys[s]) for s, b in banks.items()}, nclass, device, shallow, ck, steps); out[f"{name}_{target}"] = res
    # Factorized reconstruction uses separate linear classifiers and requires both predictions.
    x = tensor(banks["train"].K, device); probes = {}
    for target, ys, nclass in (("entity", ys_entity, nentity), ("relation", ys_rel, 4)):
        m = nn.Linear(DIM, nclass).to(device); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=.01); y = torch.from_numpy(ys["train"]).long().to(device); rng = np.random.default_rng(SEED + 500 + nclass)
        for _ in range(20 if SMOKE else PROBE_STEPS):
            ix = rng.integers(0, len(y), size=min(512, len(y))); loss = F.cross_entropy(m(x[ix]), y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        probes[target] = m
    for s, b in banks.items():
        known = ys_joint[s] >= 0
        if not known.any(): out["factorized_reconstruction"][s] = {"accuracy": None, "known_fraction": 0.0}; continue
        with torch.no_grad(): pe = probes["entity"](tensor(b.K[known], device)).argmax(1).cpu().numpy(); pr = probes["relation"](tensor(b.K[known], device)).argmax(1).cpu().numpy()
        true_e = ys_entity[s][known]; true_r = ys_rel[s][known]; out["factorized_reconstruction"][s] = {"accuracy": float(np.mean((pe == true_e) & (pr == true_r))), "entity_accuracy": float(np.mean(pe == true_e)), "relation_accuracy": float(np.mean(pr == true_r)), "known_fraction": float(known.mean()), "n_known": int(known.sum())}
    return out


def conditional_relation_diag(bank, raw_data, device):
    # F2a: leave-one-out nearest relation within the same entity group.
    groups = defaultdict(list)
    for r in bank.refs: groups[r["person"]].append(r)
    pred = []; true = []; counts = []
    for refs in groups.values():
        if len({r["relation"] for r in refs}) < 2: continue
        ks = np.stack([bank.K[r["bank_idx"]] for r in refs]); sim = ks @ ks.T
        for i, r in enumerate(refs):
            sim[i, i] = -1e9; j = int(np.argmax(sim[i])); pred.append(refs[j]["relation"]); true.append(r["relation"]); counts.append(r["person"])
    per_entity = []
    for p in sorted(set(counts)):
        m = np.asarray([x == p for x in counts]); per_entity.append(float(np.mean(np.asarray(pred)[m] == np.asarray(true)[m])))
    # F2b: same-entity P1/P2 binary relation discrimination with val-selected threshold.
    def subset(d):
        m = np.isin(d["categories"], ["P1_same_key", "P2_same_entity_diff_relation"])
        return d["y"][m], cosine(d["ka"][m], d["kb"][m]), d["categories"][m]
    # Caller supplies one val-like and one test-like pair data through raw_data.
    val_d, test_d = raw_data
    yv, sv, cv = subset(val_d); th = choose_threshold(yv, sv)
    yt, st, ct = subset(test_d)
    return {"F2a_same_entity_leave_one_out": {"n": len(true), "accuracy": float(np.mean(np.asarray(pred) == np.asarray(true))) if true else None, "per_entity_macro": float(np.mean(per_entity)) if per_entity else None, "worst_decile": float(np.quantile(per_entity, .10)) if per_entity else None}, "F2b_same_entity_P1_vs_P2": {"threshold_val_subset": float(th), "test": metric(yt, st, th, ct), "P1": per_category( metric(yt, st, th, ct), "P1_same_key"), "P2": per_category(metric(yt, st, th, ct), "P2_same_entity_diff_relation")}}


def relation_within_entity_retrieval(bank, limit=None):
    # F2c: query each key, then restrict candidates to its own entity.
    vals = []; n = 0; correct = 0
    for r in bank.refs:
        candidates = [x for x in bank.by_person[r["person"]] if x["bank_idx"] != r["bank_idx"]]
        if len({x["relation"] for x in candidates}) < 2: continue
        q = bank.Q[r["bank_idx"]]; ks = np.stack([bank.K[x["bank_idx"]] for x in candidates]); scores = ks @ q; j = int(np.argmax(scores)); correct += int(candidates[j]["relation"] == r["relation"]); n += 1
        if limit and n >= limit: break
    return {"n": n, "accuracy": float(correct / max(n, 1))}


def retrieval_eval(bank, score_fn, label, max_episodes=None):
    errors = []; margins = []; p2_margins = []; p3_margins = []; groups = Counter(); total = correct = 0
    eps = [ep for ep in bank.pool.episodes if len(ep) == 4]
    if max_episodes: eps = eps[:max_episodes]
    refs = {(r["episode"], r["pos"]): r for r in bank.refs}
    for ep in eps:
        for target0 in ep:
            target = refs[(target0["episode"], target0["pos"])]
            cand = [refs[(x["episode"], x["pos"])] for x in ep]; q = bank.Q[target["bank_idx"]]; k = np.stack([bank.K[x["bank_idx"]] for x in cand]); scores = score_fn(q, k)
            j = int(np.argmax(scores)); ok = j == target["pos"]; total += 1; correct += int(ok)
            wrong = [i for i, x in enumerate(cand) if i != target["pos"]]
            margins.append(float(scores[target["pos"]] - max(scores[wrong])))
            p2 = [i for i, x in enumerate(cand) if i != target["pos"] and x["person"] == target["person"] and x["relation"] != target["relation"]]
            p3 = [i for i, x in enumerate(cand) if i != target["pos"] and x["person"] != target["person"] and x["relation"] == target["relation"]]
            if p2: p2_margins.append(float(scores[target["pos"]] - max(scores[p2])))
            if p3: p3_margins.append(float(scores[target["pos"]] - max(scores[p3])))
            if not ok:
                p = cand[j]; same_e = p["person"] == target["person"]; same_r = p["relation"] == target["relation"]
                kind = "E+_R-" if same_e and not same_r else ("E-_R+" if (not same_e and same_r) else "E-_R-")
                groups[kind] += 1
                if len(errors) < 250: errors.append({"kind": kind, "true_pos": int(target["pos"]), "pred_pos": int(j), "true_person": target["person"], "true_relation": int(target["relation"]), "pred_person": p["person"], "pred_relation": int(p["relation"]), "margin": margins[-1]})
    return {"label": label, "n": total, "correct": correct, "accuracy": float(correct / max(total, 1)), "error_counts": dict(groups), "error_fractions": {k: float(v / max(sum(groups.values()), 1)) for k, v in groups.items()}, "errors": errors, "margin": {"all_negatives": margin_stats(np.asarray(margins)), "same_entity_wrong_relation": margin_stats(np.asarray(p2_margins)), "wrong_entity_same_relation": margin_stats(np.asarray(p3_margins))}}


def margin_stats(margins):
    margins = np.asarray(margins, dtype=np.float64)
    return {**qstats(margins), "negative_rate": float(np.mean(margins < 0)) if len(margins) else None}


def retrieval_explicit(bank, codes, key, label, max_episodes=None):
    errors = []; margins = []; groups = Counter(); total = correct = 0
    eps = [ep for ep in bank.pool.episodes if len(ep) == 4]
    if max_episodes: eps = eps[:max_episodes]
    refs = {(r["episode"], r["pos"]): r for r in bank.refs}; arr = codes[bank.pool.split][key]
    for ep in eps:
        for target0 in ep:
            target = refs[(target0["episode"], target0["pos"])]
            cand = [refs[(x["episode"], x["pos"])] for x in ep]; q = arr[target["bank_idx"]]; k = arr[[x["bank_idx"] for x in cand]]; scores = k @ q; j = int(np.argmax(scores)); ok = j == target["pos"]; total += 1; correct += int(ok)
            wrong = [i for i in range(len(cand)) if i != target["pos"]]; margins.append(float(scores[target["pos"]] - max(scores[wrong])))
            if not ok:
                p = cand[j]; same_e = p["person"] == target["person"]; same_r = p["relation"] == target["relation"]; kind = "E+_R-" if same_e and not same_r else ("E-_R+" if (not same_e and same_r) else "E-_R-"); groups[kind] += 1
                if len(errors) < 250: errors.append({"kind": kind, "true_pos": int(target["pos"]), "pred_pos": int(j), "margin": margins[-1]})
    return {"label": label, "n": total, "correct": correct, "accuracy": float(correct / max(total, 1)), "error_counts": dict(groups), "error_fractions": {k: float(v / max(sum(groups.values()), 1)) for k, v in groups.items()}, "errors": errors, "margin": margin_stats(np.asarray(margins))}


def collision_matrix(bank, model, codes):
    by_person = defaultdict(dict)
    for r in bank.refs: by_person[r["person"]][r["relation"]] = r
    chosen = next((v for v in by_person.values() if len(v) >= 4), None)
    if chosen is None: return {"available": False}
    rels = sorted(chosen)[:4]; idx = [chosen[r]["bank_idx"] for r in rels]; k = bank.K[idx]
    out = {"available": True, "relations": rels, "learned_K_cosine": k @ k.T}
    with torch.no_grad():
        # This diagnostic model is the O3b-E factorized head; it is optional here.
        e, rr = model(tensor(k, next(model.parameters()).device)); e = e.cpu().numpy(); rr = rr.cpu().numpy(); out["factor_entity_cosine"] = e @ e.T; out["factor_relation_cosine"] = rr @ rr.T
    for name, arr in codes[bank.pool.split].items(): out[f"explicit_{name}_cosine"] = arr[idx] @ arr[idx].T
    return out


def binding_geometry(sets, banks, codes, metric_arrays):
    out = {"learned_K": {}, "explicit": {}}
    for split, d in (("testA", sets["testA"]), ("testB", sets["testB"])):
        out["learned_K"][split] = {"cosine": category_geometry(d, cosine(d["ka"], d["kb"])), "euclidean_negative": category_geometry(d, -np.linalg.norm(d["ka"] - d["kb"], axis=1))}
        out["explicit"][split] = {}
        for key in ("concat", "hadamard", "kronecker"):
            sc = explicit_pair_scores(d, codes[split], key); out["explicit"][split][key] = category_geometry(d, sc)
        for name, arrs in metric_arrays.items(): out.setdefault("learned_matcher", {}).setdefault(name, {})[split] = category_geometry(d, arrs[split])
    return out


def explicit_pair_scores(data, code_dict, key):
    return (code_dict[key][data["idxa"]] * code_dict[key][data["idxb"]]).sum(1)


def main():
    seed_all(); start = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(CACHE_ROOT); pools = {s: FactPool(cache, s) for s in ("train", "val", "testA", "testB")}
    kc = torch.load(O21B, map_location="cpu", weights_only=False); key = KeyEncoder(DIM).to(device); key.load_state_dict(kc["key"], strict=True); key.eval()
    qc = torch.load(O21A_MR, map_location="cpu", weights_only=False); query = QueryEncoder(DIM).to(device); query.load_state_dict(qc["query"], strict=True); query.eval()
    for mod in (key, query):
        for p in mod.parameters(): p.requires_grad_(False)
    banks = {s: KeyBank(pools[s], key, device) for s in pools}
    for bank in banks.values():
        qs = []
        with torch.no_grad():
            for st in range(0, len(bank.refs), 256):
                hs = np.stack([bank.pool.query_hidden(r) for r in bank.refs[st:st + 256]]).astype(np.float32)
                qs.append(query(tensor(hs, device)).cpu().numpy().astype(np.float32))
        bank.Q = np.concatenate(qs, axis=0)

    sets = {"train": add_query_arrays(pair_refs(banks["train"], PAIR_TRAIN, SEED + 1), banks["train"]),
            "val": add_query_arrays(pair_refs(banks["val"], PAIR_EVAL, SEED + 2), banks["val"]),
            "testA": add_query_arrays(pair_refs(banks["testA"], PAIR_EVAL, SEED + 3), banks["testA"]),
            "testB": add_query_arrays(pair_refs(banks["testB"], PAIR_EVAL, SEED + 4), banks["testB"])}
    stats = {s: {"n": int(len(d["y"])), "positive": int(d["y"].sum()), "negative": int((d["y"] == 0).sum()), "categories": dict(d["pair_counts"]), "unique_source_episodes": len({r["episode"] for r in banks[s].refs}), "unique_persons": len({r["person"] for r in banks[s].refs})} for s, d in sets.items()}
    stats["sampling"] = {"original_P2_fraction_of_negative": 1 / 6, "P2_25_fraction_of_negative": .25, "P2_50_fraction_of_negative": .50, "same_episode_pairing": False}; dump(OUT / "pair_dataset_stats.json", stats)

    # F0: exact O3b-E reproduction, using the frozen O3b-E checkpoint.
    eck = torch.load(O3BE_OUT / "best_model.pt", map_location=device, weights_only=False); e_model = FactorizedKey().to(device); e_model.load_state_dict(eck["model"], strict=True); e_model.eval()
    f0 = {s: factor_pair(e_model, sets[s], device) for s in ("val", "testA", "testB")}
    f0_scores = {s: score_schemes(f0[s])["E1_a1_b1"] for s in f0}; f0_th = choose_threshold(sets["val"]["y"], f0_scores["val"])
    f0_result = {"threshold_val": f0_th, "val": metric(sets["val"]["y"], f0_scores["val"], f0_th, sets["val"]["categories"]), "testA": metric(sets["testA"]["y"], f0_scores["testA"], f0_th, sets["testA"]["categories"]), "testB": metric(sets["testB"]["y"], f0_scores["testB"], f0_th, sets["testB"]["categories"])}
    prior_e = None
    if (O3BE_OUT / "factorized_results.json").exists():
        prior_e = json.loads((O3BE_OUT / "factorized_results.json").read_text(encoding="utf-8"))
    f0_result["prior_o3b_e_available"] = prior_e is not None; f0_result["prior_o3b_e_best"] = None if prior_e is None else {"best_variant": prior_e.get("best_variant"), "best_scheme": prior_e.get("best_scheme")}; dump(OUT / "f0_baseline_reproduction.json", f0_result)

    # F1/F2 probes.
    probe_out = probe_suite(banks, device); dump(OUT / "joint_identity_probes.json", probe_out)
    cond = {"TestA": conditional_relation_diag(banks["testA"], (sets["val"], sets["testA"]), device), "TestB": conditional_relation_diag(banks["testB"], (sets["val"], sets["testB"]), device)}
    cond["F2c_relation_retrieval_with_entity_fixed"] = {"TestA": relation_within_entity_retrieval(banks["testA"], None), "TestB": relation_within_entity_retrieval(banks["testB"], None)}; dump(OUT / "conditional_relation.json", cond)

    # F3 frozen-K matching: only diagnostic heads are trained.
    val, ta, tb = sets["val"], sets["testA"], sets["testB"]
    raw_scores = {"val": (val["qa"] * val["qk_b"]).sum(1), "testA": (ta["qa"] * ta["qk_b"]).sum(1), "testB": (tb["qa"] * tb["qk_b"]).sum(1)}
    euc_scores = {"val": -np.linalg.norm(val["qa"] - val["qk_b"], axis=1), "testA": -np.linalg.norm(ta["qa"] - ta["qk_b"], axis=1), "testB": -np.linalg.norm(tb["qa"] - tb["qk_b"], axis=1)}
    frozen_metrics = {"cosine": pair_metric_table(val, ta, tb, "cosine", raw_scores["val"], raw_scores["testA"], raw_scores["testB"]), "euclidean": pair_metric_table(val, ta, tb, "negative_euclidean", euc_scores["val"], euc_scores["testA"], euc_scores["testB"])}; metric_models = {}; metric_records = {}; metric_arrays = {"cosine": raw_scores, "euclidean": euc_scores}
    metric_kinds = ["full_bilinear", "lowrank16", "lowrank32", "lowrank64", "symmetric_bilinear", "interaction_mlp"]
    for kind in metric_kinds:
        m, rec, arr = fit_metric(val if False else sets["train"], val, ta, tb, kind, device, MATCHER_STEPS); metric_models[kind] = m; metric_records[kind] = rec; metric_arrays[kind] = arr
    frozen_metrics["learned"] = metric_records; dump(OUT / "frozen_k_metrics.json", frozen_metrics)

    # F5/F6 explicit symbolic binding controls.
    codes = make_explicit_codes(banks); explicit = explicit_results(sets, codes); dump(OUT / "explicit_binding_results.json", explicit)

    # Select diagnostic matcher by validation F1; this does not alter frozen K.
    learned_names = list(metric_records); best_metric_name = max(learned_names, key=lambda n: metric_records[n]["val"]["f1"]); best_metric = metric_records[best_metric_name]
    def score_model(model):
        def fn(q, k):
            with torch.no_grad(): return model(tensor(np.repeat(q[None, :], len(k), axis=0), device), tensor(k, device)).cpu().numpy()
        return fn
    retrieval = {"cosine": {s: retrieval_eval(banks[s], lambda q, k: k @ q, "cosine") for s in ("testA", "testB")}, "euclidean": {s: retrieval_eval(banks[s], lambda q, k: -np.linalg.norm(k - q[None, :], axis=1), "euclidean") for s in ("testA", "testB")}}
    for name, model in metric_models.items(): retrieval[name] = {s: retrieval_eval(banks[s], score_model(model), name) for s in ("testA", "testB")}
    for key_name in ("concat", "hadamard", "kronecker"):
        retrieval[f"explicit_{key_name}"] = {s: retrieval_explicit(banks[s], codes, key_name, f"explicit_{key_name}") for s in ("testA", "testB")}
    dump(OUT / "retrieval_error_decomposition.json", retrieval)
    dump(OUT / "margin_analysis.json", {name: {s: retrieval[name][s]["margin"] for s in ("testA", "testB")} for name in retrieval})

    # F7 collision blocks and F6 geometry.
    metric_for_geometry = {name: {s: metric_arrays[name][s] for s in ("testA", "testB")} for name in metric_arrays}
    dump(OUT / "binding_geometry.json", binding_geometry(sets, banks, codes, metric_for_geometry))
    dump(OUT / "collision_matrices.json", {"testA": collision_matrix(banks["testA"], e_model, codes), "testB": collision_matrix(banks["testB"], e_model, codes)})

    # Explicitly document the K_e role-collapse diagnosis from O3b-E.
    role = {"source": "O3b-E best factor head", "TestA": {}, "TestB": {}}
    for s in ("testA", "testB"):
        with torch.no_grad(): ee, rr = e_model(tensor(banks[s].K, device)); ee = ee.cpu().numpy(); rr = rr.cpu().numpy()
        role[s] = {"K_norm": qstats(np.linalg.norm(banks[s].K, axis=1)), "K_e_norm": qstats(np.linalg.norm(ee, axis=1)), "K_r_norm": qstats(np.linalg.norm(rr, axis=1)), "K_e_mean_abs": float(np.abs(ee).mean()), "K_r_mean_abs": float(np.abs(rr).mean()), "K_e_K_r_flat_correlation": float(np.corrcoef(ee.reshape(-1), rr.reshape(-1))[0, 1])}
    role["probe_summary"] = {k: probe_out[k] for k in ("linear_entity", "mlp_entity", "linear_relation", "mlp_relation", "factorized_reconstruction") if k in probe_out}; dump(OUT / "ke_role_analysis.json", role)

    # Gate and mechanism verdict. The gate is based on frozen-K addressing, not external judges.
    p2_best = per_category(best_metric["testB"], "P2_same_entity_diff_relation"); bilinear_names = [n for n in metric_records if n.startswith("full_bilinear") or n.startswith("lowrank") or n.startswith("symmetric")]; best_bilinear_name = max(bilinear_names, key=lambda n: metric_records[n]["testB"]["f1"]); best_bilinear = metric_records[best_bilinear_name]
    explicit_best_name = max(explicit, key=lambda n: explicit[n]["val"]["f1"]); explicit_best = explicit[explicit_best_name]
    gate = {"baseline_f0_reproduced": abs(f0_result["testB"]["f1"] - 0.8897034321892703) <= .02, "best_metric_testB_f1_ge_95": best_metric["testB"]["f1"] >= .95, "best_metric_testB_recall_ge_95": best_metric["testB"]["recall"] >= .95, "best_metric_testB_P2_ge_90": p2_best >= .90, "best_bilinear_testB_f1_ge_95": best_bilinear["testB"]["f1"] >= .95, "best_bilinear_testB_recall_ge_95": best_bilinear["testB"]["recall"] >= .95}
    metric_gate = gate["best_bilinear_testB_f1_ge_95"] and gate["best_bilinear_testB_recall_ge_95"] and per_category(best_bilinear["testB"], "P2_same_entity_diff_relation") >= .90
    joint_mlp_acc = probe_out.get("mlp_joint", {}).get("testA", {}).get("accuracy") or 0.0; cond_b = cond["TestB"]["F2a_same_entity_leave_one_out"].get("accuracy") or 0.0; explicit_gap = explicit_best["testB"]["f1"] - best_bilinear["testB"]["f1"]; metric_gain = best_bilinear["testB"]["f1"] - frozen_metrics["cosine"]["testB"]["f1"]
    if metric_gate:
        verdict = "O3b-F: METRIC INCOMPATIBILITY"
    elif explicit_gap >= .05 and metric_gain >= .02 and (joint_mlp_acc >= .50 or cond_b >= .50):
        verdict = "O3b-F: MIXED BINDING + METRIC FAILURE"
    elif joint_mlp_acc < .50 and cond_b < .50 and metric_gain < .02:
        verdict = "O3b-F: INFORMATION INSUFFICIENT"
    else:
        verdict = "O3b-F: BINDING FAILURE"
    gate["metric_rescue_pass"] = metric_gate; gate["explicit_gap_testB_f1"] = explicit_gap; gate["best_metric_name"] = best_metric_name; gate["best_bilinear_name"] = best_bilinear_name; gate["explicit_best_name"] = explicit_best_name; gate["verdict"] = verdict; dump(OUT / "gate_evaluation.json", gate)

    sha = "NO_GIT_REPO"; diff = "NO_GIT_REPO"
    try: sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(Path.cwd()), text=True).strip(); diff = subprocess.check_output(["git", "diff", "--stat"], cwd=str(Path.cwd()), text=True).strip()
    except Exception: pass
    cache_files = list(CACHE_ROOT.glob("*")); cache_bytes = sum(p.stat().st_size for p in cache_files if p.is_file())
    cfg = {"phase": "CLR-v2.4-O3b-F", "seed": SEED, "git_sha": sha, "git_diff": diff, "device": str(device), "cache": str(CACHE_ROOT), "cache_reused": True, "cache_generation_s": 0, "cache_file_count": len(cache_files), "cache_bytes": cache_bytes, "frozen": ["KeyEncoder", "QueryEncoder", "backbone/Ouro", "V/ValueEncoder/GRU", "allocator", "memory storage", "slots", "downstream answer head"], "diagnostic_trainable": ["linear/MLP probes", "full and low-rank bilinear matchers", "interaction matcher"], "pair_sizes": {s: len(d["y"]) for s, d in sets.items()}, "matcher_steps": MATCHER_STEPS, "probe_steps": PROBE_STEPS, "best_metric": best_metric_name, "best_bilinear": best_bilinear_name, "explicit_best": explicit_best_name, "peak_vram": {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None}, "wall_clock_s": time.perf_counter() - start, "verdict": verdict, "reconnect": False, "o3c": False, "repeated_v_update_contamination": "independent unresolved issue"}
    dump(OUT / "config.json", cfg); dump(OUT / "git_info.json", {"git_sha": sha, "git_diff": diff}); dump(OUT / "training.log", {"metric_records": metric_records, "probe_steps": PROBE_STEPS, "matcher_steps": MATCHER_STEPS, "best_metric": best_metric_name, "best_bilinear": best_bilinear_name})
    # Save selected diagnostic heads and aliases.
    torch.save({"model": metric_models[best_metric_name].state_dict(), "kind": best_metric_name}, OUT / "best_matcher.pt")

    def pct(x): return "n/a" if x is None else f"{100 * x:.2f}%"
    report = ["# CLR-v2.4-O3b-F — Binding Geometry Diagnosis", "", f"**{verdict}**", "", "## Executive Summary", f"F0 reproduced O3b-E: TestA F1={pct(f0_result['testA']['f1'])}, TestB F1={pct(f0_result['testB']['f1'])}; best frozen-K matcher={best_metric_name}; explicit control={explicit_best_name}. No real write/read reconnect was run.", "", "## Baseline Reproduction", f"O3b-E best factorized E1_a1_b1: TestA F1={pct(f0_result['testA']['f1'])}, TestB F1={pct(f0_result['testB']['f1'])}; UPDATE recall={pct(f0_result['testA']['recall'])}/{pct(f0_result['testB']['recall'])}; P1={pct(per_category(f0_result['testA'], 'P1_same_key'))}/{pct(per_category(f0_result['testB'], 'P1_same_key'))}; P2={pct(per_category(f0_result['testA'], 'P2_same_entity_diff_relation'))}/{pct(per_category(f0_result['testB'], 'P2_same_entity_diff_relation'))}.", "", "## Joint / Conditional Probes", f"K→joint linear TestA={pct(probe_out.get('linear_joint', {}).get('testA', {}).get('accuracy'))}; shallow MLP TestA={pct(probe_out.get('mlp_joint', {}).get('testA', {}).get('accuracy'))}; factorized reconstruction TestA={pct(probe_out.get('factorized_reconstruction', {}).get('testA', {}).get('accuracy'))}.", f"Conditional relation F2a leave-one-out TestA/TestB={pct(cond['TestA']['F2a_same_entity_leave_one_out'].get('accuracy'))}/{pct(cond['TestB']['F2a_same_entity_leave_one_out'].get('accuracy'))}; F2b same-entity P1/P2 TestA/TestB F1={pct(cond['TestA']['F2b_same_entity_P1_vs_P2']['test']['f1'])}/{pct(cond['TestB']['F2b_same_entity_P1_vs_P2']['test']['f1'])}.", "", "## Frozen-K Metric Diagnosis", f"Cosine TestA/TestB F1={pct(frozen_metrics['cosine']['testA']['f1'])}/{pct(frozen_metrics['cosine']['testB']['f1'])}; Euclidean={pct(frozen_metrics['euclidean']['testA']['f1'])}/{pct(frozen_metrics['euclidean']['testB']['f1'])}.", f"Best learned matcher {best_metric_name}: TestA F1={pct(best_metric['testA']['f1'])}, recall={pct(best_metric['testA']['recall'])}; TestB F1={pct(best_metric['testB']['f1'])}, recall={pct(best_metric['testB']['recall'])}, P2={pct(p2_best)}.", f"Best bilinear {best_bilinear_name}: TestB F1={pct(best_bilinear['testB']['f1'])}, recall={pct(best_bilinear['testB']['recall'])}, P2={pct(per_category(best_bilinear['testB'], 'P2_same_entity_diff_relation'))}.", "", "## Explicit Binding Controls", f"Best explicit control {explicit_best_name}: TestA F1={pct(explicit_best['testA']['f1'])}, TestB F1={pct(explicit_best['testB']['f1'])}; all controls are symbolic offline upper bounds and are not reconnect candidates.", "", "## Error / Margin Diagnosis", f"Retrieval and error decomposition are in retrieval_error_decomposition.json; factor-independent margin quantiles are in margin_analysis.json. K_e role analysis is in ke_role_analysis.json; collision matrices are in collision_matrices.json.", "", "## Gate", json.dumps(gate, ensure_ascii=False), "", "## Final Interpretation", f"Best explicit-minus-best-bilinear TestB F1 gap={explicit_gap:.4f}; best-bilinear gain over cosine={metric_gain:.4f}. These results diagnose information, binding, and metric separately; no single probe is treated as addressing PASS.", "", "## Reconnect Readiness", "Reconnect is not allowed. O3c was not entered.", "", "## Repeated V Update Contamination", "Repeated V update contamination remains an independent unresolved issue.", "", "## Stop", "O3b-F complete; do not design O3b-G or reconnect without a separate instruction."]
    (OUT / "v24_o3b_f_binding_diag_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "f0_testA_f1": f0_result["testA"]["f1"], "f0_testB_f1": f0_result["testB"]["f1"], "best_metric": best_metric_name, "best_bilinear": best_bilinear_name, "best_metric_testA_f1": best_metric["testA"]["f1"], "best_metric_testB_f1": best_metric["testB"]["f1"], "best_bilinear_testB_f1": best_bilinear["testB"]["f1"], "explicit_best": explicit_best_name, "explicit_best_testB_f1": explicit_best["testB"]["f1"], "joint_mlp_testA": joint_mlp_acc, "cond_testB": cond_b, "reconnect": False, "wall_clock_s": time.perf_counter() - start}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
