# -*- coding: utf-8 -*-
"""CLR-v2.4-O3b-E — Relation-Aware Factorized Key.

This is an offline representation/equality experiment.  The O2.1b
KeyEncoder and O3b-D pipeline are frozen.  A tied, small factor head maps the
frozen 64-d key/query spaces to K_e and K_r; no write/read pipeline is
reconnected in this phase.
"""
import copy
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

from a1_v24_o3b import FactPool, MRCache, KeyEncoder, QueryEncoder, DIM, SLOTS
from a1_v24_o3b_diag import (
    KeyBank, sample_pair, cosine, binary_metric, choose_threshold,
    auc_binary, ece_brier, clean as clean_diag,
)


OUT = Path(os.environ.get("V24_O3BE_OUT", r"D:/training/checkpoints/clr_v24_o3b_e_relation_key"))
CACHE_ROOT = Path(os.environ.get("V24_O3BE_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
O21B = Path(os.environ.get("V24_O3BE_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
O3BD_OUT = Path(os.environ.get("V24_O3BE_O3BD_OUT", r"D:/training/checkpoints/clr_v24_o3b_diag"))
SEED = int(os.environ.get("V24_O3BE_SEED", "42"))
STEPS = int(os.environ.get("V24_O3BE_STEPS", "3000"))
EVAL_EVERY = int(os.environ.get("V24_O3BE_EVAL_EVERY", "200"))
DEVICE_NAME = os.environ.get("V24_O3BE_DEVICE", "cuda")
SMOKE = os.environ.get("V24_O3BE_SMOKE", "0") == "1" or "--smoke" in os.sys.argv
PAIR_TRAIN = 512 if SMOKE else 6000
PAIR_EVAL = 256 if SMOKE else 3000
FACTOR_DIM = 32
N_REL = 4


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


def pr_auc(y, prob):
    y = np.asarray(y, dtype=np.int64); prob = np.asarray(prob, dtype=np.float64)
    pos = int(y.sum())
    if pos == 0: return None
    order = np.argsort(-prob); ys = y[order]; tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    precision = tp / np.maximum(tp + fp, 1); recall = tp / pos
    return float(np.trapezoid(np.r_[1.0, precision], np.r_[0.0, recall]))


def metric(y, prob, threshold=0.5, categories=None):
    m = binary_metric(y, prob, threshold, categories)
    # binary_metric from O3b-D returns f1/precision/recall; PR-AUC is added here.
    m["pr_auc"] = pr_auc(y, prob)
    return m


def pair_refs(bank, count, seed, p2_fraction=None):
    """Generate the O3b-D pair protocol, optionally changing only P2 share."""
    rng = np.random.default_rng(seed); npos = count // 2; nneg = count - npos
    pairs = []
    for _ in range(npos):
        a, b = sample_pair(bank, rng, "same_key"); pairs.append((a, b, 1, "P1_same_key"))
    neg_names = {
        "same_entity_diff_relation": "P2_same_entity_diff_relation",
        "same_relation_diff_entity": "P3_same_relation_diff_entity",
        "same_value_diff_key": "P4_same_value_diff_key",
        "fully_different": "P5_fully_different",
        "lexical_hard_negative": "P6_lexical_hard_negative",
        "similar_entity_names": "P7_similar_entity_names",
    }
    cats = list(neg_names.keys())
    if p2_fraction is None:
        neg_cats = [cats[i % len(cats)] for i in range(nneg)]
    else:
        p2_n = int(round(nneg * float(p2_fraction)))
        rest = nneg - p2_n; others = [c for c in cats if c != "same_entity_diff_relation"]
        neg_cats = ["same_entity_diff_relation"] * p2_n + [others[i % len(others)] for i in range(rest)]
        rng.shuffle(neg_cats)
    for cat in neg_cats:
        a, b = sample_pair(bank, rng, cat); pairs.append((a, b, 0, neg_names[cat]))
    rng.shuffle(pairs)
    idxa = np.asarray([a["bank_idx"] for a, _, _, _ in pairs], dtype=np.int64); idxb = np.asarray([b["bank_idx"] for _, b, _, _ in pairs], dtype=np.int64)
    return {"ka": bank.K[idxa].astype(np.float32), "kb": bank.K[idxb].astype(np.float32),
            "y": np.asarray([p[2] for p in pairs], dtype=np.int64),
            "categories": np.asarray([p[3] for p in pairs]),
            "ra": np.asarray([a["relation"] for a, _, _, _ in pairs], dtype=np.int64),
            "rb": np.asarray([b["relation"] for _, b, _, _ in pairs], dtype=np.int64),
            "entity_same": np.asarray([int(a["person"] == b["person"]) for a, b, _, _ in pairs], dtype=np.int64),
            "relation_same": np.asarray([int(a["relation"] == b["relation"]) for a, b, _, _ in pairs], dtype=np.int64),
            "idxa": idxa, "idxb": idxb, "pair_counts": Counter([p[3] for p in pairs]),
            "pair_meta": [(dict(a), dict(b)) for a, b, _, _ in pairs]}


class FactorizedKey(nn.Module):
    """Small tied f_e/f_r applied to frozen K and frozen Q spaces."""
    def __init__(self):
        super().__init__()
        self.entity = nn.Linear(DIM, FACTOR_DIM)
        self.relation = nn.Linear(DIM, FACTOR_DIM)
        self.relation_classifier = nn.Linear(FACTOR_DIM, N_REL)

    def factors(self, z):
        ke = F.normalize(self.entity(z), dim=-1)
        kr = F.normalize(self.relation(z), dim=-1)
        return ke, kr

    def forward(self, z): return self.factors(z)


def t(x, device): return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


@torch.no_grad()
def factor_pair(model, data, device):
    model.eval(); out = {"se": [], "sr": [], "kea": [], "keb": [], "kra": [], "krb": []}
    for st in range(0, len(data["y"]), 1024):
        a, b = t(data["ka"][st:st + 1024], device), t(data["kb"][st:st + 1024], device); ea, ra = model(a); eb, rb = model(b)
        out["se"].append((ea * eb).sum(-1).cpu().numpy()); out["sr"].append((ra * rb).sum(-1).cpu().numpy())
        for k, v in (("kea", ea), ("keb", eb), ("kra", ra), ("krb", rb)): out[k].append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def score_schemes(f):
    se, sr = f["se"], f["sr"]
    out = {"E1_a1_b1": se + sr, "E1_a1_b2": se + 2 * sr, "E1_a1_b4": se + 4 * sr, "E1_a2_b1": 2 * se + sr}
    for tau in (1, 2, 4, 8): out[f"E2_gated_tau{tau}"] = se * (1.0 / (1.0 + np.exp(-tau * sr)))
    out["E2_linear_lambda1"] = se + sr; out["E2_linear_lambda2"] = se + 2 * sr
    full_a = np.concatenate([f["kea"], f["kra"]], 1); full_b = np.concatenate([f["keb"], f["krb"]], 1); out["E3_full_concat"] = cosine(full_a, full_b)
    return out


def evaluate_scheme(val, ta, tb, name, score_fn):
    sv, sa, sb = score_fn(val), score_fn(ta), score_fn(tb); th = choose_threshold(val["y"], sv)
    return {"name": name, "threshold_val": th, "val": metric(val["y"], sv, th, val["categories"]), "testA": metric(ta["y"], sa, th, ta["categories"]), "testB": metric(tb["y"], sb, th, tb["categories"]), "val_scores": sv, "testA_scores": sa, "testB_scores": sb}


def train_factorized(train, val, device, steps, seed, p2_fraction, ckpt):
    seed_all(seed); model = FactorizedKey().to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01); rng = np.random.default_rng(seed + 900); logs = []; best = None; t0 = time.perf_counter(); p2mask = train["categories"] == "P2_same_entity_diff_relation"
    yall = torch.from_numpy(train["y"].astype(np.float32)).to(device); rall_a = torch.from_numpy(train["ra"]).long().to(device); rall_b = torch.from_numpy(train["rb"]).long().to(device)
    bs = 256 if not SMOKE else 128
    for step in range(1, steps + 1):
        ix = rng.integers(0, len(train["y"]), size=min(bs, len(train["y"]))); a, b = t(train["ka"][ix], device), t(train["kb"][ix], device); ea, ra = model(a); eb, rb = model(b); se = (ea * eb).sum(-1); sr = (ra * rb).sum(-1)
        loss_pair = F.binary_cross_entropy_with_logits(4.0 * (0.5 * se + 0.5 * sr), yall[ix])
        loss_rel = 0.5 * (F.cross_entropy(model.relation_classifier(ra), rall_a[ix]) + F.cross_entropy(model.relation_classifier(rb), rall_b[ix]))
        local_p2 = torch.tensor(p2mask[ix], dtype=torch.bool, device=device); loss_p2 = F.softplus(4.0 * sr[local_p2]).mean() if local_p2.any() else sr.sum() * 0.0
        loss = loss_pair + 0.5 * loss_rel + 0.5 * loss_p2
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == steps:
            vf = factor_pair(model, val, device); vs = score_schemes(vf)["E1_a1_b1"]; th = choose_threshold(val["y"], vs); vm = metric(val["y"], vs, th, val["categories"]); logs.append({"step": step, "loss": float(loss.detach()), "loss_pair": float(loss_pair.detach()), "loss_relation": float(loss_rel.detach()), "loss_p2": float(loss_p2.detach()), "val": vm, "threshold": th})
            if best is None or (vm["f1"], vm["balanced_accuracy"]) > (best["val"]["f1"], best["val"]["balanced_accuracy"]): best = {"state": copy.deepcopy(model.state_dict()), "step": step, "val": vm, "threshold": th}
    model.load_state_dict(best["state"]); torch.save({"model": model.state_dict(), "step": best["step"], "threshold": best["threshold"], "params": sum(p.numel() for p in model.parameters()), "p2_fraction": p2_fraction}, ckpt)
    return model, {"best_step": best["step"], "best_val": best["val"], "threshold": best["threshold"], "steps": steps, "p2_fraction": p2_fraction, "params": sum(p.numel() for p in model.parameters()), "wall_clock_s": time.perf_counter() - t0, "logs": logs}


def scheme_table(model, val, ta, tb, device):
    vf, af, bf = factor_pair(model, val, device), factor_pair(model, ta, device), factor_pair(model, tb, device); names = score_schemes(vf).keys(); out = {}
    for name in names: out[name] = evaluate_scheme(val, ta, tb, name, lambda z, n=name: score_schemes(factor_pair(model, z, device))[n])
    return out, vf, af, bf


def geometry(data, scores=None):
    c = cosine(data["ka"], data["kb"]); out = {}
    if scores is not None: out["selected_score"] = {"mean": float(scores.mean()), "std": float(scores.std())}
    for cat in sorted(set(data["categories"].tolist())):
        m = data["categories"] == cat; d = np.linalg.norm(data["ka"] - data["kb"], axis=1)
        out[cat] = {"n": int(m.sum()), "cosine_mean": float(c[m].mean()), "cosine_std": float(c[m].std()), "l2_mean": float(d[m].mean()), "l2_std": float(d[m].std())}
    p1, p2 = c[data["categories"] == "P1_same_key"], c[data["categories"] == "P2_same_entity_diff_relation"]
    if len(p1) and len(p2):
        bins = np.linspace(-1, 1, 101); h1, _ = np.histogram(p1, bins=bins); h2, _ = np.histogram(p2, bins=bins); h1 = h1 / max(h1.sum(), 1); h2 = h2 / max(h2.sum(), 1); out["P1_vs_P2_overlap"] = {"histogram_overlap": float(np.minimum(h1, h2).sum()), "range_overlap": max(0.0, min(float(p1.max()), float(p2.max())) - max(float(p1.min()), float(p2.min())))}
    return out


def factor_geometry(data, f, score):
    out = geometry(data, score); se, sr = f["se"], f["sr"]; cats = data["categories"]
    for cat in sorted(set(cats.tolist())):
        m = cats == cat; out.setdefault(cat, {}).update({"sim_e_mean": float(se[m].mean()), "sim_e_std": float(se[m].std()), "sim_r_mean": float(sr[m].mean()), "sim_r_std": float(sr[m].std()), "selected_score_mean": float(score[m].mean()), "selected_score_std": float(score[m].std())})
    p1, p2 = score[cats == "P1_same_key"], score[cats == "P2_same_entity_diff_relation"]; bins = np.linspace(float(min(score.min(), -1)), float(max(score.max(), 1)), 101); h1, _ = np.histogram(p1, bins=bins); h2, _ = np.histogram(p2, bins=bins); h1 = h1 / max(h1.sum(), 1); h2 = h2 / max(h2.sum(), 1); out["selected_P1_vs_P2_overlap"] = float(np.minimum(h1, h2).sum())
    return out


def threshold_sensitivity(y, prob, chosen):
    vals = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    return {"chosen": float(chosen), "grid": {str(x): metric(y, prob, x) for x in vals}}


def fit_probe(xtr, ytr, xsets, nclasses, device, name, steps=300):
    seed_all(SEED + 3000 + nclasses); model = nn.Linear(xtr.shape[1], nclasses).to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01); rng = np.random.default_rng(SEED + nclasses + 991); xt = t(xtr, device); yt = torch.from_numpy(ytr.astype(np.int64)).to(device); bs = min(512, len(ytr))
    for _ in range(steps if not SMOKE else 20):
        ix = rng.integers(0, len(ytr), size=bs); loss = F.cross_entropy(model(xt[ix]), yt[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    out = {"params": sum(p.numel() for p in model.parameters())}
    for name2, x in xsets.items():
        yy = x[1]; mask = yy >= 0
        if not mask.any(): out[name2] = {"accuracy": None, "n_known": 0, "known_fraction": 0.0}; continue
        pred = model(t(x[0][mask], device)).argmax(1).detach().cpu().numpy(); out[name2] = {"accuracy": float((pred == yy[mask]).mean()), "n_known": int(mask.sum()), "known_fraction": float(mask.mean())}
    return out


def probes(model, banks, device):
    feats = {}
    for s, bank in banks.items():
        with torch.no_grad():
            e, r = model(t(bank.K, device)); e = e.cpu().numpy(); r = r.cpu().numpy()
        feats[s] = {"e": e, "r": r, "full": np.concatenate([e, r], 1)}
    train = banks["train"]; person_map = {p: i for i, p in enumerate(sorted({r["person"] for r in train.refs}))}; rel_map = {r: i for i, r in enumerate(sorted({int(x["relation"]) for x in train.refs}))}
    yperson = {}; yrel = {}
    for s, bank in banks.items():
        yperson[s] = np.asarray([person_map.get(r["person"], -1) for r in bank.refs], dtype=np.int64); yrel[s] = np.asarray([rel_map.get(int(r["relation"]), -1) for r in bank.refs], dtype=np.int64)
    out = {"relation_map": rel_map, "entity_class_count": len(person_map), "note": "TestB entity probe is closed-world; known_fraction is reported because names are unseen."}
    for target, labels, ncls in (("entity", yperson, len(person_map)), ("relation", yrel, len(rel_map))):
        for branch, dim in (("K_e", "e"), ("K_r", "r"), ("K_full", "full")):
            xtr = feats["train"][dim]; xs = {s: (feats[s][dim], labels[s]) for s in ("train", "val", "testA", "testB")}; out[f"{branch}_to_{target}"] = fit_probe(xtr, labels["train"], xs, ncls, device, f"{branch}_to_{target}", 300 if not SMOKE else 20)
    return out


def retrieval_eval(bank, model, scheme, device, limit=1000):
    pool = bank.pool; refs_by = {(r["episode"], r["pos"]): r for r in bank.refs}; out = {"n": 0, "correct": 0, "errors": [], "position_counts": Counter()}
    eps = [ep for ep in pool.episodes if len(ep) == 4][:limit]
    for ep in eps:
        for true_pos, true_ref0 in enumerate(ep):
            true_ref = refs_by[(true_ref0["episode"], true_ref0["pos"])]
            q = bank.Q[true_ref["bank_idx"]]; fact_idx = [refs_by[(r["episode"], r["pos"])] ["bank_idx"] for r in ep]
            ks = bank.K[fact_idx]
            if model is None: scores = (ks * q[None]).sum(1)
            else:
                with torch.no_grad(): qe, qr = model(t(q[None], device)); ke, kr = model(t(ks, device)); qe = qe.cpu().numpy()[0]; qr = qr.cpu().numpy()[0]; ke = ke.cpu().numpy(); kr = kr.cpu().numpy()
                if scheme.startswith("E1_a1_b"):
                    b = int(scheme.rsplit("b", 1)[1]); scores = qe @ ke.T + b * (qr @ kr.T)
                elif scheme.startswith("E1_a2_b1"): scores = 2 * (qe @ ke.T) + qr @ kr.T
                elif scheme.startswith("E2_gated_tau"):
                    tau = int(scheme.rsplit("tau", 1)[1]); se = qe @ ke.T; sr = qr @ kr.T; scores = se * (1 / (1 + np.exp(-tau * sr)))
                elif scheme == "E2_linear_lambda1": scores = qe @ ke.T + qr @ kr.T
                elif scheme == "E2_linear_lambda2": scores = qe @ ke.T + 2 * (qr @ kr.T)
                else: scores = np.concatenate([ke, kr], 1) @ np.concatenate([qe, qr], 1)
            pred_pos = int(np.argmax(scores)); out["n"] += 1; out["correct"] += int(pred_pos == true_pos); out["position_counts"][str(true_pos)] += 1
            if pred_pos != true_pos and len(out["errors"]) < 200:
                pred_ref = ep[pred_pos]; same_ent = pred_ref["person"] == true_ref["person"]; same_rel = pred_ref["relation"] == true_ref["relation"]
                kind = "correct_entity_wrong_relation" if same_ent and not same_rel else ("wrong_entity_correct_relation" if (not same_ent and same_rel) else "both_wrong")
                top2 = np.sort(scores)[-2:]; out["errors"].append({"true_pos": true_pos, "pred_pos": pred_pos, "kind": kind, "margin": float(top2[-1] - top2[-2]), "true_person": true_ref["person"], "true_relation": int(true_ref["relation"]), "pred_person": pred_ref["person"], "pred_relation": int(pred_ref["relation"])})
    out["accuracy"] = out["correct"] / max(out["n"], 1); out["position_counts"] = dict(out["position_counts"]); return out


def error_audit(retrieval):
    errs = retrieval.get("errors", [])[:100]; c = Counter(x["kind"] for x in errs); return {"sample_n": len(errs), "counts": dict(c), "fractions": {k: float(v / max(len(errs), 1)) for k, v in c.items()}, "errors": errs}


def same_entity_matrix(banks, model, device):
    bank = banks["testA"]; by_person = defaultdict(dict)
    for r in bank.refs: by_person[r["person"]][r["relation"]] = r
    selected = next((v for v in by_person.values() if len(v) >= 4), None)
    if selected is None: return {"available": False}
    rels = sorted(selected)[:4]; ks = np.stack([bank.K[selected[r]["bank_idx"]] for r in rels]); base = ks @ ks.T
    with torch.no_grad(): e, rr = model(t(ks, device)); e, rr = e.cpu().numpy(), rr.cpu().numpy()
    return {"available": True, "relations": rels, "base_cosine": base, "entity_branch_cosine": e @ e.T, "relation_branch_cosine": rr @ rr.T, "full_sum": e @ e.T + rr @ rr.T}


def same_relation_stats(banks, model, device):
    bank = banks["testA"]; by_rel = defaultdict(list)
    for r in bank.refs: by_rel[r["relation"]].append(r)
    out = {}
    for rel, refs in by_rel.items():
        refs = refs[:8]
        if len(refs) < 2: continue
        ks = np.stack([bank.K[x["bank_idx"]] for x in refs]); base = ks @ ks.T
        with torch.no_grad(): e, rr = model(t(ks, device)); e, rr = e.cpu().numpy(), rr.cpu().numpy()
        den = len(refs) * (len(refs) - 1)
        out[str(rel)] = {"n": len(refs),
                         "base_offdiag_mean": float((base.sum() - np.trace(base)) / den),
                         "entity_offdiag_mean": float(((e @ e.T).sum() - np.trace(e @ e.T)) / den),
                         "relation_offdiag_mean": float(((rr @ rr.T).sum() - np.trace(rr @ rr.T)) / den)}
    return out


class TinyJudge(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.net = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def judge_features(f, kind):
    if kind == "linear": return np.stack([f["se"], f["sr"], 1 - f["se"], 1 - f["sr"], np.linalg.norm(f["kea"] - f["keb"], axis=1), np.linalg.norm(f["kra"] - f["krb"], axis=1)], 1).astype(np.float32)
    if kind == "pair_mlp": return np.concatenate([f["kea"], f["keb"], f["kra"], f["krb"], np.abs(f["kea"] - f["keb"]), np.abs(f["kra"] - f["krb"]), f["kea"] * f["keb"], f["kra"] * f["krb"]], 1).astype(np.float32)
    return np.concatenate([np.abs(f["kea"] - f["keb"]), f["kea"] * f["keb"], np.abs(f["kra"] - f["krb"]), f["kra"] * f["krb"], (f["kea"] + f["keb"]) / 2, (f["kra"] + f["krb"]) / 2], 1).astype(np.float32)


def train_judge(train, val, ta, tb, kind, device):
    xs = {s: judge_features(d, kind) for s, d in (("train", train), ("val", val), ("testA", ta), ("testB", tb))}; mean = xs["train"].mean(0); std = xs["train"].std(0); std[std < 1e-6] = 1.0
    model = TinyJudge(xs["train"].shape[1]).to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01); rng = np.random.default_rng(SEED + 4400 + len(kind)); y = torch.from_numpy(train["y"].astype(np.float32)).to(device); best = None; steps = 1000 if not SMOKE else 20
    for step in range(1, steps + 1):
        ix = rng.integers(0, len(y), size=min(256, len(y))); x = t(((xs["train"][ix] - mean) / std).astype(np.float32), device); loss = F.binary_cross_entropy_with_logits(model(x), y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step % 200 == 0 or step == steps:
            pv = torch.sigmoid(model(t(((xs["val"] - mean) / std).astype(np.float32), device))).detach().cpu().numpy(); th = choose_threshold(val["y"], pv); vm = metric(val["y"], pv, th, val["categories"])
            if best is None or vm["f1"] > best["val"]["f1"]: best = {"state": copy.deepcopy(model.state_dict()), "val": vm, "threshold": th, "step": step}
    model.load_state_dict(best["state"]); out = {"params": sum(p.numel() for p in model.parameters()), "best_step": best["step"], "threshold": best["threshold"], "val": best["val"]}
    for s, d in (("testA", ta), ("testB", tb)):
        p = torch.sigmoid(model(t(((xs[s] - mean) / std).astype(np.float32), device))).detach().cpu().numpy(); out[s] = metric(d["y"], p, best["threshold"], d["categories"])
    return out


def main():
    seed_all(); t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(CACHE_ROOT); pools = {s: FactPool(cache, s) for s in ("train", "val", "testA", "testB")}
    kc = torch.load(O21B, map_location="cpu", weights_only=False); key = KeyEncoder(DIM).to(device); key.load_state_dict(kc["key"], strict=True); key.eval()
    qc = torch.load(r"D:\training\checkpoints\clr_v24_o21a_mr\best_model.pt", map_location="cpu", weights_only=False); query = QueryEncoder(DIM).to(device); query.load_state_dict(qc["query"], strict=True); query.eval()
    for mod in (key, query):
        for p in mod.parameters(): p.requires_grad_(False)
    banks = {s: KeyBank(pools[s], key, device) for s in pools}
    for bank in banks.values():
        qout = []
        with torch.no_grad():
            for st in range(0, len(bank.refs), 256):
                hs = np.stack([bank.pool.query_hidden(r) for r in bank.refs[st:st + 256]]).astype(np.float32); qout.append(query(t(hs, device)).cpu().numpy().astype(np.float32))
        bank.Q = np.concatenate(qout)
    # E0 uses exactly the O3b-D final pair seeds and sizes.
    sets = {"train": pair_refs(banks["train"], PAIR_TRAIN, SEED + 1), "val": pair_refs(banks["val"], PAIR_EVAL, SEED + 2), "testA": pair_refs(banks["testA"], PAIR_EVAL, SEED + 3), "testB": pair_refs(banks["testB"], PAIR_EVAL, SEED + 4)}
    stats = {s: {"n": len(d["y"]), "positive": int(d["y"].sum()), "negative": int((d["y"] == 0).sum()), "categories": dict(d["pair_counts"]), "same_episode_pairing": False, "unique_source_episodes": len({r["episode"] for r in banks[s].refs}), "unique_persons": len({r["person"] for r in banks[s].refs})} for s, d in sets.items()}; stats["hard_negative_ratio_negative"] = {s: float((d["y"] == 0).mean()) for s, d in sets.items()}; dump(OUT / "pair_dataset_stats.json", stats)
    val, ta, tb = sets["val"], sets["testA"], sets["testB"]; sv, sa, sb = cosine(val["ka"], val["kb"]), cosine(ta["ka"], ta["kb"]), cosine(tb["ka"], tb["kb"]); th0 = choose_threshold(val["y"], sv); e0 = {"chosen_threshold_val": th0, "val": metric(val["y"], sv, th0, val["categories"]), "testA": metric(ta["y"], sa, th0, ta["categories"]), "testB": metric(tb["y"], sb, th0, tb["categories"])}
    prior_path = O3BD_OUT / "cosine_baseline.json"; prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else None; repro = {"computed": e0, "prior_o3b_d": prior, "prior_available": prior is not None, "testA_f1_abs_delta": None if prior is None else abs(e0["testA"]["f1"] - prior["testA"]["f1"]), "testB_f1_abs_delta": None if prior is None else abs(e0["testB"]["f1"] - prior["testB"]["f1"]), "reproduced_within_2pp": False if prior is None else abs(e0["testA"]["f1"] - prior["testA"]["f1"]) <= .02 and abs(e0["testB"]["f1"] - prior["testB"]["f1"]) <= .02}; dump(OUT / "baseline_reproduction.json", repro)
    dsv = -np.linalg.norm(val["ka"] - val["kb"], axis=1)
    dsa = -np.linalg.norm(ta["ka"] - ta["kb"], axis=1)
    dsb = -np.linalg.norm(tb["ka"] - tb["kb"], axis=1)
    thd = choose_threshold(val["y"], dsv)
    distance = {"score": "negative_euclidean", "chosen_threshold_val": thd,
                "val": metric(val["y"], dsv, thd, val["categories"]),
                "testA": metric(ta["y"], dsa, thd, ta["categories"]),
                "testB": metric(tb["y"], dsb, thd, tb["categories"])}
    dump(OUT / "cosine_baseline.json", e0); dump(OUT / "distance_baseline.json", distance)
    variants = [("original", None, SEED + 10), ("p2_25", .25, SEED + 11), ("p2_50", .50, SEED + 12)]; train_sets = {name: pair_refs(banks["train"], PAIR_TRAIN, seed, frac) for name, frac, seed in variants}; records = {}; tables = {}; models = {}
    for name, frac, seed in variants:
        model, rec = train_factorized(train_sets[name], val, device, min(STEPS, 20) if SMOKE else STEPS, seed, frac, OUT / f"factorized_{name}_best.pt"); models[name] = model; records[name] = rec; table, vf, af, bf = scheme_table(model, val, ta, tb, device); tables[name] = {"schemes": {k: {kk: vv for kk, vv in v.items() if kk not in ("val_scores", "testA_scores", "testB_scores")} for k, v in table.items()}, "factor_val": vf, "factor_testA": af, "factor_testB": bf}
    best_name = None; best_scheme = None; best_score = -1.0
    for vn, entry in tables.items():
        for sn, res in entry["schemes"].items():
            score = res["val"]["f1"]
            if score > best_score: best_score = score; best_name = vn; best_scheme = sn
    best_model = models[best_name]; best_table = tables[best_name]["schemes"][best_scheme]; best_f = {"val": tables[best_name]["factor_val"], "testA": tables[best_name]["factor_testA"], "testB": tables[best_name]["factor_testB"]}
    equality = {"variants": tables, "best_variant": best_name, "best_scheme": best_scheme, "best_val_f1": best_score}; dump(OUT / "equality_results.json", equality); dump(OUT / "factorized_results.json", {"variants": {k: {"record": records[k], "schemes": tables[k]["schemes"]} for k in records}, "best_variant": best_name, "best_scheme": best_scheme})
    best_score_arrays = {"val": score_schemes(best_f["val"])[best_scheme], "testA": score_schemes(best_f["testA"])[best_scheme], "testB": score_schemes(best_f["testB"])[best_scheme]}; dump(OUT / "threshold_sensitivity.json", {s: threshold_sensitivity(d["y"], best_score_arrays[s], best_table["threshold_val"]) for s, d in (("val", val), ("testA", ta), ("testB", tb))})
    dump(OUT / "geometry_diagnostics.json", {"baseline_testA": geometry(ta), "baseline_testB": geometry(tb), "factorized_testA": factor_geometry(ta, best_f["testA"], best_score_arrays["testA"]), "factorized_testB": factor_geometry(tb, best_f["testB"], best_score_arrays["testB"]), "same_entity_relation_matrix": same_entity_matrix(banks, best_model, device), "same_relation_different_entity": same_relation_stats(banks, best_model, device)})
    dump(OUT / "p1_p2_histograms.json", {"baseline": {"testA": geometry(ta).get("P1_vs_P2_overlap"), "testB": geometry(tb).get("P1_vs_P2_overlap")}, "factorized": {"testA": factor_geometry(ta, best_f["testA"], best_score_arrays["testA"]).get("selected_P1_vs_P2_overlap"), "testB": factor_geometry(tb, best_f["testB"], best_score_arrays["testB"]).get("selected_P1_vs_P2_overlap")}})
    probe_out = probes(best_model, banks, device); dump(OUT / "probe_results.json", probe_out)
    retr_base = {s: retrieval_eval(banks[s], None, "baseline", device, 1000 if not SMOKE else 64) for s in ("testA", "testB")}; retr_fact = {s: retrieval_eval(banks[s], best_model, best_scheme, device, 1000 if not SMOKE else 64) for s in ("testA", "testB")}; dump(OUT / "retrieval_results.json", {"baseline": retr_base, "factorized": retr_fact})
    audit = {s: error_audit(retr_fact[s]) for s in retr_fact}; dump(OUT / "nearest_neighbor_audit.json", audit)
    judges = {}
    def judge_data(raw, factors):
        d = dict(factors)
        d["y"] = raw["y"]
        d["categories"] = raw["categories"]
        return d
    factor_train = judge_data(train_sets[best_name], factor_pair(best_model, train_sets[best_name], device))
    factor_val = judge_data(val, best_f["val"])
    factor_ta = judge_data(ta, best_f["testA"])
    factor_tb = judge_data(tb, best_f["testB"])
    for kind in ("linear", "pair_mlp", "symmetric"):
        judges[kind] = train_judge(factor_train, factor_val, factor_ta, factor_tb, kind, device)
    dump(OUT / "judge_ladder.json", judges)
    conf = {"best_factorized": {s: {"threshold": best_table["threshold_val"], "metrics": metric(d["y"], best_score_arrays[s], best_table["threshold_val"], d["categories"])} for s, d in (("val", val), ("testA", ta), ("testB", tb))}, "external_judges": judges}; dump(OUT / "confusion_matrices.json", conf)
    # Probe gate is relation probe on K_r; choose the held-out TestB relation result.
    kr_rel = probe_out["K_r_to_relation"]; p2a = best_table["testA"]["per_category_accuracy"].get("P2_same_entity_diff_relation", {}).get("accuracy", 0); p2b = best_table["testB"]["per_category_accuracy"].get("P2_same_entity_diff_relation", {}).get("accuracy", 0); gate = {"baseline_reproduced": repro["reproduced_within_2pp"], "testA_f1_ge_95": best_table["testA"]["f1"] >= .95, "testB_f1_ge_95": best_table["testB"]["f1"] >= .95, "testA_update_recall_ge_95": best_table["testA"]["recall"] >= .95, "testB_update_recall_ge_95": best_table["testB"]["recall"] >= .95, "P2_testA_ge_95": p2a >= .95, "P2_testB_ge_90": p2b >= .90, "K_r_to_relation_testB_ge_90": (kr_rel["testB"]["accuracy"] or 0) >= .90}; passed = all(gate.values())
    reconnect = {"allowed": False, "reason": "O3b-E is offline only; no reconnect was performed even if a future rerun passes."}; dump(OUT / "reconnect_readiness.json", {"gate": gate, "passed": passed, "reconnect": reconnect, "o3c": "not entered"})
    git_sha = "NO_GIT_REPO"; git_diff = "NO_GIT_REPO"
    try: git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(Path.cwd()), text=True).strip(); git_diff = subprocess.check_output(["git", "diff", "--stat"], cwd=str(Path.cwd()), text=True).strip()
    except Exception: pass
    cfg = {"phase": "CLR-v2.4-O3b-E", "seed": SEED, "device": str(device), "source_key_checkpoint": str(O21B), "source_o3b_d_report": str(O3BD_OUT / "v24_o3b_diag_report.md"), "cache": str(CACHE_ROOT), "git_sha": git_sha, "git_diff": git_diff, "frozen": ["original KeyEncoder", "V/Value path", "GRU", "QueryEncoder", "O3a allocator", "backbone", "memory/slots", "downstream answer head"], "trainable": ["tied Linear K_e projection", "tied Linear K_r projection", "relation auxiliary classifier"], "factor_dim": FACTOR_DIM, "query_projection": "tied to key factor projections on frozen QueryEncoder output", "loss": "L_pair + 0.5 L_relation + 0.5 L_P2; no answer loss", "variants": ["original", "P2 25% of negatives", "P2 50% of negatives"], "steps": STEPS, "best_variant": best_name, "best_scheme": best_scheme, "factor_params": records[best_name]["params"], "wall_clock_s": time.perf_counter() - t0, "peak_vram": {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None}}
    torch.save({"model": best_model.state_dict(), "best_variant": best_name,
                "best_scheme": best_scheme, "threshold_val": best_table["threshold_val"],
                "factor_dim": FACTOR_DIM}, OUT / "best_model.pt")
    torch.save({"model": best_model.state_dict(), "best_variant": best_name,
                "best_scheme": best_scheme, "threshold_val": best_table["threshold_val"],
                "factor_dim": FACTOR_DIM}, OUT / "final_model.pt")
    train_log = {"variants": records, "best_variant": best_name, "best_scheme": best_scheme}
    dump(OUT / "config.json", cfg); dump(OUT / "git_info.json", {"git_sha": git_sha, "git_diff": git_diff})
    dump(OUT / "training.log", train_log); dump(OUT / "train_log.json", train_log)
    verdict = "O3b-E PASS: RELATION-AWARE KEY REPRESENTATION SUFFICIENT" if passed else ("O3b-E FAIL: RELATION-AWARE FACTORIZATION INSUFFICIENT" if (kr_rel["testB"]["accuracy"] or 0) < .90 else "O3b-E FAIL: RELATION IDENTITY PRESENT BUT NOT GEOMETRICALLY USABLE")
    pct = lambda x: f"{100*x:.2f}%" if x is not None else "n/a"
    report = ["# CLR-v2.4-O3b-E — Relation-Aware Factorized Key", "", f"**{verdict}**", "", "## Executive Summary", f"Best variant={best_name}, best score={best_scheme}; factor head parameters={records[best_name]['params']}. Original KeyEncoder parameters were frozen; no real write/read reconnect was run.", "", "## Baseline Reproduction", f"O3b-D raw cosine reproduction: TestA F1={pct(e0['testA']['f1'])}, TestB F1={pct(e0['testB']['f1'])}; prior deltas={repro['testA_f1_abs_delta']}/{repro['testB_f1_abs_delta']}.", "", "## Main Results"]
    for vn, entry in tables.items():
        rows = []
        for sn, r in entry["schemes"].items(): rows.append(f"{sn}: A F1={pct(r['testA']['f1'])}, recall={pct(r['testA']['recall'])}; B F1={pct(r['testB']['f1'])}, recall={pct(r['testB']['recall'])}; P2 A/B={pct(r['testA']['per_category_accuracy'].get('P2_same_entity_diff_relation',{}).get('accuracy'))}/{pct(r['testB']['per_category_accuracy'].get('P2_same_entity_diff_relation',{}).get('accuracy'))}")
        report.append(f"{vn}: " + " | ".join(rows))
    report += ["", "## Best Variant / Probes", f"Best={best_name}/{best_scheme}. TestA F1={pct(best_table['testA']['f1'])}, TestB F1={pct(best_table['testB']['f1'])}; UPDATE recall A/B={pct(best_table['testA']['recall'])}/{pct(best_table['testB']['recall'])}; P1 A/B={pct(best_table['testA']['per_category_accuracy'].get('P1_same_key',{}).get('accuracy'))}/{pct(best_table['testB']['per_category_accuracy'].get('P1_same_key',{}).get('accuracy'))}; P2 A/B={pct(p2a)}/{pct(p2b)}.", f"K_r→relation TestA/TestB={pct(kr_rel['testA']['accuracy'])}/{pct(kr_rel['testB']['accuracy'])}; full probe details are in probe_results.json.", "", "## Retrieval / Error Audit", f"Frozen K-only retrieval baseline A/B={pct(retr_base['testA']['accuracy'])}/{pct(retr_base['testB']['accuracy'])}; factorized={pct(retr_fact['testA']['accuracy'])}/{pct(retr_fact['testB']['accuracy'])}. Nearest-neighbor error categories are in nearest_neighbor_audit.json.", "", "## Gate", json.dumps(gate, ensure_ascii=False), "", "## Boundary", "Repeated V update contamination remains an independent unresolved issue. This experiment did not modify V, GRU, allocator, memory storage, or answer supervision. No reconnect and no O3c were entered.", "", "## Stop", "O3b-E complete; await a separate instruction before any reconnect or O3c."]
    (OUT / "v24_o3b_e_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "best_variant": best_name, "best_scheme": best_scheme, "testA_f1": best_table["testA"]["f1"], "testB_f1": best_table["testB"]["f1"], "testA_recall": best_table["testA"]["recall"], "testB_recall": best_table["testB"]["recall"], "P2_testA": p2a, "P2_testB": p2b, "Kr_relation_testA": kr_rel["testA"]["accuracy"], "Kr_relation_testB": kr_rel["testB"]["accuracy"], "baseline_reproduced": repro["reproduced_within_2pp"], "reconnect": False, "wall_clock_s": time.perf_counter() - t0}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
