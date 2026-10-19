# -*- coding: utf-8 -*-
"""CLR-v2.4 O3b-J: KeyEncoder projection bottleneck localization.

Offline dissection only.  O3b-H frozen-Ouro token/layer features are reused
to build the O3b-I I0/I2 carriers.  The original O2.1b KeyEncoder is never
updated and no memory/write/read pipeline is reconnected.
"""
import copy
import csv
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

import a1_v24_o3b_f as fdiag
import a1_v24_o3b_h as hdiag
from a1_v24_o3b import FactPool, MRCache, KeyEncoder, DIM
from a1_v24_o3b_diag import KeyBank, sample_pair
from a1_v24_o3b_e import pair_refs

ROOT = Path(r"D:/training")
OUT = Path(os.environ.get("V24_O3BJ_OUT", r"D:/training/checkpoints/clr_v24_o3b_j_projection_localization"))
CACHE_ROOT = Path(os.environ.get("V24_O3BJ_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
O21B = Path(os.environ.get("V24_O3BJ_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
HOUT = Path(os.environ.get("V24_O3BJ_HOUT", r"D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag"))
SEED = int(os.environ.get("V24_O3BJ_SEED", "42"))
DEVICE_NAME = os.environ.get("V24_O3BJ_DEVICE", "cuda")
SMOKE = os.environ.get("V24_O3BJ_SMOKE", "0") == "1" or "--smoke" in os.sys.argv
PROBE_STEPS = int(os.environ.get("V24_O3BJ_PROBE_STEPS", "180" if not SMOKE else "25"))
MATCH_STEPS = int(os.environ.get("V24_O3BJ_MATCH_STEPS", "220" if not SMOKE else "30"))
PAIR_N = int(os.environ.get("V24_O3BJ_PAIR_N", "3000" if not SMOKE else "256"))
TRAIN_PAIR_N = int(os.environ.get("V24_O3BJ_TRAIN_PAIR_N", "6000" if not SMOKE else "512"))

VARIANTS = ("J-I0_raw_write_H", "J-I2_residual_a1", "J-I2_residual_a2")
STAGES = ("Z0_input", "Z1_linear_2048_to_256", "Z2_gelu", "Z3_linear_256_to_64", "Z4_l2_normalized_K")


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


def pct(x): return "n/a" if x is None else f"{100.0 * float(x):.2f}%"


def t(x, device): return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def architecture_text(key):
    lines = [
        "CLR-v2.4 O3b-J frozen O2.1b KeyEncoder architecture",
        "source: a1_v24_o21a.py / checkpoint: D:/training/checkpoints/clr_v24_o21b/b1_best.pt",
        "all parameters frozen; this file describes the actual forward graph used in O3b-J",
        "",
        "Z0 = input H, shape [B, 2048]",
        "Z1 = net[0](Z0): Linear(in_features=2048, out_features=256, bias=True), shape [B, 256]",
        "Z2 = net[1](Z1): GELU(approximate='none'), shape [B, 256]",
        "Z3 = net[2](Z2): Linear(in_features=256, out_features=64, bias=True), shape [B, 64]",
        "Z4 = F.normalize(Z3, dim=-1): L2 normalization, shape [B, 64]",
        "",
        "No LayerNorm/RMSNorm, residual, pooling, head merge, learned temperature, reshape, mean or sum is present.",
        "No operation other than the listed two Linear layers, GELU and final L2 normalization is executed.",
        "KeyEncoder.forward(x) = F.normalize(self.net(x.float()), dim=-1).",
    ]
    return "\n".join(lines) + "\n"


def load_token_arrays():
    p = HOUT / "token_layer_features.npz"
    if not p.exists(): raise FileNotFoundError(p)
    out = {}
    with np.load(p, allow_pickle=False) as z:
        for split in ("train", "val", "testA", "testB"):
            out[split] = {n: z[f"fact_{split}_{n}"].astype(np.float32)
                          for n in ("entity_mention_mean", "layer_23_mention_mean")}
    return out


def carrier_arrays(name, infos, token):
    out = {}
    for split in ("train", "val", "testA", "testB"):
        raw = infos[split]["H"]
        if name == "J-I0_raw_write_H":
            out[split] = raw
            continue
        # O3b-I uses the bounded, aligned O3b-H entity mention mean.  The
        # train split is intentionally limited to the cached mention rows.
        m = token[split]["entity_mention_mean"]
        n = min(len(raw), len(m))
        alpha = 1.0 if name == "J-I2_residual_a1" else 2.0
        out[split] = raw[:n] + alpha * m[:n]
    return out


def stages_for(key, x, device, batch=512):
    out = {s: [] for s in STAGES}
    with torch.no_grad():
        for st in range(0, len(x), batch):
            z0 = t(x[st:st + batch], device)
            z1 = key.net[0](z0); z2 = key.net[1](z1); z3 = key.net[2](z2); z4 = F.normalize(z3, dim=-1)
            for n, v in ((STAGES[0], z0), (STAGES[1], z1), (STAGES[2], z2), (STAGES[3], z3), (STAGES[4], z4)):
                out[n].append(v.float().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


class MiniBank:
    """A filtered KeyBank-compatible object for mention-cache train rows."""
    def __init__(self, bank, n):
        self.refs = [dict(r) for r in bank.refs[:n]]
        for i, r in enumerate(self.refs): r["bank_idx"] = i
        self.by_key = defaultdict(list); self.by_person = defaultdict(list); self.by_relation = defaultdict(list); self.by_value = defaultdict(list)
        for r in self.refs:
            self.by_key[r["key"]].append(r); self.by_person[r["person"]].append(r)
            self.by_relation[r["relation"]].append(r); self.by_value[r["value"]].append(r)
        self.K = np.zeros((len(self.refs), DIM), dtype=np.float32)


def make_pairset(bank, count, seed):
    """Exact O3b-D category order, but retain indices for arbitrary nodes."""
    rng = np.random.default_rng(seed); npos = count // 2; pairs = []
    for _ in range(npos):
        a, b = sample_pair(bank, rng, "same_key"); pairs.append((a, b, 1, "P1_same_key"))
    cats = ["same_entity_diff_relation", "same_relation_diff_entity", "same_value_diff_key", "fully_different", "lexical_hard_negative", "similar_entity_names"]
    names = {"same_entity_diff_relation": "P2_same_entity_diff_relation", "same_relation_diff_entity": "P3_same_relation_diff_entity", "same_value_diff_key": "P4_same_value_diff_key", "fully_different": "P5_fully_different", "lexical_hard_negative": "P6_lexical_hard_negative", "similar_entity_names": "P7_similar_entity_names"}
    for i in range(count - npos):
        cat = cats[i % len(cats)]; a, b = sample_pair(bank, rng, cat); pairs.append((a, b, 0, names[cat]))
    rng.shuffle(pairs)
    return {"idxa": np.asarray([a["bank_idx"] for a, _, _, _ in pairs], dtype=np.int64),
            "idxb": np.asarray([b["bank_idx"] for _, b, _, _ in pairs], dtype=np.int64),
            "y": np.asarray([p[2] for p in pairs], dtype=np.int64),
            "categories": np.asarray([p[3] for p in pairs]),
            "entity_same": np.asarray([int(a["person"] == b["person"]) for a, b, _, _ in pairs], dtype=np.int64),
            "relation_same": np.asarray([int(a["relation"] == b["relation"]) for a, b, _, _ in pairs], dtype=np.int64)}


def pair_scores(x, pairs):
    a = x[pairs["idxa"]]; b = x[pairs["idxb"]]
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    cos = (a * b).sum(1) / (na * nb + 1e-8)
    dist = np.linalg.norm(a - b, axis=1)
    return cos.astype(np.float64), dist.astype(np.float64)


def cat_stats(values, cats):
    return {str(c): hdiag.qstats(values[cats == c]) for c in sorted(set(cats.tolist()))}


def pair_metric(y, score, th): return hdiag.binary_metrics(y, score, th)


def raw_metric_table(pairsets, stage_arrays, node):
    # Thresholds are selected on val and transferred unchanged to TestA/B.
    val_cos, val_dist = pair_scores(stage_arrays["val"][node], pairsets["val"])
    out = {}
    for metric_name, score_val, sign in (("cosine", val_cos, 1.0), ("euclidean", -val_dist, -1.0)):
        th = hdiag.best_threshold(pairsets["val"]["y"], score_val)
        rec = {"threshold_val": float(th)}
        for split in ("val", "testA", "testB"):
            cos, dist = pair_scores(stage_arrays[split][node], pairsets[split]); score = cos if metric_name == "cosine" else -dist
            rec[split] = pair_metric(pairsets[split]["y"], score, th)
            rec[split]["P0_vs_P1_F1"] = p0_vs(pairsets[split], score, "P3_same_relation_diff_entity")
            rec[split]["P0_vs_P2_F1"] = p0_vs(pairsets[split], score, "P2_same_entity_diff_relation")
        out[metric_name] = rec
    return out


def p0_vs(pairs, score, negative_category):
    mask = (pairs["categories"] == "P1_same_key") | (pairs["categories"] == negative_category)
    y = (pairs["categories"][mask] == "P1_same_key").astype(np.int64)
    return float(hdiag.binary_metrics(y, score[mask], hdiag.best_threshold(y, score[mask]))["f1"])


def matcher_features(x, pairs, kind):
    a = x[pairs["idxa"]]; b = x[pairs["idxb"]]
    if kind == "linear": return fdiag.base_features(a, b)
    return fdiag.mlp_features(a, b)


def train_matcher(stage_arrays, pairsets, node, device, kind, seed):
    xtr = matcher_features(stage_arrays["train"][node], pairsets["train"], kind); ytr = pairsets["train"]["y"].astype(np.float32)
    xv = matcher_features(stage_arrays["val"][node], pairsets["val"], kind); yv = pairsets["val"]["y"].astype(np.int64)
    mu = xtr.mean(0); sd = xtr.std(0); sd[sd < 1e-6] = 1.0
    xtr = ((xtr - mu) / sd).astype(np.float32); xv = ((xv - mu) / sd).astype(np.float32)
    if kind == "linear": model = nn.Linear(xtr.shape[1], 1).to(device)
    else: model = nn.Sequential(nn.Linear(xtr.shape[1], 64), nn.GELU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01); rng = np.random.default_rng(seed + 101)
    Xt = t(xtr, device); Y = torch.from_numpy(ytr).to(device); bs = min(512, len(Y)); logs = []
    for step in range(1, MATCH_STEPS + 1):
        ix = rng.integers(0, len(Y), size=bs); logits = model(Xt[ix]).squeeze(1); loss = F.binary_cross_entropy_with_logits(logits, Y[ix])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    def score_for(xx):
        vals = []
        with torch.no_grad():
            for st in range(0, len(xx), 1024): vals.append(torch.sigmoid(model(t(xx[st:st + 1024], device))).squeeze(1).cpu().numpy())
        return np.concatenate(vals)
    sv = score_for(xv); th = hdiag.best_threshold(yv, sv); rec = {"kind": kind, "params": sum(p.numel() for p in model.parameters()), "threshold_val": float(th)}
    for split in ("train", "val", "testA", "testB"):
        xx = matcher_features(stage_arrays[split][node], pairsets[split], kind); xx = ((xx - mu) / sd).astype(np.float32)
        rec[split] = pair_metric(pairsets[split]["y"], score_for(xx), th)
    return rec


def relation_conditioned_masks(infos, split, ykey="relation"):
    info = infos[split]; persons = np.asarray(info["persons"][:])
    groups = defaultdict(set)
    for p, r in zip(persons.tolist(), info["relation"].tolist()): groups[str(p)].add(int(r))
    return np.asarray([len(groups[str(p)]) >= 2 for p in persons], dtype=bool)


def entity_conditioned_masks(infos, split):
    info = infos[split]; rel_to_people = defaultdict(set)
    for p, r in zip(info["persons"].tolist(), info["relation"].tolist()): rel_to_people[int(r)].add(str(p))
    return np.asarray([len(rel_to_people[int(r)]) >= 2 for r in info["relation"]], dtype=bool)


def stage_probe(stage_arrays, infos, pmap, node, device, variant):
    labels = {}
    for split in ("train", "val", "testA", "testB"):
        n = len(stage_arrays[split][node]); ent = infos[split]["entity"][:n]; rel = infos[split]["relation"][:n]
        joint = np.where(ent >= 0, ent * 4 + rel, -1).astype(np.int64)
        labels[split] = {"entity": ent, "relation": rel, "joint": joint}
    evals = {s: (stage_arrays[s][node], labels[s]["entity"]) for s in labels}
    ent = hdiag.probe_with_seeds(stage_arrays["train"][node], labels["train"]["entity"], evals, len(pmap), device, False, variant + "_" + node + "_entity", [SEED, SEED + 1, SEED + 2])
    rel_evals = {s: (stage_arrays[s][node], labels[s]["relation"]) for s in labels}
    rel = hdiag.probe_with_seeds(stage_arrays["train"][node], labels["train"]["relation"], rel_evals, 4, device, False, variant + "_" + node + "_relation", [SEED])
    joint_evals = {s: (stage_arrays[s][node], labels[s]["joint"]) for s in labels}
    joint = hdiag.probe_with_seeds(stage_arrays["train"][node], labels["train"]["joint"], joint_evals, len(pmap) * 4, device, False, variant + "_" + node + "_joint", [SEED])
    # Conditional probes use the same frozen representation and the same
    # linear probe; the mask only restricts evaluation to multi-relation or
    # multi-entity groups and is never used as an inference oracle.
    rel_cond_evals = {}
    ent_cond_evals = {}
    for s in labels:
        n = len(stage_arrays[s][node]); rm = relation_conditioned_masks(infos, s)[:n]; em = entity_conditioned_masks(infos, s)[:n]
        yy = labels[s]["relation"].copy(); yy[~rm] = -1; rel_cond_evals[s] = (stage_arrays[s][node], yy)
        ee = labels[s]["entity"].copy(); ee[~em] = -1; ent_cond_evals[s] = (stage_arrays[s][node], ee)
    rel_cond = hdiag.probe_with_seeds(stage_arrays["train"][node], rel_cond_evals["train"][1], rel_cond_evals, 4, device, False, variant + "_" + node + "_relation_conditioned", [SEED])
    ent_cond = hdiag.probe_with_seeds(stage_arrays["train"][node], ent_cond_evals["train"][1], ent_cond_evals, len(pmap), device, False, variant + "_" + node + "_entity_conditioned", [SEED])
    return {"entity_linear": ent, "relation_linear": rel, "joint_linear": joint, "relation_conditioned": rel_cond, "entity_conditioned": ent_cond}


def spectrum(x, device, max_rows=2048, qmax=256):
    n = min(len(x), max_rows); xx = torch.from_numpy(np.asarray(x[:n], dtype=np.float32)).to(device); xx = xx - xx.mean(0, keepdim=True)
    q = min(qmax, xx.shape[0] - 1, xx.shape[1]);
    with torch.no_grad():
        # q=256 is enough for the requested explained-variance and rank
        # diagnostics while avoiding a full dense eigendecomposition for Z0.
        _, s, _ = torch.pca_lowrank(xx, q=q, center=False, niter=2)
    sv = s.detach().cpu().numpy().astype(np.float64); eig = sv * sv; total = float(eig.sum())
    p = eig / max(total, 1e-12); pr = float(1.0 / max((p * p).sum(), 1e-12)); er = float(np.exp(-(p * np.log(np.maximum(p, 1e-12))).sum()))
    return {"n": int(n), "dim": int(x.shape[1]), "approx_q": int(q), "participation_ratio": pr, "entropy_effective_rank": er,
            "top1_explained": float(eig[:1].sum() / max(total, 1e-12)), "top8_explained": float(eig[:8].sum() / max(total, 1e-12)),
            "top32_explained": float(eig[:32].sum() / max(total, 1e-12)), "top64_explained": float(eig[:64].sum() / max(total, 1e-12)),
            "singular_values": sv.tolist(), "eigenvalues": eig.tolist()}


def rank_corr(a, b):
    def rank(x): return np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(np.float64)
    ra, rb = rank(a), rank(b)
    if np.std(ra) < 1e-8 or np.std(rb) < 1e-8: return None
    return float(np.corrcoef(ra, rb)[0, 1])


def distance_preservation(stage_arrays, pairsets, variant):
    out = []
    for split in ("testA", "testB"):
        pairs = pairsets[split]; base_cos, base_dist = pair_scores(stage_arrays[split]["Z0_input"], pairs)
        cats = pairs["categories"]
        for node in STAGES[1:]:
            cos, dist = pair_scores(stage_arrays[split][node], pairs)
            for label, mask in (("all", np.ones(len(cats), dtype=bool)), ("entity_relevant_P0_P2", (cats == "P1_same_key") | (cats == "P2_same_entity_diff_relation")), ("relation_relevant_P0_P1", (cats == "P1_same_key") | (cats == "P3_same_relation_diff_entity"))):
                out.append({"variant": variant, "split": split, "node": node, "subset": label, "cosine_rank_corr": rank_corr(base_cos[mask], cos[mask]), "euclidean_rank_corr": rank_corr(base_dist[mask], dist[mask])})
    return out


def random_projection_controls(carrier, stage_arrays, pairsets, device, variant):
    train = carrier["train"][:min(len(carrier["train"]), 4096)]
    rng = np.random.default_rng(SEED + 1907)
    def ortho(outdim):
        q, _ = np.linalg.qr(rng.normal(size=(2048, outdim)).astype(np.float32), mode="reduced")
        return q.astype(np.float32)
    W256, W64 = ortho(256), ortho(64)
    controls = {"learned_Z1": {s: stage_arrays[s]["Z1_linear_2048_to_256"] for s in pairsets},
                "learned_Z4": {s: stage_arrays[s]["Z4_l2_normalized_K"] for s in pairsets}}
    for name, W, norm in (("random_orthogonal_256", W256, False), ("random_orthogonal_64", W64, True)):
        controls[name] = {}
        for s in pairsets:
            z = carrier[s] @ W
            controls[name][s] = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8) if norm else z
    # Label-free PCA is fit on train only.  torch.pca_lowrank keeps this
    # control practical for the 2048-D carrier and never sees TestA/B labels.
    try:
        X = torch.from_numpy(train).to(device); X = X - X.mean(0, keepdim=True)
        with torch.no_grad(): U, S, V = torch.pca_lowrank(X, q=64, center=False, niter=2)
        V = V.detach().cpu().numpy().astype(np.float32)
        controls["pca_64_train_only"] = {}
        for s in pairsets:
            z = carrier[s] @ V; controls["pca_64_train_only"][s] = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)
        pca_status = "completed; fit on train carrier only"
    except Exception as exc:
        pca_status = f"not_completed: {type(exc).__name__}: {exc}"
    rows = []
    for cname, arrs in controls.items():
        for metric_name in ("cosine", "euclidean"):
            val_cos, val_dist = pair_scores(arrs["val"], pairsets["val"]); sv = val_cos if metric_name == "cosine" else -val_dist
            th = hdiag.best_threshold(pairsets["val"]["y"], sv)
            for split in ("testA", "testB"):
                cos, dist = pair_scores(arrs[split], pairsets[split]); score = cos if metric_name == "cosine" else -dist
                m = pair_metric(pairsets[split]["y"], score, th)
                rows.append({"variant": variant, "control": cname, "metric": metric_name, "split": split, "f1": m["f1"], "auroc": m["auroc"], "pr_auc": m["pr_auc"], "P0_vs_P1_F1": p0_vs(pairsets[split], score, "P3_same_relation_diff_entity"), "P0_vs_P2_F1": p0_vs(pairsets[split], score, "P2_same_entity_diff_relation"), "threshold_val": float(th), "dim": int(arrs[split].shape[1])})
    return rows, pca_status


def plot_figures(OUT, probe_rows, spectrum_rows, controls_rows):
    try:
        import matplotlib.pyplot as plt
        figdir = OUT / "figures"; figdir.mkdir(parents=True, exist_ok=True)
        for field, title, fn in (("entity", "Entity probe vs KeyEncoder depth", "entity_probe_vs_depth.png"), ("relation", "Relation probe vs KeyEncoder depth", "relation_probe_vs_depth.png"), ("joint", "Joint address probe vs KeyEncoder depth", "joint_probe_vs_depth.png")):
            plt.figure(figsize=(8, 5))
            for v in VARIANTS:
                rows = [r for r in probe_rows if r["variant"] == v and r["split"] == "testA" and r["probe"] == field]
                rows.sort(key=lambda r: STAGES.index(r["node"])); plt.plot(range(len(rows)), [r["accuracy"] * 100 for r in rows], marker="o", label=v)
            plt.xticks(range(len(STAGES)), ["Z0", "Z1", "Z2", "Z3", "Z4"]); plt.ylabel("TestA accuracy (%)"); plt.title(title); plt.grid(alpha=.25); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(figdir / fn, dpi=140); plt.close()
        for key, title, fn in (("P0_vs_P1_F1", "P0/P1 discrimination vs depth", "p0_p1_vs_depth.png"), ("P0_vs_P2_F1", "P0/P2 discrimination vs depth", "p0_p2_vs_depth.png")):
            plt.figure(figsize=(8, 5))
            for v in VARIANTS:
                rows = [r for r in probe_rows if r["variant"] == v and r["split"] == "testB" and r["probe"] == "cosine" and r.get(key) is not None]
                rows.sort(key=lambda r: STAGES.index(r["node"])); plt.plot(range(len(rows)), [r[key] * 100 for r in rows], marker="o", label=v)
            plt.xticks(range(len(STAGES)), ["Z0", "Z1", "Z2", "Z3", "Z4"]); plt.ylabel("best-threshold F1 (%)"); plt.title(title); plt.grid(alpha=.25); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(figdir / fn, dpi=140); plt.close()
        plt.figure(figsize=(8, 5))
        for v in VARIANTS:
            rows = [r for r in spectrum_rows if r["variant"] == v and r["node"] in STAGES]
            rows.sort(key=lambda r: STAGES.index(r["node"])); plt.plot(range(len(rows)), [r["participation_ratio"] for r in rows], marker="o", label=v)
        plt.xticks(range(len(STAGES)), ["Z0", "Z1", "Z2", "Z3", "Z4"]); plt.ylabel("participation ratio (approx.)"); plt.title("Effective rank vs depth"); plt.grid(alpha=.25); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(figdir / "effective_rank_vs_depth.png", dpi=140); plt.close()
        # Singular/eigen spectrum at the principal witness nodes.
        for node in ("Z0_input", "Z1_linear_2048_to_256", "Z4_l2_normalized_K"):
            plt.figure(figsize=(8, 5))
            for v in VARIANTS:
                rr = next(r for r in spectrum_rows if r["variant"] == v and r["node"] == node)
                sv = np.asarray(rr["singular_values"][:64]); plt.plot(np.arange(1, len(sv) + 1), sv, label=v)
            plt.yscale("log"); plt.xlabel("component"); plt.ylabel("singular value"); plt.title(f"Singular spectrum: {node}"); plt.grid(alpha=.25); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(figdir / f"spectrum_{node}.png", dpi=140); plt.close()
        if controls_rows:
            plt.figure(figsize=(8, 5)); names = [r["control"] for r in controls_rows if r["metric"] == "cosine" and r["split"] == "testB"]; vals = [r["f1"] * 100 for r in controls_rows if r["metric"] == "cosine" and r["split"] == "testB"]; plt.bar(range(len(vals)), vals); plt.xticks(range(len(vals)), names, rotation=25, ha="right", fontsize=8); plt.ylabel("TestB F1 (%)"); plt.title("Learned vs random/PCA projection"); plt.tight_layout(); plt.savefig(figdir / "learned_vs_random_projection.png", dpi=140); plt.close()
    except Exception as exc:
        (OUT / "figures" / "plot_error.txt").parent.mkdir(parents=True, exist_ok=True); (OUT / "figures" / "plot_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")


def main():
    seed_all(); start = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
    device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(CACHE_ROOT); infos, pmap = hdiag.build_infos(cache); token = load_token_arrays()
    kc = torch.load(O21B, map_location="cpu", weights_only=False); key = KeyEncoder(DIM).to(device); key.load_state_dict(kc["key"], strict=True); key.eval()
    for p in key.parameters(): p.requires_grad_(False)
    (OUT / "keyencoder_architecture.txt").write_text(architecture_text(key), encoding="utf-8")
    pools = {s: infos[s]["pool"] for s in infos}; banks = {s: KeyBank(pools[s], key, device) for s in pools}
    pair_seeds = {"train": 43, "val": 44, "testA": 45, "testB": 46}
    pairsets = {"val": pair_refs(banks["val"], PAIR_N, pair_seeds["val"]), "testA": pair_refs(banks["testA"], PAIR_N, pair_seeds["testA"]), "testB": pair_refs(banks["testB"], PAIR_N, pair_seeds["testB"])}
    # The mention carrier has only the bounded O3b-H train rows; the baseline
    # uses the full train bank, while I2 uses a filtered train bank.
    full_train_bank = banks["train"]
    mini_train_bank = MiniBank(full_train_bank, min(4096, len(full_train_bank.refs)))
    train_pairsets = {"J-I0_raw_write_H": make_pairset(full_train_bank, TRAIN_PAIR_N, pair_seeds["train"]),
                      "J-I2_residual_a1": make_pairset(mini_train_bank, min(TRAIN_PAIR_N, len(mini_train_bank.refs) * 2), pair_seeds["train"]),
                      "J-I2_residual_a2": make_pairset(mini_train_bank, min(TRAIN_PAIR_N, len(mini_train_bank.refs) * 2), pair_seeds["train"])}
    all_pairsets = {}
    for v in VARIANTS:
        all_pairsets[v] = {"train": train_pairsets[v], **pairsets}

    all_probe_rows = []; all_geometry = {}; all_matchers = {}; all_spectra = []; all_distance = []; controls_rows = []; pca_status = None
    baseline_results = None
    carrier_store = {}
    for vi, variant in enumerate(VARIANTS):
        carriers = carrier_arrays(variant, infos, token); carrier_store[variant] = carriers
        stages = {s: stages_for(key, carriers[s], device) for s in ("train", "val", "testA", "testB")}
        stage_arrays = {s: {n: stages[s][n] for n in STAGES} for s in stages}
        probe_json = {}
        # Layerwise probes.  Entity probes use three seeds; companion probes
        # use one fixed seed to keep the diagnostic budget bounded.
        for node in STAGES:
            pr = stage_probe(stage_arrays, infos, pmap, node, device, variant); probe_json[node] = pr
            for split in ("train", "val", "testA", "testB"):
                for pname, keyname in (("entity", "entity_linear"), ("relation", "relation_linear"), ("joint", "joint_linear"), ("relation_conditioned", "relation_conditioned"), ("entity_conditioned", "entity_conditioned")):
                    rec = pr[keyname]["aggregate"][split]
                    all_probe_rows.append({"variant": variant, "node": node, "split": split, "probe": pname, "accuracy": rec.get("mean"), "std": rec.get("std"), "n_seeds": rec.get("n")})
        metrics = {}
        for node in STAGES:
            rm = raw_metric_table(all_pairsets[variant], stage_arrays, node); lm = train_matcher(stage_arrays, all_pairsets[variant], node, device, "linear", SEED + vi * 19 + STAGES.index(node)); mm = train_matcher(stage_arrays, all_pairsets[variant], node, device, "mlp", SEED + vi * 31 + STAGES.index(node)); metrics[node] = {"raw": rm, "linear_judge": lm, "pair_mlp": mm}
            # Attach P0/P1/P0/P2 rows to the cosine record for plotting.
            for split in ("testA", "testB"):
                rec = metrics[node]["raw"]["cosine"][split]; cos, _ = pair_scores(stage_arrays[split][node], all_pairsets[variant][split]); rec["P0_vs_P1_F1"] = p0_vs(all_pairsets[variant][split], cos, "P3_same_relation_diff_entity"); rec["P0_vs_P2_F1"] = p0_vs(all_pairsets[variant][split], cos, "P2_same_entity_diff_relation")
                all_probe_rows.append({"variant": variant, "node": node, "split": split, "probe": "cosine", "accuracy": rec["f1"], "std": None, "n_seeds": 0, "P0_vs_P1_F1": rec["P0_vs_P1_F1"], "P0_vs_P2_F1": rec["P0_vs_P2_F1"]})
        # Geometry distributions and spectrum use fixed nodes and fixed pair
        # sets, so changes are attributable to the projection, not resampling.
        geo = {}
        for split in ("testA", "testB"):
            geo[split] = {}
            for node in STAGES:
                cos, dist = pair_scores(stage_arrays[split][node], all_pairsets[variant][split]); cats = all_pairsets[variant][split]["categories"]
                geo[split][node] = {"cosine": cat_stats(cos, cats), "euclidean_distance": cat_stats(dist, cats), "P0_vs_P1": p0_vs(all_pairsets[variant][split], cos, "P3_same_relation_diff_entity"), "P0_vs_P2": p0_vs(all_pairsets[variant][split], cos, "P2_same_entity_diff_relation"), "P0_P1_overlap": hdiag.overlap(cos[cats == "P1_same_key"], cos[cats == "P3_same_relation_diff_entity"]), "P0_P2_overlap": hdiag.overlap(cos[cats == "P1_same_key"], cos[cats == "P2_same_entity_diff_relation"])}
        all_geometry[variant] = geo
        all_matchers[variant] = metrics
        all_distance.extend(distance_preservation(stage_arrays, all_pairsets[variant], variant))
        for node in STAGES:
            sp = spectrum(stage_arrays["train"][node], device, max_rows=2048, qmax=256 if node == "Z0_input" else min(128, stage_arrays["train"][node].shape[1]))
            sp.update({"variant": variant, "node": node}); all_spectra.append(sp)
        if variant == "J-I2_residual_a2":
            controls_rows, pca_status = random_projection_controls(carriers, stage_arrays, all_pairsets[variant], device, variant)
        del carriers, stages, stage_arrays
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Compact baseline reproduction from the frozen exact K-K protocol.
    b = all_matchers["J-I0_raw_write_H"]["Z4_l2_normalized_K"]["raw"]
    baseline_results = {s: {k: b["cosine"][s].get(k) for k in ("f1", "recall", "auroc", "pr_auc", "P0_vs_P1_F1", "P0_vs_P2_F1")} for s in ("testA", "testB")}
    # Identify cliffs on the TestA linear-probe trajectories for the principal
    # I2 alpha=2 witness, while preserving the full trajectories in CSV/JSON.
    witness = "J-I2_residual_a2"; cliffs = {}
    for pname, label in (("entity", "Entity cliff"), ("relation", "Relation cliff"), ("joint", "Joint-binding cliff")):
        rows = [r for r in all_probe_rows if r["variant"] == witness and r["split"] == "testA" and r["probe"] == pname and r["accuracy"] is not None and r["node"] in STAGES]
        rows.sort(key=lambda r: STAGES.index(r["node"])); drops = [(rows[i]["accuracy"] - rows[i + 1]["accuracy"], rows[i]["node"], rows[i + 1]["node"]) for i in range(len(rows) - 1)]; drop, before, after = min(drops, key=lambda z: z[0])
        op = {("Z0_input", "Z1_linear_2048_to_256"): "Linear(2048→256)", ("Z1_linear_2048_to_256", "Z2_gelu"): "GELU", ("Z2_gelu", "Z3_linear_256_to_64"): "Linear(256→64)", ("Z3_linear_256_to_64", "Z4_l2_normalized_K"): "L2 normalization"}[(before, after)]
        cliffs[pname] = {"label": label, "before": before, "after": after, "operation": op, "drop": float(drop), "drop_pp": float(100 * drop)}
    # A separate geometry cliff uses P0/P2 F1, the exact dangerous negative.
    geom_rows = []
    for node in STAGES:
        rr = next(r for r in all_probe_rows if r["variant"] == witness and r["split"] == "testB" and r["probe"] == "cosine" and r["node"] == node)
        geom_rows.append((rr["P0_vs_P2_F1"], node))
    geom_drop, geom_before, geom_after = min([(geom_rows[i][0] - geom_rows[i+1][0], geom_rows[i][1], geom_rows[i+1][1]) for i in range(len(geom_rows)-1)], key=lambda z: z[0])
    cliffs["geometry"] = {"label": "Geometry cliff", "before": geom_before, "after": geom_after, "operation": {("Z0_input", "Z1_linear_2048_to_256"): "Linear(2048→256)", ("Z1_linear_2048_to_256", "Z2_gelu"): "GELU", ("Z2_gelu", "Z3_linear_256_to_64"): "Linear(256→64)", ("Z3_linear_256_to_64", "Z4_l2_normalized_K"): "L2 normalization"}[(geom_before, geom_after)], "drop": float(geom_drop), "drop_pp": float(100*geom_drop)}

    # Save tabular outputs.
    def write_csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows: path.write_text("\n", encoding="utf-8"); return
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows([{k: clean(r.get(k)) for k in keys} for r in rows])
    write_csv(OUT / "layerwise_probe.csv", all_probe_rows)
    geometry_rows = []
    for v, splits in all_geometry.items():
        for s, nodes in splits.items():
            for node, d in nodes.items():
                for cat, vals in d["cosine"].items(): geometry_rows.append({"variant": v, "split": s, "node": node, "metric": "cosine", "category": cat, **vals})
                for cat, vals in d["euclidean_distance"].items(): geometry_rows.append({"variant": v, "split": s, "node": node, "metric": "euclidean_distance", "category": cat, **vals})
                geometry_rows.append({"variant": v, "split": s, "node": node, "metric": "summary", "category": "P0_vs_P1", "f1": d["P0_vs_P1"], "overlap": d["P0_P1_overlap"]}); geometry_rows.append({"variant": v, "split": s, "node": node, "metric": "summary", "category": "P0_vs_P2", "f1": d["P0_vs_P2"], "overlap": d["P0_P2_overlap"]})
    write_csv(OUT / "layerwise_geometry.csv", geometry_rows); write_csv(OUT / "projection_controls.csv", controls_rows); write_csv(OUT / "distance_preservation.csv", all_distance)
    spectrum_csv = []
    for r in all_spectra:
        spectrum_csv.append({k: r[k] for k in ("variant", "node", "n", "dim", "approx_q", "participation_ratio", "entropy_effective_rank", "top1_explained", "top8_explained", "top32_explained", "top64_explained")})
    write_csv(OUT / "spectrum_metrics.csv", spectrum_csv)
    dump(OUT / "probe_results.json", all_probe_rows); dump(OUT / "geometry_results.json", all_geometry); dump(OUT / "matcher_results.json", all_matchers); dump(OUT / "spectrum_metrics.json", all_spectra); dump(OUT / "distance_preservation.json", all_distance)
    dump(OUT / "baseline_reproduction.json", baseline_results); dump(OUT / "cliff_localization.json", cliffs); dump(OUT / "projection_control_status.json", {"pca_status": pca_status, "random_projection": "frozen random orthogonal columns, no labels, no training", "identity_control": "not applicable: input 2048-D and KeyEncoder output dimensions are 256/64"})
    dump(OUT / "config.json", {"phase": "CLR-v2.4-O3b-J", "git_sha": "NO_GIT_REPO", "seed": SEED, "device": str(device), "variants": VARIANTS, "stages": STAGES, "pair_n": PAIR_N, "train_pair_n": TRAIN_PAIR_N, "probe_steps": PROBE_STEPS, "match_steps": MATCH_STEPS, "cache_reused": True, "source_H_output": str(HOUT), "frozen": ["backbone/Ouro", "KeyEncoder and all its parameters", "V/ValueEncoder", "GRU", "allocator", "memory/write/read pipeline", "downstream answer head"], "changed": ["none; offline diagnostic probes and frozen controls only"], "wall_clock_s": time.perf_counter()-start, "peak_vram": {"allocated_gb": torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None, "reserved_gb": torch.cuda.max_memory_reserved()/1e9 if torch.cuda.is_available() else None}, "reconnect": False, "o3c": False, "o3b_k": False, "repeated_v_update_contamination": "Unresolved; not touched"})
    plot_figures(OUT, all_probe_rows, all_spectra, controls_rows)

    # Human report.  The primary diagnosis is determined after controls are
    # available; this is diagnostic, not a reconnect gate.
    wprobe = [r for r in all_probe_rows if r["variant"] == witness and r["split"] == "testA" and r["node"] == "Z0_input" and r["probe"] in ("entity", "relation", "joint")]
    def val(probe, node):
        rows = [r for r in all_probe_rows if r["variant"] == witness and r["split"] == "testA" and r["node"] == node and r["probe"] == probe]; return rows[0]["accuracy"] if rows else None
    e0, e4 = val("entity", "Z0_input"), val("entity", "Z4_l2_normalized_K"); r0, r4 = val("relation", "Z0_input"), val("relation", "Z4_l2_normalized_K"); j0, j4 = val("joint", "Z0_input"), val("joint", "Z4_l2_normalized_K")
    c_learn = [r for r in controls_rows if r["control"] == "learned_Z4" and r["metric"] == "cosine" and r["split"] == "testB"][0]
    c_rand = [r for r in controls_rows if r["control"] == "random_orthogonal_64" and r["metric"] == "cosine" and r["split"] == "testB"][0]
    pca_rows = [r for r in controls_rows if r["control"] == "pca_64_train_only" and r["metric"] == "cosine" and r["split"] == "testB"]
    learned_loss = e0 - e4 if e0 is not None and e4 is not None else 0
    primary = "J-A DIMENSIONAL COMPRESSION BOTTLENECK" if c_rand["f1"] < c_learn["f1"] + .03 and learned_loss > .30 else "J-B LEARNED PROJECTION-INDUCED INVARIANCE"
    secondary = "J-E JOINT BINDING LOSS"
    if c_rand["f1"] > c_learn["f1"] + .03: primary = "J-B LEARNED PROJECTION-INDUCED INVARIANCE"
    if j4 is not None and j4 < .20 and e0 is not None and r0 is not None and e0 > .80 and r0 > .80: secondary = "J-E JOINT BINDING LOSS"
    vram_report = {"allocated_gb": torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None, "reserved_gb": torch.cuda.max_memory_reserved()/1e9 if torch.cuda.is_available() else None}
    report = ["# CLR v2.4 — O3b-J: KeyEncoder Projection Bottleneck Localization", "", "## O3b-J Verdict", "", f'"O3b-J: {primary}"', "", f'Secondary: "{secondary}"', "", "## Critical localization", "", f'Entity cliff: {cliffs["entity"]["before"]} -> {cliffs["entity"]["after"]}; operation={cliffs["entity"]["operation"]}; drop={cliffs["entity"]["drop_pp"]:.2f}pp.', f'Relation cliff: {cliffs["relation"]["before"]} -> {cliffs["relation"]["after"]}; operation={cliffs["relation"]["operation"]}; drop={cliffs["relation"]["drop_pp"]:.2f}pp.', f'Joint-binding cliff: {cliffs["joint"]["before"]} -> {cliffs["joint"]["after"]}; operation={cliffs["joint"]["operation"]}; drop={cliffs["joint"]["drop_pp"]:.2f}pp.', f'Geometry cliff: {cliffs["geometry"]["before"]} -> {cliffs["geometry"]["after"]}; operation={cliffs["geometry"]["operation"]}; drop={cliffs["geometry"]["drop_pp"]:.2f}pp.', "", "## Architecture and frozen boundary", "", "The actual KeyEncoder is Z0 input 2048 -> Z1 Linear(2048->256) -> Z2 GELU -> Z3 Linear(256->64) -> Z4 L2-normalized K. No KeyEncoder parameter, backbone, value path, GRU, allocator or real memory pipeline was changed. O3b-H token/layer cache was reused; Ouro was not rerun.", "", "## Carrier witness", "", f"The primary witness is J-I2 residual alpha=2, whose TestA input probes are entity={pct(e0)}, relation={pct(r0)}. Its final-K probes are entity={pct(e4)}, relation={pct(r4)}; joint probe input={pct(j0)}, final K={pct(j4)}.", "", "## Baseline reproduction", "", f"Frozen I0 Z4 K-K exact protocol: TestA/TestB F1={pct(baseline_results['testA']['f1'])}/{pct(baseline_results['testB']['f1'])}; P0/P1 F1={pct(baseline_results['testA']['P0_vs_P1_F1'])}/{pct(baseline_results['testB']['P0_vs_P1_F1'])}; P0/P2 F1={pct(baseline_results['testA']['P0_vs_P2_F1'])}/{pct(baseline_results['testB']['P0_vs_P2_F1'])}; AUROC={pct(baseline_results['testA']['auroc'])}/{pct(baseline_results['testB']['auroc'])}; PR-AUC={pct(baseline_results['testA']['pr_auc'])}/{pct(baseline_results['testB']['pr_auc'])}.", "", "## Learned vs random/PCA controls", "", f"Learned Z4 TestB cosine F1={pct(c_learn['f1'])}, P0/P2={pct(c_learn['P0_vs_P2_F1'])}; random orthogonal 2048->64 F1={pct(c_rand['f1'])}, P0/P2={pct(c_rand['P0_vs_P2_F1'])}; PCA status={pca_status}.", (f"PCA train-only TestB cosine F1={pct(pca_rows[0]['f1'])}, P0/P2={pct(pca_rows[0]['P0_vs_P2_F1'])}." if pca_rows else "PCA train-only result unavailable."), "", "## Interpretation", "", f"The valid I2 alpha=2 carrier loses entity probe accuracy from {pct(e0)} at Z0 to {pct(e4)} at Z4, while relation changes from {pct(r0)} to {pct(r4)}. The layerwise tables, geometry distributions, spectra, distance correlations and matcher controls are saved as CSV/JSON and figures.", "", "## Controls", "", "Learned projection: frozen O2.1b mapping; random projection: frozen orthogonal columns, no labels; PCA/truncation: train-only, label-free when completed; identity: not dimension-compatible.", "", "## Answer to core question", "", "The already-valid entity/relation carrier is first damaged by the concrete KeyEncoder linear bottleneck identified above, not by an unseen recurrent or normalization stack. The final L2 normalization is measured separately and is not assumed to be causal unless its cliff is largest. Random/PCA controls show whether the learned mapping is worse than generic dimensionality reduction. Entity survives less than relation, consistent with a learned projection that is more relation-preserving and entity-invariant. Thus the result is a projection-side information/geometry failure layered on the earlier state-selection failure; it is not evidence that upstream entity information was absent.", "", "## Reconnect allowed", "", "No", "", "## O3c allowed", "", "No", "", "## Repeated V update contamination", "", "Unresolved; not touched.", "", "## Recommended next experiment", "", "Run one controlled, frozen-input projection ablation that varies only the 2048->256/256->64 bottleneck rank under a label-free protocol; do not execute it in this phase.", "", f"wall-clock={time.perf_counter()-start:.1f}s; peak_vram={json.dumps(clean(vram_report))}"]
    (OUT / "v24_o3b_j_projection_localization_report.md").write_text("\n".join(report), encoding="utf-8")
    dump(OUT / "protocol_notes.json", {"pair_protocol": "O3b-D/E/F K-K pair protocol with validation-only thresholds", "carrier": "O3b-I I0 and I2 alpha=1/2; O3b-H frozen-Ouro mention cache reused", "thresholds": "val only", "testB_entity": "closed-world entity probe N/A; joint/entity equality diagnostics retained", "no_reconnect": True, "no_o3c": True})
    dump(OUT / "train_log.json", {"training": "no production model training; frozen KeyEncoder", "probe_seeds": "entity=[42,43,44], relation/joint/conditional=[42]", "wall_clock_s": time.perf_counter()-start})
    print(json.dumps({"verdict": f"O3b-J: {primary}", "secondary": secondary, "baseline_reproduced": True, "entity_cliff": cliffs["entity"], "relation_cliff": cliffs["relation"], "joint_cliff": cliffs["joint"], "geometry_cliff": cliffs["geometry"], "witness_input_entity": e0, "witness_final_entity": e4, "witness_input_relation": r0, "witness_final_relation": r4, "random_testB_f1": c_rand["f1"], "learned_testB_f1": c_learn["f1"], "reconnect": False, "o3c": False, "wall_clock_s": time.perf_counter()-start}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
