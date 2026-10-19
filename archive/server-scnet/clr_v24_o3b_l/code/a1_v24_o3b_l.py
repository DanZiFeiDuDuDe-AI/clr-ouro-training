# -*- coding: utf-8 -*-
"""CLR-v2.4 O3b-L: matched singular-value weighting ablation.

This is a frozen, diagnostic-only experiment.  It reuses the O3b-H carrier
cache and the frozen O2.1b KeyEncoder first Linear.  The only trained objects
are temporary probes; no production path, checkpoint, matcher, memory policy,
or architecture is changed.
"""
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import a1_v24_o3b_j as j
from a1_v24_o3b import MRCache, KeyEncoder, DIM


OUT = Path(os.environ.get("V24_O3BL_OUT", r"D:/training/checkpoints/clr_v24_o3b_l_matched_sigma_ablation"))
CACHE_ROOT = Path(os.environ.get("V24_O3BL_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
HOUT = Path(os.environ.get("V24_O3BL_HOUT", r"D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag"))
O21B = Path(os.environ.get("V24_O3BL_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
SEED = int(os.environ.get("V24_O3BL_SEED", "42"))
PROBE_STEPS = int(os.environ.get("V24_O3BL_PROBE_STEPS", "180"))
N_SEEDS = int(os.environ.get("V24_O3BL_N_SEEDS", "10"))
N_PERM = int(os.environ.get("V24_O3BL_N_PERM", "20"))
PAIR_N = int(os.environ.get("V24_O3BL_PAIR_N", "1800"))
LONGER_STEPS = int(os.environ.get("V24_O3BL_LONGER_STEPS", "500"))
LONGER_SEEDS = int(os.environ.get("V24_O3BL_LONGER_SEEDS", "5"))
DEVICE_NAME = os.environ.get("V24_O3BL_DEVICE", "cuda")
SPLITS = ("train", "val", "testA", "testB")
SEEDS = [11, 23, 37, 53, 71, 89, 107, 131, 157, 181][:N_SEEDS]
if len(SEEDS) < N_SEEDS:
    SEEDS += [211 + 17 * i for i in range(N_SEEDS - len(SEEDS))]


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer, np.floating)): return x.item()
    if isinstance(x, Path): return str(x)
    return x


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def stage(label, started):
    dump(OUT / "progress.txt", {"stage": label, "elapsed_s": time.perf_counter() - started})
    print(json.dumps({"stage": label, "elapsed_s": time.perf_counter() - started}, ensure_ascii=False), flush=True)


def pct(x):
    return "n/a" if x is None else f"{100.0 * float(x):.2f}%"


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def bootstrap_ci(values, seed=991, draws=5000):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0: return (None, None)
    if len(a) == 1: return (float(a[0]), float(a[0]))
    rng = np.random.default_rng(seed)
    means = np.mean(a[rng.integers(0, len(a), size=(draws, len(a)))], axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def aggregate(values, seed=991):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "ci95_low": None, "ci95_high": None, "n": 0}
    lo, hi = bootstrap_ci(a, seed=seed)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0)), "min": float(a.min()), "max": float(a.max()), "ci95_low": lo, "ci95_high": hi, "n": int(len(a))}


def np_corr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12: return None
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64); i = 0
    while i < len(order):
        q = i + 1
        while q < len(order) and x[order[q]] == x[order[i]]: q += 1
        r[order[i:q]] = .5 * (i + 1 + q); i = q
    return r


def rank_corr(a, b):
    return np_corr(rankdata(a), rankdata(b))


def fit_seed_batched(reps, labels, nclass, device, steps, seeds, task_offset=0, eval_splits=SPLITS):
    """Identical linear probes across representations and seeds.

    The same minibatch indices are used for every representation and seed;
    only the explicitly listed initialization seed changes.  Models are
    batched by representation within each seed block for speed.
    """
    names = list(reps.keys())
    ytr_all = np.asarray(labels["train"], dtype=np.int64)
    train_mask = ytr_all >= 0
    ytr = ytr_all[train_mask]
    xnp = np.stack([np.asarray(reps[n]["train"], dtype=np.float32)[train_mask] for n in names], axis=0)
    r, n, d = xnp.shape
    m = r * len(seeds)
    # Models are laid out as [seed0: all reps, seed1: all reps, ...], so
    # replicate the representation batch over the same seed-major axis.
    x_models = np.tile(xnp, (len(seeds), 1, 1))
    X = torch.from_numpy(x_models).to(device)
    Y = torch.from_numpy(ytr).long().to(device)
    weight = nn.Parameter(torch.empty(m, nclass, d, device=device))
    bias = nn.Parameter(torch.empty(m, nclass, device=device))
    bound = 1.0 / math.sqrt(max(d, 1))
    with torch.no_grad():
        for si, seed in enumerate(seeds):
            torch.manual_seed(int(seed) + task_offset * 1009)
            st, en = si * r, (si + 1) * r
            weight[st:en].uniform_(-bound, bound)
            bias[st:en].uniform_(-bound, bound)
    opt = torch.optim.AdamW([weight, bias], lr=3e-4, weight_decay=.01)
    rng = np.random.default_rng(SEED + 913 + task_offset)
    bs = min(1024, n)
    for _ in range(max(1, steps)):
        ix = rng.integers(0, n, size=bs)
        ti = torch.as_tensor(ix, dtype=torch.long, device=device)
        xb = X[:, ti, :]
        logits = torch.bmm(xb, weight.transpose(1, 2)) + bias[:, None, :]
        target = Y[ti].unsqueeze(0).expand(m, -1)
        loss = F.cross_entropy(logits.reshape(m * bs, nclass), target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    out = {n_: {seed: {} for seed in seeds} for n_ in names}
    chunk = 256
    with torch.no_grad():
        for split in eval_splits:
            yy = np.asarray(labels[split], dtype=np.int64)
            mask = yy >= 0
            if not mask.any():
                for name in names:
                    for seed in seeds: out[name][seed][split] = None
                continue
            xev_base = np.stack([np.asarray(reps[name][split], dtype=np.float32)[mask] for name in names], axis=0)
            xev_np = np.tile(xev_base, (len(seeds), 1, 1))
            correct = np.zeros(m, dtype=np.int64); total = 0
            for st in range(0, len(yy[mask]), chunk):
                en = min(st + chunk, len(yy[mask]))
                xev = torch.from_numpy(xev_np[:, st:en]).to(device)
                pred = torch.bmm(xev, weight.transpose(1, 2)) + bias[:, None, :]
                pred = pred.argmax(2).cpu().numpy()
                correct += np.sum(pred == yy[mask][st:en][None, :], axis=1)
                total += en - st
            vals = correct.astype(np.float64) / max(total, 1)
            for si, seed in enumerate(seeds):
                for ri, name in enumerate(names): out[name][seed][split] = float(vals[si * r + ri])
    del opt, weight, bias, X, Y
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def task_labels(infos, jlabels, pmap, lengths):
    return {
        "entity": ({s: infos[s]["entity"][:lengths[s]] for s in SPLITS}, len(pmap), 1),
        "relation": ({s: infos[s]["relation"][:lengths[s]] for s in SPLITS}, 4, 2),
        "joint": ({s: jlabels[s][:lengths[s]] for s in SPLITS}, len(set(jlabels["train"][jlabels["train"] >= 0])), 3),
    }


def run_probe_suite(rep_arrs, infos, jlabels, pmap, device, steps, seeds, eval_splits=SPLITS):
    by_task = {}
    lengths = {s: len(next(iter(rep_arrs.values()))[s]) for s in SPLITS}
    for task, (labels, nclass, offset) in task_labels(infos, jlabels, pmap, lengths).items():
        by_task[task] = fit_seed_batched(rep_arrs, labels, nclass, device, steps, seeds, task_offset=offset, eval_splits=eval_splits)
    return by_task


def flatten_probe_results(probe, norm, suite, seeds):
    rows = []
    for task, by_rep in probe.items():
        for rep, by_seed in by_rep.items():
            for seed in seeds:
                for split in SPLITS:
                    rows.append({"suite": suite, "normalization": norm, "representation": rep, "task": task, "seed": seed, "split": split, "accuracy": by_seed[seed].get(split)})
    return rows


def aggregate_probe_rows(rows):
    out = []
    keys = sorted({(r["suite"], r["normalization"], r["representation"], r["task"], r["split"]) for r in rows})
    for suite, norm, rep, task, split in keys:
        vals = [r["accuracy"] for r in rows if (r["suite"], r["normalization"], r["representation"], r["task"], r["split"]) == (suite, norm, rep, task, split) and r["accuracy"] is not None]
        out.append({"suite": suite, "normalization": norm, "representation": rep, "task": task, "split": split, **aggregate(vals, seed=1301 + len(out))})
    return out


def paired_effect_rows(raw_probe, norm, suite, seeds, pairs):
    rows = []
    for task, by_rep in raw_probe.items():
        for split in ("val", "testA", "testB"):
            for lhs, rhs, label in pairs:
                if lhs not in by_rep or rhs not in by_rep: continue
                vals = []
                for seed in seeds:
                    a = by_rep[lhs][seed].get(split); b = by_rep[rhs][seed].get(split)
                    if a is not None and b is not None: vals.append(float(a - b))
                rows.append({"suite": suite, "normalization": norm, "task": task, "split": split, "effect": label, "lhs": lhs, "rhs": rhs, **aggregate(vals, seed=1901 + len(rows))})
    return rows


def standardize(rep_arrs):
    out = {}; stats = {}
    for name, arrs in rep_arrs.items():
        mu = np.asarray(arrs["train"], dtype=np.float64).mean(0).astype(np.float32)
        sd = np.asarray(arrs["train"], dtype=np.float64).std(0).astype(np.float32)
        sd[sd < 1e-8] = 1.0
        out[name] = {s: ((np.asarray(arrs[s], dtype=np.float32) - mu[None, :]) / sd[None, :]).astype(np.float32) for s in SPLITS}
        stats[name] = {"mean_abs": float(np.mean(np.abs(mu))), "std_min": float(sd.min()), "std_max": float(sd.max()), "zero_std_dims": int(np.sum(sd == 1.0))}
    return out, stats


def covariance_metrics(X):
    X = np.asarray(X, dtype=np.float64)
    xc = X - X.mean(0, keepdims=True)
    cov = (xc.T @ xc) / max(len(X) - 1, 1)
    eig = np.linalg.eigvalsh(cov)
    pos = eig[eig > max(float(eig.max()) * 1e-10, 1e-12)] if len(eig) else eig
    p = pos / max(float(pos.sum()), 1e-30)
    eff = float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-30))))) if len(p) else 0.0
    var = np.diag(cov)
    return {"n": int(len(X)), "dim": int(X.shape[1]), "cov_eigen_min": float(eig.min()), "cov_eigen_max": float(eig.max()), "condition_number": float(eig.max() / max(eig[eig > max(float(eig.max()) * 1e-10, 1e-12)].min(), 1e-30)), "effective_rank": eff, "variance_min": float(var.min()), "variance_max": float(var.max()), "variance_max_min_ratio": float(var.max() / max(var[var > 1e-12].min(), 1e-30)), "covariance_spectrum": eig[::-1].tolist()}


def geometry_rows(rep_arrs, pairsets, device):
    rows = []
    invs = {}
    for name, arrs in rep_arrs.items():
        X = np.asarray(arrs["train"], dtype=np.float64)
        xc = X - X.mean(0, keepdims=True)
        cov = (xc.T @ xc) / max(len(X) - 1, 1)
        ridge = max(float(np.trace(cov) / max(cov.shape[0], 1)) * 1e-4, 1e-8)
        invs[name] = np.linalg.inv(cov + ridge * np.eye(cov.shape[0]))
    for name, arrs in rep_arrs.items():
        for split in ("val", "testA", "testB"):
            X = arrs[split]
            for metric in ("cosine", "euclidean", "mahalanobis"):
                def scores(typ):
                    ab = pairsets[split][typ]; a = X[ab[:, 0]]; b = X[ab[:, 1]]
                    if metric == "cosine": return j.cosine(a, b)
                    d = a - b
                    if metric == "euclidean": dist = np.linalg.norm(d, axis=1)
                    else: dist = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", d, invs[name], d), 0.0))
                    return -dist
                p0 = scores("P0_same_entity_same_relation")
                for neg in ("P1_diff_entity_same_relation", "P2_same_entity_diff_relation"):
                    pn = scores(neg)
                    vp0 = None; vpn = None
                    if split in ("testA", "testB"):
                        vp0 = geometry_score(rep_arrs[name]["val"], pairsets["val"], metric, "P0_same_entity_same_relation", invs[name])
                        vpn = geometry_score(rep_arrs[name]["val"], pairsets["val"], metric, neg, invs[name])
                    y = np.r_[np.ones(len(p0), dtype=np.int64), np.zeros(len(pn), dtype=np.int64)]
                    sc = np.r_[p0, pn]
                    if vp0 is not None and len(vp0) and len(vpn):
                        th = j.best_threshold(np.r_[np.ones(len(vp0), dtype=np.int64), np.zeros(len(vpn), dtype=np.int64)], np.r_[vp0, vpn])
                    else:
                        th = j.best_threshold(y, sc)
                    bm = j.binary_metrics(y, sc, th)
                    rows.append({"representation": name, "split": split, "metric": metric, "contrast": f"P0_vs_{neg}", "n": len(p0) + len(pn), "auroc": bm["auroc"], "best_f1": bm["f1"], "balanced_accuracy": bm["balanced_accuracy"], "overlap": j.overlap(p0, pn), "median_margin": float(np.median(p0) - np.median(pn)), "p0_median": float(np.median(p0)), "negative_median": float(np.median(pn))})
    return rows


def geometry_score(X, pairset, metric, typ, inv):
    ab = pairset[typ]; a = X[ab[:, 0]]; b = X[ab[:, 1]]
    if metric == "cosine": return j.cosine(a, b)
    d = a - b
    if metric == "euclidean": return -np.linalg.norm(d, axis=1)
    return -np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", d, inv, d), 0.0))


def sigma_semantics(R0, infos, jlabels):
    X = np.asarray(R0["train"], dtype=np.float64)
    result = []; summaries = []
    for task, y in (("entity", infos["train"]["entity"][:len(X)]), ("relation", infos["train"]["relation"][:len(X)]), ("joint", jlabels["train"][:len(X)])):
        y = np.asarray(y, dtype=np.int64); scores = []
        mask = y >= 0; xx = X[mask]; yy = y[mask]; classes = np.unique(yy); overall = xx.mean(0)
        between = np.zeros(X.shape[1], dtype=np.float64); within = np.zeros(X.shape[1], dtype=np.float64)
        for c in classes:
            xc = xx[yy == c]; delta = xc.mean(0) - overall
            between += len(xc) * delta * delta
            within += np.sum((xc - xc.mean(0)) ** 2, axis=0)
        score = between / np.maximum(within / max(len(yy) - len(classes), 1), 1e-12)
        scores.append(score)
        s = np.asarray(current_singular_values, dtype=np.float64)
        summary = {"task": task, "corr_sigma_score_pearson": np_corr(s, score), "corr_sigma_score_spearman": rank_corr(s, score), "score_mean": float(score.mean()), "score_max": float(score.max())}
        summaries.append(summary)
        for i, val in enumerate(score): result.append({"row_type": "direction", "task": task, "direction": i, "sigma": float(s[i]), "semantic_score": float(val), "sigma_rank_desc": int(np.argsort(-s).tolist().index(i) + 1)})
    for x in summaries: result.append({"row_type": "summary", **x})
    return result, summaries


def svg_chart(path, title, series, labels, ymin=0.0, ymax=1.0, ytitle="accuracy"):
    W, H, L, R, T, B = 1000, 600, 82, 28, 58, 78
    pw, ph = W - L - R, H - T - B
    labels = [str(x) for x in labels]
    def x(i): return L + (pw * i / max(len(labels) - 1, 1))
    def y(v): return T + ph * (1.0 - (float(v) - ymin) / max(ymax - ymin, 1e-12))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{W/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>']
    out += [f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="#444"/><line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="#444"/>']
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        yy = y(v); out.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{L+pw}" y2="{yy:.1f}" stroke="#eee"/>')
        out.append(f'<text x="{L-8}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{v:.2f}</text>')
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for si, (name, vals) in enumerate(series):
        pts = []
        for i, val in enumerate(vals):
            if val is not None and np.isfinite(val): pts.append((x(i), y(val)))
        if len(pts) >= 2: out.append(f'<polyline fill="none" stroke="{colors[si%len(colors)]}" stroke-width="3" points="{" ".join(f"{a:.1f},{b:.1f}" for a,b in pts)}"/>')
        for a, b in pts: out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{colors[si%len(colors)]}"/>')
        out.append(f'<text x="{L + 12 + si*150}" y="{H-25}" font-family="Arial" font-size="12" fill="{colors[si%len(colors)]}">{name}</text>')
    for i, lab in enumerate(labels): out.append(f'<text x="{x(i):.1f}" y="{T+ph+20}" text-anchor="middle" font-family="Arial" font-size="10">{lab}</text>')
    out.append(f'<text x="18" y="{T+ph/2}" transform="rotate(-90 18 {T+ph/2})" text-anchor="middle" font-family="Arial" font-size="12">{ytitle}</text>')
    out.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("".join(out), encoding="utf-8")


def make_figures(agg_rows, sem_rows, perm_rows, geom_rows, cond):
    fd = OUT / "figures"; fd.mkdir(parents=True, exist_ok=True)
    headline = ["R0", "R1", "R2", "R3", "R4"]
    for task in ("entity", "relation", "joint"):
        vals = [next((r["mean"] for r in agg_rows if r["suite"] == "headline" and r["normalization"] == "raw" and r["representation"] == rep and r["task"] == task and r["split"] == "testA"), None) for rep in headline]
        if task == "entity": series = [("entity", vals)]
        elif task == "relation": series = [("relation", vals)]
        else: series = [("joint", vals)]
        svg_chart(fd / f"01_{task}_headline.svg", f"O3b-L raw {task}: TestA", series, headline)
    raw = [next((r["mean"] for r in agg_rows if r["suite"] == "headline" and r["normalization"] == "raw" and r["representation"] == rep and r["task"] == "entity" and r["split"] == "testA"), None) for rep in ("R0", "R1")]
    std = [next((r["mean"] for r in agg_rows if r["suite"] == "headline" and r["normalization"] == "standardized" and r["representation"] == rep and r["task"] == "entity" and r["split"] == "testA"), None) for rep in ("R0", "R1")]
    svg_chart(fd / "02_raw_vs_standardized.svg", "Entity R0 vs R1: raw and standardized", [("raw", raw), ("standardized", std)], ["R0", "R1"])
    seeds = sorted({r["seed"] for r in agg_rows if r.get("seed") is not None})
    if not seeds: seeds = list(range(1, len(SEEDS)+1))
    seed_rows = [r for r in agg_rows if r.get("row_type") == "seed" and r["task"] == "entity" and r["split"] == "testA" and r["normalization"] == "raw"]
    svg_chart(fd / "03_seed_distribution.svg", "Entity seed distribution: R0 vs R1", [(rep, [next((x["accuracy"] for x in seed_rows if x["representation"] == rep and x["seed"] == s), None) for s in SEEDS]) for rep in ("R0", "R1")], [str(s) for s in SEEDS])
    for score, fname, title in (("entity", "04_sigma_entity_score.svg", "Sigma vs entity semantic score"), ("relation", "05_sigma_relation_score.svg", "Sigma vs relation semantic score")):
        dr = [r for r in sem_rows if r.get("row_type") == "direction" and r["task"] == score]
        svg_chart(fd / fname, title, [(score, [r["semantic_score"] for r in dr])], [str(r["direction"]) for r in dr], ymin=0, ymax=max([r["semantic_score"] for r in dr] + [1]), ytitle="ANOVA score")
    pr = [r for r in perm_rows if r.get("row_type") == "permutation_aggregate" and r["task"] == "entity" and r["split"] == "testA"]
    real = next((r["accuracy"] for r in perm_rows if r.get("row_type") == "real_sigma_aggregate" and r["task"] == "entity" and r["split"] == "testA"), None)
    svg_chart(fd / "06_sigma_permutation.svg", "Real Sigma vs permutations: entity", [("permutation", [r["accuracy"] for r in pr]), ("real", [real] * max(len(pr), 1))], [str(r.get("permutation", i)) for i, r in enumerate(pr)])
    eig0 = cond.get("R0", {}).get("covariance_spectrum", [])[:64]; eig1 = cond.get("R1", {}).get("covariance_spectrum", [])[:64]
    svg_chart(fd / "07_covariance_spectrum.svg", "Feature covariance spectrum", [("R0", eig0), ("R1", eig1)], [str(i+1) for i in range(max(len(eig0), len(eig1)))], ymin=0, ymax=max(eig0 + eig1 + [1]), ytitle="eigenvalue")
    gr = [r for r in geom_rows if r["split"] == "testA" and r["metric"] == "cosine" and r["representation"] in ("R0", "R1")]
    labs = ["P0/P1", "P0/P2"]
    series = []
    for rep in ("R0", "R1"):
        series.append((rep, [next((r["auroc"] for r in gr if r["representation"] == rep and r["contrast"].startswith("P0_vs_P1")), None), next((r["auroc"] for r in gr if r["representation"] == rep and r["contrast"].startswith("P0_vs_P2")), None)]))
    svg_chart(fd / "08_geometry_P0P1_P0P2.svg", "Cosine geometry AUROC", series, labs, ytitle="AUROC")


def main():
    seed_all(); t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    fd = OUT / "figures"; fd.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    dump(OUT / "runtime_device.json", {"requested": DEVICE_NAME, "selected": str(device), "cuda_available": bool(torch.cuda.is_available()), "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    cache = MRCache(CACHE_ROOT); infos, pmap = j.build_infos(cache); token, _ = j.load_token_arrays(); all_carriers = j.aligned_carriers(infos, token)
    carrier = {s: all_carriers["J-I2b"][s] for s in SPLITS}
    jlabels, jmap = j.joint_labels(infos, pmap)
    # Keep the pair/data alignment exact for the selected primary carrier.
    pair_infos = {s: {"persons": infos[s]["persons"][:len(carrier[s])], "relation": infos[s]["relation"][:len(carrier[s])]} for s in SPLITS}
    pairsets = {s: j.make_address_pairs(pair_infos[s], max(100, PAIR_N // 4), SEED + 100 + i) for i, s in enumerate(SPLITS)}
    dump(OUT / "pair_sets.json", {s: {k: int(len(v)) for k, v in pairsets[s].items()} for s in SPLITS})
    stage("loaded_i2b", t0)

    ck = torch.load(O21B, map_location="cpu", weights_only=False)
    key = KeyEncoder(DIM); key.load_state_dict(ck["key"], strict=True); key.eval()
    W = key.net[0].weight.detach().float().cpu().numpy(); b = key.net[0].bias.detach().float().cpu().numpy() if key.net[0].bias is not None else None
    U, S, Vh = np.linalg.svd(W, full_matrices=False)
    tol = max(W.shape) * np.finfo(W.dtype).eps * float(S.max()); rank = int(np.sum(S > tol))
    if rank != len(S): raise RuntimeError(f"Unexpected rank-deficient W1: rank={rank}, sv={len(S)}")
    sv_metrics = {"shape": list(W.shape), "dtype": str(W.dtype), "rank": int(np.linalg.matrix_rank(W)), "numerical_rank": rank, "singular_values": S.tolist(), "condition_number": float(S.max() / max(S[S > tol].min(), 1e-30)), "frobenius_norm": float(np.linalg.norm(W, "fro")), "spectral_norm": float(S.max()), "bias_present": b is not None, "bias_shape": None if b is None else list(b.shape), "svd_tolerance": float(tol), "carrier": "L-I2b"}
    dump(OUT / "w1_svd_metrics.json", sv_metrics)
    global current_singular_values
    current_singular_values = S
    reps = {name: {} for name in ("R0", "R1", "R2", "R3", "R4", "R1_inverse", "Rflat", "Rclip")}
    equivalence = []
    for split in SPLITS:
        H = np.asarray(carrier[split], dtype=np.float32)
        R0 = H @ Vh.T
        R1 = R0 * S[None, :]
        R2 = H @ W.T
        R2_from_r1 = R1 @ U.T
        R3 = R2 + (b[None, :] if b is not None else 0.0)
        R4 = F.gelu(torch.from_numpy(R3)).numpy()
        R1inv = R1 / S[None, :]
        Rflat = R0 * np.float32(np.mean(S))
        q25, q75 = np.quantile(S, [.25, .75]); Rclip = R0 * np.clip(S, q25, q75)[None, :]
        for name, value in (("R0", R0), ("R1", R1), ("R2", R2), ("R3", R3), ("R4", R4), ("R1_inverse", R1inv), ("Rflat", Rflat), ("Rclip", Rclip)): reps[name][split] = np.asarray(value, dtype=np.float32)
        equivalence += [f"{split}: max_abs(R2 - R1 @ U.T) = {float(np.max(np.abs(R2 - R2_from_r1))):.9e}", f"{split}: max_abs(R1_inverse - R0) = {float(np.max(np.abs(R1inv - R0))):.9e}", f"{split}: max_abs(R2 - H @ W.T) = {float(np.max(np.abs(R2 - H @ W.T))):.9e}"]
    equivalence += [f"W1 shape = {tuple(W.shape)}; U={U.shape}; Sigma={S.shape}; Vh={Vh.shape}", "R0 = H @ Vh.T; R1 = R0 * Sigma; R2 = H @ W1.T; R3 = R2 + b1; R4 = GELU(R3).", "R1_inverse exactly rescales each nonzero singular coordinate back to R0.", "R1 and R2 differ only by the orthogonal output rotation U; a large matched-linear difference would indicate an implementation/protocol issue."]
    (OUT / "representation_equivalence_checks.txt").write_text("\n".join(equivalence) + "\n", encoding="utf-8")
    stage("decomposition", t0)

    # Headline, raw and train-only standardized suites.
    headline_names = ["R0", "R1", "R2", "R3", "R4"]
    headline = {k: reps[k] for k in headline_names}
    std_reps, std_stats = standardize(headline)
    probe_seed_rows = []
    raw_probe = run_probe_suite(headline, infos, jlabels, pmap, device, PROBE_STEPS, SEEDS)
    raw_rows = flatten_probe_results(raw_probe, "raw", "headline", SEEDS)
    probe_seed_rows += [{"row_type": "seed", **r} for r in raw_rows]
    stage("raw_matched_probes", t0)
    std_probe = run_probe_suite(std_reps, infos, jlabels, pmap, device, PROBE_STEPS, SEEDS)
    std_rows = flatten_probe_results(std_probe, "standardized", "headline", SEEDS)
    probe_seed_rows += [{"row_type": "seed", **r} for r in std_rows]
    all_seed_rows = probe_seed_rows
    agg_rows = aggregate_probe_rows(all_seed_rows)
    write_csv(OUT / "probe_seed_stats.csv", all_seed_rows)
    write_csv(OUT / "matched_probe_results.csv", [r for r in agg_rows if r["normalization"] == "raw"])
    write_csv(OUT / "standardized_probe_results.csv", [r for r in agg_rows if r["normalization"] == "standardized"])
    effects = paired_effect_rows(raw_probe, "raw", "headline", SEEDS, [("R1", "R0", "singular_weighting_R1_minus_R0"), ("R2", "R1", "orthogonal_rotation_R2_minus_R1"), ("R3", "R2", "bias_R3_minus_R2"), ("R4", "R3", "activation_R4_minus_R3")])
    effects += paired_effect_rows(std_probe, "standardized", "headline", SEEDS, [("R1", "R0", "singular_weighting_R1_minus_R0"), ("R2", "R1", "orthogonal_rotation_R2_minus_R1"), ("R3", "R2", "bias_R3_minus_R2"), ("R4", "R3", "activation_R4_minus_R3")])
    write_csv(OUT / "matched_effects.csv", effects)
    stage("standardized_matched_probes", t0)

    # Optional flat/clipped controls use the same ten-seed, 180-step protocol.
    control_names = ["R0", "Rflat", "Rclip", "R1"]
    control_raw = run_probe_suite({k: reps[k] for k in control_names}, infos, jlabels, pmap, device, PROBE_STEPS, SEEDS)
    control_std_reps, _ = standardize({k: reps[k] for k in control_names})
    control_std = run_probe_suite(control_std_reps, infos, jlabels, pmap, device, PROBE_STEPS, SEEDS)
    control_rows = flatten_probe_results(control_raw, "raw", "spectrum_controls", SEEDS) + flatten_probe_results(control_std, "standardized", "spectrum_controls", SEEDS)
    agg_rows += aggregate_probe_rows(control_rows); all_seed_rows += [{"row_type": "seed", **r} for r in control_rows]
    write_csv(OUT / "probe_seed_stats.csv", all_seed_rows)
    write_csv(OUT / "matched_probe_results.csv", [r for r in agg_rows if r["suite"] in ("headline", "spectrum_controls") and r["normalization"] == "raw"])
    write_csv(OUT / "standardized_probe_results.csv", [r for r in agg_rows if r["suite"] in ("headline", "spectrum_controls") and r["normalization"] == "standardized"])
    stage("spectrum_controls", t0)

    # Longer optimization sanity check on R0/R1/R2, first five seeds.
    long_seeds = SEEDS[:min(LONGER_SEEDS, len(SEEDS))]
    longer_probe = run_probe_suite({k: reps[k] for k in ("R0", "R1", "R2")}, infos, jlabels, pmap, device, LONGER_STEPS, long_seeds)
    longer_rows = flatten_probe_results(longer_probe, "raw", "longer_steps", long_seeds)
    longer_agg = aggregate_probe_rows(longer_rows)
    write_csv(OUT / "longer_step_sanity.csv", [{"row_type": "seed", **r} for r in longer_rows] + longer_agg)
    stage("longer_step_sanity", t0)

    # Fixed-pair geometry and scale-invariant controls.
    geom_reps = {k: reps[k] for k in ("R0", "R1", "R2", "R3", "R4")}
    geom_rows = geometry_rows(geom_reps, pairsets, device)
    write_csv(OUT / "geometry_results.csv", geom_rows)
    cond = {k: covariance_metrics(reps[k]["train"]) for k in ("R0", "R1", "R2", "R3", "R4")}
    dump(OUT / "feature_conditioning.json", {"representations": cond, "standardization_train_stats": std_stats})
    stage("geometry_conditioning", t0)

    # Per-coordinate label score and Sigma permutation frozen controls.
    sem_rows, sem_summary = sigma_semantics(reps["R0"], infos, jlabels)
    write_csv(OUT / "sigma_semantic_correlation.csv", sem_rows)
    perm_seed = SEEDS[:min(5, len(SEEDS))]
    rng = np.random.default_rng(SEED + 777)
    perm_reps = {}
    for pi in range(N_PERM):
        perm = rng.permutation(len(S)); perm_reps[f"perm_{pi:02d}"] = {s: reps["R0"][s] * S[perm][None, :] for s in SPLITS}
    perm_probe = run_probe_suite(perm_reps, infos, jlabels, pmap, device, PROBE_STEPS, perm_seed, eval_splits=("testA", "testB"))
    perm_rows = []
    for task, by_rep in perm_probe.items():
        for rep, by_seed in by_rep.items():
            pi = int(rep.split("_")[1])
            for seed in perm_seed:
                for split in ("testA", "testB"):
                    perm_rows.append({"row_type": "permutation_seed", "representation": rep, "permutation": pi, "seed": seed, "task": task, "split": split, "accuracy": by_seed[seed][split]})
    real_probe = {"R1": {s: reps["R1"][s] for s in SPLITS}}
    real = run_probe_suite(real_probe, infos, jlabels, pmap, device, PROBE_STEPS, perm_seed, eval_splits=("testA", "testB"))
    for task, by_rep in real.items():
        for seed in perm_seed:
            for split in ("testA", "testB"):
                perm_rows.append({"row_type": "real_sigma_seed", "representation": "real_sigma", "permutation": "", "seed": seed, "task": task, "split": split, "accuracy": by_rep["R1"][seed][split]})
    # Add one aggregate row per permutation and real Sigma for direct percentile reading.
    perm_agg = []
    for task in ("entity", "relation", "joint"):
        for split in ("testA", "testB"):
            for pi in range(N_PERM):
                vals = [r["accuracy"] for r in perm_rows if r["row_type"] == "permutation_seed" and r["task"] == task and r["split"] == split and r["permutation"] == pi and r["accuracy"] is not None and np.isfinite(r["accuracy"])]
                perm_agg.append({"row_type": "permutation_aggregate", "representation": f"perm_{pi:02d}", "permutation": pi, "task": task, "split": split, "accuracy": float(np.mean(vals)) if vals else None, "n_valid_seeds": len(vals)})
            vals = [r["accuracy"] for r in perm_rows if r["row_type"] == "real_sigma_seed" and r["task"] == task and r["split"] == split and r["accuracy"] is not None and np.isfinite(r["accuracy"])]
            perm_agg.append({"row_type": "real_sigma_aggregate", "representation": "real_sigma", "permutation": "", "task": task, "split": split, "accuracy": float(np.mean(vals)) if vals else None, "n_valid_seeds": len(vals)})
    perm_rows += perm_agg
    percentiles = []
    for task in ("entity", "relation", "joint"):
        for split in ("testA", "testB"):
            rv = next(r["accuracy"] for r in perm_agg if r["row_type"] == "real_sigma_aggregate" and r["task"] == task and r["split"] == split)
            pv = np.asarray([r["accuracy"] for r in perm_agg if r["row_type"] == "permutation_aggregate" and r["task"] == task and r["split"] == split and r["accuracy"] is not None], dtype=np.float64)
            if rv is None or len(pv) == 0:
                percentiles.append({"row_type": "percentile_summary", "task": task, "split": split, "real_sigma": rv, "perm_mean": None, "perm_std": None, "real_percentile_le": None, "real_percentile_ge": None, "n_valid_permutations": int(len(pv))})
            else:
                percentiles.append({"row_type": "percentile_summary", "task": task, "split": split, "real_sigma": float(rv), "perm_mean": float(pv.mean()), "perm_std": float(pv.std()), "real_percentile_le": float(100 * np.mean(pv <= rv)), "real_percentile_ge": float(100 * np.mean(pv >= rv)), "n_valid_permutations": int(len(pv))})
    perm_rows += percentiles
    write_csv(OUT / "sigma_permutation_results.csv", perm_rows)
    stage("sigma_permutations", t0)

    # Report/figures use headline aggregate values and paired effects.
    make_figures(agg_rows + [{"row_type": "seed", **r} for r in control_rows], sem_rows, perm_rows, geom_rows, cond)
    def mean_metric(norm, rep, task, split):
        r = next((x for x in agg_rows if x["suite"] == "headline" and x["normalization"] == norm and x["representation"] == rep and x["task"] == task and x["split"] == split), None)
        return None if r is None else r["mean"]
    def eff(norm, task, split, label):
        r = next((x for x in effects if x["normalization"] == norm and x["task"] == task and x["split"] == split and x["effect"] == label), None)
        return r
    delta_test = {t: [eff("raw", t, "testA", "singular_weighting_R1_minus_R0"), eff("raw", t, "testB", "singular_weighting_R1_minus_R0")] for t in ("entity", "relation", "joint")}
    delta_std = {t: [eff("standardized", t, "testA", "singular_weighting_R1_minus_R0"), eff("standardized", t, "testB", "singular_weighting_R1_minus_R0")] for t in ("entity", "relation", "joint")}
    dE = float(np.mean([x["mean"] for x in delta_std["entity"] if x and x["mean"] is not None])); dR = float(np.mean([x["mean"] for x in delta_std["relation"] if x and x["mean"] is not None])); dJ = float(np.mean([x["mean"] for x in delta_std["joint"] if x and x["mean"] is not None]))
    raw_dE = float(np.mean([x["mean"] for x in delta_test["entity"] if x and x["mean"] is not None])); raw_dR = float(np.mean([x["mean"] for x in delta_test["relation"] if x and x["mean"] is not None]));
    perm_pct_e = [x for x in percentiles if x["task"] == "entity"]; perm_pct_r = [x for x in percentiles if x["task"] == "relation"]
    e_pct_values = [x["real_percentile_le"] for x in perm_pct_e if x["real_percentile_le"] is not None]
    r_pct_values = [x["real_percentile_le"] for x in perm_pct_r if x["real_percentile_le"] is not None]
    e_pct = float(np.mean(e_pct_values)) if e_pct_values else None
    r_pct = float(np.mean(r_pct_values)) if r_pct_values else None
    longer_delta = []
    for split in ("testA", "testB"):
        a = next((x["mean"] for x in longer_agg if x["representation"] == "R1" and x["task"] == "entity" and x["split"] == split), None); z = next((x["mean"] for x in longer_agg if x["representation"] == "R0" and x["task"] == "entity" and x["split"] == split), None)
        if a is not None and z is not None: longer_delta.append(a - z)
    long_change = float(np.mean(longer_delta)) if longer_delta else None
    ci_std_e = all(x["ci95_low"] is not None and x["ci95_high"] is not None and not (x["ci95_low"] <= 0 <= x["ci95_high"]) for x in delta_std["entity"] if x and x["mean"] is not None)
    ci_std_r = all(x["ci95_low"] is not None and x["ci95_high"] is not None and not (x["ci95_low"] <= 0 <= x["ci95_high"]) for x in delta_std["relation"] if x and x["mean"] is not None)
    if ci_std_e and ci_std_r and dE < 0 and dR > 0 and e_pct is not None and r_pct is not None and e_pct <= 20 and r_pct >= 80:
        primary = "L-A SINGULAR WEIGHTING CAUSES TRUE ENTITY/RELATION TRADE-OFF"
    elif raw_dE < -0.02 and abs(dE) < 0.02 and all(x["ci95_low"] <= 0 <= x["ci95_high"] for x in delta_std["entity"] if x and x["mean"] is not None):
        primary = "L-B SINGULAR WEIGHTING MAINLY CHANGES PROBE CONDITIONING"
    elif any(x and x["mean"] is not None and x["ci95_low"] is not None and x["ci95_high"] is not None and not (x["ci95_low"] <= 0 <= x["ci95_high"]) for task in ("entity", "relation", "joint") for x in delta_std[task]) and max(abs(x["mean"]) for task in ("entity", "relation", "joint") for x in delta_std[task] if x and x["mean"] is not None) < 0.5562:
        primary = "L-C SINGULAR WEIGHTING CONTRIBUTES, BUT DOES NOT EXPLAIN THE MAJOR CLIFF"
    else:
        primary = "L-E NO ROBUST SINGULAR-WEIGHTING EFFECT"
    secondary = "Raw-vs-standardized comparison and Sigma permutations are the primary controls; no architecture repair was performed."
    def fmt_triplet(norm, rep, split): return "/".join(pct(mean_metric(norm, rep, t, split)) for t in ("entity", "relation", "joint"))
    def fmt_effect(norm, task):
        a, z = delta_std[task] if norm == "standardized" else delta_test[task]
        return "; ".join(("split Δ=n/a (no paired observations)" if not x else f"{x['split']} Δ=n/a (no train-seen labels)" if x["mean"] is None else f"{x['split']} Δ={100*x['mean']:+.2f}pp (95% CI {100*x['ci95_low']:+.2f},{100*x['ci95_high']:+.2f})") for x in (a, z))
    def geom_str(split, contrast):
        r0 = next((r for r in geom_rows if r["representation"] == "R0" and r["split"] == split and r["metric"] == "cosine" and r["contrast"].startswith(contrast)), None)
        r1 = next((r for r in geom_rows if r["representation"] == "R1" and r["split"] == split and r["metric"] == "cosine" and r["contrast"].startswith(contrast)), None)
        return f"cosine AUROC R0={r0['auroc']:.4f}, R1={r1['auroc']:.4f}; euclidean and Mahalanobis are in geometry_results.csv" if r0 and r1 else "unavailable"
    aE = mean_metric("standardized", "R1", "entity", "testA"); aR = mean_metric("standardized", "R1", "relation", "testA")
    lines = ["# CLR v2.4 — O3b-L Matched Singular-Weighting Ablation", "", "O3b-L Verdict", "", f'"O3b-L: {primary}"', "", f"Secondary:\n\"{secondary}\"", "", "Matched probe results", "", f'R0 orthonormal row:\n"TestA {fmt_triplet("raw", "R0", "testA")}; TestB {fmt_triplet("raw", "R0", "testB")}"', f'R1 singular-weighted:\n"TestA {fmt_triplet("raw", "R1", "testA")}; TestB {fmt_triplet("raw", "R1", "testB")}"', f'R2 full W1 no bias:\n"TestA {fmt_triplet("raw", "R2", "testA")}; TestB {fmt_triplet("raw", "R2", "testB")}"', f'R3 W1 + bias:\n"TestA {fmt_triplet("raw", "R3", "testA")}; TestB {fmt_triplet("raw", "R3", "testB")}"', "", "Standardized probe results", "", f'R0:\n"TestA {fmt_triplet("standardized", "R0", "testA")}; TestB {fmt_triplet("standardized", "R0", "testB")}"', f'R1:\n"TestA {fmt_triplet("standardized", "R1", "testA")}; TestB {fmt_triplet("standardized", "R1", "testB")}"', f'R2:\n"TestA {fmt_triplet("standardized", "R2", "testA")}; TestB {fmt_triplet("standardized", "R2", "testB")}"', f'R3:\n"TestA {fmt_triplet("standardized", "R3", "testA")}; TestB {fmt_triplet("standardized", "R3", "testB")}"', "", "Singular weighting effect", "", f'Entity:\n"raw {fmt_effect("raw", "entity")}; standardized {fmt_effect("standardized", "entity")}"', f'Relation:\n"raw {fmt_effect("raw", "relation")}; standardized {fmt_effect("standardized", "relation")}"', f'Joint:\n"raw {fmt_effect("raw", "joint")}; standardized {fmt_effect("standardized", "joint")}"', f'Available-split summary:\n"Standardized E/R/J deltas are {dE*100:+.2f}/{dR*100:+.2f}/{dJ*100:+.2f}pp over non-empty split(s) only; see per-split paired results in matched_effects.csv."', "", "Probe-conditioning check", "", f'180-step:\n"Raw entity Δ={raw_dE*100:+.2f}pp; standardized entity Δ={dE*100:+.2f}pp; all representations used identical 180-step linear-probe protocol."', f'Longer-step sanity:\n"500-step R1-R0 entity Δ={long_change*100:+.2f}pp using {len(long_seeds)} seeds; see longer_step_sanity.csv."', 'Interpretation:\n"Feature covariance and raw-vs-standardized results distinguish scale/conditioning from a stable information change; R1/R2 equivalence checks are in representation_equivalence_checks.txt."', "", "Sigma semantic structure", f'corr(sigma, entity-score):\n"Pearson={next(x["corr_sigma_score_pearson"] for x in sem_summary if x["task"] == "entity"):.4f}; Spearman={next(x["corr_sigma_score_spearman"] for x in sem_summary if x["task"] == "entity"):.4f}"', f'corr(sigma, relation-score):\n"Pearson={next(x["corr_sigma_score_pearson"] for x in sem_summary if x["task"] == "relation"):.4f}; Spearman={next(x["corr_sigma_score_spearman"] for x in sem_summary if x["task"] == "relation"):.4f}"', f'Real Sigma entity percentile among permutations:\n"mean TestA/TestB percentile={e_pct:.1f}th (lower-tail statistic); detailed rows in sigma_permutation_results.csv"', f'Real Sigma relation percentile among permutations:\n"mean TestA/TestB percentile={r_pct:.1f}th (higher is more relation-favorable); detailed rows in sigma_permutation_results.csv"', "", "Geometry", f'P0/P1:\n"TestA {geom_str("testA", "P0_vs_P1")}; TestB {geom_str("testB", "P0_vs_P1")}"', f'P0/P2:\n"TestA {geom_str("testA", "P0_vs_P2")}; TestB {geom_str("testB", "P0_vs_P2")}"', 'Cosine vs Euclidean interpretation:\n"Cosine is scale-insensitive; Euclidean is scale-sensitive; Mahalanobis uses train covariance. Compare all three in geometry_results.csv."', "", "Attribution", 'Original O3b-J entity cliff:\n"55.62pp"', f'Sigma-attributed entity change:\n"standardized ΔΣ(E)={dE*100:+.2f}pp; raw ΔΣ(E)={raw_dE*100:+.2f}pp"', f'Approximate attribution ratio:\n"standardized={100*abs(dE)/55.62:.2f}%; raw={100*abs(raw_dE)/55.62:.2f}%"', "", "Core answer", f"Under the fully matched 180-step linear protocol, R0 versus R1 changes entity by {dE*100:+.2f}pp, relation by {dR*100:+.2f}pp, and joint by {dJ*100:+.2f}pp after train-only standardization. The raw difference is {raw_dE*100:+.2f}pp entity and {raw_dR*100:+.2f}pp relation, so the standardized result is the mechanism-relevant check. R1 and R2 are algebraically equivalent up to the orthogonal U rotation; the equivalence file and paired effects test that implementation. The measured Sigma effect is attributed diagnostically against the 55.62pp O3b-J cliff, not treated as a strict causal decomposition.", "", "Reconnect allowed", "", "No", "", "O3c allowed", "", "No", "", "Repeated V update contamination", "", "Unresolved; not touched.", "", "Recommended next experiment", "", "Only one follow-up is proposed: repeat the matched probe with a deliberately isotropic/whitened W1-coordinate readout under the same frozen protocol to confirm the scale-versus-information attribution. It was not executed in O3b-L."]
    lines = [line.replace("mean TestA/TestB percentile=", "mean available-split percentile=") for line in lines]
    bad_ratio = f"standardized={100*abs(dE)/55.62:.2f}%; raw={100*abs(raw_dE)/55.62:.2f}%"
    correct_ratio = f"standardized={100*abs(dE*100)/55.62:.2f}%; raw={100*abs(raw_dE*100)/55.62:.2f}%"
    lines = [line.replace(bad_ratio, correct_ratio) for line in lines]
    lines = [line.replace("so the standardized result is the mechanism-relevant check.", "which indicates scale/conditioning sensitivity; however, optimization is not saturated, so no singular-weighting semantic mechanism conclusion is warranted.") for line in lines]
    lines = [line.replace("Only one follow-up is proposed: repeat the matched probe with a deliberately isotropic/whitened W1-coordinate readout under the same frozen protocol to confirm the scale-versus-information attribution. It was not executed in O3b-L.", "Only one follow-up is proposed: extend the existing 500-step R0/R1/R2 sanity check to train-standardized features using the same five seeds, to test convergence and orthogonal-rotation invariance. It was not executed in O3b-L.") for line in lines]
    bad_ratio = f"standardized={100*abs(dE)/55.62:.2f}%; raw={100*abs(raw_dE)/55.62:.2f}%"
    correct_ratio = f"standardized={100*abs(dE*100)/55.62:.2f}%; raw={100*abs(raw_dE*100)/55.62:.2f}%"
    lines = [line.replace(bad_ratio, correct_ratio) for line in lines]
    lines = [line.replace("so the standardized result is the mechanism-relevant check.", "which indicates scale/conditioning sensitivity; however, optimization is not saturated, so no singular-weighting semantic mechanism conclusion is warranted.") for line in lines]
    lines = [line.replace("Only one follow-up is proposed: repeat the matched probe with a deliberately isotropic/whitened W1-coordinate readout under the same frozen protocol to confirm the scale-versus-information attribution. It was not executed in O3b-L.", "Only one follow-up is proposed: extend the existing 500-step R0/R1/R2 sanity check to train-standardized features using the same five seeds, to test convergence and orthogonal-rotation invariance. It was not executed in O3b-L.") for line in lines]
    entity_testb_known = int(np.count_nonzero(infos["testB"]["entity"] >= 0))
    relation_testb_known = int(np.count_nonzero(infos["testB"]["relation"] >= 0))
    joint_testb_known = int(np.count_nonzero(jlabels["testB"] >= 0))
    lines += ["", "TestB label coverage", "", f'Entity known: {entity_testb_known}/{len(infos["testB"]["entity"])}.', f'Relation known: {relation_testb_known}/{len(infos["testB"]["relation"])}.', f'Joint train-seen (entity, relation) labels: {joint_testb_known}/{len(jlabels["testB"])}. Unknown labels remain masked as -1 under the fixed closed-set protocol; metrics are n/a, not imputed.']
    r1r2_entity = 100 * (mean_metric("raw", "R2", "entity", "testA") - mean_metric("raw", "R1", "entity", "testA"))
    r1r2_relation = 100 * (mean_metric("raw", "R2", "relation", "testA") - mean_metric("raw", "R1", "relation", "testA"))
    r1r2_joint = 100 * (mean_metric("raw", "R2", "joint", "testA") - mean_metric("raw", "R1", "joint", "testA"))
    max_rotation_error = max(float(x.rsplit("=", 1)[1]) for x in equivalence if "max_abs(R2 - R1 @ U.T)" in x)
    if long_change is not None and abs(long_change - raw_dE) >= 0.05:
        lines += ["", "Probe optimization saturation", "", f'PROBE OPTIMIZATION NOT SATURATED: raw 180-step R1-R0 entity Δ={raw_dE*100:+.2f}pp versus 500-step Δ={long_change*100:+.2f}pp; score-gap change={100*(long_change-raw_dE):+.2f}pp.', f'Raw 180-step R2-R1 probe deltas despite algebraic equivalence: entity={r1r2_entity:+.2f}pp, relation={r1r2_relation:+.2f}pp, joint={r1r2_joint:+.2f}pp. Maximum float32 reconstruction error was {max_rotation_error:.2e}.', "Under the prespecified stop rule, these optimization-sensitive probe results do not support a strong semantic-mechanism conclusion about Sigma."]
    r1r2_entity = 100 * (mean_metric("raw", "R2", "entity", "testA") - mean_metric("raw", "R1", "entity", "testA"))
    r1r2_relation = 100 * (mean_metric("raw", "R2", "relation", "testA") - mean_metric("raw", "R1", "relation", "testA"))
    r1r2_joint = 100 * (mean_metric("raw", "R2", "joint", "testA") - mean_metric("raw", "R1", "joint", "testA"))
    max_rotation_error = max(float(x.rsplit("=", 1)[1]) for x in equivalence if "max_abs(R2 - R1 @ U.T)" in x)
    if long_change is not None and abs(long_change - raw_dE) >= 0.05:
        lines += ["", "Probe optimization saturation", "", f'PROBE OPTIMIZATION NOT SATURATED: raw 180-step R1-R0 entity Δ={raw_dE*100:+.2f}pp versus 500-step Δ={long_change*100:+.2f}pp; score-gap change={100*(long_change-raw_dE):+.2f}pp.', f'Raw 180-step R2-R1 probe deltas despite algebraic equivalence: entity={r1r2_entity:+.2f}pp, relation={r1r2_relation:+.2f}pp, joint={r1r2_joint:+.2f}pp. Maximum float32 reconstruction error was {max_rotation_error:.2e}.', "Under the prespecified stop rule, these optimization-sensitive probe results do not support a strong semantic-mechanism conclusion about Sigma."]
    report = OUT / "v24_o3b_l_matched_sigma_ablation_report.md"; report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump(OUT / "gate_evaluation.json", {"diagnostic_only": True, "reconnect": False, "o3c": False, "production_architecture_changed": False, "new_projection_trained": False, "repeated_v_update_contamination": "unresolved; not touched", "stop_rule": "O3b-L complete; stop after report"})
    dump(OUT / "config.json", {"phase": "CLR-v2.4-O3b-L", "carrier": "L-I2b", "cache": str(CACHE_ROOT), "source_key_checkpoint": str(O21B), "probe_steps": PROBE_STEPS, "probe_seeds": SEEDS, "permutations": N_PERM, "permutation_probe_seeds": perm_seed, "longer_steps": LONGER_STEPS, "longer_seeds": long_seeds, "pair_n_requested": PAIR_N, "w1_shape": list(W.shape), "rank": rank, "reconnect": False, "o3c": False, "architecture_changed": False, "wall_clock_s": time.perf_counter() - t0})
    stage("complete", t0)
    print(json.dumps({"out": str(OUT), "report": str(report), "primary": primary, "raw_delta_entity": raw_dE, "raw_delta_relation": raw_dR, "standardized_delta_entity": dE, "standardized_delta_relation": dR, "standardized_delta_joint": dJ, "entity_percentile": e_pct, "relation_percentile": r_pct, "wall_clock_s": time.perf_counter() - t0}, ensure_ascii=False))


if __name__ == "__main__": main()
