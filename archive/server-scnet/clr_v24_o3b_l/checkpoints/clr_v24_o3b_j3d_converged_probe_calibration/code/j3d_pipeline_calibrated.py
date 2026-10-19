from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CACHE = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache")
OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3d_converged_probe_calibration")
SPLITS = ("train", "val", "testA", "testB")
SEEDS = [11, 23, 37, 53, 71]
TASKS = ("entity", "relation", "joint")
TASK_OFFSET = {"entity": 1, "relation": 2, "joint": 3}
LABEL_FIELD = {"entity": "entity_class_id", "relation": "relation_class_id", "joint": "joint_class_id"}
MIN_STEPS = 1000
MAX_STEPS = 50000
EVAL_EVERY = 100
PATIENCE = 30
MIN_REL_LOSS_GAIN = 1e-5
MIN_ACC_GAIN = 1e-4
STD_EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOOT_DRAWS = 20000
BOOT_SEED = 18431


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    return obj


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status,reason\nNOT_RUN,no rows produced\n", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ci(values: list[float], seed: int) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "t95_low": None, "t95_high": None, "bootstrap95_low": None, "bootstrap95_high": None}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
    lo, hi = (mean, mean) if n == 1 else (mean - tcrit * sd / math.sqrt(n), mean + tcrit * sd / math.sqrt(n))
    if n == 1:
        blo, bhi = mean, mean
    else:
        rng = np.random.default_rng(seed)
        boot = x[rng.integers(0, n, size=(BOOT_DRAWS, n))].mean(1)
        blo, bhi = np.quantile(boot, [0.025, 0.975])
    return {"n": n, "mean": mean, "std": sd, "t95_low": float(lo), "t95_high": float(hi),
            "bootstrap95_low": float(blo), "bootstrap95_high": float(bhi)}


def label_arrays(split: str, tasks=TASKS) -> dict[str, np.ndarray]:
    rows = read_csv(CACHE / f"labels_{split}.csv")
    return {t: np.asarray([int(r[LABEL_FIELD[t]]) for r in rows], dtype=np.int64) for t in tasks}


def feature_arrays(stage: str, splits=("train", "val")) -> dict[str, np.ndarray]:
    return {s: np.asarray(np.load(CACHE / f"{stage}_{s}.npy", allow_pickle=False), dtype=np.float32) for s in splits}


def build_stats_and_arrays(stages: list[str], splits=("train", "val")) -> tuple[dict, dict]:
    stats, arrays = {}, {}
    for stage in stages:
        raw = feature_arrays(stage, splits)
        train = raw["train"].astype(np.float64)
        mu = train.mean(0)
        sd = train.std(0)
        tiny = sd < STD_EPS
        sd[tiny] = 1.0
        std = {s: ((x.astype(np.float64) - mu) / sd).astype(np.float32) for s, x in raw.items()}
        arrays[stage] = {"raw": raw, "standardized": std}
        stats[stage] = {
            "train_only": True, "mean": mu.tolist(), "std": sd.tolist(),
            "near_zero_std_replaced_dims": int(tiny.sum()), "epsilon": STD_EPS,
            "train_rows": int(len(train)), "dimension": int(train.shape[1]),
            "raw_train_feature_norm_mean": float(np.linalg.norm(train, axis=1).mean()),
            "raw_train_feature_norm_std": float(np.linalg.norm(train, axis=1).std()),
        }
    return stats, arrays


def task_class_info(ytr: np.ndarray, yval: np.ndarray, task: str) -> dict:
    known = ytr[ytr >= 0]
    if len(known) != len(ytr):
        raise ValueError(f"training labels contain unknowns for {task}")
    classes = np.unique(known)
    if not np.array_equal(classes, np.arange(int(classes[-1]) + 1)):
        raise ValueError(f"non-contiguous frozen class IDs for {task}; do not remap: {classes[:10]} ...")
    if len(classes) < 2:
        raise ValueError(f"fewer than two train classes for {task}")
    val_known = int(np.sum((yval >= 0) & (yval < len(classes))))
    if val_known == 0:
        raise ValueError(f"no train-known validation labels for {task}; cannot calibrate stopping without leakage")
    return {"num_classes": int(len(classes)), "train_known": int(len(known)),
            "val_known": val_known,
            "val_unknown": int(np.sum((yval < 0) | (yval >= len(classes)))),
            "class_ids_unchanged": True}


def make_optimizer(kind: str, params, lr: float, wd: float):
    if kind == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if kind == "SGD_momentum_0.9":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(kind)


def init_params(nseed: int, nrep: int, nclass: int, dim: int, seeds: list[int], task: str):
    ws, bs = [], []
    bound = 1.0 / math.sqrt(dim)
    for seed in seeds:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + TASK_OFFSET[task] * 1009)
        w = torch.empty(nclass, dim).uniform_(-bound, bound, generator=g)
        b = torch.empty(nclass).uniform_(-bound, bound, generator=g)
        ws.append(w.unsqueeze(0).repeat(nrep, 1, 1))
        bs.append(b.unsqueeze(0).repeat(nrep, 1))
    w = torch.cat(ws, dim=0).to(DEVICE)
    b = torch.cat(bs, dim=0).to(DEVICE)
    return torch.nn.Parameter(w), torch.nn.Parameter(b)


@torch.no_grad()
def eval_models(X: torch.Tensor, y: torch.Tensor, W: torch.Tensor, B: torch.Tensor, chunk: int = 128):
    m, n, _ = X.shape
    nclass = W.shape[1]
    known = (y >= 0) & (y < nclass)
    if not bool(known.any()):
        return [None] * m, [None] * m
    sums = torch.zeros(m, device=X.device, dtype=torch.float64)
    correct = torch.zeros(m, device=X.device, dtype=torch.long)
    total = 0
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        mask = known[start:end]
        if not bool(mask.any()):
            continue
        xb = X[:, start:end][..., :]
        logits = torch.bmm(xb, W.transpose(1, 2)) + B[:, None, :]
        yy = y[start:end][mask]
        logits = logits[:, mask]
        flat = logits.reshape(m * len(yy), nclass)
        tgt = yy.unsqueeze(0).expand(m, -1).reshape(-1)
        losses = F.cross_entropy(flat, tgt, reduction="none").reshape(m, -1)
        sums += losses.double().sum(1)
        correct += (logits.argmax(-1) == yy.unsqueeze(0)).sum(1)
        total += len(yy)
    return (sums.div(total).cpu().tolist(), correct.float().div(total).cpu().tolist())


def curve_slopes(points: list[dict], index: int) -> tuple[float | None, float | None]:
    if len(points) < 4:
        return None, None
    tail = points[max(0, len(points) - 10):]
    x = np.asarray([r["step"] / EVAL_EVERY for r in tail], dtype=np.float64)
    loss = np.asarray([r["val_loss"] for r in tail], dtype=np.float64)
    acc = np.asarray([r["val_accuracy"] for r in tail], dtype=np.float64)
    ls = float(np.polyfit(x, loss, 1)[0] / max(abs(float(loss.mean())), 1e-12))
    ac = float(np.polyfit(x, acc, 1)[0])
    return ls, ac


def fit_group(task: str, rep_inputs: dict[str, tuple[np.ndarray, np.ndarray]], ytr_np: np.ndarray,
              yval_np: np.ndarray, seeds: list[int], config: dict, tag: str,
              max_steps: int = MAX_STEPS, early_stop: bool = True):
    rep_names = list(rep_inputs)
    nrep = len(rep_names)
    info = task_class_info(ytr_np, yval_np, task)
    nclass = info["num_classes"]
    dim = int(rep_inputs[rep_names[0]][0].shape[1])
    if any(x[0].shape[1] != dim or x[1].shape[1] != dim for x in rep_inputs.values()):
        raise ValueError(f"group mixes dimensions: {[(k, v[0].shape[1]) for k,v in rep_inputs.items()]}")
    if any(len(v[0]) != len(ytr_np) or len(v[1]) != len(yval_np) for v in rep_inputs.values()):
        raise ValueError("feature/label count mismatch")
    xbase = torch.from_numpy(np.stack([rep_inputs[k][0] for k in rep_names])).to(DEVICE)
    xvbase = torch.from_numpy(np.stack([rep_inputs[k][1] for k in rep_names])).to(DEVICE)
    nseed = len(seeds)
    X = xbase.unsqueeze(0).expand(nseed, -1, -1, -1).reshape(nseed * nrep, len(ytr_np), dim).contiguous()
    XV = xvbase.unsqueeze(0).expand(nseed, -1, -1, -1).reshape(nseed * nrep, len(yval_np), dim).contiguous()
    ytr = torch.from_numpy(ytr_np).long().to(DEVICE)
    yval = torch.from_numpy(yval_np).long().to(DEVICE)
    W, B = init_params(nseed, nrep, nclass, dim, seeds, task)
    opt = make_optimizer(config["optimizer"], [W, B], float(config["lr"]), float(config["weight_decay"]))
    rng = np.random.default_rng(42 + 913 + TASK_OFFSET[task])
    batch = min(1024, int(np.sum(ytr_np >= 0)))
    model_count = nseed * nrep
    curves: list[dict] = []
    model_state = [{"best_loss": float("inf"), "best_acc": -1.0, "best_acc_seen": -1.0,
                    "best_step": 0, "best_w": None, "best_b": None, "patience": 0,
                    "stopped": False, "stop_step": None, "stop_reason": None,
                    "best_train_loss": None, "best_train_acc": None, "best_grad_norm": None,
                    "best_weight_norm": None, "best_bias_norm": None, "points": []}
                   for _ in range(model_count)]
    train_loss_window, train_acc_window, grad_window = [], [], []

    def evaluate(step: int):
        val_losses, val_accs = eval_models(XV, yval, W.detach(), B.detach())
        with torch.no_grad():
            wnorm = torch.sqrt((W.detach() ** 2).sum((1, 2))).cpu().numpy()
            bnorm = torch.sqrt((B.detach() ** 2).sum(1)).cpu().numpy()
        avg_train_loss = float(np.mean(train_loss_window[-EVAL_EVERY:])) if train_loss_window else None
        avg_train_acc = float(np.mean(train_acc_window[-EVAL_EVERY:])) if train_acc_window else None
        avg_grad = float(np.mean(grad_window[-EVAL_EVERY:])) if grad_window else None
        for mi in range(model_count):
            state = model_state[mi]
            vl, va = float(val_losses[mi]), float(val_accs[mi])
            point = {"step": step, "val_loss": vl, "val_accuracy": va}
            state["points"].append(point)
            improved_loss = vl < state["best_loss"]
            if improved_loss:
                prev = state["best_loss"]
                significant_loss = (not math.isfinite(prev)) or ((prev - vl) / max(abs(prev), 1e-12) >= MIN_REL_LOSS_GAIN)
                state["best_loss"] = vl
                state["best_acc"] = va
                state["best_step"] = step
                state["best_w"] = W.detach()[mi].cpu().numpy().copy()
                state["best_b"] = B.detach()[mi].cpu().numpy().copy()
                state["best_train_loss"] = avg_train_loss
                state["best_train_acc"] = avg_train_acc
                state["best_grad_norm"] = avg_grad
                state["best_weight_norm"] = float(wnorm[mi])
                state["best_bias_norm"] = float(bnorm[mi])
            else:
                significant_loss = False
            significant_acc = va - state["best_acc_seen"] >= MIN_ACC_GAIN
            state["best_acc_seen"] = max(state["best_acc_seen"], va)
            if early_stop and step >= MIN_STEPS:
                if significant_loss or significant_acc:
                    state["patience"] = 0
                else:
                    state["patience"] += 1
                if state["patience"] >= PATIENCE and not state["stopped"]:
                    state["stopped"] = True
                    state["stop_step"] = step
                    state["stop_reason"] = "validation_patience"
            curves.append({"phase": tag, "task": task, "representation": rep_names[mi % nrep],
                           "seed": seeds[mi // nrep], "step": step,
                           "optimizer": config["optimizer"], "lr": config["lr"], "weight_decay": config["weight_decay"],
                           "train_minibatch_loss_mean": avg_train_loss, "train_minibatch_accuracy_mean": avg_train_acc,
                           "validation_loss": vl, "validation_accuracy": va,
                           "gradient_norm_mean": avg_grad, "weight_norm": float(wnorm[mi]), "bias_norm": float(bnorm[mi]),
                           "best_step_so_far": state["best_step"], "patience_counter": state["patience"]})

    evaluate(0)
    for step in range(1, max_steps + 1):
        ix = torch.as_tensor(rng.integers(0, len(ytr_np), size=batch), dtype=torch.long, device=DEVICE)
        xb = X[:, ix, :]
        yb = ytr[ix]
        logits = torch.bmm(xb, W.transpose(1, 2)) + B[:, None, :]
        losses = F.cross_entropy(logits.reshape(model_count * batch, nclass),
                                 yb.unsqueeze(0).expand(model_count, -1).reshape(-1), reduction="none").reshape(model_count, batch)
        loss = losses.mean(1).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            gn = torch.sqrt((W.grad ** 2).sum((1, 2)) + (B.grad ** 2).sum(1))
            grad_window.append(float(gn.mean().item()))
            pred = logits.detach().argmax(-1)
            train_acc_window.append(float((pred == yb.unsqueeze(0)).float().mean().item()))
            train_loss_window.append(float(losses.detach().mean().item()))
        opt.step()
        if step % EVAL_EVERY == 0:
            evaluate(step)
            if step % 1000 == 0:
                active = sum(not s["stopped"] for s in model_state) if early_stop else model_count
                print(f"PROGRESS tag={tag} task={task} step={step}/{max_steps} active={active}/{model_count}", flush=True)
            if early_stop and all(s["stopped"] for s in model_state):
                break

    last_step = min(max_steps, (step // EVAL_EVERY) * EVAL_EVERY)
    for state in model_state:
        if state["stop_step"] is None:
            state["stop_step"] = last_step
            state["stop_reason"] = "max_steps"
    results, states = [], {}
    for mi, state in enumerate(model_state):
        ls, ac = curve_slopes(state["points"], mi)
        stop_idx = next((i for i, p in enumerate(state["points"]) if p["step"] == state["stop_step"]), len(state["points"]) - 1)
        plateau = (state["stop_reason"] == "validation_patience" and ls is not None and ac is not None
                   and abs(ls) <= 1e-4 and abs(ac) <= 1e-4
                   and state["best_step"] <= state["stop_step"] - 5 * EVAL_EVERY)
        converged = bool(plateau)
        rep = rep_names[mi % nrep]
        seed = seeds[mi // nrep]
        results.append({"phase": tag, "task": task, "representation": rep, "seed": seed,
                        "input_dim": dim, "num_classes": nclass, "optimizer": config["optimizer"],
                        "lr": config["lr"], "weight_decay": config["weight_decay"], "batch_size": batch,
                        "best_step": state["best_step"], "stop_step": state["stop_step"],
                        "stop_reason": state["stop_reason"], "convergence_flag": "CONVERGED" if converged else "NOT_CONVERGED",
                        "plateau_confirmed": plateau, "best_val_loss": state["best_loss"],
                        "best_val_accuracy": state["best_acc"], "best_train_minibatch_loss": state["best_train_loss"],
                        "best_train_minibatch_accuracy": state["best_train_acc"], "best_gradient_norm": state["best_grad_norm"],
                        "best_weight_norm": state["best_weight_norm"], "best_bias_norm": state["best_bias_norm"],
                        "last10_relative_val_loss_slope_per_eval": ls, "last10_val_accuracy_slope_per_eval": ac,
                        "validation_known_rows": info["val_known"], "class_ids_unchanged": True})
        states[(rep, seed)] = {"weight": state["best_w"], "bias": state["best_b"], "best_step": state["best_step"]}
    del opt, X, XV, xbase, xvbase, W, B
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, curves, states


def cond_accuracy(x: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray, chunk=256):
    known = (y >= 0) & (y < w.shape[0])
    if not known.any():
        return {"known_rows": 0, "loss": None, "accuracy": None}
    ce_sum, correct = 0.0, 0
    for start in range(0, len(y), chunk):
        end = min(start + chunk, len(y))
        mask = known[start:end]
        if not mask.any():
            continue
        logits = x[start:end][mask] @ w.T + b
        yy = y[start:end][mask]
        row_max = logits.max(1, keepdims=True)
        logsum = np.log(np.exp(logits - row_max).sum(1)) + row_max[:, 0]
        ce_sum += float((logsum - logits[np.arange(len(yy)), yy]).sum())
        correct += int((logits.argmax(1) == yy).sum())
    n = int(known.sum())
    return {"known_rows": n, "loss": ce_sum / n, "accuracy": correct / n}


def evaluate_group_tests(task: str, states: dict, rep_inputs: dict[str, dict[str, np.ndarray]],
                         label_by_split: dict[str, dict[str, np.ndarray]], phase: str,
                         config: dict) -> list[dict]:
    rows = []
    for (rep, seed), state in states.items():
        w, b = state["weight"], state["bias"]
        for split in ("train", "val", "testA", "testB"):
            if split not in label_by_split or split not in rep_inputs[rep]:
                continue
            met = cond_accuracy(rep_inputs[rep][split], label_by_split[split][task], w, b)
            rows.append({"phase": phase, "task": task, "representation": rep, "seed": seed,
                         "split": split, "known_rows": met["known_rows"], "cross_entropy": met["loss"],
                         "accuracy": met["accuracy"], "best_step": state["best_step"],
                         "optimizer": config["optimizer"], "lr": config["lr"], "weight_decay": config["weight_decay"]})
    return rows


def paired_summary(rows: list[dict], left: str, right: str, task: str, norm: str,
                   split: str, tolerance: float, seed: int) -> dict:
    values = {}
    for r in rows:
        if r.get("task") != task or r.get("split") != split or r.get("normalization") != norm:
            continue
        if r.get("representation") in (left, right) and r.get("accuracy") is not None:
            values[(r["representation"], int(r["seed"]))] = float(r["accuracy"])
    common = sorted(set(s for rep, s in values if rep == left) & set(s for rep, s in values if rep == right))
    d = [values[(left, s)] - values[(right, s)] for s in common]
    stats = ci(d, seed)
    compatible = bool(stats["n"] and stats["bootstrap95_low"] <= tolerance and stats["bootstrap95_high"] >= -tolerance
                      and stats["t95_low"] <= tolerance and stats["t95_high"] >= -tolerance)
    return {"task": task, "normalization": norm, "split": split, "left": left, "right": right,
            "paired_deltas_left_minus_right": d, "tolerance": tolerance, **stats,
            "mean_within_tolerance": stats["mean"] is not None and abs(stats["mean"]) <= tolerance,
            "ci_compatible_with_tolerance_band": compatible}


def record_tests_with_norm(rows: list[dict], norm_by_rep: dict[str, str]) -> list[dict]:
    return [{**r, "normalization": norm_by_rep[r["representation"]]} for r in rows]


def load_transform() -> dict:
    with np.load(CACHE / "w1_svd_matrices.npz", allow_pickle=False) as f:
        keys = list(f.files)
        vals = {k: np.asarray(f[k]) for k in keys}
    # Confirm unique matrices by shape; J3C stores U/S/Vh under named keys.
    u = vals.get("U")
    if u is None:
        u = vals.get("u")
    if u is None:
        raise KeyError(f"U missing from w1_svd_matrices.npz; keys={keys}")
    s = vals.get("S", vals.get("s", vals.get("singular_values")))
    if s is None:
        raise KeyError(f"singular values missing; keys={keys}")
    return {"U": u.astype(np.float64), "S": np.asarray(s, dtype=np.float64), "keys": keys}


def save_curves(path: Path, curves: list[dict]) -> None:
    write_csv(path, curves)


def make_svg(path: Path, title: str, series: list[tuple[str, list[tuple[float, float]]]], y_label: str):
    width, height = 1000, 560
    l, r, t, b = 80, 30, 65, 75
    vals = [v for _, points in series for _, v in points if v is not None and np.isfinite(v)]
    xs = [x for _, points in series for x, _ in points]
    if not vals or not xs:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="30" y="50">{title}: NOT_RUN</text></svg>'
        path.write_text(svg, encoding="utf-8")
        return
    ymin, ymax = min(vals), max(vals)
    if ymax == ymin:
        ymax = ymin + 1
    xmin, xmax = min(xs), max(xs)
    if xmax == xmin:
        xmax = xmin + 1
    colors = ["#1769aa", "#ef6c00", "#2e7d32", "#8e24aa", "#c62828", "#00838f", "#6d4c41"]
    def px(x): return l + (x - xmin) / (xmax - xmin) * (width - l - r)
    def py(y): return t + (ymax - y) / (ymax - ymin) * (height - t - b)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{l}" y="30" font-family="Arial" font-size="19">{title}</text>']
    for i in range(6):
        y = t + i * (height - t - b) / 5
        val = ymax - i * (ymax - ymin) / 5
        parts.append(f'<line x1="{l}" y1="{y:.2f}" x2="{width-r}" y2="{y:.2f}" stroke="#dddddd"/>')
        parts.append(f'<text x="8" y="{y+4:.2f}" font-family="Arial" font-size="12">{val:.4g}</text>')
    for j, (label, points) in enumerate(series):
        points = [(x, y) for x, y in points if y is not None and np.isfinite(y)]
        if not points:
            continue
        coords = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in points)
        color = colors[j % len(colors)]
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        ly = t + 18 * (j + 1)
        parts.append(f'<line x1="{width-260}" y1="{ly-4}" x2="{width-235}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{width-228}" y="{ly}" font-family="Arial" font-size="12">{label}</text>')
    parts += [f'<text x="{width/2}" y="{height-22}" font-family="Arial" text-anchor="middle" font-size="14">step</text>',
              f'<text x="18" y="{height/2}" font-family="Arial" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="14">{y_label}</text>', '</svg>']
    path.write_text("\n".join(parts), encoding="utf-8")


def mark_not_run(path: Path, reason: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"status,reason\nNOT_RUN,{json.dumps(reason)}\n", encoding="utf-8")


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    integrity_path = OUT / "cache_integrity_recheck.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if not integrity.get("pass"):
        finalize_early("J3D-F — CERTIFIED CACHE INTEGRITY CHECK FAILED", "Stage 0 certified-cache integrity recheck failed", t0)
        return
    transform_audit_path = OUT / "sanity" / "stage0_transform_audit.json"
    if not transform_audit_path.is_file() or not json.loads(transform_audit_path.read_text(encoding="utf-8")).get("pass"):
        finalize_early("J3D-F — STAGE 0 ROW-TRANSFORM AUDIT FAILED",
                       "R0/R1/R2 definitions or analytical transfer did not pass the pre-probe Stage 0 audit", t0)
        return
    prereg = json.loads((OUT / "configs" / "protocol_selection_preregistered.json").read_text(encoding="utf-8"))
    interpretation_rules = json.loads((OUT / "configs" / "j_cliff_interpretation_preregistered.json").read_text(encoding="utf-8"))
    # Record exact train-only preprocessing before optimizer calibration. Test arrays are not loaded here.
    prep_stats, rep_arrays = build_stats_and_arrays(["R0", "R1", "R2"], splits=("train", "val"))
    dump(OUT / "preprocessing_manifest.json", {
        "conditions": ["raw", "train_only_standardized"], "epsilon": STD_EPS,
        "std_floor_policy": "train std < 1e-8 replaced with 1.0", "statistics_source": "train only",
        "test_features_loaded_before_protocol_freeze": False, "representations": prep_stats})
    ytr = label_arrays("train")
    yval = label_arrays("val")
    class_info = {t: task_class_info(ytr[t], yval[t], t) for t in TASKS}
    dump(OUT / "sanity" / "frozen_label_task_definitions.json", {
        "label_fields_used_without_remap": LABEL_FIELD, "tasks": class_info,
        "semantic_class_ids_reencoded": False, "train_and_val_only": True})

    # Candidate screen uses only train/validation and the preregistered matrix.
    candidate_rows = []
    candidate_curve_rows = []
    candidate_scores: dict[str, list[dict]] = {t: [] for t in TASKS}
    norm_rep_inputs = {}
    for stage in ("R0", "R1"):
        for norm in ("raw", "standardized"):
            norm_rep_inputs[f"{stage}_{norm}"] = (rep_arrays[stage][norm]["train"], rep_arrays[stage][norm]["val"])
    candidates = prereg["optimizer_selection"]["candidates"]
    for task in TASKS:
        for ci_idx, candidate in enumerate(candidates):
            results, curves, _ = fit_group(task, norm_rep_inputs, ytr[task], yval[task], [11], candidate,
                                            f"screen_{task}_{ci_idx:02d}", max_steps=3000, early_stop=False)
            candidate_curve_rows.extend(curves)
            write_csv(OUT / "tables" / "optimizer_calibration_curves.csv", candidate_curve_rows)
            tail_rows = [r for r in curves if r["step"] >= 2600]
            by_rep = {}
            for rep in norm_rep_inputs:
                rr = [r for r in tail_rows if r["representation"] == rep]
                by_rep[rep] = {"val_accuracy_last5_mean": float(np.mean([x["validation_accuracy"] for x in rr])),
                               "val_loss_last5_mean": float(np.mean([x["validation_loss"] for x in rr]))}
            score_acc = float(np.mean([v["val_accuracy_last5_mean"] for v in by_rep.values()]))
            score_loss = float(np.mean([v["val_loss_last5_mean"] for v in by_rep.values()]))
            row = {"task": task, "candidate_index": ci_idx, **candidate,
                   "screen_seed": 11, "screen_steps": 3000, "mean_val_accuracy_last5_four_conditions": score_acc,
                   "mean_val_cross_entropy_last5_four_conditions": score_loss,
                   "all_finite": bool(np.isfinite(score_acc) and np.isfinite(score_loss)),
                   "by_condition_json": json.dumps(by_rep, ensure_ascii=False)}
            candidate_rows.append(row)
            candidate_scores[task].append(row)
            write_csv(OUT / "tables" / "optimizer_calibration.csv", candidate_rows)
            dump(OUT / "sanity" / "optimizer_calibration_progress.json", {
                "task": task, "candidates_completed_for_task": ci_idx + 1, "candidate_count": len(candidates),
                "completed_rows": len(candidate_rows), "latest": row})
            print(f"SCREEN task={task} candidate={ci_idx+1}/{len(candidates)} optimizer={candidate['optimizer']} lr={candidate['lr']} wd={candidate['weight_decay']} val_acc={score_acc:.6f} val_loss={score_loss:.6f}", flush=True)

    selected, alternate_family, wd_control = {}, {}, {}
    for task in TASKS:
        good = [r for r in candidate_scores[task] if r["all_finite"]]
        if not good:
            finalize_early("J3D-F — MATCHED CONVERGED PROBE FAILS R0/R1 CONTROL", f"All optimizer screen candidates were non-finite for {task}", t0)
            return
        order = {i: i for i in range(len(candidates))}
        ranked = sorted(good, key=lambda r: (-r["mean_val_accuracy_last5_four_conditions"],
                                             r["mean_val_cross_entropy_last5_four_conditions"], r["candidate_index"]))
        selected[task] = {k: ranked[0][k] for k in ("optimizer", "lr", "weight_decay", "candidate_index")}
        for family in ("AdamW", "SGD_momentum_0.9"):
            fam = [r for r in good if r["optimizer"] == family]
            alternate_family[(task, family)] = {k: sorted(fam, key=lambda r: (-r["mean_val_accuracy_last5_four_conditions"], r["mean_val_cross_entropy_last5_four_conditions"], r["candidate_index"]))[0][k]
                                                for k in ("optimizer", "lr", "weight_decay", "candidate_index")} if fam else None
        same_lr_nonzero = [r for r in good if r["optimizer"] == selected[task]["optimizer"] and r["lr"] == selected[task]["lr"] and r["weight_decay"] > 0]
        same_lr_zero = [r for r in good if r["optimizer"] == selected[task]["optimizer"] and r["lr"] == selected[task]["lr"] and r["weight_decay"] == 0]
        nonzero = sorted(same_lr_nonzero, key=lambda r: (-r["mean_val_accuracy_last5_four_conditions"], r["mean_val_cross_entropy_last5_four_conditions"], r["candidate_index"]))
        zero = sorted(same_lr_zero, key=lambda r: (-r["mean_val_accuracy_last5_four_conditions"], r["mean_val_cross_entropy_last5_four_conditions"], r["candidate_index"]))
        wd_control[task] = {"zero_wd": {k: zero[0][k] for k in ("optimizer", "lr", "weight_decay", "candidate_index")} if zero else None,
                            "nonzero_wd": {k: nonzero[0][k] for k in ("optimizer", "lr", "weight_decay", "candidate_index")} if nonzero else None}
    frozen = {
        **prereg,
        "status": "FINAL_PROTOCOL_FROZEN_BEFORE_TEST_METRIC_EVALUATION",
        "optimizer_selected_per_task_from_train_val_only": selected,
        "best_screened_AdamW_per_task": {t: alternate_family[(t, "AdamW")] for t in TASKS},
        "best_screened_SGD_per_task": {t: alternate_family[(t, "SGD_momentum_0.9")] for t in TASKS},
        "weight_decay_matched_controls_per_task": wd_control,
        "j_cliff_interpretation_rules": interpretation_rules,
        "freeze_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "testA_testB_metric_evaluation_started": False,
        "selected_using_only": ["train", "val"],
    }
    dump(OUT / "probe_protocol_frozen.json", frozen)
    dump(OUT / "configs" / "optimizer_screen_selection.json", {"selection_method": prereg["optimizer_selection"],
                                                                  "selected": selected, "alternates": frozen["best_screened_AdamW_per_task"],
                                                                  "weight_decay_controls": wd_control,
                                                                  "all_selection_data_splits": ["train", "val"]})
    summary_lines = ["# Optimizer calibration (train/validation only)", "",
                     "Candidate screen: 18 preregistered AdamW/SGD × LR × weight-decay settings, 3,000 steps, seed 11, averaged across R0/R1 raw and train-standardized validation conditions. No TestA/TestB metrics were read.", "",
                     "| task | chosen optimizer | LR | WD | mean last-5 val accuracy | mean last-5 val CE |", "|---|---|---:|---:|---:|---:|"]
    for task in TASKS:
        r = next(x for x in candidate_scores[task] if x["candidate_index"] == selected[task]["candidate_index"])
        summary_lines.append(f"| {task} | {r['optimizer']} | {r['lr']} | {r['weight_decay']} | {r['mean_val_accuracy_last5_four_conditions']:.6f} | {r['mean_val_cross_entropy_last5_four_conditions']:.6f} |")
    (OUT / "report" / "optimizer_calibration_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (OUT / "optimizer_calibration_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Frozen primary convergence fits; still train/validation only.
    all_main_results, all_main_curves, main_states = [], [], {}
    norm_rep_inputs3 = {}
    norm_by_rep3 = {}
    for stage in ("R0", "R1", "R2"):
        for norm in ("raw", "standardized"):
            key = f"{stage}_{norm}"
            norm_rep_inputs3[key] = (rep_arrays[stage][norm]["train"], rep_arrays[stage][norm]["val"])
            norm_by_rep3[key] = norm
    for task in TASKS:
        cfg = selected[task]
        rs = {k: v for k, v in norm_rep_inputs3.items() if k.startswith("R0_") or k.startswith("R1_")}
        res, curves, states = fit_group(task, rs, ytr[task], yval[task], SEEDS, cfg, f"r0_r1_{task}")
        all_main_results.extend(res); all_main_curves.extend(curves)
        main_states[task] = {k: v for k, v in states.items()}
        write_csv(OUT / "tables" / "r0_r1_converged_results.csv", all_main_results)
        save_curves(OUT / "tables" / "r0_r1_convergence_curves.csv", all_main_curves)
        print(f"FIT_DONE r0_r1 task={task} converged={sum(r['convergence_flag']=='CONVERGED' for r in res)}/{len(res)}", flush=True)

    # Only now, after the final protocol has been frozen, load held-out splits for evaluation.
    label_all = {s: label_arrays(s) for s in SPLITS}
    test_rep_arrays = {}
    for stage in ("R0", "R1", "R2"):
        test_rep_arrays[stage] = {}
        for norm in ("raw", "standardized"):
            stat = prep_stats[stage]
            raw_all = {s: np.asarray(np.load(CACHE / f"{stage}_{s}.npy", allow_pickle=False), dtype=np.float32) for s in ("testA", "testB")}
            if norm == "raw":
                test_rep_arrays[stage][norm] = raw_all
            else:
                mu, sd = np.asarray(stat["mean"], dtype=np.float64), np.asarray(stat["std"], dtype=np.float64)
                test_rep_arrays[stage][norm] = {s: ((x.astype(np.float64) - mu) / sd).astype(np.float32) for s, x in raw_all.items()}
    r0r1_test_rows = []
    for task in TASKS:
        inp = {}
        for stage in ("R0", "R1"):
            for norm in ("raw", "standardized"):
                key = f"{stage}_{norm}"
                inp[key] = {"train": norm_rep_inputs3[key][0], "val": norm_rep_inputs3[key][1], **test_rep_arrays[stage][norm]}
        tested = evaluate_group_tests(task, main_states[task], inp, label_all, "r0_r1_primary", selected[task])
        r0r1_test_rows.extend(record_tests_with_norm(tested, norm_by_rep3))
    write_csv(OUT / "tables" / "r0_r1_converged_results.csv", all_main_results + r0r1_test_rows)
    r0r1_gate = evaluate_gate("R0", "R1", r0r1_test_rows, all_main_results, "standardized")
    dump(OUT / "sanity" / "r0_r1_invariance_gate.json", r0r1_gate)
    if not r0r1_gate["pass"]:
        finalize_early("J3D-F — MATCHED CONVERGED PROBE FAILS R0/R1 CONTROL", "R0/R1 standardized positive-control gate failed or primary convergence/plateau was not established", t0, r0r1_gate)
        return

    # R0/R1 passed; proceed to independent R1/R2 refits under the exact frozen protocol.
    r1r2_results, r1r2_curves, r1r2_states = [], [], {}
    r12_inputs = {k: v for k, v in norm_rep_inputs3.items() if k.startswith("R1_") or k.startswith("R2_")}
    for task in TASKS:
        cfg = selected[task]
        res, curves, states = fit_group(task, r12_inputs, ytr[task], yval[task], SEEDS, cfg, f"r1_r2_{task}")
        r1r2_results.extend(res); r1r2_curves.extend(curves); r1r2_states[task] = states
        write_csv(OUT / "tables" / "r1_r2_converged_results.csv", r1r2_results)
        save_curves(OUT / "tables" / "r1_r2_convergence_curves.csv", r1r2_curves)
        print(f"FIT_DONE r1_r2 task={task} converged={sum(r['convergence_flag']=='CONVERGED' for r in res)}/{len(res)}", flush=True)

    r1r2_test_rows = []
    for task in TASKS:
        inp = {}
        for stage in ("R1", "R2"):
            for norm in ("raw", "standardized"):
                key = f"{stage}_{norm}"
                inp[key] = {"train": norm_rep_inputs3[key][0], "val": norm_rep_inputs3[key][1], **test_rep_arrays[stage][norm]}
        tested = evaluate_group_tests(task, r1r2_states[task], inp, label_all, "r1_r2_primary", selected[task])
        r1r2_test_rows.extend(record_tests_with_norm(tested, norm_by_rep3))
    write_csv(OUT / "tables" / "r1_r2_converged_results.csv", r1r2_results + r1r2_test_rows)

    # Exact transform and bidirectional analytical classifier transfer, raw and train-standardized.
    svd = load_transform()
    U, singular = svd["U"], svd["S"]
    if U.shape != (256, 256) or len(singular) != 256 or np.linalg.matrix_rank(U) != 256:
        raise ValueError(f"unexpected U/S shape or rank: {U.shape}, {singular.shape}, rank={np.linalg.matrix_rank(U)}")
    r1tr = feature_arrays("R1", ("train",))["train"]
    r2tr = feature_arrays("R2", ("train",))["train"]
    r1u = r1tr.astype(np.float64) @ U.T
    transform = {"U_shape": list(U.shape), "U_rank": int(np.linalg.matrix_rank(U)),
                 "U_condition_number": float(np.linalg.cond(U)), "singular_value_count": int(len(singular)),
                 "R2_vs_R1UT_train_max_abs": float(np.max(np.abs(r2tr.astype(np.float64) - r1u))),
                 "R2_vs_R1UT_train_mean_abs": float(np.mean(np.abs(r2tr.astype(np.float64) - r1u))),
                 "svd_file_keys": svd["keys"]}
    transfer_rows = []
    transfer_pass = True
    # Assemble train stats for standardized affine transform; raw uses A=U.T and reverse A=U.
    for task in TASKS:
        stats1 = prep_stats["R1"]; stats2 = prep_stats["R2"]
        mu1, sd1 = np.asarray(stats1["mean"]), np.asarray(stats1["std"])
        mu2, sd2 = np.asarray(stats2["mean"]), np.asarray(stats2["std"])
        for norm in ("raw", "standardized"):
            for seed in SEEDS:
                a = r1r2_states[task][(f"R1_{norm}", seed)]
                bstate = r1r2_states[task][(f"R2_{norm}", seed)]
                for src, dst, src_state, A in (("R1", "R2", a, U.T), ("R2", "R1", bstate, U)):
                    Wsrc, bsrc = src_state["weight"].astype(np.float64), src_state["bias"].astype(np.float64)
                    if norm == "raw":
                        # Row-vector features satisfy x_dst = x_src @ A.  For
                        # logits x @ W.T, the equivalent destination classifier
                        # is W_dst = W_src @ A^{-T}; A is orthogonal here.
                        Wdst = Wsrc @ A
                        bdst = bsrc.copy()
                        affine_offset = np.zeros(256, dtype=np.float64)
                    else:
                        ms, ss = (mu1, sd1) if src == "R1" else (mu2, sd2)
                        md, sd = (mu2, sd2) if dst == "R2" else (mu1, sd1)
                        Bmat = (ss[:, None] * A) / sd[None, :]
                        offset = (ms @ A - md) / sd
                        # x_dst_std = x_src_std @ Bmat + offset.
                        # Solve Bmat @ W_dst.T = W_src.T to preserve logits.
                        Wdst = np.linalg.solve(Bmat, Wsrc.T).T
                        bdst = bsrc - offset @ Wdst.T
                        affine_offset = offset
                    # Directly compare all rows in train/val/test; labels only used for accuracy reporting.
                    max_err, sum_err, nlogits, pred_mismatch = 0.0, 0.0, 0, 0
                    for split in SPLITS:
                        xsrc = rep_arrays["R1" if src == "R1" else "R2"][norm].get(split)
                        xdst = rep_arrays["R2" if dst == "R2" else "R1"][norm].get(split)
                        if xsrc is None:
                            raw_src = np.asarray(np.load(CACHE / f"{src}_{split}.npy", allow_pickle=False), dtype=np.float32)
                            raw_dst = np.asarray(np.load(CACHE / f"{dst}_{split}.npy", allow_pickle=False), dtype=np.float32)
                            if norm == "standardized":
                                raw_src = ((raw_src.astype(np.float64) - ms) / ss).astype(np.float32)
                                raw_dst = ((raw_dst.astype(np.float64) - md) / sd).astype(np.float32)
                            xsrc, xdst = raw_src, raw_dst
                        elif split in ("testA", "testB"):
                            xsrc = test_rep_arrays[src][norm][split]
                            xdst = test_rep_arrays[dst][norm][split]
                        Lsrc, Ldst = [], []
                        for st in range(0, len(xsrc), 256):
                            qsrc = xsrc[st:st+256].astype(np.float64) @ Wsrc.T + bsrc
                            qdst = xdst[st:st+256].astype(np.float64) @ Wdst.T + bdst
                            diff = np.abs(qsrc - qdst)
                            max_err = max(max_err, float(diff.max(initial=0)))
                            sum_err += float(diff.sum())
                            nlogits += int(diff.size)
                            pred_mismatch += int(np.sum(qsrc.argmax(1) != qdst.argmax(1)))
                        ys = label_all[split][task]
                        sm = cond_accuracy(xsrc, ys, Wsrc.astype(np.float32), bsrc.astype(np.float32))
                        tm = cond_accuracy(xdst, ys, Wdst.astype(np.float32), bdst.astype(np.float32))
                        transfer_rows.append({"task": task, "normalization": norm, "direction": f"{src}_to_{dst}",
                                              "seed": seed, "split": split,
                                              "source_known_rows": sm["known_rows"], "source_accuracy": sm["accuracy"],
                                              "transferred_accuracy": tm["accuracy"],
                                              "accuracy_delta": None if sm["accuracy"] is None or tm["accuracy"] is None else tm["accuracy"] - sm["accuracy"],
                                              "max_logit_abs_error_all_splits_so_far": max_err})
                    mean_err = sum_err / max(1, nlogits)
                    mismatch_rate = pred_mismatch / max(1, nlogits // Wsrc.shape[0])
                    passed = max_err <= 1e-3 and mismatch_rate == 0.0
                    transfer_pass &= passed
                    transform.update({f"{norm}_{src}_to_{dst}_seed{seed}_max_logit_abs_error": max_err,
                                      f"{norm}_{src}_to_{dst}_seed{seed}_mean_logit_abs_error": mean_err,
                                      f"{norm}_{src}_to_{dst}_seed{seed}_prediction_mismatch_rate": mismatch_rate,
                                      f"{norm}_{src}_to_{dst}_seed{seed}_pass": passed})
    write_csv(OUT / "tables" / "weight_transfer_control.csv", transfer_rows)
    dump(OUT / "sanity" / "representation_transform_recheck.json", transform)
    if not transfer_pass:
        dump(OUT / "sanity" / "weight_transfer_gate.json", {"pass": False, "reason": "analytical transfer exceeded preregistered logit/prediction tolerance", "transform": transform})
        finalize_early("J3D-F — ANALYTICAL WEIGHT-TRANSFER CONTROL FAILED", "R1/R2 analytically transferred linear classifier failed numerical equivalence", t0)
        return
    dump(OUT / "sanity" / "weight_transfer_gate.json", {"pass": True, "transform": transform})

    # Compare independent R1/R2 fits, after exact-transfer parity has been checked.
    basis_gate = evaluate_gate("R1", "R2", r1r2_test_rows, r1r2_results, "standardized")
    dump(OUT / "basis_invariance_gate.json", basis_gate)

    # Weight-decay control: wd=0 versus matched best-screened nonzero wd for selected optimizer/LR.
    wd_results, wd_curves, wd_tests = [], [], []
    wd_control_configs = {}
    for task in TASKS:
        pair = wd_control[task]
        # Selected protocol and its matched counterpart share optimizer/LR by construction.
        chosen = selected[task]
        if chosen["weight_decay"] == 0:
            counterpart = pair["nonzero_wd"]
        else:
            counterpart = pair["zero_wd"]
        if counterpart is None:
            continue
        wd_control_configs[task] = {"primary": chosen, "counterpart": counterpart}
        # Only run the counterpart if it is not the already-fitted primary configuration.
        if float(counterpart["weight_decay"]) == float(chosen["weight_decay"]):
            continue
        inputs = {k: v for k, v in r12_inputs.items() if k in ("R1_standardized", "R2_standardized")}
        res, curves, states = fit_group(task, inputs, ytr[task], yval[task], SEEDS, counterpart,
                                        f"weight_decay_control_{task}")
        wd_results.extend(res); wd_curves.extend(curves)
        inp_test = {k: {"train": r12_inputs[k][0], "val": r12_inputs[k][1], **test_rep_arrays["R1" if k.startswith("R1") else "R2"]["standardized"]}
                    for k in inputs}
        tested = evaluate_group_tests(task, states, inp_test, label_all, "weight_decay_counterpart", counterpart)
        wd_tests.extend(tested)
    write_csv(OUT / "tables" / "weight_decay_controls.csv", wd_results + wd_tests)
    dump(OUT / "sanity" / "weight_decay_control_protocol.json", wd_control_configs)

    # Limited alternate optimizer robustness is only needed if independent basis gate fails.
    optimizer_diag_rows = []
    if not basis_gate["pass"]:
        for task in TASKS:
            primary = selected[task]
            other_family = "SGD_momentum_0.9" if primary["optimizer"] == "AdamW" else "AdamW"
            alternate = alternate_family[(task, other_family)]
            if alternate is None:
                continue
            inputs = {k: v for k, v in r12_inputs.items() if k in ("R1_standardized", "R2_standardized")}
            res, curves, states = fit_group(task, inputs, ytr[task], yval[task], SEEDS, alternate,
                                            f"alternate_optimizer_{task}")
            inp_test = {k: {"train": r12_inputs[k][0], "val": r12_inputs[k][1], **test_rep_arrays["R1" if k.startswith("R1") else "R2"]["standardized"]}
                        for k in inputs}
            tested = evaluate_group_tests(task, states, inp_test, label_all, "alternate_optimizer", alternate)
            optimizer_diag_rows.extend(res + tested)
    write_csv(OUT / "tables" / "optimizer_robustness_if_needed.csv", optimizer_diag_rows)

    # If any calibration gate fails, there is no downstream J stagewise attribution.
    if not basis_gate["pass"]:
        dump(OUT / "sanity" / "measurement_gate.json", {"status": "CLOSED", "r0_r1_pass": True,
                                                           "r1_r2_pass": False, "weight_transfer_pass": True,
                                                           "stagewise_j_authorized": False})
        finalize_early("J3D-F — CONVERGED MATCHED PROBE REMAINS BASIS-SENSITIVE",
                       "R1/R2 standardized independent-refit gate failed despite analytical classifier transfer", t0,
                       {"r0_r1_gate": r0r1_gate, "r1_r2_gate": basis_gate, "weight_transfer_pass": transfer_pass,
                        "weight_decay_controls": wd_control_configs,
                        "optimizer_robustness_rows": len(optimizer_diag_rows)})
        return

    dump(OUT / "sanity" / "measurement_gate.json", {"status": "OPEN", "r0_r1_pass": True,
                                                       "r1_r2_pass": True, "weight_transfer_pass": True,
                                                       "stagewise_j_authorized": True})
    # J stagewise linear probes are permitted only after all calibration gates passed.
    stage_definition = json.loads((CACHE / "stage_definition.json").read_text(encoding="utf-8"))
    stages = list(stage_definition.keys())
    if stages != ["Z0", "Z1", "Z2", "Z3", "Z4"]:
        # Preserve true order from JSON but do not infer or manufacture stages.
        stages = list(stage_definition.keys())
    stage_stats, stage_arrays = build_stats_and_arrays(stages, splits=("train", "val"))
    prep_full = json.loads((OUT / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    prep_full["representations"].update(stage_stats)
    prep_full["all_test_metric_evaluation_after_final_protocol_freeze"] = True
    dump(OUT / "preprocessing_manifest.json", prep_full)
    stagewise_results, stagewise_curves = [], []
    stage_states_by_task: dict[str, dict] = {t: {} for t in TASKS}
    dim_groups: dict[int, list[str]] = {}
    for stage in stages:
        dim_groups.setdefault(stage_arrays[stage]["raw"]["train"].shape[1], []).append(stage)
    for task in TASKS:
        for norm in ("raw", "standardized"):
            for dim, group in dim_groups.items():
                inputs = {f"{stage}_{norm}": (stage_arrays[stage][norm]["train"], stage_arrays[stage][norm]["val"]) for stage in group}
                res, curves, states = fit_group(task, inputs, ytr[task], yval[task], SEEDS, selected[task],
                                                f"stagewise_{task}_{norm}_d{dim}")
                stagewise_results.extend(res); stagewise_curves.extend(curves)
                stage_states_by_task[task].update(states)
                write_csv(OUT / "tables" / "stagewise_linear_results.csv", stagewise_results)
                save_curves(OUT / "tables" / "stagewise_linear_convergence.csv", stagewise_curves)
                print(f"FIT_DONE stagewise task={task} norm={norm} dims={dim} stages={group} converged={sum(r['convergence_flag']=='CONVERGED' for r in res)}/{len(res)}", flush=True)
    # Now evaluate all J stages after fits and frozen protocol; held-out metrics are not used for tuning.
    stage_test_rows = []
    all_stage_labels = {s: label_arrays(s) for s in SPLITS}
    for task in TASKS:
        for norm in ("raw", "standardized"):
            for stage in stages:
                state_subset = {(f"{stage}_{norm}", seed): stage_states_by_task[task][(f"{stage}_{norm}", seed)] for seed in SEEDS}
                test_inputs = {"train": stage_arrays[stage][norm]["train"], "val": stage_arrays[stage][norm]["val"]}
                for sp in ("testA", "testB"):
                    xx = np.asarray(np.load(CACHE / f"{stage}_{sp}.npy", allow_pickle=False), dtype=np.float32)
                    if norm == "standardized":
                        mu, sd = np.asarray(stage_stats[stage]["mean"]), np.asarray(stage_stats[stage]["std"])
                        xx = ((xx.astype(np.float64) - mu) / sd).astype(np.float32)
                    test_inputs[sp] = xx
                inp = {f"{stage}_{norm}": test_inputs}
                tested = evaluate_group_tests(task, state_subset, inp, all_stage_labels, "stagewise", selected[task])
                stage_test_rows.extend(record_tests_with_norm(tested, {f"{stage}_{norm}": norm}))
    write_csv(OUT / "tables" / "stagewise_linear_results.csv", stagewise_results + stage_test_rows)

    # Activation metrics and matched linear probe deltas are descriptive only.
    activation_rows = []
    for task in TASKS:
        for norm in ("raw", "standardized"):
            for split in ("val", "testA", "testB"):
                by = {(r["representation"], int(r["seed"])): r for r in stage_test_rows
                      if r["task"] == task and r["normalization"] == norm and r["split"] == split}
                seeds_common = sorted(s for s in SEEDS if ("Z1_" + norm, s) in by and ("Z2_" + norm, s) in by)
                diffs = [by[("Z2_" + norm, s)]["accuracy"] - by[("Z1_" + norm, s)]["accuracy"]
                         for s in seeds_common if by[("Z1_" + norm, s)]["accuracy"] is not None and by[("Z2_" + norm, s)]["accuracy"] is not None]
                activation_rows.append({"task": task, "normalization": norm, "split": split,
                                        "Z1_to_Z2_accuracy_delta_Z2_minus_Z1": ci(diffs, BOOT_SEED + len(activation_rows))})
    # GELU output statistics on train only; H_mention semantics are intentionally not conflated with write state.
    for stage in ("Z1", "Z2"):
        x = stage_arrays[stage]["raw"]["train"].astype(np.float64)
        activation_rows.append({"task": "all", "normalization": "raw", "split": "train", "stage": stage,
                                "train_mean_norm": float(np.linalg.norm(x, axis=1).mean()),
                                "train_feature_norm_std": float(np.linalg.norm(x, axis=1).std()),
                                "train_fraction_positive": float((x > 0).mean()),
                                "train_fraction_zero": float((x == 0).mean()),
                                "train_effective_rank_cov": effective_rank(x)})
    write_csv(OUT / "tables" / "activation_results.csv", activation_rows)

    # Trigger a small nonlinear entity recoverability diagnostic only from the preregistered rule.
    trigger = nonlinear_trigger(stage_test_rows, stagewise_results)
    dump(OUT / "sanity" / "nonlinear_trigger.json", trigger)
    nonlinear_rows = []
    if trigger["trigger"]:
        nonlinear_rows = run_nonlinear_entity(stage_arrays, stage_stats, ytr, yval, label_all, selected)
    write_csv(OUT / "tables" / "stagewise_nonlinear_results.csv", nonlinear_rows)

    final_verdict = "J3D-P — CALIBRATED PIPELINE PASSES; RESIDUAL LINEAR ENTITY-ACCESSIBILITY CLIFF REPRODUCED"
    j_cliff = summarize_j_cliff(stage_test_rows)
    if j_cliff.get("std_entity_testA_delta_mean") is not None and j_cliff["std_entity_testA_delta_mean"] > -0.10:
        final_verdict = "J3D-P — CALIBRATED PIPELINE PASSES; HISTORICAL LARGE PROJECTION CLIFF NOT REPRODUCED"
    elif j_cliff.get("std_entity_testA_delta_mean") is not None and j_cliff["std_entity_testA_delta_mean"] <= -0.40:
        final_verdict = "J3D-P — CALIBRATED PIPELINE PASSES; STRONG LINEAR ENTITY-ACCESSIBILITY CLIFF REPRODUCED"
    report = build_final_report(final_verdict, r0r1_gate, basis_gate, transfer_pass, all_main_results,
                                r1r2_results, r0r1_test_rows, r1r2_test_rows, stagewise_results,
                                stage_test_rows, activation_rows, nonlinear_rows, j_cliff, trigger,
                                selected, wd_control_configs, transform, t0)
    (OUT / "final_report.md").write_text(report, encoding="utf-8")
    write_csv(OUT / "tables" / "stagewise_summary.csv", summarize_rows(stage_test_rows, stages))
    dump(OUT / "FINAL_VERDICT.json", {"primary": final_verdict, "r0_r1_pass": r0r1_gate["pass"],
                                      "r1_r2_pass": basis_gate["pass"], "weight_transfer_pass": transfer_pass,
                                      "measurement_gate": "OPEN", "reconnect": False, "o3c": False,
                                      "checkpoint_modified": False, "data_leakage_detected": False,
                                      "j_cliff": j_cliff, "runtime_seconds": time.time() - t0})
    (OUT / "FINAL_VERDICT.txt").write_text(final_verdict + "\n", encoding="utf-8")
    write_required_not_run_files(None)
    materialize_root_deliverables()
    refresh_plots()
    print(json.dumps({"stage": "j3d_complete", "verdict": final_verdict,
                      "r0_r1_pass": r0r1_gate["pass"], "r1_r2_pass": basis_gate["pass"],
                      "elapsed_seconds": time.time() - t0}, ensure_ascii=False), flush=True)


def evaluate_gate(left: str, right: str, test_rows: list[dict], fit_rows: list[dict], norm: str) -> dict:
    thresholds = {"entity": 0.005, "relation": 0.005, "joint": 0.01}
    pair_name_left = left
    pair_name_right = right
    # Result names are R0_raw / R1_standardized etc; filter standardized explicitly.
    all_converged = all(r["convergence_flag"] == "CONVERGED" for r in fit_rows)
    details, pass_all = [], all_converged
    for task in TASKS:
        for split in ("testA", "testB"):
            summary = paired_summary(test_rows, f"{left}_standardized", f"{right}_standardized", task, norm,
                                     split, thresholds[task], BOOT_SEED + len(details))
            available = summary["n"] > 0
            item_pass = (not available) or (summary["mean_within_tolerance"] and summary["ci_compatible_with_tolerance_band"])
            if available and split == "testA" and not item_pass:
                pass_all = False
            if available and split == "testB" and not item_pass:
                pass_all = False
            details.append({**summary, "available": available, "pass": item_pass})
    return {"left": pair_name_left, "right": pair_name_right, "normalization": norm,
            "all_primary_runs_converged": all_converged, "thresholds": thresholds,
            "paired_comparisons": details, "pass": bool(pass_all and all_converged)}


def effective_rank(x: np.ndarray) -> float:
    x = x - x.mean(0, keepdims=True)
    cov = x.T @ x / max(1, len(x) - 1)
    eig = np.linalg.eigvalsh(cov)
    pos = eig[eig > max(float(eig.max()) * 1e-10, 1e-12)]
    p = pos / max(float(pos.sum()), 1e-30)
    return float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-30)))))


def nonlinear_trigger(test_rows: list[dict], fit_rows: list[dict]) -> dict:
    # Predeclared: standardized entity Z0->Z1 must be <= -5pp on val and every available test split,
    # with seed-paired 95% bootstrap CI strictly below zero; no Test metric tunes this diagnostic.
    deltas = {}
    for split in ("val", "testA", "testB"):
        source = fit_rows if split == "val" else test_rows
        rows = [r for r in source if r.get("task") == "entity" and r.get("normalization") == "standardized" and r.get("split") == split]
        if split == "val":
            # convergence result rows retain prefixed representation names, val metric is stored as best_val_accuracy.
            vals = {(r["representation"], int(r["seed"])): r["best_val_accuracy"] for r in fit_rows
                    if r["task"] == "entity" and r["representation"] in ("Z0_standardized", "Z1_standardized")}
        else:
            vals = {(r["representation"], int(r["seed"])): r["accuracy"] for r in rows}
        ds = [vals[("Z1_standardized", s)] - vals[("Z0_standardized", s)] for s in SEEDS
              if ("Z0_standardized", s) in vals and ("Z1_standardized", s) in vals
              and vals[("Z0_standardized", s)] is not None and vals[("Z1_standardized", s)] is not None]
        if ds:
            deltas[split] = ci(ds, BOOT_SEED + len(deltas))
        else:
            deltas[split] = None
    required = [deltas["val"], deltas["testA"]]
    if deltas["testB"] is not None:
        required.append(deltas["testB"])
    ok = all(d is not None and d["mean"] <= -0.05 and d["bootstrap95_high"] < 0 for d in required)
    return {"trigger": bool(ok), "rule": "standardized Z0->Z1 entity delta <= -5pp and paired bootstrap CI strictly below 0 for val and TestA, plus TestB when known labels exist",
            "deltas": deltas}


def run_nonlinear_entity(stage_arrays, stage_stats, ytr, yval, labels_all, selected):
    # This path is intentionally small and only runs when the frozen trigger is satisfied.
    # Independent per-representation fits retain the same seed list, linear-probe optimizer family/LR/WD and val stopping rule.
    rows, curves = [], []
    for norm in ("raw", "standardized"):
        for stage in ("Z0", "Z1"):
            xtr = stage_arrays[stage][norm]["train"]
            xval = stage_arrays[stage][norm]["val"]
            nclass = task_class_info(ytr["entity"], yval["entity"], "entity")["num_classes"]
            for seed in SEEDS:
                torch.manual_seed(seed + 7001)
                model = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, nclass)).to(DEVICE)
                opt = make_optimizer(selected["entity"]["optimizer"], model.parameters(), float(selected["entity"]["lr"]), float(selected["entity"]["weight_decay"]))
                Xt = torch.from_numpy(xtr).to(DEVICE); Xv = torch.from_numpy(xval).to(DEVICE)
                yt = torch.from_numpy(ytr["entity"]).long().to(DEVICE); yv = torch.from_numpy(yval["entity"]).long().to(DEVICE)
                batch = min(1024, len(ytr["entity"]))
                rng = np.random.default_rng(42 + 913 + TASK_OFFSET["entity"])
                best_loss, best_state, best_step, best_acc_seen, patience = float("inf"), None, 0, -1.0, 0
                points = []
                converged = False
                for step in range(1, MAX_STEPS + 1):
                    ix = torch.as_tensor(rng.integers(0, len(yt), size=batch), dtype=torch.long, device=DEVICE)
                    model.train(); logits = model(Xt[ix]); loss = F.cross_entropy(logits, yt[ix])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    if step % EVAL_EVERY:
                        continue
                    model.eval()
                    with torch.no_grad():
                        val_logits = model(Xv); mask = (yv >= 0) & (yv < nclass)
                        vl = float(F.cross_entropy(val_logits[mask], yv[mask]).item())
                        va = float((val_logits[mask].argmax(1) == yv[mask]).float().mean().item())
                    points.append({"step": step, "val_loss": vl, "val_accuracy": va})
                    old = best_loss
                    if vl < best_loss:
                        best_loss = vl; best_step = step; best_acc = va
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    sig_loss = old == float("inf") or (old - vl) / max(abs(old), 1e-12) >= MIN_REL_LOSS_GAIN
                    sig_acc = va - best_acc_seen >= MIN_ACC_GAIN
                    best_acc_seen = max(best_acc_seen, va)
                    patience = 0 if (sig_loss or sig_acc) else patience + 1
                    curves.append({"task": "entity", "representation": stage, "normalization": norm, "seed": seed,
                                   "step": step, "train_minibatch_loss": float(loss.item()), "validation_loss": vl, "validation_accuracy": va})
                    if step >= MIN_STEPS and patience >= PATIENCE:
                        slopes = points[-10:]
                        if len(slopes) >= 4:
                            sx = np.asarray([p["step"] / EVAL_EVERY for p in slopes], dtype=np.float64)
                            sl = np.asarray([p["val_loss"] for p in slopes], dtype=np.float64)
                            sa = np.asarray([p["val_accuracy"] for p in slopes], dtype=np.float64)
                            loss_slope = float(np.polyfit(sx, sl, 1)[0] / max(abs(float(sl.mean())), 1e-12))
                            acc_slope = float(np.polyfit(sx, sa, 1)[0])
                            converged = (abs(loss_slope) <= 1e-4 and abs(acc_slope) <= 1e-4
                                         and best_step <= step - 5 * EVAL_EVERY)
                        break
                if best_state is None:
                    continue
                model.load_state_dict(best_state); model.eval()
                result = {"task": "entity", "representation": stage, "normalization": norm, "seed": seed,
                          "best_step": best_step, "stop_step": step,
                          "convergence_flag": "CONVERGED" if converged else "NOT_CONVERGED",
                          "best_val_loss": best_loss}
                for split in ("testA", "testB"):
                    x = np.asarray(np.load(CACHE / f"{stage}_{split}.npy", allow_pickle=False), dtype=np.float32)
                    if norm == "standardized":
                        mu = np.asarray(stage_stats[stage]["mean"]); sd = np.asarray(stage_stats[stage]["std"])
                        x = ((x.astype(np.float64) - mu) / sd).astype(np.float32)
                    with torch.no_grad():
                        pred = model(torch.from_numpy(x).to(DEVICE)).argmax(1).cpu().numpy()
                    yy = labels_all[split]["entity"]; known = (yy >= 0) & (yy < nclass)
                    result[split] = None if not known.any() else float((pred[known] == yy[known]).mean())
                rows.append(result)
    write_csv(OUT / "tables" / "stagewise_nonlinear_convergence.csv", curves)
    return rows


def summarize_j_cliff(rows: list[dict]) -> dict:
    out = {}
    for norm in ("raw", "standardized"):
        for split in ("val", "testA", "testB"):
            if split == "val":
                # val values are taken from best-val rows; same frozen stopping rule across stages.
                source = [r for r in rows if r.get("task") == "entity" and r.get("normalization") == norm and r.get("split") == "val"]
            else:
                source = [r for r in rows if r.get("task") == "entity" and r.get("normalization") == norm and r.get("split") == split]
            by = {(r["representation"], int(r["seed"])): r["accuracy"] for r in source if r.get("representation") in (f"Z0_{norm}", f"Z1_{norm}")}
            ds = [by[(f"Z1_{norm}", s)] - by[(f"Z0_{norm}", s)] for s in SEEDS
                  if (f"Z0_{norm}", s) in by and (f"Z1_{norm}", s) in by and by[(f"Z0_{norm}", s)] is not None and by[(f"Z1_{norm}", s)] is not None]
            out[f"{norm}_entity_{split}_delta_z1_minus_z0"] = ci(ds, BOOT_SEED + len(out)) if ds else None
            if norm == "standardized" and split == "testA":
                out["std_entity_testA_delta_mean"] = None if not ds else float(np.mean(ds))
    return out


def summarize_rows(rows: list[dict], stages: list[str]) -> list[dict]:
    out = []
    for task in TASKS:
        for norm in ("raw", "standardized"):
            for stage in stages:
                for split in ("val", "testA", "testB"):
                    rr = [r for r in rows if r.get("task") == task and r.get("normalization") == norm
                          and r.get("representation") == f"{stage}_{norm}" and r.get("split") == split and r.get("accuracy") is not None]
                    out.append({"task": task, "normalization": norm, "representation": stage, "split": split,
                                **ci([r["accuracy"] for r in rr], BOOT_SEED + len(out))})
    return out


def build_final_report(verdict, r0r1, r1r2, transfer_pass, main_fit, r12_fit,
                       r0r1_tests, r1r2_tests, stage_fit, stage_rows,
                       activation, nonlinear, j_cliff, trigger, selected,
                       wd_control_configs, transform, start_time):
    def gate_text(g): return "PASS" if g.get("pass") else "FAIL"
    lines = ["# O3b-J3D — Converged matched linear-probe calibration", "", "## FINAL VERDICT", "", f"**{verdict}**", "",
             "## Executive findings", "",
             f"- Certified cache integrity: PASS; 59 cache inputs independently rehashed and 18,661 feature IDs verified globally unique with row/label/canonical-ID alignment.",
             f"- R0/R1 standardized gate: **{gate_text(r0r1)}**; all primary runs converged: {r0r1.get('all_primary_runs_converged')}.",
             f"- R1/R2 analytical transfer: **{'PASS' if transfer_pass else 'FAIL'}**; `R2 ≈ R1 @ U.T`, train max abs error {transform.get('R2_vs_R1UT_train_max_abs'):.3g}.",
             f"- R1/R2 standardized independent-refit gate: **{gate_text(r1r2)}**.",
             f"- Measurement gate: **{'OPEN' if r0r1.get('pass') and r1r2.get('pass') and transfer_pass else 'CLOSED'}**.", "",
             "## Frozen protocol", "",
             f"Seeds `{SEEDS}` (exact O3b-L longer seeds); affine float32 linear softmax; train-only per-dimension standardization with epsilon {STD_EPS}; validation-only early stopping, evaluation every {EVAL_EVERY} steps, min {MIN_STEPS}, max {MAX_STEPS}, patience {PATIENCE}, restore best validation CE; 5 seeds; fixed minibatch ordering. Hyperparameters were selected from train/validation only and frozen before any TestA/TestB metrics.", "",
             "| task | optimizer | LR | weight decay |", "|---|---|---:|---:|"]
    for task in TASKS:
        c = selected[task]
        lines.append(f"| {task} | {c['optimizer']} | {c['lr']} | {c['weight_decay']} |")
    lines += ["", "## R0/R1 control", "", "Paired standardized TestA/TestB R1−R0 gate details are in `sanity/r0_r1_invariance_gate.json`; per-seed results and full convergence traces are in `tables/r0_r1_converged_results.csv` and `tables/r0_r1_convergence_curves.csv`.", "",
              "## R1/R2 basis invariance and weight transfer", "",
              "The analytical control transfers both classifier directions in raw and train-standardized coordinates. Detailed max/mean logit error, prediction mismatch and accuracy parity for train/val/TestA/TestB are in `tables/weight_transfer_control.csv`. Independent refit gate paired intervals are in `basis_invariance_gate.json`.", "",
              "Matched regularization controls are recorded in `tables/weight_decay_controls.csv`; alternate optimizer diagnostics, if executed, are in `tables/optimizer_robustness_if_needed.csv`.", "",
              "## Fresh certified J stagewise results", "",
              "Stagewise probes were run only if both calibration gates and analytical transfer passed. Actual stage definitions were loaded from J3C `stage_definition.json`; `H_mention` remains a mention representation and is not a memory write state. Primary Z0→Z1 entity TestA/TestB paired deltas:", ""]
    for key, val in j_cliff.items():
        if "entity_" in key:
            lines.append(f"- `{key}`: `{val}`")
    lines += ["", "Stagewise linear results: `tables/stagewise_linear_results.csv`; convergence curves: `tables/stagewise_linear_convergence.csv`; activation comparisons and train-only geometry stats: `tables/activation_results.csv`.", "",
              f"Nonlinear diagnostic trigger: `{json.dumps(trigger, ensure_ascii=False)}`.",
              f"Nonlinear results: {'ran' if nonlinear else 'NOT_RUN (trigger not met)'}.", "",
              "## Interpretation limits", "",
              "Probe accuracy is linear accessibility under a specified optimizer, regularization geometry and stopping rule; it is not an information-content measure. A basis-sensitive independent refit with a successful analytical transfer does not imply representation information loss. Historical O3b-H/J/L/L2 caches remain not row-level provenance-certified; these measurements use only the J3C certified cache.", "",
              "## Required questions", "",
              "1. Certified cache integrity: PASS.",
              f"2. Frozen probe protocol: see `probe_protocol_frozen.json` (per-task settings listed above).",
              f"3. Main probe runs validation-converged: R0/R1={r0r1.get('all_primary_runs_converged')}; R1/R2={r1r2.get('all_primary_runs_converged')}.",
              f"4. Standardized R0/R1 invariance gate: {gate_text(r0r1)}.",
              f"5. R1/R2 analytical transfer: {'PASS' if transfer_pass else 'FAIL'}.",
              f"6. Standardized R1/R2 independent-refit gate: {gate_text(r1r2)}.",
              f"7. Failure mechanism evidence: see weight-decay controls and, when needed, preregistered alternate optimizer fits; do not infer representational information loss.",
              f"8. Measurement gate: {'OPEN' if r0r1.get('pass') and r1r2.get('pass') and transfer_pass else 'CLOSED'}."]
    if not (r0r1.get("pass") and r1r2.get("pass") and transfer_pass):
        lines += [f"{i}. NOT_RUN — calibration gate closed." for i in range(9, 18)]
    else:
        lines += [f"9. Fresh certified Z0→Z1 entity raw cliff: `{j_cliff.get('raw_entity_testA_delta_z1_minus_z0')}` on TestA; TestB `{j_cliff.get('raw_entity_testB_delta_z1_minus_z0')}`.",
                  f"10. Fresh certified standardized cliff: `{j_cliff.get('standardized_entity_testA_delta_z1_minus_z0')}` on TestA; TestB `{j_cliff.get('standardized_entity_testB_delta_z1_minus_z0')}`.",
                  f"11. Historical −55.62pp phenomenon: {'NOT REPRODUCED AT COMPARABLE MAGNITUDE' if j_cliff.get('std_entity_testA_delta_mean', 0) > -0.10 else ('STRONG CLIFF REPRODUCED' if j_cliff.get('std_entity_testA_delta_mean', 0) <= -0.40 else 'PARTIALLY REPRODUCED')} (historical cache itself is not retroactively certified).",
                  "12. Robust linear entity-accessibility reduction: see paired standardized Z0→Z1 CIs in the stagewise result table.",
                  f"13. Nonlinear recoverability: {'tested; see nonlinear table' if nonlinear else 'NOT RUN; preregistered trigger not met'}.",
                  "14. Activation as a major suspect: judged from matched Z1→Z2 entity/relation deltas; see activation table.",
                  "15. ‘W1 projection destroys entity information’: NOT SUPPORTED by probe evidence alone.",
                  "16. ‘W1 projection reduces linear entity accessibility’: assessed independently on this fresh cache; see Z0→Z1 paired deltas.",
                  "17. Safest mechanism statement: W1 may alter conditioning and linear accessibility; neither probe scores nor basis-sensitive refits establish entity information destruction or learned entity invariance."]
    lines += ["18. Reconnect allowed: NO.", "19. O3c allowed: NO.", "",
              "## Exactly one recommended next experiment", "",
              ("Use a single frozen geometric/optimization control targeted at the observed Z0→Z1 result; do not reconnect or change architecture." if r0r1.get("pass") and r1r2.get("pass") and transfer_pass else "Run one targeted measurement follow-up on the first failing calibration contrast; keep the certified cache frozen and do not run J stagewise attribution."), "",
              "## Operational status", "",
              f"Checkpoint/model modified: NO. Backbone re-forwarded: NO. Reconnect: NO. O3c: NO. Repeated-V contamination touched: NO. Data leakage: NO. Runtime: {time.time()-start_time:.1f}s at report generation. Device: `{DEVICE}` ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}).", ""]
    return "\n".join(lines)


def finalize_early(verdict: str, reason: str, t0: float, gate_detail=None):
    (OUT / "FINAL_VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")
    dump(OUT / "FINAL_VERDICT.json", {"primary": verdict, "reason": reason, "measurement_gate": "CLOSED",
                                      "reconnect": False, "o3c": False, "checkpoint_modified": False,
                                      "data_leakage_detected": False, "runtime_seconds": time.time() - t0,
                                      "gate_detail": gate_detail})
    write_required_not_run_files(reason)
    materialize_root_deliverables()
    refresh_plots()
    gate_status = "PASS" if json.loads((OUT / "cache_integrity_recheck.json").read_text()).get("pass") else "FAIL"
    (OUT / "final_report.md").write_text(
        f"# O3b-J3D — Final report\n\n## FINAL VERDICT\n\n**{verdict}**\n\nReason: {reason}.\n\n"
        f"Q1. Certified cache integrity: {gate_status}.\nQ2. Frozen probe protocol: see `probe_protocol_frozen.json`; if absent, `NOT_RUN` (stopped before selection).\n"
        "Q3. All main runs converged: no, or unresolved as shown in convergence tables.\n"
        "Q4. Standardized R0/R1 invariance: FAIL/UNRESOLVED per `sanity/r0_r1_invariance_gate.json`.\n"
        "Q5. Analytical R1/R2 transfer: NOT_RUN unless `weight_transfer_control.csv` contains results.\n"
        "Q6. Standardized independent R1/R2 gate: NOT_RUN because an earlier gate closed.\n"
        "Q7. Failure-mode attribution: only reported if train/validation weight-decay or optimizer controls ran; otherwise unresolved.\n"
        "Q8. Measurement gate: CLOSED.\n"
        + "\n".join(f"Q{i}. NOT_RUN — calibration gate closed." for i in range(9, 16))
        + "\nQ16. ‘W1 reduces linear entity accessibility’: UNRESOLVED on the calibrated fresh-cache pipeline; no stagewise result was authorized.\n"
        "Q17. Safest combined statement: O3b-K did not find direct row-space evidence of selective entity rejection; O3b-L/L2 showed probe conditioning and basis-sensitive fitting; J3C certifies new row provenance but does not itself establish a mechanism. No information destruction or learned invariance is established.\n"
        "Q18. Reconnect allowed: NO.\nQ19. O3c allowed: NO.\n\n"
        "Stagewise Z attribution: **NOT_RUN** because a preregistered calibration gate did not pass. Checkpoint/model unchanged; no reconnect/O3c.\n\n"
        "Exactly one next experiment: target the first failing calibration contrast using the same certified cache and train/validation-only diagnosis; do not run J stagewise attribution until it passes.\n\n"
        "See `cache_integrity_recheck.json`, `probe_protocol_frozen.json` if available, and per-stage files for exact evidence.\n", encoding="utf-8")
    print(json.dumps({"stage": "j3d_stopped", "verdict": verdict, "reason": reason,
                      "elapsed_seconds": time.time() - t0}, ensure_ascii=False), flush=True)


def write_required_not_run_files(reason: str | None):
    why = reason or "gate passed only for documented stages; file is not applicable/not triggered"
    required = ["tables/r0_r1_converged_results.csv", "tables/r0_r1_convergence_curves.csv",
                "tables/r1_r2_converged_results.csv", "tables/r1_r2_convergence_curves.csv",
                "tables/weight_transfer_control.csv", "basis_invariance_gate.json",
                "tables/stagewise_linear_results.csv", "tables/stagewise_linear_convergence.csv",
                "tables/stagewise_nonlinear_results.csv", "tables/stagewise_nonlinear_convergence.csv",
                "tables/activation_results.csv"]
    for rel in required:
        p = OUT / rel
        if p.exists():
            continue
        if p.suffix == ".json":
            dump(p, {"status": "NOT_RUN", "reason": why})
        else:
            mark_not_run(p, why)


def materialize_root_deliverables():
    import shutil
    mapping = {
        "tables/optimizer_calibration.csv": "optimizer_calibration.csv",
        "tables/optimizer_calibration_curves.csv": "optimizer_calibration_curves.csv",
        "report/optimizer_calibration_summary.md": "optimizer_calibration_summary.md",
        "tables/r0_r1_converged_results.csv": "r0_r1_converged_results.csv",
        "tables/r0_r1_convergence_curves.csv": "r0_r1_convergence_curves.csv",
        "tables/r1_r2_converged_results.csv": "r1_r2_converged_results.csv",
        "tables/r1_r2_convergence_curves.csv": "r1_r2_convergence_curves.csv",
        "tables/weight_transfer_control.csv": "weight_transfer_control.csv",
        "tables/stagewise_linear_results.csv": "stagewise_linear_results.csv",
        "tables/stagewise_linear_convergence.csv": "stagewise_linear_convergence.csv",
        "tables/stagewise_nonlinear_results.csv": "stagewise_nonlinear_results.csv",
        "tables/stagewise_nonlinear_convergence.csv": "stagewise_nonlinear_convergence.csv",
        "tables/activation_results.csv": "activation_results.csv",
    }
    for source, destination in mapping.items():
        src = OUT / source
        dst = OUT / destination
        if src.exists():
            shutil.copyfile(src, dst)
        elif not dst.exists():
            mark_not_run(dst, "not run because a preregistered gate closed before this stage")
    verdict = OUT / "FINAL_VERDICT.json"
    if verdict.exists():
        shutil.copyfile(verdict, OUT / "verdict.json")
    final_report = OUT / "final_report.md"
    if final_report.exists():
        shutil.copyfile(final_report, OUT / "report" / "10_final_report.md")


def refresh_plots():
    plot_dir = OUT / "plots"
    plot_dir.mkdir(exist_ok=True)
    paths = [OUT / "tables" / "r0_r1_convergence_curves.csv", OUT / "tables" / "stagewise_linear_convergence.csv"]
    for file_idx, path in enumerate(paths):
        if not path.exists() or path.stat().st_size < 20:
            make_svg(plot_dir / ("calibration_convergence.svg" if file_idx == 0 else "stagewise_convergence.svg"),
                     "Probe validation convergence", [], "validation accuracy")
            continue
        try:
            rows = read_csv(path)
            subset = [r for r in rows if r.get("task") == "entity" and r.get("representation") in
                      (("R0_raw", "R1_raw", "R0_standardized", "R1_standardized") if file_idx == 0 else
                       ("Z0_standardized", "Z1_standardized", "Z2_standardized"))]
            series = []
            for rep in sorted({r["representation"] for r in subset}):
                by_step = {}
                for row in subset:
                    if row["representation"] == rep:
                        by_step.setdefault(float(row["step"]), []).append(float(row["validation_accuracy"]))
                series.append((rep, [(x, float(np.mean(v))) for x, v in sorted(by_step.items())]))
            make_svg(plot_dir / ("calibration_convergence.svg" if file_idx == 0 else "stagewise_convergence.svg"),
                     "Entity validation accuracy vs optimization step", series, "validation accuracy")
        except Exception as exc:
            (plot_dir / f"plot_{file_idx}_error.txt").write_text(repr(exc) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        OUT.mkdir(parents=True, exist_ok=True)
        dump(OUT / "sanity" / "unhandled_error.json", {"error": repr(exc), "traceback": traceback.format_exc()})
        raise
