from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ["LD_LIBRARY_PATH"] = "/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import torch
import torch.nn.functional as F


CACHE = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache")
OUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3e_joint_convergence_diag")
SEEDS = [11, 23, 37, 53, 71]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
TASK_OFFSET = 2


def dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data(include_test: bool = False):
    tr_raw = {r: np.asarray(np.load(CACHE / f"{r}_train.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64)
              for r in ("R1", "R2")}
    means = {r: x.mean(0) for r, x in tr_raw.items()}
    stds = {r: x.std(0) for r, x in tr_raw.items()}
    for r in stds:
        stds[r][stds[r] < EPS] = 1.0
    xs = {}
    for r in ("R1", "R2"):
        xs[(r, "train")] = ((tr_raw[r] - means[r]) / stds[r]).astype(np.float32)
        del tr_raw[r]
        raw = np.asarray(np.load(CACHE / f"{r}_val.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64)
        xs[(r, "val")] = ((raw - means[r]) / stds[r]).astype(np.float32)
        del raw
        if include_test:
            raw = np.asarray(np.load(CACHE / f"{r}_testA.npy", mmap_mode="r", allow_pickle=False), dtype=np.float64)
            xs[(r, "testA")] = ((raw - means[r]) / stds[r]).astype(np.float32)
            del raw
    ys = {s: np.asarray([int(r["joint_class_id"]) for r in read_csv(CACHE / f"labels_{s}.csv")], dtype=np.int64)
          for s in (("train", "val", "testA", "testB") if include_test else ("train", "val"))}
    return xs, ys, means, stds


def make_transform(means: dict, stds: dict):
    with np.load(CACHE / "w1_svd_matrices.npz", allow_pickle=False) as z:
        U = np.asarray(z["U"], dtype=np.float64)
    A = (stds["R1"][:, None] * U.T) / stds["R2"][None, :]
    off = (means["R1"] @ U.T - means["R2"]) / stds["R2"]
    A_inv = np.linalg.inv(A)
    off_inv = -off @ A_inv
    return {("R1", "R2"): (A, off), ("R2", "R1"): (A_inv, off_inv)}


def transfer(W: np.ndarray, b: np.ndarray, src: str, dst: str, maps: dict):
    A, off = maps[(src, dst)]
    Wd = np.linalg.solve(A, W.T).T
    bd = b - off @ Wd.T
    return Wd, bd


def make_tensors(xs: dict, ys: dict, device=DEVICE):
    xtr = torch.from_numpy(np.stack([xs[("R1", "train")], xs[("R2", "train")]])).to(device)
    xval = torch.from_numpy(np.stack([xs[("R1", "val")], xs[("R2", "val")]])).to(device)
    ytr = torch.from_numpy(ys["train"]).long().to(device)
    yval = torch.from_numpy(ys["val"]).long().to(device)
    nclass = int(ys["train"].max()) + 1
    return xtr, xval, ytr, yval, nclass


def init_params(nmodels: int, nclass: int, dim: int, seeds: list[int], nrep: int = 2):
    weights, biases = [], []
    bound = 1.0 / math.sqrt(dim)
    for seed in seeds:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + TASK_OFFSET * 1009)
        w = torch.empty(nclass, dim).uniform_(-bound, bound, generator=g)
        b = torch.empty(nclass).uniform_(-bound, bound, generator=g)
        for _ in range(nrep):
            weights.append(w.clone())
            biases.append(b.clone())
    if len(weights) != nmodels:
        raise ValueError("initializer model count mismatch")
    return torch.nn.Parameter(torch.stack(weights).to(DEVICE)), torch.nn.Parameter(torch.stack(biases).to(DEVICE))


def evaluate(x: torch.Tensor, y: torch.Tensor, W: torch.Tensor, b: torch.Tensor, chunk: int = 512):
    m, n, _ = x.shape
    nclass = W.shape[1]
    known = (y >= 0) & (y < nclass)
    sums = torch.zeros(m, dtype=torch.float64, device=x.device)
    correct = torch.zeros(m, dtype=torch.float64, device=x.device)
    total = 0
    maxlogit = torch.zeros(m, dtype=torch.float64, device=x.device)
    margin_sum = torch.zeros(m, dtype=torch.float64, device=x.device)
    margin_sq = torch.zeros(m, dtype=torch.float64, device=x.device)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        mask = known[lo:hi]
        if not bool(mask.any()):
            continue
        xb = x[:, lo:hi][..., :]
        logits = torch.bmm(xb, W.transpose(1, 2)) + b[:, None, :]
        logits = logits[:, mask]
        yy = y[lo:hi][mask]
        flat = logits.reshape(m * len(yy), nclass)
        target = yy.unsqueeze(0).expand(m, -1).reshape(-1)
        loss = F.cross_entropy(flat, target, reduction="none").reshape(m, -1)
        sums += loss.double().sum(1)
        pred = logits.argmax(-1)
        correct += (pred == yy.unsqueeze(0)).double().sum(1)
        top2 = logits.topk(2, dim=-1).values
        margin = top2[..., 0] - top2[..., 1]
        margin_sum += margin.double().sum(1)
        margin_sq += (margin.double() ** 2).sum(1)
        maxlogit = torch.maximum(maxlogit, logits.abs().amax(dim=(1, 2)).double())
        total += len(yy)
    means = sums / max(total, 1)
    acc = correct / max(total, 1)
    mmean = margin_sum / max(total, 1)
    mstd = torch.sqrt(torch.clamp(margin_sq / max(total, 1) - mmean ** 2, min=0))
    return [{"loss": float(means[i].cpu()), "accuracy": float(acc[i].cpu()),
             "max_logit_abs": float(maxlogit[i].cpu()), "margin_mean": float(mmean[i].cpu()),
             "margin_std": float(mstd[i].cpu()), "known_rows": int(total)} for i in range(m)]


def optimizer(kind: str, params, lr: float, wd: float):
    if kind == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)


def screen_configs():
    out = []
    for kind, lrs in (("AdamW", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]),
                      ("SGD_momentum_0.9", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])):
        for lr in lrs:
            for wd in (0.0, 1e-6, 1e-5, 1e-4, 1e-3):
                out.append({"optimizer": kind, "lr": lr, "weight_decay": wd})
    return out


def init_cuda_data():
    xs, ys, means, stds = load_data(False)
    xtr, xval, ytr, yval, nclass = make_tensors(xs, ys)
    return xs, ys, means, stds, xtr, xval, ytr, yval, nclass


def search() -> int:
    xs, ys, means, stds, xtr, xval, ytr, yval, nclass = init_cuda_data()
    config_rows, curve_rows = [], []
    configs = screen_configs()
    dump_json(OUT / "configs" / "search_preregistered.json", {
        "scope": "standardized R1/R2 + Joint only; train/validation only",
        "phase_a_steps": 250, "eval_every": 50, "seed": 11, "batch_size": 1024,
        "configs": configs, "selection_metric": "mean of R1/R2 validation CE over final two evaluations",
        "test_metrics_read": False, "optimizer_state_reset_per_config": True})
    for ci, cfg in enumerate(configs, 1):
        torch.manual_seed(11 + TASK_OFFSET * 1009)
        W, B = init_params(2, nclass, 256, [11])
        opt = optimizer(cfg["optimizer"], [W, B], cfg["lr"], cfg["weight_decay"])
        rng = np.random.default_rng(73019)
        batch = 1024
        eval_steps = [0, 50, 100, 150, 200, 250]
        points = {"R1": [], "R2": []}
        best_w = W.detach().clone()
        best_b = B.detach().clone()
        diverged = False
        for step in range(251):
            if step in eval_steps:
                mets = evaluate(xval, yval, W.detach(), B.detach())
                for ri, rep in enumerate(("R1", "R2")):
                    p = {"step": step, "validation_loss": mets[ri]["loss"], "validation_accuracy": mets[ri]["accuracy"]}
                    points[rep].append(p)
                    curve_rows.append({"phase": "screen", "optimizer": cfg["optimizer"], "lr": cfg["lr"],
                                       "weight_decay": cfg["weight_decay"], "seed": 11, "representation": rep,
                                       **p})
                    if not math.isfinite(p["validation_loss"]):
                        diverged = True
                if step == 250:
                    break
            ix = torch.as_tensor(rng.integers(0, len(ys["train"]), size=batch), dtype=torch.long, device=DEVICE)
            xb = xtr[:, ix, :]
            logits = torch.bmm(xb, W.transpose(1, 2)) + B[:, None, :]
            loss = F.cross_entropy(logits.reshape(2 * batch, nclass), ytr[ix].repeat(2), reduction="none").reshape(2, batch).mean(1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if not bool(torch.isfinite(W).all()) or not bool(torch.isfinite(B).all()):
                diverged = True
                break
        last_losses = [p["validation_loss"] for rep in ("R1", "R2") for p in points[rep][-2:]]
        last_acc = [p["validation_accuracy"] for rep in ("R1", "R2") for p in points[rep][-2:]]
        score = float(np.mean(last_losses)) if last_losses else float("inf")
        config_rows.append({**cfg, "screen_seed": 11, "steps": step, "batch_size": batch,
                            "validation_ce_last2_mean_R1_R2": score,
                            "validation_accuracy_last2_mean_R1_R2": float(np.mean(last_acc)) if last_acc else None,
                            "diverged": diverged,
                            "screening_only": True, "test_metrics_used": False})
        print(f"SCREEN {ci}/{len(configs)} {cfg['optimizer']} lr={cfg['lr']:g} wd={cfg['weight_decay']:g} val_ce={score:.6f} diverged={diverged}", flush=True)
        del W, B, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(OUT / "optimizer_search.csv", config_rows)
    write_csv(OUT / "convergence_curves.csv", curve_rows)
    viable = [r for r in config_rows if not r["diverged"] and math.isfinite(r["validation_ce_last2_mean_R1_R2"])]
    candidates = []
    for fam in ("AdamW", "SGD_momentum_0.9"):
        fam_rows = [r for r in viable if r["optimizer"] == fam]
        candidates.append(min(fam_rows, key=lambda r: r["validation_ce_last2_mean_R1_R2"]))
        zero_rows = [r for r in fam_rows if r["weight_decay"] == 0.0]
        if zero_rows:
            best0 = min(zero_rows, key=lambda r: r["validation_ce_last2_mean_R1_R2"])
            if not any(c["optimizer"] == best0["optimizer"] and c["lr"] == best0["lr"] and c["weight_decay"] == best0["weight_decay"] for c in candidates):
                candidates.append(best0)
    # Ensure a matched weight-decay contrast at the best AdamW learning rate.
    best_adam = next(c for c in candidates if c["optimizer"] == "AdamW")
    contrast = [r for r in viable if r["optimizer"] == "AdamW" and r["lr"] == best_adam["lr"] and r["weight_decay"] != best_adam["weight_decay"]]
    if contrast:
        # If the best AdamW run is regularized, explicitly retain its no-decay counterpart.
        # Otherwise retain the strongest same-LR nonzero-decay candidate.
        zeros = [r for r in contrast if r["weight_decay"] == 0.0]
        candidates.append(min(zeros or contrast, key=lambda r: r["validation_ce_last2_mean_R1_R2"]))
    # Deduplicate and keep the screen deliberately small for the full convergence phase.
    unique = {}
    for c in candidates:
        unique[(c["optimizer"], c["lr"], c["weight_decay"])] = c
    chosen = list(unique.values())
    selection = {"selection_is_train_validation_only": True, "test_metrics_read": False,
                 "screening_steps": 250, "selection_rule": "lowest R1/R2 mean validation CE among quick-screen candidates; retain best AdamW, best SGD, each family's best WD=0 if distinct, plus same-LR AdamW WD contrast",
                 "phase_b_candidates": [{k: c[k] for k in ("optimizer", "lr", "weight_decay", "validation_ce_last2_mean_R1_R2", "validation_accuracy_last2_mean_R1_R2")} for c in chosen]}
    dump_json(OUT / "configs" / "phase_b_candidates.json", selection)
    print(json.dumps({"search_complete": True, "candidate_count": len(chosen), "candidates": selection["phase_b_candidates"]}), flush=True)
    return 0


def plateau_status(points: list[dict], eval_every: int, min_steps: int = 2000, window: int = 50):
    if len(points) < window:
        return {"plateau_confirmed": False, "plateau_step": None, "window_evaluations": len(points),
                "window_relative_ce_improvement": None, "window_accuracy_range": None,
                "loss_slope_relative_per_eval": None, "accuracy_slope_per_eval": None}
    w = points[-window:]
    if w[-1]["step"] < min_steps:
        return {"plateau_confirmed": False, "plateau_step": None, "window_evaluations": window,
                "window_relative_ce_improvement": None, "window_accuracy_range": None,
                "loss_slope_relative_per_eval": None, "accuracy_slope_per_eval": None}
    start_loss = float(w[0]["validation_loss"])
    best_loss = min(float(p["validation_loss"]) for p in w)
    rel_improvement = max(0.0, start_loss - best_loss) / max(abs(start_loss), 1e-12)
    acc_change = max(float(p["validation_accuracy"]) for p in w) - min(float(p["validation_accuracy"]) for p in w)
    xx = np.arange(len(w), dtype=np.float64)
    loss = np.asarray([p["validation_loss"] for p in w], dtype=np.float64)
    acc = np.asarray([p["validation_accuracy"] for p in w], dtype=np.float64)
    loss_slope = float(np.polyfit(xx, loss, 1)[0] / max(abs(float(loss.mean())), 1e-12))
    acc_slope = float(np.polyfit(xx, acc, 1)[0])
    passed = rel_improvement < 1e-5 and acc_change < 1e-4 and abs(loss_slope) < 1e-5 and abs(acc_slope) < 1e-4
    if passed:
        return {"plateau_confirmed": True, "plateau_step": int(w[-1]["step"]),
                "window_evaluations": window, "window_relative_ce_improvement": rel_improvement,
                "window_accuracy_range": acc_change, "loss_slope_relative_per_eval": loss_slope,
                "accuracy_slope_per_eval": acc_slope}
    return {"plateau_confirmed": False, "plateau_step": None,
            "window_evaluations": window, "window_relative_ce_improvement": rel_improvement,
            "window_accuracy_range": acc_change, "loss_slope_relative_per_eval": loss_slope,
            "accuracy_slope_per_eval": acc_slope}


def train_config(cfg: dict, max_steps: int, eval_every: int, seeds: list[int], batch: int,
                 phase: str, paired_init: bool = False, full_batch: bool = False,
                 save_state: bool = True):
    xs, ys, means, stds, xtr, xval, ytr, yval, nclass = init_cuda_data()
    nrep, nmodels = 2, len(seeds) * 2
    W, B = init_params(nmodels, nclass, 256, seeds)
    maps = make_transform(means, stds)
    if paired_init:
        with torch.no_grad():
            for si in range(len(seeds)):
                wsrc = W[si*2].detach().cpu().numpy().astype(np.float64)
                bsrc = B[si*2].detach().cpu().numpy().astype(np.float64)
                wdst, bdst = transfer(wsrc, bsrc, "R1", "R2", maps)
                W[si*2+1].copy_(torch.from_numpy(wdst.astype(np.float32)).to(DEVICE))
                B[si*2+1].copy_(torch.from_numpy(bdst.astype(np.float32)).to(DEVICE))
    opt = optimizer(cfg["optimizer"], [W, B], float(cfg["lr"]), float(cfg["weight_decay"]))
    rng = np.random.default_rng(73019)
    bsz = len(ys["train"]) if full_batch else min(batch, len(ys["train"]))
    curves, per_model_points = [], [[] for _ in range(nmodels)]
    best_losses = np.full(nmodels, np.inf)
    best_ws, best_bs = [None] * nmodels, [None] * nmodels
    plateau_first = [None] * nmodels
    last_grad = np.full(nmodels, np.nan)
    t0 = time.time()
    for step in range(max_steps + 1):
        if step % eval_every == 0:
            train_metrics = evaluate(xtr, ytr, W.detach(), B.detach())
            val_metrics = evaluate(xval, yval, W.detach(), B.detach())
            for mi in range(nmodels):
                rep = "R1" if mi % 2 == 0 else "R2"
                seed = seeds[mi // 2]
                point = {"step": step, "validation_loss": val_metrics[mi]["loss"],
                         "validation_accuracy": val_metrics[mi]["accuracy"]}
                per_model_points[mi].append(point)
                if val_metrics[mi]["loss"] < best_losses[mi]:
                    best_losses[mi] = val_metrics[mi]["loss"]
                    best_ws[mi] = W[mi].detach().cpu().numpy().copy()
                    best_bs[mi] = B[mi].detach().cpu().numpy().copy()
                pstat = plateau_status(per_model_points[mi], eval_every)
                if pstat["plateau_confirmed"] and plateau_first[mi] is None:
                    plateau_first[mi] = pstat["plateau_step"]
                curves.append({"phase": phase, "optimizer": cfg["optimizer"], "lr": cfg["lr"],
                               "weight_decay": cfg["weight_decay"], "batch_size": bsz,
                               "full_batch": full_batch, "paired_initialization": paired_init,
                               "seed": seed, "representation": rep, "step": step,
                               "train_loss": train_metrics[mi]["loss"],
                               "train_accuracy": train_metrics[mi]["accuracy"],
                               "validation_loss": val_metrics[mi]["loss"],
                               "validation_accuracy": val_metrics[mi]["accuracy"],
                               "gradient_norm": float(last_grad[mi]) if np.isfinite(last_grad[mi]) else None,
                               "weight_norm": float(torch.linalg.vector_norm(W[mi]).detach().cpu()),
                               "bias_norm": float(torch.linalg.vector_norm(B[mi]).detach().cpu()),
                               "plateau_first_step": plateau_first[mi]})
            if step > 0 and step % 2000 == 0:
                nplateau = sum(x is not None for x in plateau_first)
                print(f"PROGRESS {phase} step={step}/{max_steps} plateau={nplateau}/{nmodels} elapsed={time.time()-t0:.1f}s", flush=True)
            if step >= 2000 and all(p is not None for p in plateau_first):
                break
            if step == max_steps:
                break
        if full_batch:
            ix = torch.arange(len(ys["train"]), dtype=torch.long, device=DEVICE)
        else:
            ix = torch.as_tensor(rng.integers(0, len(ys["train"]), size=bsz), dtype=torch.long, device=DEVICE)
        xb = xtr[:, ix, :]
        yb = ytr[ix]
        logits = torch.bmm(xb, W.transpose(1, 2)) + B[:, None, :]
        loss = F.cross_entropy(logits.reshape(nmodels * len(ix), nclass), yb.repeat(nmodels), reduction="none").reshape(nmodels, len(ix)).mean(1).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            gn = torch.sqrt((W.grad ** 2).sum((1, 2)) + (B.grad ** 2).sum(1))
            last_grad[:] = gn.detach().cpu().numpy()
        opt.step()
        if not bool(torch.isfinite(W).all()) or not bool(torch.isfinite(B).all()):
            print(f"DIVERGED {phase} step={step+1}", flush=True)
            break
    summaries, states = [], {}
    for mi in range(nmodels):
        rep = "R1" if mi % 2 == 0 else "R2"
        seed = seeds[mi // 2]
        pstat = plateau_status(per_model_points[mi], eval_every)
        best_w, best_b = best_ws[mi], best_bs[mi]
        summaries.append({"phase": phase, "optimizer": cfg["optimizer"], "lr": cfg["lr"],
                          "weight_decay": cfg["weight_decay"], "batch_size": bsz,
                          "full_batch": full_batch, "paired_initialization": paired_init,
                          "seed": seed, "representation": rep, "steps_run": curves[-1]["step"] if curves else 0,
                          "plateau_confirmed": bool(plateau_first[mi] is not None),
                          "plateau_step": plateau_first[mi], "final_window_plateau": pstat["plateau_confirmed"],
                          "best_val_loss": float(best_losses[mi]),
                          "best_val_accuracy": float(max(p["validation_accuracy"] for p in per_model_points[mi])),
                          "final_train_loss": curves[-1]["train_loss"] if curves else None,
                          "final_val_loss": per_model_points[mi][-1]["validation_loss"] if per_model_points[mi] else None,
                          "last_eval_gradient_norm": float(last_grad[mi]),
                          **{k: v for k, v in pstat.items() if k != "plateau_confirmed"}})
        states[(rep, seed)] = {"weight": best_w, "bias": best_b,
                               "best_step": min((p["step"] for p in per_model_points[mi]
                                                if p["validation_loss"] == best_losses[mi]), default=None)}
    if save_state:
        safe = f"{phase}_{cfg['optimizer']}_lr{cfg['lr']:g}_wd{cfg['weight_decay']:g}".replace(".", "p")
        sp = OUT / "probe_runs" / f"{safe}.npz"
        np.savez_compressed(sp, **{f"{rep}_seed{seed}_{k}": v for (rep, seed), st in states.items() for k, v in st.items() if k in ("weight", "bias")})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summaries, curves, states


def main_convergence() -> int:
    cand_file = json.loads((OUT / "configs" / "phase_b_candidates.json").read_text(encoding="utf-8"))
    candidates = cand_file["phase_b_candidates"]
    curves_all, summaries_all = [], []
    # Full five-seed matched convergence for the screened AdamW/SGD leaders and a weight-decay control.
    for i, raw in enumerate(candidates, 1):
        cfg = {k: raw[k] for k in ("optimizer", "lr", "weight_decay")}
        summaries, curves, _ = train_config(cfg, 100000, 200, SEEDS, 1024, f"phase_b_{i}")
        summaries_all.extend(summaries)
        curves_all.extend(curves)
        write_csv(OUT / "convergence_summary.csv", summaries_all)
        write_csv(OUT / "convergence_curves.csv", read_csv(OUT / "convergence_curves.csv") + curves if (OUT / "convergence_curves.csv").is_file() else curves)
        print(f"PHASE_B_DONE {i}/{len(candidates)} plateau={sum(bool(r['plateau_confirmed']) for r in summaries)}/{len(summaries)}", flush=True)
    protocols = []
    for key in {(r["optimizer"], r["lr"], r["weight_decay"]) for r in summaries_all}:
        group = [r for r in summaries_all if (r["optimizer"], r["lr"], r["weight_decay"]) == key]
        plateaus = sum(bool(r["plateau_confirmed"]) for r in group)
        protocols.append({"optimizer": key[0], "lr": key[1], "weight_decay": key[2],
                          "plateau_count": plateaus, "total_runs": len(group),
                          "mean_best_val_loss": float(np.mean([r["best_val_loss"] for r in group])),
                          "mean_best_val_accuracy": float(np.mean([r["best_val_accuracy"] for r in group])),
                          "mean_abs_R1_R2_val_CE_gap": float(abs(np.mean([r["best_val_loss"] for r in group if r["representation"] == "R1"]) - np.mean([r["best_val_loss"] for r in group if r["representation"] == "R2"]))),
                          "all_seeds_both_representations_plateau": plateaus == len(group)})
    dump_json(OUT / "configs" / "phase_b_protocol_comparison.json", {"selection_metrics": "train/validation only",
              "test_metrics_read": False, "candidates": protocols})
    return 0


def conditioning() -> dict:
    xs, ys, means, stds = load_data(False)
    info = {"standardization_statistics_source": "train only", "epsilon": EPS, "representations": {}}
    for rep in ("R1", "R2"):
        x = xs[(rep, "train")].astype(np.float64)
        cov = np.cov(x, rowvar=False)
        eig = np.linalg.eigvalsh(cov)
        eig = np.clip(eig, 0, None)[::-1]
        p = eig / max(eig.sum(), 1e-30)
        eff = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
        info["representations"][rep] = {"mean_abs_feature_mean": float(np.mean(np.abs(x.mean(0)))),
                "min_dimension_std": float(x.std(0).min()), "max_dimension_std": float(x.std(0).max()),
                "covariance_condition_number": float(eig[0] / max(eig[-1], 1e-30)),
                "effective_rank_entropy": eff, "participation_ratio": float(eig.sum()**2 / max((eig**2).sum(), 1e-30)),
                "top_pc_explained_variance": [float(eig[:k].sum()/eig.sum()) for k in (1,5,10,32,64,128,256)],
                "feature_norm_mean": float(np.linalg.norm(x, axis=1).mean()),
                "feature_norm_std": float(np.linalg.norm(x, axis=1).std()),
                "covariance_eigenvalues": eig.tolist()}
    return info


def control_runs() -> int:
    # Matched full-batch deterministic SGD and paired transformed initialization (train/validation only).
    cfgf = json.loads((OUT / "configs" / "phase_b_candidates.json").read_text(encoding="utf-8"))["phase_b_candidates"]
    adam = next((r for r in cfgf if r["optimizer"] == "AdamW"), None)
    sgd = next((r for r in cfgf if r["optimizer"] == "SGD_momentum_0.9"), None)
    base = adam or sgd or {"optimizer": "AdamW", "lr": 0.003, "weight_decay": 0.0}
    rows = []
    full_cfg = {"optimizer": "SGD_momentum_0.9", "lr": float((sgd or {}).get("lr", 0.03)), "weight_decay": 0.0}
    sm, cr, _ = train_config(full_cfg, 5000, 100, [11], 1024, "full_batch_sgd", full_batch=True)
    rows.extend(sm)
    write_csv(OUT / "full_batch_control.csv", rows)
    paired_cfg = {k: base[k] for k in ("optimizer", "lr", "weight_decay")}
    pm, pc, _ = train_config(paired_cfg, 10000, 200, SEEDS, 1024, "paired_initialization", paired_init=True)
    write_csv(OUT / "paired_initialization_control.csv", pm)
    # Convex full-batch LBFGS, one paired seed per representation, fixed features; maximum 300 optimizer iterations.
    xs, ys, means, stds = load_data(False)
    xtr, xval, ytr, yval, nclass = make_tensors(xs, ys)
    lbfgs_rows = []
    seed = 11
    for ri, rep in enumerate(("R1", "R2")):
        torch.manual_seed(seed + TASK_OFFSET * 1009)
        bound = 1.0 / math.sqrt(256)
        w = torch.nn.Parameter(torch.empty(nclass, 256, device=DEVICE).uniform_(-bound, bound))
        b = torch.nn.Parameter(torch.empty(nclass, device=DEVICE).uniform_(-bound, bound))
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=1, history_size=10,
                                tolerance_grad=1e-7, tolerance_change=1e-12, line_search_fn="strong_wolfe")
        train_x = xtr[ri]
        valid_x = xval[ri]
        last = {}
        start = time.time()
        for it in range(1, 301):
            def closure():
                opt.zero_grad(set_to_none=True)
                logits = train_x @ w.T + b
                loss = F.cross_entropy(logits, ytr)
                loss.backward()
                return loss
            opt.step(closure)
            if it % 10 == 0 or it == 1:
                with torch.no_grad():
                    tr = F.cross_entropy(train_x @ w.T + b, ytr).item()
                    va_logits = valid_x @ w.T + b
                    known = (yval >= 0) & (yval < nclass)
                    va = F.cross_entropy(va_logits[known], yval[known]).item()
                    vacc = (va_logits[known].argmax(-1) == yval[known]).float().mean().item()
                    gn = math.nan if w.grad is None else float(torch.sqrt((w.grad**2).sum() + (b.grad**2).sum()).item())
                last = {"iteration": it, "train_loss": tr, "validation_loss": va,
                        "validation_accuracy": vacc, "gradient_norm": gn}
            if it % 50 == 0:
                print(f"LBFGS {rep} iter={it} train={last.get('train_loss')} val={last.get('validation_loss')}", flush=True)
        lbfgs_rows.append({"representation": rep, "seed": seed, "optimizer": "torch.LBFGS_strong_wolfe",
                           "iterations": 300, **last, "wall_seconds": time.time()-start,
                           "converged_by_grad_tolerance": bool(last.get("gradient_norm", 1) < 1e-5),
                           "test_metrics_read": False})
        del opt, w, b
    write_csv(OUT / "convex_solver_control.csv", lbfgs_rows)
    # Warm-start each target representation from the best validation classifier trained in the source coordinates.
    selected = pick_protocol()
    warm_rows = []
    for src, dst in (("R1", "R2"), ("R2", "R1")):
        npz = np.load(state_path(selected), allow_pickle=False)
        for seed in SEEDS:
            Ws = npz[f"{src}_seed{seed}_weight"].astype(np.float64)
            bs = npz[f"{src}_seed{seed}_bias"].astype(np.float64)
            Wd, bd = transfer(Ws, bs, src, dst, make_transform(means, stds))
            ri = 0 if dst == "R1" else 1
            wt = torch.nn.Parameter(torch.from_numpy(Wd.astype(np.float32)).to(DEVICE))
            bt = torch.nn.Parameter(torch.from_numpy(bd.astype(np.float32)).to(DEVICE))
            opt = optimizer(selected, [wt, bt], selected["lr"], selected["weight_decay"])
            xdst = xtr[ri]
            target = ytr
            checkpoints = {0, 10, 100, 1000}
            def metric(step):
                with torch.no_grad():
                    tr = F.cross_entropy(xdst @ wt.T + bt, target).item()
                    vv = xval[ri] @ wt.T + bt
                    kn = (yval >= 0) & (yval < nclass)
                    vl = F.cross_entropy(vv[kn], yval[kn]).item()
                    va = (vv[kn].argmax(-1) == yval[kn]).float().mean().item()
                warm_rows.append({"source": src, "target": dst, "seed": seed, "step": step,
                                  "train_loss": tr, "validation_loss": vl, "validation_accuracy": va})
            metric(0)
            rng = np.random.default_rng(73019)
            for step in range(1, 1001):
                ix = torch.as_tensor(rng.integers(0, len(target), size=1024), dtype=torch.long, device=DEVICE)
                logits = xdst[ix] @ wt.T + bt
                loss = F.cross_entropy(logits, target[ix])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if step in checkpoints:
                    metric(step)
            del opt, wt, bt
        npz.close()
    write_csv(OUT / "transfer_warmstart_control.csv", warm_rows)
    dump_json(OUT / "feature_conditioning_summary.json", conditioning())
    write_csv(OUT / "convergence_curves.csv", []) if not (OUT / "convergence_curves.csv").exists() else None
    return 0


def pick_protocol() -> dict:
    rows = read_csv(OUT / "convergence_summary.csv")
    keys = sorted({(r["optimizer"], float(r["lr"]), float(r["weight_decay"])) for r in rows})
    candidates = []
    for key in keys:
        grp = [r for r in rows if (r["optimizer"], float(r["lr"]), float(r["weight_decay"])) == key]
        candidates.append({"optimizer": key[0], "lr": key[1], "weight_decay": key[2],
                           "plateau_count": sum(r["plateau_confirmed"] in (True, "True") for r in grp),
                           "runs": len(grp), "mean_val_ce": float(np.mean([float(r["best_val_loss"]) for r in grp])),
                           "mean_val_accuracy": float(np.mean([float(r["best_val_accuracy"]) for r in grp]))})
    candidates.sort(key=lambda r: (-r["plateau_count"], r["mean_val_ce"], r["mean_val_accuracy"]))
    c = candidates[0]
    return {"optimizer": c["optimizer"], "lr": c["lr"], "weight_decay": c["weight_decay"],
            "batch_size": 1024, "seed_list": SEEDS, "initialization": "uniform[-1/sqrt(256),+1/sqrt(256)] with seed+2018; matched R1/R2 starts unless paired-init control",
            "preprocessing": "train-only per-dimension standardization; std<1e-8 set to 1; epsilon=1e-8",
            "training_steps_max": 100000, "minimum_steps": 2000, "evaluation_interval": 200,
            "plateau_rule": "50 consecutive validation evaluations; relative best CE improvement <1e-5, accuracy range <0.01pp, normalized CE slope <1e-5/eval, accuracy slope <0.01pp/eval",
            "selection_basis": "train/validation only; prefer plateau count then validation CE; TestA excluded",
            "selected_from_converged_candidates": {k: c[k] for k in ("plateau_count", "runs", "mean_val_ce", "mean_val_accuracy")}}


def state_path(protocol: dict) -> Path:
    summaries = read_csv(OUT / "convergence_summary.csv")
    matching = [r for r in summaries if r["optimizer"] == protocol["optimizer"] and abs(float(r["lr"])-float(protocol["lr"])) < 1e-12 and abs(float(r["weight_decay"])-float(protocol["weight_decay"])) < 1e-12]
    if not matching:
        raise FileNotFoundError("no trained state for selected protocol")
    phase = matching[0]["phase"]
    safe = f"{phase}_{protocol['optimizer']}_lr{float(protocol['lr']):g}_wd{float(protocol['weight_decay']):g}".replace(".", "p")
    return OUT / "probe_runs" / f"{safe}.npz"


def aggregate(values: list[float], seed: int = 20260925) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"n": 0, "mean": None, "std": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    boots = np.asarray([rng.choice(a, size=len(a), replace=True).mean() for _ in range(20000)])
    return {"n": len(a), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def frozen_final() -> int:
    selected = pick_protocol()
    selected["protocol_frozen_before_TestA"] = True
    selected["testB_joint"] = "NOT EVALUABLE — NO KNOWN JOINT LABELS (Stage 0 found 0 known rows)"
    dump_json(OUT / "j3e_probe_protocol_frozen.json", selected)
    xs, ys, means, stds = load_data(True)
    maps = make_transform(means, stds)
    npz = np.load(state_path(selected), allow_pickle=False)
    final_rows, transfer_rows, gap_rows = [], [], []
    split_arr = {"train": ys["train"], "val": ys["val"], "testA": ys["testA"]}
    split_x = {s: {r: xs[(r, s)] for r in ("R1", "R2")} for s in ("train", "val", "testA")}
    y_tensors = {s: torch.from_numpy(y).long().to(DEVICE) for s, y in split_arr.items()}
    for seed in SEEDS:
        weights = {r: npz[f"{r}_seed{seed}_weight"].astype(np.float64) for r in ("R1", "R2")}
        biases = {r: npz[f"{r}_seed{seed}_bias"].astype(np.float64) for r in ("R1", "R2")}
        for src, dst in (("R1", "R2"), ("R2", "R1")):
            wt, bt = transfer(weights[src], biases[src], src, dst, maps)
            for split in ("train", "val", "testA"):
                xt = torch.from_numpy(split_x[split][src]).to(DEVICE)
                xd = torch.from_numpy(split_x[split][dst]).to(DEVICE)
                yy = y_tensors[split]
                with torch.no_grad():
                    log_s = xt @ torch.from_numpy(weights[src].T.astype(np.float32)).to(DEVICE) + torch.from_numpy(biases[src].astype(np.float32)).to(DEVICE)
                    log_d = xd @ torch.from_numpy(wt.T.astype(np.float32)).to(DEVICE) + torch.from_numpy(bt.astype(np.float32)).to(DEVICE)
                    known = (yy >= 0) & (yy < log_s.shape[1])
                    mismatch = float((log_s.argmax(-1)[known] != log_d.argmax(-1)[known]).float().mean().cpu())
                    maxerr = float((log_s[known] - log_d[known]).abs().max().cpu())
                    meanerr = float((log_s[known] - log_d[known]).abs().mean().cpu())
                    ce = F.cross_entropy(log_s[known], yy[known]).item()
                    acc = (log_s[known].argmax(-1) == yy[known]).float().mean().item()
                transfer_rows.append({"seed": seed, "source": src, "target": dst, "split": split,
                                      "max_logit_error": maxerr, "mean_logit_error": meanerr,
                                      "prediction_mismatch_rate": mismatch, "source_solution_ce": ce,
                                      "source_solution_accuracy": acc, "known_rows": int(known.sum().item())})
        for rep in ("R1", "R2"):
            W = weights[rep]
            b = biases[rep]
            for split in ("train", "val", "testA"):
                x = split_x[split][rep]
                yy = split_arr[split]
                xx = torch.from_numpy(x).to(DEVICE)
                yt = y_tensors[split]
                with torch.no_grad():
                    logits = xx @ torch.from_numpy(W.T.astype(np.float32)).to(DEVICE) + torch.from_numpy(b.astype(np.float32)).to(DEVICE)
                    known = (yt >= 0) & (yt < logits.shape[1])
                    ce = F.cross_entropy(logits[known], yt[known]).item()
                    pred = logits[known].argmax(-1)
                    acc = (pred == yt[known]).float().mean().item()
                    top2 = logits[known].topk(2, dim=-1).values
                    margins = top2[:, 0] - top2[:, 1]
                final_rows.append({"seed": seed, "representation": rep, "split": split,
                                   "cross_entropy": ce, "accuracy": acc, "known_rows": int(known.sum().item()),
                                   "max_logit_abs": float(logits[known].abs().max().item()),
                                   "margin_mean": float(margins.mean().item()), "margin_std": float(margins.std().item())})
    npz.close()
    # Both directional transferred solution gaps against the independent target fit.
    by = {(r["seed"], r["representation"], r["split"]): r for r in final_rows}
    trby = {(r["seed"], r["source"], r["target"], r["split"]): r for r in transfer_rows}
    for seed in SEEDS:
        for src, dst in (("R1", "R2"), ("R2", "R1")):
            for split in ("train", "val", "testA"):
                independent = by[(seed, dst, split)]
                transferred = trby[(seed, src, dst, split)]
                gap_rows.append({"seed": seed, "source": src, "target": dst, "split": split,
                                 "independent_ce": independent["cross_entropy"],
                                 "transferred_ce": transferred["source_solution_ce"],
                                 "loss_gap_independent_minus_transferred": independent["cross_entropy"] - transferred["source_solution_ce"],
                                 "independent_accuracy": independent["accuracy"],
                                 "transferred_accuracy": transferred["source_solution_accuracy"],
                                 "accuracy_gap_transferred_minus_independent": transferred["source_solution_accuracy"] - independent["accuracy"]})
    write_csv(OUT / "final_r1_r2_joint_results.csv", final_rows)
    write_csv(OUT / "weight_transfer_control.csv", transfer_rows)
    write_csv(OUT / "known_good_solution_gap.csv", gap_rows)
    # Paired seed deltas for headline final standardized TestA.
    test = {(r["seed"], r["representation"]): r for r in final_rows if r["split"] == "testA"}
    delta = [test[(s, "R1")]["accuracy"] - test[(s, "R2")]["accuracy"] for s in SEEDS]
    result = {"primary_verdict": None, "selected_protocol": selected,
              "testA_joint_r1_minus_r2_accuracy_pp": aggregate([100*x for x in delta]),
              "testA_r1_accuracy": aggregate([100*test[(s,"R1")]["accuracy"] for s in SEEDS]),
              "testA_r2_accuracy": aggregate([100*test[(s,"R2")]["accuracy"] for s in SEEDS]),
              "all_weight_transfer_prediction_mismatches_zero": all(r["prediction_mismatch_rate"] == 0 for r in transfer_rows),
              "testB_joint": "NOT EVALUABLE — NO KNOWN JOINT LABELS",
              "test_metrics_used_for_protocol_selection": False}
    summaries = read_csv(OUT / "convergence_summary.csv")
    target = [r for r in summaries if r["optimizer"] == selected["optimizer"] and abs(float(r["lr"])-float(selected["lr"])) < 1e-12 and abs(float(r["weight_decay"])-float(selected["weight_decay"])) < 1e-12]
    allplateau = len(target) == 10 and all(str(r["plateau_confirmed"]).lower() == "true" for r in target)
    dmean = abs(result["testA_joint_r1_minus_r2_accuracy_pp"]["mean"])
    if not allplateau:
        verdict = "J3E-F — STANDARDIZED JOINT PROBE STILL FAILS TO REACH VERIFIED CONVERGENCE"
    elif dmean <= 1.0 and result["testA_joint_r1_minus_r2_accuracy_pp"]["ci95"][0] <= 1.0 and result["testA_joint_r1_minus_r2_accuracy_pp"]["ci95"][1] >= -1.0:
        verdict = "J3E-P — STANDARDIZED R1/R2 JOINT PROBE CONVERGENCE AND BASIS-INVARIANCE PASS"
    else:
        verdict = "J3E-F — CONVERGED JOINT PROBE REMAINS BASIS-SENSITIVE DESPITE TRANSFERABLE SOLUTIONS"
    result["primary_verdict"] = verdict
    dump_json(OUT / "final_verdict.json", result)
    (OUT / "FINAL_VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")
    report = ["# O3b-J3E final report", "", f"## Verdict\n\n**{verdict}**", "",
              "## Scope and integrity", "", "Only train-standardized R1/R2 Joint linear probes were fitted. No Entity/Relation, R0, Z-stage, nonlinear, activation, model, checkpoint, memory-write, reconnect, or O3c work was performed. Protocol selection used train/validation only. TestA was opened only after the frozen protocol file was written; TestB Joint is not evaluable because Stage 0 found zero known labels.", "",
              "## Frozen protocol", "", "```json", json.dumps(selected, ensure_ascii=False, indent=2), "```", "",
              "## Frozen TestA results", "", f"R1 TestA Joint accuracy: {result['testA_r1_accuracy']}", f"R2 TestA Joint accuracy: {result['testA_r2_accuracy']}", f"Paired R1−R2 delta (pp): {result['testA_joint_r1_minus_r2_accuracy_pp']}", "",
              "## Analytical transfer and known-good gaps", "", f"All transfer prediction mismatches zero: {result['all_weight_transfer_prediction_mismatches_zero']}", "See `weight_transfer_control.csv` and `known_good_solution_gap.csv` for per-seed/split logits, losses, accuracies, and gaps.", "",
              "## Required stage statuses", "", "Optimizer search: completed using train/validation only. Full-batch control: see `full_batch_control.csv`. Convex/LBFGS control: see `convex_solver_control.csv`. Paired initialization: see `paired_initialization_control.csv`. Warm-start: see `transfer_warmstart_control.csv`. Convergence curves and plateau statuses: see `convergence_curves.csv` and `convergence_summary.csv`.", "",
              "## Interpretation", "", "A successful analytical classifier transfer establishes equality of the affine linear hypothesis classes under the verified coordinate transform; it does not guarantee that independently initialized optimizers reach equivalent parameterizations. Any independent-fit gap is therefore interpreted first as optimization/regularization/initialization procedure sensitivity, not loss of Joint information in a representation.", "",
              "## Stop boundary", "", "Reconnect allowed: No. O3c allowed: No. No post-pass J3 calibration stages were launched.", ""]
    (OUT / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("search", "main", "controls", "final"))
    a = p.parse_args()
    if DEVICE.type == "cuda":
        print(f"DEVICE cuda name={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)
    else:
        print(f"DEVICE cpu torch={torch.__version__}", flush=True)
    if a.stage == "search":
        return search()
    if a.stage == "main":
        return main_convergence()
    if a.stage == "controls":
        return control_runs()
    return frozen_final()


if __name__ == "__main__":
    raise SystemExit(main())
