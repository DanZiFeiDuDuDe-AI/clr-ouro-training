from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SPLITS = ("train", "val", "testA", "testB")
PAIR_NAMES = ("P1", "P2", "P3", "P4")
TYPE_ID = {name: i for i, name in enumerate(PAIR_NAMES)}
SEEDS = (17, 29, 43, 61, 79)
TRAIN_CAP = 8000
EVAL_CAP = 2000
BATCH = 1024
MAX_EPOCHS = 40
PATIENCE = 7
MIN_EPOCHS = 12
LR = {"A": 0.03, "B": 0.001, "C": 0.03}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status,detail\nNOT_RUN,no rows\n", encoding="utf-8-sig")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_inputs(cache: Path, out: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    integrity: dict[str, Any] = {
        "experiment": "O3b-M input integrity",
        "cache_path": str(cache),
        "cache_certificate": "J3C-certified cache reused; no backbone forward performed",
        "feature_sources": {},
        "splits": {},
        "source_id_overlap": {},
        "pass": False,
    }
    certificate_file = cache / "FINAL_VERDICT.txt"
    certificate = certificate_file.read_text(encoding="utf-8", errors="replace").strip() if certificate_file.exists() else "MISSING"
    integrity["cache_certificate"] = certificate
    if "J3C-P" not in certificate:
        raise RuntimeError(f"J3C certification verdict missing or not PASS: {certificate!r}")

    all_data: dict[str, dict[str, Any]] = {}
    source_ids: dict[str, set[str]] = {}
    file_manifest: list[dict[str, Any]] = []
    for split in SPLITS:
        labels_path = cache / f"labels_{split}.csv"
        sidecar_path = cache / f"H_fresh_{split}_sidecar.csv"
        e_path = cache / f"H_mention_fresh_{split}.npy"
        r_path = cache / f"H_base_I0_{split}.npy"
        missing = [str(p) for p in (labels_path, sidecar_path, e_path, r_path) if not p.exists()]
        if missing:
            raise FileNotFoundError("missing certified cache input(s): " + ", ".join(missing))
        labels, side = read_csv_rows(labels_path), read_csv_rows(sidecar_path)
        if len(labels) != len(side):
            raise ValueError(f"{split}: labels/sidecar row count mismatch {len(labels)} != {len(side)}")
        fields = ("canonical_fact_id", "feature_row_id")
        for i, (a, b) in enumerate(zip(labels, side)):
            for field in fields:
                if a.get(field) != b.get(field):
                    raise ValueError(f"{split} row {i}: {field} mismatch")
            if not a.get("entity_semantic_value") or not a.get("relation_semantic_value"):
                raise ValueError(f"{split} row {i}: missing entity/relation semantic identity")
        e = np.load(e_path, mmap_mode="r", allow_pickle=False)
        r = np.load(r_path, mmap_mode="r", allow_pickle=False)
        if e.ndim != 2 or r.ndim != 2 or e.shape != r.shape or e.shape[0] != len(labels):
            raise ValueError(f"{split}: feature shape/row mismatch entity={e.shape}, relation={r.shape}, rows={len(labels)}")
        if not np.isfinite(e[: min(32, len(e))]).all() or not np.isfinite(r[: min(32, len(r))]).all():
            raise FloatingPointError(f"{split}: non-finite feature sample")
        ids = {x["canonical_fact_id"] for x in labels}
        feature_ids = {x["feature_row_id"] for x in labels}
        record_hashes = {x.get("stable_source_record_sha256", "") for x in side if x.get("stable_source_record_sha256")}
        source_ids[split] = ids | {"FEATURE:" + x for x in feature_ids} | {"RECORD:" + x for x in record_hashes}
        all_data[split] = {"labels": labels, "sidecar": side, "entity": e, "relation": r}
        integrity["splits"][split] = {
            "rows": len(labels),
            "feature_shape": list(e.shape),
            "entity_dtype": str(e.dtype),
            "relation_dtype": str(r.dtype),
            "canonical_fact_id_unique": len(ids) == len(labels),
            "feature_row_id_unique": len(feature_ids) == len(labels),
            "entity_count": len({x["entity_semantic_value"] for x in labels}),
            "relation_count": len({x["relation_semantic_value"] for x in labels}),
            "repeat_key_count": sum(v > 1 for v in Counter((x["entity_semantic_value"], x["relation_semantic_value"]) for x in labels).values()),
        }
        for p in (labels_path, sidecar_path, e_path, r_path):
            file_manifest.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)})
    overlap = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            n = len(source_ids[a] & source_ids[b])
            overlap[f"{a}-{b}"] = n
            if n:
                raise ValueError(f"source-level leakage: {a}/{b} have {n} overlapping row/feature/source IDs")
    integrity["source_id_overlap"] = overlap
    integrity["feature_sources"] = {
        "entity": {"file_pattern": "H_mention_fresh_{split}.npy", "semantic_source": "fresh exact entity-mention mean hidden state", "dimension": int(all_data["train"]["entity"].shape[1]), "normalized_in_cache": False, "frozen": True},
        "relation": {"file_pattern": "H_base_I0_{split}.npy", "semantic_source": "certified MR fact_H addressed by canonical source fact row", "dimension": int(all_data["train"]["relation"].shape[1]), "normalized_in_cache": False, "frozen": True},
    }
    integrity["files"] = file_manifest
    integrity["pass"] = True
    write_json(out / "input_integrity.json", integrity)
    return all_data, integrity


def pair_type(e1: str, r1: str, e2: str, r2: str) -> int:
    if e1 == e2 and r1 == r2:
        return 0
    if e1 == e2:
        return 1
    if r1 == r2:
        return 2
    return 3


def all_p1_p2(labels: list[dict[str, str]]) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_entity: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(labels):
        e, r = row["entity_semantic_value"], row["relation_semantic_value"]
        by_key[(e, r)].append(i)
        by_entity[e][r].append(i)
    p1_changed: list[tuple[int, int, int]] = []
    p1_same_value: list[tuple[int, int, int]] = []
    values = [row.get("object_value", "") for row in labels]
    for group in by_key.values():
        for i, j in itertools.combinations(group, 2):
            target = p1_changed if values[i] != values[j] else p1_same_value
            target.append((i, j, 0))
    p2: list[tuple[int, int, int]] = []
    for relations in by_entity.values():
        for rel1, rel2 in itertools.combinations(sorted(relations), 2):
            for i in relations[rel1]:
                for j in relations[rel2]:
                    p2.append((min(i, j), max(i, j), 1))
    return p1_changed + p1_same_value, p2


def random_category_pairs(labels: list[dict[str, str]], type_id: int, n: int, rng: np.random.Generator) -> list[tuple[int, int, int]]:
    pairs: set[tuple[int, int]] = set()
    total = len(labels)
    attempts = 0
    limit = max(10000, n * 100)
    while len(pairs) < n and attempts < limit:
        attempts += 1
        i, j = rng.integers(0, total, size=2).tolist()
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if (i, j) in pairs:
            continue
        a, b = labels[i], labels[j]
        if pair_type(a["entity_semantic_value"], a["relation_semantic_value"], b["entity_semantic_value"], b["relation_semantic_value"]) == type_id:
            pairs.add((i, j))
    if len(pairs) < n:
        raise RuntimeError(f"could only sample {len(pairs)}/{n} of P{type_id+1} after {attempts} attempts")
    return [(i, j, type_id) for i, j in sorted(pairs)]


def make_pairs(split: str, labels: list[dict[str, str]], cap: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    p1, p2 = all_p1_p2(labels)
    available = {0: len(p1), 1: len(p2)}
    n = min(cap, available[0], available[1])
    if n < 1:
        raise RuntimeError(f"{split}: insufficient P1/P2 source-separated pairs")
    p1_changed = [(i, j, t) for i, j, t in p1 if labels[i].get("object_value", "") != labels[j].get("object_value", "")]
    rng.shuffle(p1_changed)
    p1_same = [(i, j, t) for i, j, t in p1 if labels[i].get("object_value", "") == labels[j].get("object_value", "")]
    rng.shuffle(p1_same)
    selected_p1 = (p1_changed[:n] + p1_same[: max(0, n - len(p1_changed[:n]))])[:n]
    rng.shuffle(p2)
    selected_p2 = p2[:n]
    selected_p3 = random_category_pairs(labels, 2, n, rng)
    selected_p4 = random_category_pairs(labels, 3, n, rng)
    pairs = np.asarray(selected_p1 + selected_p2 + selected_p3 + selected_p4, dtype=np.int64)
    rng.shuffle(pairs)
    info = {
        "split": split,
        "source_rows": len(labels),
        "sampled_pairs_per_type": n,
        "sampled_total": int(len(pairs)),
        "p1_available": available[0],
        "p1_different_object_value_available": len(p1_changed),
        "p2_available": available[1],
        "p3_sampled": n,
        "p4_sampled": n,
        "pair_types_balanced": True,
        "pair_endpoint_ids_unique_to_this_source_split": True,
    }
    return pairs, info


class FactorizedCosine(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale_e = nn.Parameter(torch.tensor(math.log(8.0)))
        self.bias_e = nn.Parameter(torch.tensor(0.0))
        self.log_scale_r = nn.Parameter(torch.tensor(math.log(8.0)))
        self.bias_r = nn.Parameter(torch.tensor(0.0))

    def from_cosines(self, se: torch.Tensor, sr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        le = self.log_scale_e.clamp(-2.0, 4.0).exp() * se + self.bias_e
        lr = self.log_scale_r.clamp(-2.0, 4.0).exp() * sr + self.bias_r
        return le + lr, le, lr

    def forward(self, eq: torch.Tensor, em: torch.Tensor, rq: torch.Tensor, rm: torch.Tensor):
        se = F.cosine_similarity(eq, em, dim=-1, eps=1e-8)
        sr = F.cosine_similarity(rq, rm, dim=-1, eps=1e-8)
        return self.from_cosines(se, sr)


class FactorizedPairMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.entity_key = nn.Linear(dim, 64)
        self.relation_key = nn.Linear(dim, 64)
        self.pair_head = nn.Sequential(nn.Linear(64 * 4, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, e: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.entity_key(e), dim=-1, eps=1e-8), F.normalize(self.relation_key(r), dim=-1, eps=1e-8)

    def score_embeddings(self, eq: torch.Tensor, em: torch.Tensor, rq: torch.Tensor, rm: torch.Tensor) -> torch.Tensor:
        symmetric = torch.cat(((eq - em).abs(), eq * em, (rq - rm).abs(), rq * rm), dim=-1)
        return self.pair_head(symmetric).squeeze(-1)

    def forward(self, eq: torch.Tensor, em: torch.Tensor, rq: torch.Tensor, rm: torch.Tensor) -> torch.Tensor:
        keq, krq = self.encode(eq, rq)
        kem, krm = self.encode(em, rm)
        return self.score_embeddings(keq, kem, krq, krm)


def torch_rows(array: np.ndarray, idx: np.ndarray, device: str, mu: torch.Tensor | None = None, sd: torch.Tensor | None = None) -> torch.Tensor:
    x = torch.from_numpy(np.array(array[idx], dtype=np.float32, copy=True)).to(device)
    if mu is not None and sd is not None:
        x = (x - mu) / sd
    return x


def metric_pack(types: np.ndarray, scores: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = (types == 0).astype(np.int64)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    out: dict[str, float] = {
        "accuracy": (tp + tn) / max(1, len(types)),
        "same_key_f1": f1,
        "precision": precision,
        "update_recall": recall,
        "new_false_update_rate": fp / max(1, fp + tn),
    }
    for type_id, name in enumerate(PAIR_NAMES):
        mask = types == type_id
        out[f"{name.lower()}_accuracy"] = float((pred[mask] == (type_id == 0)).mean()) if mask.any() else float("nan")
    out["p2_false_update_rate"] = float(pred[types == 1].mean()) if (types == 1).any() else float("nan")
    out["operational_score"] = float(np.mean([out["same_key_f1"], out["update_recall"], out["p2_accuracy"], 1.0 - out["new_false_update_rate"]]))
    return out


def threshold_search(name: str, probs: np.ndarray, types: np.ndarray, pe: np.ndarray | None = None, pr: np.ndarray | None = None) -> tuple[dict[str, float], dict[str, float]]:
    best: tuple[float, float, float] | None = None
    best_metrics: dict[str, float] = {}
    best_thresholds: dict[str, float] = {}
    grid = np.linspace(0.05, 0.95, 91)
    if name != "C":
        for t in grid:
            pred = (probs >= t).astype(np.int64)
            m = metric_pack(types, probs, pred)
            key = (m["operational_score"], m["same_key_f1"], m["p2_accuracy"])
            if best is None or key > best:
                best, best_metrics, best_thresholds = key, m, {"threshold": float(t)}
    else:
        assert pe is not None and pr is not None
        for te in grid[::2]:
            p_e = pe >= te
            for tr in grid[::2]:
                pred = (p_e & (pr >= tr)).astype(np.int64)
                m = metric_pack(types, pe * pr, pred)
                key = (m["operational_score"], m["same_key_f1"], m["p2_accuracy"])
                if best is None or key > best:
                    best, best_metrics, best_thresholds = key, m, {"entity_threshold": float(te), "relation_threshold": float(tr)}
    return best_metrics, best_thresholds


def standardization(train_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(train_array, dtype=np.float64)
    mu = x.mean(axis=0).astype(np.float32)
    sd = x.std(axis=0).astype(np.float32)
    sd[sd < 1e-5] = 1.0
    return mu, sd


def make_model(name: str, dim: int) -> nn.Module:
    if name in ("A", "C"):
        return FactorizedCosine()
    if name == "B":
        return FactorizedPairMLP(dim)
    raise ValueError(name)


def outputs_for_pairs(name: str, model: nn.Module, e: torch.Tensor, r: torch.Tensor, pairs: np.ndarray, batch: int = 2048) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    all_joint: list[np.ndarray] = []
    all_e: list[np.ndarray] = []
    all_r: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch):
            p = pairs[start : start + batch]
            i = torch.as_tensor(p[:, 0], dtype=torch.long, device=e.device)
            j = torch.as_tensor(p[:, 1], dtype=torch.long, device=e.device)
            if name == "B":
                assert isinstance(model, FactorizedPairMLP)
                ki_e, ki_r = model.encode(e[i], r[i])
                kj_e, kj_r = model.encode(e[j], r[j])
                logits = model.score_embeddings(ki_e, kj_e, ki_r, kj_r)
                all_joint.append(logits.sigmoid().cpu().numpy())
            else:
                assert isinstance(model, FactorizedCosine)
                joint, le, lr = model(e[i], e[j], r[i], r[j])
                all_joint.append(joint.sigmoid().cpu().numpy())
                all_e.append(le.sigmoid().cpu().numpy())
                all_r.append(lr.sigmoid().cpu().numpy())
    return np.concatenate(all_joint), np.concatenate(all_e) if all_e else None, np.concatenate(all_r) if all_r else None


def train_one(name: str, seed: int, data: dict[str, dict[str, Any]], pairs: dict[str, np.ndarray], e_mu_np: np.ndarray, e_sd_np: np.ndarray, r_mu_np: np.ndarray, r_sd_np: np.ndarray, out: Path) -> dict[str, Any]:
    seed_all(seed)
    dev = DEVICE
    model = make_model(name, int(data["train"]["entity"].shape[1])).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=1e-4)
    train_e = torch.from_numpy(np.array(data["train"]["entity"], dtype=np.float32, copy=True)).to(dev)
    train_r = torch.from_numpy(np.array(data["train"]["relation"], dtype=np.float32, copy=True)).to(dev)
    val_e = torch.from_numpy(np.array(data["val"]["entity"], dtype=np.float32, copy=True)).to(dev)
    val_r = torch.from_numpy(np.array(data["val"]["relation"], dtype=np.float32, copy=True)).to(dev)
    if name == "B":
        emu, esd = torch.as_tensor(e_mu_np, device=dev), torch.as_tensor(e_sd_np, device=dev)
        rmu, rsd = torch.as_tensor(r_mu_np, device=dev), torch.as_tensor(r_sd_np, device=dev)
        train_e, train_r = (train_e - emu) / esd, (train_r - rmu) / rsd
        val_e, val_r = (val_e - emu) / esd, (val_r - rmu) / rsd
    train_p = pairs["train"]
    val_p = pairs["val"]
    rng = np.random.default_rng(seed + 901)
    best_score, best_epoch, stale = -1.0, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    best_thresholds: dict[str, float] = {}
    best_val_metrics: dict[str, float] = {}
    curves: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_p))
        loss_sum, seen = 0.0, 0
        for start in range(0, len(order), BATCH):
            batch_ids = order[start : start + BATCH]
            bp = train_p[batch_ids]
            ii = torch.as_tensor(bp[:, 0], dtype=torch.long, device=dev)
            jj = torch.as_tensor(bp[:, 1], dtype=torch.long, device=dev)
            typ = torch.as_tensor(bp[:, 2], dtype=torch.long, device=dev)
            if name == "B":
                logits = model(train_e[ii], train_e[jj], train_r[ii], train_r[jj])
                loss = F.binary_cross_entropy_with_logits(logits, (typ == 0).float())
            else:
                joint, le, lr = model(train_e[ii], train_e[jj], train_r[ii], train_r[jj])
                y = (typ == 0).float()
                ye = ((typ == 0) | (typ == 1)).float()
                yr = ((typ == 0) | (typ == 2)).float()
                if name == "A":
                    loss = F.binary_cross_entropy_with_logits(joint, y) + 0.25 * (F.binary_cross_entropy_with_logits(le, ye) + F.binary_cross_entropy_with_logits(lr, yr))
                else:
                    loss = F.binary_cross_entropy_with_logits(le, ye) + F.binary_cross_entropy_with_logits(lr, yr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(bp)
            seen += len(bp)

        probs, pe, pr = outputs_for_pairs(name, model, val_e, val_r, val_p)
        vm, thresholds = threshold_search(name, probs, val_p[:, 2], pe, pr)
        curves.append({"candidate": name, "seed": seed, "epoch": epoch, "train_loss": loss_sum / max(1, seen), "val_operational_score": vm["operational_score"], "val_f1": vm["same_key_f1"], "val_update_recall": vm["update_recall"], "val_p2_accuracy": vm["p2_accuracy"], **thresholds})
        if vm["operational_score"] > best_score + 1e-8:
            best_score, best_epoch, stale = vm["operational_score"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thresholds, best_val_metrics = thresholds, vm
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"{name}/{seed}: no validation checkpoint")
    model.load_state_dict(best_state)
    model_path = out / "models" / f"candidate_{name.lower()}_seed_{seed}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"candidate": name, "seed": seed, "state_dict": best_state, "thresholds": best_thresholds, "best_epoch": best_epoch, "best_val_metrics": best_val_metrics, "input_dim": int(data["train"]["entity"].shape[1]), "normalization": "train-only per dimension" if name == "B" else "L2 cosine within scorer", "feature_stats": {"entity_mean": e_mu_np, "entity_std": e_sd_np, "relation_mean": r_mu_np, "relation_std": r_sd_np} if name == "B" else None}, model_path)
    write_csv(out / "logs" / f"candidate_{name.lower()}_seed_{seed}_curve.csv", curves)
    return {"candidate": name, "seed": seed, "model_path": str(model_path), "best_epoch": best_epoch, "best_val_score": best_score, "best_val_metrics": best_val_metrics, "thresholds": best_thresholds, "epochs_run": len(curves), "wall_seconds": time.perf_counter() - t0}


def load_model(checkpoint: Path, name: str, dim: int, device: str) -> tuple[nn.Module, dict[str, Any]]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model(name, dim)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device).eval()
    return model, ck


def eval_saved(name: str, seed: int, checkpoint: Path, split: str, data: dict[str, dict[str, Any]], pairs: np.ndarray, out: Path, do_sweep: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    model, ck = load_model(checkpoint, name, int(data[split]["entity"].shape[1]), DEVICE)
    e = torch.from_numpy(np.array(data[split]["entity"], dtype=np.float32, copy=True)).to(DEVICE)
    r = torch.from_numpy(np.array(data[split]["relation"], dtype=np.float32, copy=True)).to(DEVICE)
    if name == "B":
        stats = ck["feature_stats"]
        e = (e - torch.as_tensor(stats["entity_mean"], device=DEVICE)) / torch.as_tensor(stats["entity_std"], device=DEVICE)
        r = (r - torch.as_tensor(stats["relation_mean"], device=DEVICE)) / torch.as_tensor(stats["relation_std"], device=DEVICE)
    probs, pe, pr = outputs_for_pairs(name, model, e, r, pairs)
    types = pairs[:, 2]
    th = ck["thresholds"]
    if name == "C":
        pred = ((pe >= th["entity_threshold"]) & (pr >= th["relation_threshold"])).astype(np.int64)
    else:
        pred = (probs >= th["threshold"]).astype(np.int64)
    metrics = metric_pack(types, probs, pred)
    row = {"candidate": name, "seed": seed, "split": split, **th, **metrics}
    sweep_rows: list[dict[str, Any]] = []
    if do_sweep:
        grid = (0.3, 0.4, 0.5, 0.6, 0.7)
        if name == "C":
            for te, tr in itertools.product(grid, repeat=2):
                pm = ((pe >= te) & (pr >= tr)).astype(np.int64)
                sweep_rows.append({"candidate": name, "seed": seed, "split": split, "entity_threshold": te, "relation_threshold": tr, **metric_pack(types, pe * pr, pm)})
        else:
            for t in grid:
                pm = (probs >= t).astype(np.int64)
                sweep_rows.append({"candidate": name, "seed": seed, "split": split, "threshold": t, **metric_pack(types, probs, pm)})
    return row, {"probs": probs, "entity_probs": pe, "relation_probs": pr, "sweep": sweep_rows}


def prepare_action_features(name: str, model: nn.Module, ck: dict[str, Any], split_data: dict[str, Any]) -> dict[str, torch.Tensor]:
    e = torch.from_numpy(np.array(split_data["entity"], dtype=np.float32, copy=True)).to(DEVICE)
    r = torch.from_numpy(np.array(split_data["relation"], dtype=np.float32, copy=True)).to(DEVICE)
    with torch.no_grad():
        if name in ("A", "C"):
            return {"e": F.normalize(e, dim=-1, eps=1e-8), "r": F.normalize(r, dim=-1, eps=1e-8)}
        stats = ck["feature_stats"]
        emu = torch.as_tensor(stats["entity_mean"], device=DEVICE)
        esd = torch.as_tensor(stats["entity_std"], device=DEVICE)
        rmu = torch.as_tensor(stats["relation_mean"], device=DEVICE)
        rsd = torch.as_tensor(stats["relation_std"], device=DEVICE)
        keys_e, keys_r = [], []
        model.eval()
        for start in range(0, len(e), 512):
            ke, kr = model.encode((e[start : start + 512] - emu) / esd, (r[start : start + 512] - rmu) / rsd)
            keys_e.append(ke)
            keys_r.append(kr)
        return {"e": torch.cat(keys_e), "r": torch.cat(keys_r)}


def action_scores(name: str, model: nn.Module, feature: dict[str, torch.Tensor], q: int, slots: list[int]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    idx = torch.as_tensor(slots, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        if name == "B":
            q_e = feature["e"][q].expand(len(slots), -1)
            q_r = feature["r"][q].expand(len(slots), -1)
            logits = model.score_embeddings(q_e, feature["e"][idx], q_r, feature["r"][idx])
            return logits.sigmoid().cpu().numpy(), None, None
        assert isinstance(model, FactorizedCosine)
        se = feature["e"][idx] @ feature["e"][q]
        sr = feature["r"][idx] @ feature["r"][q]
        joint, le, lr = model.from_cosines(se, sr)
        return joint.sigmoid().cpu().numpy(), le.sigmoid().cpu().numpy(), lr.sigmoid().cpu().numpy()


def simulate_actions(name: str, seed: int, ck: dict[str, Any], model: nn.Module, split: str, data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    d = data[split]
    features = prepare_action_features(name, model, ck, d)
    rows, labels = d["sidecar"], d["labels"]
    order = sorted(range(len(rows)), key=lambda i: (int(rows[i].get("episode_index") or 0), int(rows[i].get("item_index") or 0), int(rows[i].get("fact_index") or i)))
    slot_rows: list[int] = []
    slot_keys: list[tuple[str, str]] = []
    counts = Counter()
    target_total = 0
    for q in order:
        lab = labels[q]
        key = (lab["entity_semantic_value"], lab["relation_semantic_value"])
        existing = [i for i, k in enumerate(slot_keys) if k == key]
        true_update = bool(existing)
        if not slot_rows:
            slot_rows.append(q)
            slot_keys.append(key)
            counts["true_new"] += 1
            counts["pred_new_correct"] += 1
            continue
        scores, pe, pr = action_scores(name, model, features, q, slot_rows)
        ck_th = ck["thresholds"]
        if name == "C":
            valid = (pe >= ck_th["entity_threshold"]) & (pr >= ck_th["relation_threshold"])
            rank_score = pe * pr
        else:
            valid = scores >= ck_th["threshold"]
            rank_score = scores
        best_i = int(np.argmax(rank_score))
        predict_update = bool(valid[best_i])
        if true_update:
            counts["true_update"] += 1
        else:
            counts["true_new"] += 1
            p2_case = any(k[0] == key[0] and k[1] != key[1] for k in slot_keys)
            p3_case = any(k[1] == key[1] and k[0] != key[0] for k in slot_keys)
            if p2_case:
                counts["p2_new_cases"] += 1
            if p3_case:
                counts["p3_new_cases"] += 1
        if predict_update:
            if true_update:
                counts["pred_update_action_correct"] += 1
                target_total += 1
                if slot_keys[best_i] == key:
                    counts["correct_update_target"] += 1
                else:
                    counts["wrong_update_target"] += 1
            else:
                counts["false_update"] += 1
                if slot_keys[best_i][0] == key[0] and slot_keys[best_i][1] != key[1]:
                    counts["p2_wrong_update"] += 1
                if slot_keys[best_i][1] == key[1] and slot_keys[best_i][0] != key[0]:
                    counts["p3_wrong_update"] += 1
        else:
            if true_update:
                counts["duplicate_key_creation"] += 1
                slot_rows.append(q)
                slot_keys.append(key)
            else:
                counts["pred_new_correct"] += 1
                slot_rows.append(q)
                slot_keys.append(key)
    true_updates = counts["true_update"]
    true_news = counts["true_new"]
    out = {
        "candidate": name,
        "seed": seed,
        "split": split,
        "facts_processed": len(order),
        "action_accuracy": (counts["pred_update_action_correct"] + counts["pred_new_correct"]) / max(1, len(order)),
        "update_target_slot_recall": counts["correct_update_target"] / max(1, true_updates),
        "new_collision_rate": counts["false_update"] / max(1, true_news),
        "duplicate_same_key_creation_rate": counts["duplicate_key_creation"] / max(1, true_updates),
        "p2_wrong_update_rate": counts["p2_wrong_update"] / max(1, counts["p2_new_cases"]),
        "wrong_update_target_rate_on_predicted_update": counts["wrong_update_target"] / max(1, target_total),
        "unique_slots_at_end": len(slot_rows),
        "counts": dict(counts),
    }
    return out


def hash_outputs(out: Path) -> list[dict[str, Any]]:
    result = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "artifact_sha256_manifest.json":
            result.append({"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    write_json(out / "artifact_sha256_manifest.json", {"root": str(out), "files": result})
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cache, out = args.cache, args.out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    logs_dir = out / "logs"
    logs_dir.mkdir(exist_ok=True)
    print(f"O3b-M runner starting; device={DEVICE}; cache={cache}; out={out}", flush=True)
    data, integrity = load_inputs(cache, out)

    # Fit preprocessing only on unique TRAIN source facts; validation/test are never consulted.
    e_mu, e_sd = standardization(data["train"]["entity"])
    r_mu, r_sd = standardization(data["train"]["relation"])
    write_json(out / "feature_standardization.json", {"fit_split": "train only", "entity_mean": e_mu.tolist(), "entity_std": e_sd.tolist(), "relation_mean": r_mu.tolist(), "relation_std": r_sd.tolist()})

    pairs: dict[str, np.ndarray] = {}
    pair_manifest: dict[str, Any] = {"source_id_overlap": integrity["source_id_overlap"], "pair_construction": {}, "split_pair_counts": {}}
    for i, split in enumerate(SPLITS):
        cap = TRAIN_CAP if split == "train" else EVAL_CAP
        pairs[split], info = make_pairs(split, data[split]["labels"], cap, 2409 + i)
        pair_manifest["pair_construction"][split] = info
        pair_manifest["split_pair_counts"][split] = {name: int((pairs[split][:, 2] == tid).sum()) for tid, name in enumerate(PAIR_NAMES)}
        pair_rows = []
        for left, right, typ in pairs[split]:
            a, b = data[split]["labels"][int(left)], data[split]["labels"][int(right)]
            pair_rows.append({"split": split, "pair_type": PAIR_NAMES[int(typ)], "same_key": int(typ == 0), "same_entity": int(a["entity_semantic_value"] == b["entity_semantic_value"]), "same_relation": int(a["relation_semantic_value"] == b["relation_semantic_value"]), "left_fact_id": a["canonical_fact_id"], "right_fact_id": b["canonical_fact_id"], "left_feature_row_id": a["feature_row_id"], "right_feature_row_id": b["feature_row_id"], "left_object_value": a.get("object_value", ""), "right_object_value": b.get("object_value", "")})
        write_csv(out / f"pairs_{split}.csv", pair_rows)
    write_json(out / "pair_dataset_manifest.json", pair_manifest)

    # The pair construction is frozen now; all model/threshold selection below uses TRAIN+VAL only.
    initial_rows: list[dict[str, Any]] = []
    for name in ("A", "B", "C"):
        for seed in SEEDS:
            rec = train_one(name, seed, data, pairs, e_mu, e_sd, r_mu, r_sd, out)
            initial_rows.append(rec)
            print(f"trained candidate={name} seed={seed} best_epoch={rec['best_epoch']} val_score={rec['best_val_score']:.4f}", flush=True)
    write_json(out / "training_summary.json", {"device": DEVICE, "seed_count": len(SEEDS), "runs": initial_rows, "wall_seconds": time.perf_counter() - t0})

    candidate_val = {}
    for name in ("A", "B", "C"):
        runs = [x for x in initial_rows if x["candidate"] == name]
        candidate_val[name] = float(np.mean([x["best_val_score"] for x in runs]))
    winner_name = max(("A", "B", "C"), key=lambda x: (candidate_val[x], x == "A"))
    winner_runs = sorted([x for x in initial_rows if x["candidate"] == winner_name], key=lambda x: x["best_val_score"])
    winner_run = winner_runs[len(winner_runs) // 2]  # pre-registered median validation-score seed; no ensemble
    winner_checkpoint = Path(winner_run["model_path"])
    winner_config = {
        "selected_by": "TRAIN/VAL only",
        "candidate_val_mean_operational_score": candidate_val,
        "winner": winner_name,
        "deployed_seed": winner_run["seed"],
        "seed_rule": "median validation operational score among the winner candidate's five seeds; no ensemble",
        "checkpoint": str(winner_checkpoint),
        "validation_thresholds": winner_run["thresholds"],
        "validation_metrics": winner_run["best_val_metrics"],
        "training_protocol": {"optimizer": "AdamW", "learning_rate": LR[winner_name], "weight_decay": 0.0001, "batch_size": BATCH, "max_epochs": MAX_EPOCHS, "early_stop_patience": PATIENCE},
    }
    write_json(out / "winner_config.json", winner_config)

    # First and only access to TestA/TestB scores: all architectures/checkpoints/thresholds are frozen.
    test_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    selected_artifacts: dict[str, tuple[nn.Module, dict[str, Any]]] = {}
    for name in ("A", "B", "C"):
        for run in [x for x in initial_rows if x["candidate"] == name]:
            model, ck = load_model(Path(run["model_path"]), name, int(data["train"]["entity"].shape[1]), DEVICE)
            if name == winner_name and run["seed"] == winner_run["seed"]:
                selected_artifacts[name] = (model, ck)
            for split in ("testA", "testB"):
                row, extra = eval_saved(name, int(run["seed"]), Path(run["model_path"]), split, data, pairs[split], out, do_sweep=True)
                test_rows.append(row)
                sweep_rows.extend(extra["sweep"])
    for name in ("A", "B", "C"):
        write_csv(out / f"candidate_{name.lower()}_results.csv", [r for r in test_rows if r["candidate"] == name])
    write_csv(out / "threshold_sweep.csv", sweep_rows)
    comparison_rows = []
    for name in ("A", "B", "C"):
        vruns = [x for x in initial_rows if x["candidate"] == name]
        row: dict[str, Any] = {"candidate": name, "val_operational_score_mean": candidate_val[name], "val_operational_score_std": float(np.std([x["best_val_score"] for x in vruns], ddof=1)), "val_f1_mean": float(np.mean([x["best_val_metrics"]["same_key_f1"] for x in vruns]))}
        for split in ("testA", "testB"):
            subset = [x for x in test_rows if x["candidate"] == name and x["split"] == split]
            for metric in ("same_key_f1", "precision", "update_recall", "p1_accuracy", "p2_accuracy", "p3_accuracy", "p4_accuracy", "new_false_update_rate"):
                vals = [x[metric] for x in subset]
                row[f"{split}_{metric}_mean"] = float(np.mean(vals))
                row[f"{split}_{metric}_std"] = float(np.std(vals, ddof=1))
        comparison_rows.append(row)
    write_csv(out / "candidate_comparison.csv", comparison_rows)

    selected_test = {(r["split"]): r for r in test_rows if r["candidate"] == winner_name and r["seed"] == winner_run["seed"]}
    required = {
        "testA_same_key_f1": selected_test["testA"]["same_key_f1"] >= 0.95,
        "testB_same_key_f1": selected_test["testB"]["same_key_f1"] >= 0.95,
        "testA_update_recall": selected_test["testA"]["update_recall"] >= 0.95,
        "testB_update_recall": selected_test["testB"]["update_recall"] >= 0.95,
        "testA_p2_accuracy": selected_test["testA"]["p2_accuracy"] >= 0.95,
        "testB_p2_accuracy": selected_test["testB"]["p2_accuracy"] >= 0.90,
        "testA_new_false_update": selected_test["testA"]["new_false_update_rate"] <= 0.05,
        "testB_new_false_update": selected_test["testB"]["new_false_update_rate"] <= 0.05,
    }
    offline_pass = all(required.values())
    offline_gate = {"winner": winner_name, "deployed_seed": winner_run["seed"], "selection_split": "validation only", "thresholds": winner_run["thresholds"], "testA": selected_test["testA"], "testB": selected_test["testB"], "conditions": required, "pass": offline_pass, "testB_joint_label": "NOT_USED; same-key uses preserved semantic entity/relation identity, not closed-world joint classes"}
    write_json(out / "offline_behavior_gate.json", offline_gate)

    action_results: list[dict[str, Any]] = []
    action_pass = False
    if offline_pass:
        model, ck = selected_artifacts[winner_name]
        for split in ("testA", "testB"):
            action_results.append(simulate_actions(winner_name, int(winner_run["seed"]), ck, model, split, data))
        write_csv(out / "action_simulation_results.csv", [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v) for k, v in row.items()} for row in action_results])
        action_conditions = {
            f"{x['split']}_update_target_slot_recall": x["update_target_slot_recall"] >= 0.95 for x in action_results
        }
        for x in action_results:
            action_conditions[f"{x['split']}_new_collision_rate"] = x["new_collision_rate"] <= 0.05
            action_conditions[f"{x['split']}_duplicate_same_key_creation_rate"] = x["duplicate_same_key_creation_rate"] <= 0.05
            action_conditions[f"{x['split']}_p2_wrong_update_rate"] = x["p2_wrong_update_rate"] <= 0.05
        action_pass = all(action_conditions.values())
        write_json(out / "action_gate.json", {"winner": winner_name, "conditions": action_conditions, "results": action_results, "pass": action_pass})
    else:
        write_csv(out / "action_simulation_results.csv", [{"status": "NOT_RUN", "reason": "No validation-selected candidate passed the frozen offline TestA/TestB behavior gate."}])
        write_json(out / "action_gate.json", {"status": "NOT_RUN", "reason": "Offline behavior gate failed; reconnect/action gate is locked.", "pass": False})

    reconnect_status = "NOT_RUN: real O3b reconnect is locked unless both offline behavior and action-simulation gates PASS."
    if offline_pass and action_pass:
        reconnect_status = "AUTHORIZED_NOT_EXECUTED: both behavior gates passed; next run requires a separate scoped integration/evaluation stage. O3c remains prohibited."
    write_csv(out / "reconnect_results.csv", [{"status": reconnect_status}])
    write_csv(out / "regression_check.csv", [{"status": "NOT_RUN: no real reconnect was executed."}])

    # Additional required files are explicit and machine-readable even when their stage is locked.
    final_verdict = "O3b-M-P — OPERATIONAL SAME-KEY ADDRESSING PASS" if offline_pass and action_pass else ("O3b-M-F — SAME-KEY CLASSIFICATION PASSES BUT ACTION-LEVEL ADDRESSING FAILS" if offline_pass else "O3b-M-F — NO MINIMAL ADDRESS HEAD PASSES SAME-KEY BEHAVIOR GATE")
    report = [
        "# O3b-M — Operational Same-Key Addressing Rescue", "", f"**{final_verdict}**", "",
        "## Scope and data", "", "J/K/L/J3 attribution is archived as unresolved and was not resumed. The J3C provenance-certified frozen cache was reused; no backbone forward, model/checkpoint edit, read-path change, value-path change, or O3c run occurred.",
        "", f"Entity-side input: `{integrity['feature_sources']['entity']['file_pattern']}` ({integrity['feature_sources']['entity']['dimension']}D, frozen). Relation-side input: `{integrity['feature_sources']['relation']['file_pattern']}` ({integrity['feature_sources']['relation']['dimension']}D, frozen). Pair datasets were built within each fixed source split; canonical fact, feature-row, and stable source-record IDs had zero cross-split overlap.",
        "", "## Candidate comparison", "", f"Validation selected Candidate {winner_name}, deployed seed {winner_run['seed']}, without ensembling. Validation operational scores by candidate: " + ", ".join(f"{k}={v:.4f}" for k, v in candidate_val.items()) + ". TestA/TestB rows in `candidate_comparison.csv` are evaluation-only and were not used for selection.",
        "", "## Offline behavior gate", "", "```json", json.dumps({"pass": offline_pass, "winner": winner_name, "seed": winner_run["seed"], "testA": {k: selected_test["testA"][k] for k in ("same_key_f1", "precision", "update_recall", "p1_accuracy", "p2_accuracy", "p3_accuracy", "p4_accuracy", "new_false_update_rate")}, "testB": {k: selected_test["testB"][k] for k in ("same_key_f1", "precision", "update_recall", "p1_accuracy", "p2_accuracy", "p3_accuracy", "p4_accuracy", "new_false_update_rate")}, "conditions": required}, indent=2), "```",
        "", "## Action simulation and reconnect", "", "```json", json.dumps({"action_simulation": "PASS" if action_pass else ("FAIL" if offline_pass else "NOT_RUN"), "reconnect": reconnect_status, "O3c": "NO"}, indent=2), "```",
        "", "## Guardrails", "", "- No TestA/TestB threshold tuning or candidate selection.", "- No key/projection attribution or representation-mechanism claim.", "- No modification of Ouro, KeyEncoder checkpoint, backbone, Value/Read path, memory capacity, allocator, or original O3b code.", "- Real reconnect is conditional and is not executed by this offline/action harness; if both gates pass, it is authorized as the single next scoped stage, not O3c.",
        "", "## Artifacts", "", "See `input_integrity.json`, `pair_dataset_manifest.json`, `pairs_*.csv`, `candidate_*_results.csv`, `candidate_comparison.csv`, `threshold_sweep.csv`, `offline_behavior_gate.json`, `action_simulation_results.csv`, `action_gate.json`, `reconnect_results.csv`, `regression_check.csv`, and `artifact_sha256_manifest.json`.",
    ]
    (out / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (out / "FINAL_VERDICT.txt").write_text(final_verdict + "\n", encoding="utf-8")
    write_json(out / "run_summary.json", {"verdict": final_verdict, "offline_pass": offline_pass, "action_pass": action_pass, "reconnect": reconnect_status, "o3c": "NO", "model_or_checkpoint_modified": False, "data_leakage": False, "total_wall_seconds": time.perf_counter() - t0, "device": DEVICE})
    hash_outputs(out)
    print(json.dumps({"verdict": final_verdict, "winner": winner_name, "seed": winner_run["seed"], "offline_pass": offline_pass, "action_pass": action_pass, "reconnect": reconnect_status, "wall_seconds": time.perf_counter() - t0}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
