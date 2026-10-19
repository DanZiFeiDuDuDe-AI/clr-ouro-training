from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SPLITS = ("train", "val", "testA", "testB")
CAPS = (1, 2, 4)
SEED = 42
STEPS = 3000
EVAL_EVERY = 100
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 0.01
TEMP = 0.1


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status", "detail"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        if rows:
            w.writerows(rows)
        else:
            w.writerow({"status": "NOT_RUN", "detail": "no rows"})


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


class KeyEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2048, 256), nn.GELU(), nn.Linear(256, dim))

    def forward(self, x):
        return F.normalize(self.net(x.float()), dim=-1)


class QueryEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2048, 256), nn.GELU(), nn.Linear(256, dim))

    def forward(self, x):
        return F.normalize(self.net(x.float()), dim=-1)


class KeyHeads(nn.Module):
    def __init__(self, nentity, dim=64):
        super().__init__()
        self.entity = nn.Linear(dim, nentity)
        self.query_entity = nn.Linear(dim, nentity)


class Cache:
    def __init__(self, root: Path, splits):
        self.root = root
        self.data = {}
        self.meta = {}
        self.rows_by_cap = {}
        for split in splits:
            with np.load(root / f"{split}_data.npz", allow_pickle=False) as z:
                self.data[split] = {k: z[k] for k in z.files}
            self.meta[split] = [json.loads(x) for x in (root / f"{split}_meta.jsonl").read_text(encoding="utf-8").splitlines() if x]
            d = self.data[split]
            if len(self.meta[split]) != len(d["N"]):
                raise ValueError(f"{split}: metadata/episode row count mismatch")
            if not np.array_equal(d["Y"], d["fact_Y"][d["fact_offsets"][1:] - 1]):
                # Query target may be anywhere in the episode; validate the exact target below.
                for i, q in enumerate(d["query_idx"]):
                    lo, hi = int(d["fact_offsets"][i]), int(d["fact_offsets"][i + 1])
                    if int(d["fact_Y"][lo + int(q)]) != int(d["Y"][i]):
                        raise ValueError(f"{split}: query/value label mismatch at episode {i}")
            for cap in CAPS:
                self.rows_by_cap[(split, cap)] = np.flatnonzero(d["N"] == cap).astype(np.int64)

    def batch(self, split, rows):
        rows = np.asarray(rows, dtype=np.int64)
        d = self.data[split]
        offsets = d["fact_offsets"]
        fact = np.stack([d["fact_H"][int(offsets[i]):int(offsets[i + 1])] for i in rows]).astype(np.float32, copy=False)
        ent = np.stack([d["fact_entity"][int(offsets[i]):int(offsets[i + 1])] for i in rows]).astype(np.int64, copy=False)
        qh = d["query_H"][rows].astype(np.float32, copy=False)
        qidx = d["query_idx"][rows].astype(np.int64, copy=False)
        qent = d["query_entity"][rows].astype(np.int64, copy=False)
        return fact, ent, qh, qidx, qent


def batch_tensors(cache, split, rows, device):
    fact, ent, qh, qidx, qent = cache.batch(split, rows)
    return tuple(torch.as_tensor(x, device=device) for x in (fact, ent, qh, qidx, qent))


@torch.no_grad()
def eval_capacity(key, query, cache, split, cap, rows, device):
    n = correct = 0
    for start in range(0, len(rows), 128):
        rb = rows[start:start + 128]
        facts, _, qh, qidx, _ = batch_tensors(cache, split, rb, device)
        b, c, _ = facts.shape
        k = key(facts.reshape(-1, 2048)).reshape(b, c, 64)
        q = query(qh)
        pred = (q[:, None] * k).sum(-1).argmax(-1)
        correct += int((pred == qidx).sum())
        n += len(rb)
    return {"count": n, "retrieval": correct / max(n, 1)}


@torch.no_grad()
def eval_split(key, query, cache, split, device):
    out = {}
    for cap in CAPS:
        rows = cache.rows_by_cap[(split, cap)]
        out[str(cap)] = eval_capacity(key, query, cache, split, cap, rows, device)
    out["macro_retrieval"] = float(np.mean([out[str(c)]["retrieval"] for c in CAPS]))
    for kind in ("same_entity", "same_relation"):
        rows = np.asarray([i for i, m in enumerate(cache.meta[split]) if int(m["capacity"]) == 4 and m["episode_type"] == kind], dtype=np.int64)
        out[kind] = eval_capacity(key, query, cache, split, 4, rows, device)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {"experiment": "CLR-O3-REBASE-1/O21A", "seed": SEED, "device": str(device), "architecture": "2048-256-GELU-64-L2 for independent K/Q encoders", "objective": "retrieval CE + 0.5*in-batch positive alignment CE + 0.1*(fact entity CE + query entity CE)", "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0, "batch_size": BATCH_SIZE, "steps": STEPS, "eval_every": EVAL_EVERY, "selection": "best validation macro retrieval over capacities 1,2,4", "cache_root": str(args.cache_root), "test_policy": "TestA/TestB are not loaded until the VAL gate passes"}
    dump_json(args.out_root / "o21a_rebase_config.json", config)
    cache = Cache(args.cache_root, ("train", "val"))
    train, val = cache.data["train"], cache.data["val"]
    nentity = len(json.loads((args.cache_root / "train_persons.json").read_text(encoding="utf-8")))
    key, query, heads = KeyEncoder().to(device), QueryEncoder().to(device), KeyHeads(nentity).to(device)
    params = list(key.parameters()) + list(query.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED + 800)
    train_rows = {c: cache.rows_by_cap[("train", c)] for c in CAPS}
    best = -1.0
    best_step = 0
    curves = []
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        cap = int(rng.choice(CAPS))
        rr = train_rows[cap]
        rows = rr[rng.integers(0, len(rr), size=min(BATCH_SIZE, len(rr)))]
        facts, ent, qh, qidx, qent = batch_tensors(cache, "train", rows, device)
        b, c, _ = facts.shape
        k = key(facts.reshape(-1, 2048)).reshape(b, c, 64)
        q = query(qh)
        retrieval_logits = (q[:, None] * k).sum(-1) / TEMP
        loss_retrieval = F.cross_entropy(retrieval_logits, qidx)
        pos = k[torch.arange(b, device=device), qidx]
        loss_align = F.cross_entropy((q @ pos.T) / TEMP, torch.arange(b, device=device))
        loss_entity = F.cross_entropy(heads.entity(k.reshape(-1, 64)), ent.reshape(-1))
        loss_query_entity = F.cross_entropy(heads.query_entity(q), qent)
        loss = loss_retrieval + 0.5 * loss_align + 0.1 * (loss_entity + loss_query_entity)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == STEPS:
            key.eval(); query.eval(); heads.eval()
            val_metrics = eval_split(key, query, cache, "val", device)
            row = {"step": step, "train_loss": float(loss.detach()), "loss_retrieval": float(loss_retrieval.detach()), "loss_align": float(loss_align.detach()), "loss_entity": float(loss_entity.detach()), "loss_query_entity": float(loss_query_entity.detach()), "val_macro_retrieval": val_metrics["macro_retrieval"], "val_N1": val_metrics["1"]["retrieval"], "val_N2": val_metrics["2"]["retrieval"], "val_N4": val_metrics["4"]["retrieval"], "val_same_entity_N4": val_metrics["same_entity"]["retrieval"], "val_same_relation_N4": val_metrics["same_relation"]["retrieval"]}
            curves.append(row)
            print(json.dumps({"stage": "O21A_REBASE", **row}, ensure_ascii=False), flush=True)
            if val_metrics["macro_retrieval"] > best:
                best = val_metrics["macro_retrieval"]
                best_step = step
                payload = {"key": key.state_dict(), "query": query.state_dict(), "heads": heads.state_dict(), "step": step, "val": val_metrics, "config": config}
                torch.save(payload, args.out_root / "o21a_rebase.pt")
            key.train(); query.train(); heads.train()
    write_csv(args.out_root / "o21a_rebase_learning_curve.csv", curves)
    payload = torch.load(args.out_root / "o21a_rebase.pt", map_location="cpu", weights_only=False)
    key.load_state_dict(payload["key"], strict=True); query.load_state_dict(payload["query"], strict=True)
    key.eval(); query.eval(); heads.eval()
    # Reload the auxiliary heads only to preserve an exact inference bundle; not used by retrieval.
    heads.load_state_dict(payload["heads"], strict=True)
    val_final = eval_split(key, query, cache, "val", device)
    val_pass = val_final["4"]["retrieval"] >= 0.90
    result_rows = []
    for split in ("train", "val"):
        metrics = eval_split(key, query, cache, split, device)
        for cap in CAPS:
            result_rows.append({"split": split, "capacity": cap, "episode_type": "all", **metrics[str(cap)]})
        for kind in ("same_entity", "same_relation"):
            result_rows.append({"split": split, "capacity": 4, "episode_type": kind, **metrics[kind]})
    gate = {"experiment": "CLR-O3-REBASE-1/O21A", "protocol_frozen": True, "best_step": best_step, "best_val_macro": best, "val_N4_retrieval": val_final["4"]["retrieval"], "val_gate_pass": val_pass, "test_evaluation": "NOT_RUN_VAL_GATE_FAILED" if not val_pass else "PENDING_FINAL_FROZEN_EVALUATION", "gate_pass": False}
    if val_pass:
        # Test files are first opened here, after training, checkpoint selection, and the VAL gate are frozen.
        test_cache = Cache(args.cache_root, ("testA", "testB"))
        test_metrics = {}
        for split in ("testA", "testB"):
            m = eval_split(key, query, test_cache, split, device)
            test_metrics[split] = m
            for cap in CAPS:
                result_rows.append({"split": split, "capacity": cap, "episode_type": "all", **m[str(cap)]})
            for kind in ("same_entity", "same_relation"):
                result_rows.append({"split": split, "capacity": 4, "episode_type": kind, **m[kind]})
        tests_pass = (test_metrics["testA"]["4"]["retrieval"] >= 0.90 and test_metrics["testB"]["4"]["retrieval"] >= 0.85 and
                      test_metrics["testA"]["same_entity"]["retrieval"] >= 0.90 and test_metrics["testB"]["same_entity"]["retrieval"] >= 0.90 and
                      test_metrics["testA"]["same_relation"]["retrieval"] >= 0.90 and test_metrics["testB"]["same_relation"]["retrieval"] >= 0.90)
        gate.update({"test_metrics": test_metrics, "test_gate_pass": tests_pass, "gate_pass": tests_pass, "test_evaluation": "COMPLETED_FROZEN_EVALUATION"})
    write_csv(args.out_root / "o21a_rebase_results.csv", result_rows)
    dump_json(args.out_root / "o21a_rebase_gate.json", gate)
    dump_json(args.out_root / "o21a_rebase_runtime.json", {"device": str(device), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "wall_clock_s": time.perf_counter() - started, "best_step": best_step, "steps": STEPS})
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
