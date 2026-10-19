from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CAPS = (1, 2, 4)
SEED = 42
STEPS = 10_000
EVAL_EVERY = 100
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
SNAPSHOT_STEPS = {0, 100, 200, 500, 1000, 1500, 2000, 2500, 3000,
                  4000, 5000, 6000, 7000, 8000, 9000, 10000}
EXPECTED_3K_CONFIG_SHA = "5be7a10274df9b1ae7689f71cc1bb3726fd03cdf605303988b06fa434b576b8b"
EXPECTED_3K_SCRIPT_SHA = "179a9473810ad62bc2e890b6620f0e383970e0c0a0ace2809bc0063c27e545be"
EXPECTED_O21A_SHA = "dadf66a59c8722ef86604aff67528f415acb18f3e72ba4ccfac415df37e90ced"


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status", "detail"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if rows:
            writer.writerows(rows)
        else:
            writer.writerow({"status": "NOT_RUN", "detail": "no rows"})


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class OracleLocalCore(nn.Module):
    def __init__(self, hidden_dim=2048, slots=4, dim=64):
        super().__init__()
        self.hidden_dim, self.slots, self.dim = hidden_dim, slots, dim
        self.content_read = nn.Linear(hidden_dim, dim)
        self.shared_gru = nn.GRUCell(dim, dim)

    def zero_state(self, batch_size, device=None, dtype=None):
        p = next(self.parameters())
        return torch.zeros(batch_size, self.slots, self.dim,
                           device=device or p.device, dtype=dtype or p.dtype)

    def write_to_slot(self, state, h, target_slot):
        z = self.content_read(h.to(device=state.device, dtype=state.dtype))
        new_state = state.clone()
        new_state[:, target_slot, :] = self.shared_gru(z, state[:, target_slot, :])
        return new_state


class MRValueHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(64, 64) for _ in range(4)])

    def logits(self, state, slot):
        return self.heads[int(slot)](state[:, int(slot), :].float())


def build_update_pairs(labels, sidecar, values):
    groups = defaultdict(list)
    for i, (lab, side) in enumerate(zip(labels, sidecar)):
        key = (lab.get("entity_semantic_value", ""), lab.get("relation_semantic_value", ""))
        obj = side.get("object_value", "")
        if key[0] and key[1] and obj:
            groups[key].append((i, obj, int(values[i])))
    pairs = []
    for rows in groups.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a[0] != b[0] and a[1] != b[1] and a[2] != b[2]:
                    pairs.append((a[0], b[0], a[2], b[2]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 4)


class SplitData:
    def __init__(self, split, mr_root: Path, j3c_root: Path):
        self.split = split
        with np.load(mr_root / f"{split}_data.npz", allow_pickle=False) as z:
            self.d = {k: z[k] for k in z.files}
        self.labels = read_csv(j3c_root / f"labels_{split}.csv")
        self.sidecar = read_csv(j3c_root / f"H_fresh_{split}_sidecar.csv")
        rel_path = j3c_root / f"H_base_I0_{split}.npy"
        mention_path = j3c_root / f"H_mention_fresh_{split}.npy"
        Hrel = np.load(rel_path, mmap_mode="r", allow_pickle=False)
        Hmention = np.load(mention_path, mmap_mode="r", allow_pickle=False)
        n = len(self.d["fact_H"])
        if self.d["fact_H"].shape != Hrel.shape or not np.array_equal(self.d["fact_H"], Hrel):
            raise ValueError(f"{split}: MR fact_H and certified relation carrier rows do not match exactly")
        if Hmention.shape != Hrel.shape or len(self.labels) != n or len(self.sidecar) != n:
            raise ValueError(f"{split}: feature/label/sidecar row count mismatch")
        if any(a.get("canonical_fact_id") != b.get("canonical_fact_id")
               for a, b in zip(self.labels, self.sidecar)):
            raise ValueError(f"{split}: canonical fact ID mismatch between labels and feature sidecar")
        ids = [x.get("canonical_fact_id", "") for x in self.labels]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{split}: canonical fact IDs are empty or not unique")
        self.fact_ids = ids
        self.index = {fid: i for i, fid in enumerate(ids)}
        self.pairs = build_update_pairs(self.labels, self.sidecar, self.d["fact_Y"])
        if len(self.pairs) == 0:
            raise ValueError(f"{split}: no source-separated same-key changed-value update pairs")
        self.rows_by_cap = {cap: np.flatnonzero(self.d["N"] == cap).astype(np.int64) for cap in CAPS}
        self.episode_ids = None
        for key in self.d:
            if key.lower() in {"episode_ids", "episode_id"}:
                self.episode_ids = np.asarray(self.d[key]).reshape(-1).astype(str).tolist()
                break
        self.files = {
            "mr_npz": mr_root / f"{split}_data.npz",
            "labels": j3c_root / f"labels_{split}.csv",
            "sidecar": j3c_root / f"H_fresh_{split}_sidecar.csv",
            "relation_features": rel_path,
            "entity_features": mention_path,
        }
        self.audit = {
            "split": split,
            "episodes": int(len(self.d["N"])),
            "fact_rows": n,
            "relation_carrier_exact_to_fact_H": True,
            "entity_carrier_shape": list(Hmention.shape),
            "label_sidecar_ids_exact": True,
            "canonical_fact_id_unique": True,
            "episode_ids_present": self.episode_ids is not None,
            "changed_value_update_pair_count": int(len(self.pairs)),
            "episode_counts_by_N": {str(c): int(len(self.rows_by_cap[c])) for c in CAPS},
        }


def batch_episode(data: SplitData, rows):
    rows = np.asarray(rows, dtype=np.int64)
    offsets = data.d["fact_offsets"]
    hs = np.stack([data.d["fact_H"][int(offsets[i]):int(offsets[i + 1])] for i in rows]).astype(np.float32, copy=False)
    ys = np.stack([data.d["fact_Y"][int(offsets[i]):int(offsets[i + 1])] for i in rows]).astype(np.int64, copy=False)
    return hs, ys


def episode_loss(core, heads, facts, targets, slot_ids, device):
    b, _, _ = facts.shape
    state = core.zero_state(b, device=device)
    for pos, slot in enumerate(slot_ids):
        h = torch.as_tensor(facts[:, pos], dtype=torch.float32, device=device)
        state = core.write_to_slot(state, h, int(slot))
    losses = []
    for pos, slot in enumerate(slot_ids):
        logits = heads.logits(state, int(slot))
        y = torch.as_tensor(targets[:, pos], dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits, y))
    return torch.stack(losses).mean()


def update_loss(core, heads, data: SplitData, pair_ids, rng, device):
    p = data.pairs[pair_ids]
    d = data.d["fact_H"]
    h1 = torch.as_tensor(np.array(d[p[:, 0]], dtype=np.float32, copy=True), device=device)
    h2 = torch.as_tensor(np.array(d[p[:, 1]], dtype=np.float32, copy=True), device=device)
    y1 = torch.as_tensor(p[:, 2], dtype=torch.long, device=device)
    y2 = torch.as_tensor(p[:, 3], dtype=torch.long, device=device)
    slot = int(rng.integers(0, 4))
    state = core.zero_state(len(p), device=device)
    state1 = core.write_to_slot(state, h1, slot)
    logits1 = heads.logits(state1, slot)
    state2 = core.write_to_slot(state1, h2, slot)
    logits2 = heads.logits(state2, slot)
    return 0.5 * (F.cross_entropy(logits1, y1) + F.cross_entropy(logits2, y2))


@torch.no_grad()
def eval_single(core, heads, data: SplitData, device, seed, per_cap_limit=None, collect=False):
    rng = np.random.default_rng(seed)
    out = {}
    total_good = total_n = 0
    loss_sum = loss_n = 0.0
    details = []
    for cap in CAPS:
        good = n = 0
        rows = data.rows_by_cap[cap]
        if per_cap_limit is not None:
            rows = rows[:per_cap_limit]
        for start in range(0, len(rows), BATCH_SIZE):
            rb = rows[start:start + BATCH_SIZE]
            facts, targets = batch_episode(data, rb)
            slots = rng.choice(4, size=cap, replace=False).tolist()
            state = core.zero_state(len(rb), device=device)
            for pos, slot in enumerate(slots):
                h = torch.as_tensor(facts[:, pos], dtype=torch.float32, device=device)
                state = core.write_to_slot(state, h, int(slot))
            for pos, slot in enumerate(slots):
                logits = heads.logits(state, int(slot))
                y = torch.as_tensor(targets[:, pos], dtype=torch.long, device=device)
                pred = logits.argmax(-1)
                good += int((pred == y).sum())
                n += len(rb)
                loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
                loss_n += len(rb)
                if collect:
                    yv, pv = y.cpu().numpy(), pred.cpu().numpy()
                    fact_rows = data.d["fact_offsets"]
                    for j, episode_row in enumerate(rb):
                        flat_idx = int(fact_rows[int(episode_row)] + pos)
                        details.append({"N": cap, "fact_index": flat_idx, "value_class": int(yv[j]), "correct": int(yv[j] == pv[j])})
        out[str(cap)] = {"correct": good, "count": n, "answer_accuracy": good / max(n, 1)}
        total_good += good
        total_n += n
    out["overall"] = {"correct": total_good, "count": total_n, "answer_accuracy": total_good / max(total_n, 1)}
    out["loss"] = loss_sum / max(loss_n, 1)
    if collect:
        out["details"] = details
    return out


@torch.no_grad()
def eval_updates(core, heads, data: SplitData, device, batch_size=BATCH_SIZE, pair_limit=None, collect=False):
    old_ok = new_ok = n = 0
    old_loss_sum = new_loss_sum = 0.0
    non_target_changed = False
    target_ids = np.arange(len(data.pairs), dtype=np.int64)
    if pair_limit is not None:
        target_ids = target_ids[:pair_limit]
    details = []
    for start in range(0, len(target_ids), batch_size):
        p = data.pairs[target_ids[start:start + batch_size]]
        hs = data.d["fact_H"]
        h1 = torch.as_tensor(np.array(hs[p[:, 0]], dtype=np.float32, copy=True), device=device)
        h2 = torch.as_tensor(np.array(hs[p[:, 1]], dtype=np.float32, copy=True), device=device)
        y1 = torch.as_tensor(p[:, 2], dtype=torch.long, device=device)
        y2 = torch.as_tensor(p[:, 3], dtype=torch.long, device=device)
        slot = (start // batch_size) % 4
        state0 = core.zero_state(len(p), device=device)
        state1 = core.write_to_slot(state0, h1, slot)
        state2 = core.write_to_slot(state1, h2, slot)
        logits_old = heads.logits(state1, slot)
        logits_new = heads.logits(state2, slot)
        pred_old, pred_new = logits_old.argmax(-1), logits_new.argmax(-1)
        old_ok += int((pred_old == y1).sum())
        new_ok += int((pred_new == y2).sum())
        old_loss_sum += float(F.cross_entropy(logits_old, y1, reduction="sum"))
        new_loss_sum += float(F.cross_entropy(logits_new, y2, reduction="sum"))
        others = [j for j in range(4) if j != slot]
        if not torch.equal(state0[:, others], state1[:, others]) or not torch.equal(state1[:, others], state2[:, others]):
            non_target_changed = True
        if collect:
            y1v, y2v = y1.cpu().numpy(), y2.cpu().numpy()
            p1v, p2v = pred_old.cpu().numpy(), pred_new.cpu().numpy()
            for i, pair in enumerate(p):
                details.append({"pair_index": int(target_ids[start + i]), "old_value_class": int(y1v[i]),
                                "old_correct": int(y1v[i] == p1v[i]), "latest_value_class": int(y2v[i]),
                                "latest_correct": int(y2v[i] == p2v[i]), "fact1": int(pair[0]), "fact2": int(pair[1])})
        n += len(p)
    return {
        "pair_count": n,
        "old_value_accuracy_after_first_write": old_ok / max(n, 1),
        "latest_value_accuracy_after_update": new_ok / max(n, 1),
        "old_correct": old_ok,
        "latest_correct": new_ok,
        "old_loss": old_loss_sum / max(n, 1),
        "latest_loss": new_loss_sum / max(n, 1),
        "loss": 0.5 * (old_loss_sum + new_loss_sum) / max(2 * n, 1),
        "non_target_slot_mutation": non_target_changed,
        **({"details": details} if collect else {}),
    }


def evaluate(core, heads, data, device, split, diagnostic=False, collect=False):
    if split == "val":
        eval_seed = SEED + 70000
        single = eval_single(core, heads, data, device, eval_seed, collect=collect)
        updates = eval_updates(core, heads, data, device, collect=collect)
    elif split == "train_diag":
        single = eval_single(core, heads, data, device, SEED + 60000,
                             per_cap_limit=128, collect=collect)
        updates = eval_updates(core, heads, data, device, pair_limit=256, collect=collect)
    elif split == "testA":
        single = eval_single(core, heads, data, device, SEED + 80000, collect=collect)
        updates = eval_updates(core, heads, data, device, collect=collect)
    elif split == "testB":
        single = eval_single(core, heads, data, device, SEED + 90000, collect=collect)
        updates = eval_updates(core, heads, data, device, collect=collect)
    else:
        raise ValueError(split)
    val_loss = 0.5 * single["loss"] + 0.5 * updates["loss"]
    return single, updates, val_loss


def score_key(single, updates, val_loss):
    return (int(updates["latest_correct"]), int(single["4"]["correct"]),
            int(single["overall"]["correct"]), -float(val_loss))


def metric_row(step, train_last, train_roll, train_diag_loss, single, updates, val_loss, task):
    return {
        "step": int(step),
        "train_task": task,
        "train_loss_last_batch": "" if train_last is None else float(train_last),
        "train_loss_mean_last_100_steps": "" if train_roll is None else float(train_roll),
        "train_loss_fixed_diagnostic_subset": float(train_diag_loss),
        "validation_loss_mixture_50_50": float(val_loss),
        "validation_single_write_overall": single["overall"]["answer_accuracy"],
        "validation_single_write_N1": single["1"]["answer_accuracy"],
        "validation_single_write_N2": single["2"]["answer_accuracy"],
        "validation_single_write_N4": single["4"]["answer_accuracy"],
        "validation_update_old_value": updates["old_value_accuracy_after_first_write"],
        "validation_update_latest_value": updates["latest_value_accuracy_after_update"],
        "validation_update_non_target_slot_mutation": updates["non_target_slot_mutation"],
        "single_overall_correct": single["overall"]["correct"],
        "single_overall_count": single["overall"]["count"],
        "single_N4_correct": single["4"]["correct"],
        "single_N4_count": single["4"]["count"],
        "update_latest_correct": updates["latest_correct"],
        "update_pair_count": updates["pair_count"],
        "validation_single_write_cross_entropy": single["loss"],
        "validation_update_cross_entropy": updates["loss"],
    }


def detailed_error_rows(core, heads, data, device, checkpoint, out_root):
    single = eval_single(core, heads, data, device, SEED + 70000, collect=True)
    updates = eval_updates(core, heads, data, device, collect=True)
    records = []
    for row in single["details"]:
        label = data.labels[row["fact_index"]]
        for name, value in (("N", row["N"]), ("value_class", row["value_class"]),
                            ("entity_semantic_value", label.get("entity_semantic_value", "")),
                            ("relation_semantic_value", label.get("relation_semantic_value", ""))):
            records.append((f"single_N{row['N']}", name, str(value), row["correct"]))
    for row in updates["details"]:
        label = data.labels[row["fact1"]]
        for scope, class_field, correct_field in (("update_old", "old_value_class", "old_correct"),
                                                   ("update_latest", "latest_value_class", "latest_correct")):
            for name, value in (("value_class", row[class_field]),
                                ("entity_semantic_value", label.get("entity_semantic_value", "")),
                                ("relation_semantic_value", label.get("relation_semantic_value", ""))):
                records.append((scope, name, str(value), row[correct_field]))
    counts = defaultdict(lambda: [0, 0])
    for scope, category, group, correct in records:
        key = (scope, category, group)
        counts[key][0] += 1
        counts[key][1] += int(correct)
    return [{"checkpoint": checkpoint, "split": "val", "scope": k[0], "breakdown": k[1],
             "group": k[2], "count": v[0], "correct": v[1], "accuracy": v[1] / max(v[0], 1)}
            for k, v in sorted(counts.items())]


def freeze_and_audit(args, out_root):
    old_cfg_path = args.previous_config
    old_script_path = args.previous_script
    if not old_cfg_path.is_file() or not old_script_path.is_file():
        raise FileNotFoundError("3K config or source script missing; cannot exact-reproduce")
    cfg_sha, src_sha = sha256(old_cfg_path), sha256(old_script_path)
    if cfg_sha != EXPECTED_3K_CONFIG_SHA or src_sha != EXPECTED_3K_SCRIPT_SHA:
        raise RuntimeError(f"3K artifact hash mismatch: config={cfg_sha}, script={src_sha}")
    previous = json.loads(old_cfg_path.read_text(encoding="utf-8"))
    old_core = {
        "experiment": "CLR-O3-REBASE-1/VGen2", "seed": SEED,
        "architecture": "OracleLocalCore: Linear(2048,64) + GRUCell(64,64), plus four Linear(64,64) slot decoders",
        "initialization": "fresh random initialization; no missing R2/VGen state substituted",
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP, "batch_size": BATCH_SIZE, "eval_every": EVAL_EVERY,
        "training_mixture": "50% source-separated multi-slot episodes; 50% source-separated same-key changed-value update pairs",
        "episode_objective": "mean per-written-slot 64-class cross entropy",
        "update_objective": "mean CE after first write and after same-slot second write; latest value is supervised",
        "mr_cache": str(args.mr_cache), "j3c_cache": str(args.j3c_cache),
    }
    mismatches = {k: {"expected": v, "actual": previous.get(k)} for k, v in old_core.items() if previous.get(k) != v}
    if mismatches:
        raise RuntimeError("3K training config differs from frozen expected config: " + canonical_json(mismatches))
    if previous.get("steps") != 3000 or previous.get("device") != "cuda":
        raise RuntimeError("Previous 3K horizon/device does not match expected source run")
    if previous.get("frozen_dependency", {}).get("o21a_checkpoint_sha256") != EXPECTED_O21A_SHA:
        raise RuntimeError("Prior run does not point to the expected frozen O21A checkpoint")
    old_o21a_root = args.o21a_root
    old_gate = json.loads((old_o21a_root / "o21a_rebase_gate.json").read_text(encoding="utf-8"))
    old_ckpt = old_o21a_root / "o21a_rebase.pt"
    if not old_gate.get("gate_pass") or sha256(old_ckpt) != EXPECTED_O21A_SHA:
        raise RuntimeError("Frozen O21A PASS/checkpoint SHA verification failed")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen config requires CUDA/ROCm device, but torch.cuda is unavailable")
    device_name = torch.cuda.get_device_name(0)
    frozen = dict(previous)
    frozen.update({
        "steps": STEPS,
        "selection": "validation-only lexicographic priority: UPDATE latest-value accuracy, then N=4 single-write accuracy, then overall single-write accuracy, then lower validation loss",
        "checkpoint_evaluation_schedule": sorted(SNAPSHOT_STEPS),
        "curve_eval_every": EVAL_EVERY,
        "evaluation_randomness": "fixed deterministic validation slot permutations reused at every checkpoint; training RNG/objective unchanged",
        "validation_loss": "0.5 * full validation episode CE + 0.5 * full validation update CE; update CE averages first-write and latest-value CE",
        "train_loss_reporting": "last minibatch, rolling mean over preceding 100 optimization steps, and deterministic train-only diagnostic subset loss",
        "test_policy": "TestA/TestB files are not opened unless selected validation checkpoint passes all three frozen >=95% gates",
        "reproduction_parent_config_sha256": cfg_sha,
        "reproduction_parent_script_sha256": src_sha,
        "frozen_o21a_checkpoint_sha256_verified": EXPECTED_O21A_SHA,
        "training_hyperparameters_unchanged_from_3k": True,
        "selection_policy_changed_only_as_explicitly_required": True,
        "device_name": device_name,
    })
    dump_json(out_root / "vgen2_10k_frozen_config.json", frozen)
    return previous, frozen, {"previous_config_sha256": cfg_sha, "previous_script_sha256": src_sha,
                              "o21a_checkpoint_sha256": EXPECTED_O21A_SHA, "device_name": device_name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mr-cache", type=Path, required=True)
    ap.add_argument("--j3c-cache", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--o21a-root", type=Path, required=True)
    ap.add_argument("--previous-config", type=Path, required=True)
    ap.add_argument("--previous-script", type=Path, required=True)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    allowed_staging = {"code", "logs", "snapshots", "vgen2_10k_frozen_config.json"}
    unexpected = [p.name for p in args.out_root.iterdir() if p.name not in allowed_staging]
    if unexpected:
        raise RuntimeError(f"Output directory contains unexpected prior-run artifacts; refusing to overwrite: {unexpected}")
    for name in ("code", "logs", "snapshots"):
        stage_dir = args.out_root / name
        stage_dir.mkdir(exist_ok=True)
        if name == "code":
            unexpected_code = [p.name for p in stage_dir.iterdir()
                               if p.name not in {"train_vgen2_10k_diag.py", "__pycache__"}]
            if unexpected_code:
                raise RuntimeError(f"Code staging directory has unexpected files: {unexpected_code}")
        elif name == "logs":
            unexpected_logs = [p.name for p in stage_dir.iterdir()
                               if p.name not in {"training.log", "preflight_guard_failure.log"}]
            if unexpected_logs:
                raise RuntimeError(f"Log staging directory has unexpected files: {unexpected_logs}")
        elif any(stage_dir.iterdir()):
            raise RuntimeError(f"Snapshot staging directory is not empty; refusing to overwrite: {stage_dir}")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    previous, frozen, source_hashes = freeze_and_audit(args, args.out_root)

    # Deliberately instantiate only train and validation. TestA/TestB are not touched before the gate.
    train = SplitData("train", args.mr_cache, args.j3c_cache)
    val = SplitData("val", args.mr_cache, args.j3c_cache)
    train_ids, val_ids = set(train.fact_ids), set(val.fact_ids)
    fact_overlap = sorted(train_ids & val_ids)
    episode_overlap = None
    if train.episode_ids is not None and val.episode_ids is not None:
        episode_overlap = sorted(set(train.episode_ids) & set(val.episode_ids))
    if fact_overlap or episode_overlap:
        raise ValueError(f"train/validation source leakage: fact_overlap={len(fact_overlap)}, episode_overlap={len(episode_overlap or [])}")
    input_files = {}
    for split_data in (train, val):
        for name, path in split_data.files.items():
            input_files[f"{split_data.split}/{name}"] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    integrity = {
        "status": "PASS", "test_split_files_opened": False,
        "train": train.audit, "validation": val.audit,
        "train_validation_canonical_fact_id_overlap": 0,
        "train_validation_episode_id_overlap": None if episode_overlap is None else 0,
        "source_separation_verified": True,
        "input_file_hashes_train_validation_only": input_files,
        "parent_artifact_hashes": source_hashes,
        "o21a_parent_gate": "PASS; frozen artifact read-only",
    }
    dump_json(args.out_root / "input_integrity.json", integrity)
    audit_lines = [
        "# Stage 0 — Exact Rebuild Audit", "",
        "Status: PASS. The original 3K config and final source script hashes match the archived local/remote artifacts.",
        "", "Training configuration reused exactly:",
        f"- Architecture: {previous['architecture']}",
        f"- Seed / initialization: {SEED} / {previous['initialization']}",
        f"- Optimizer: AdamW, lr={LR}, weight_decay={WEIGHT_DECAY}, gradient clip={GRADIENT_CLIP}",
        f"- Batch size: {BATCH_SIZE}; same 50/50 episode/update mixture; same losses and raw frozen features.",
        f"- Source config SHA-256: `{source_hashes['previous_config_sha256']}`",
        f"- Source training script SHA-256: `{source_hashes['previous_script_sha256']}`",
        f"- Frozen O21A checkpoint SHA-256 verified: `{source_hashes['o21a_checkpoint_sha256']}`",
        "- Only horizon changed from 3,000 to 10,000 total steps. Selection ordering and fixed validation slot permutations follow this round's explicit requirements; neither changes training updates.",
        "- TestA/TestB caches were not opened during Stage 0.",
        "", "Input join and source separation:",
        f"- Train: {train.audit['episodes']} episodes / {train.audit['fact_rows']} facts; update pairs={train.audit['changed_value_update_pair_count']}.",
        f"- Validation: {val.audit['episodes']} episodes / {val.audit['fact_rows']} facts; update pairs={val.audit['changed_value_update_pair_count']}.",
        f"- Canonical fact overlap: {len(fact_overlap)}; episode overlap: {'not exposed in NPZ schema' if episode_overlap is None else len(episode_overlap)}.",
        "- MR `fact_H` equals certified J3C `H_base_I0` bitwise per split; sidecar and label IDs match rowwise.",
    ]
    (args.out_root / "stage0_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    core, heads = OracleLocalCore().to(device), MRValueHeads().to(device)
    params = list(core.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED + 2400)
    train_rows = {c: train.rows_by_cap[c] for c in CAPS}
    curves, single_rows, update_rows, checkpoint_rows = [], [], [], []
    rolling_losses = deque(maxlen=100)
    best_key, best_step, best_metrics = None, None, None
    best_candidate = args.out_root / "best_val_candidate.pt"
    start_time = time.perf_counter()

    def run_eval(step, train_last, task):
        nonlocal best_key, best_step, best_metrics
        core.eval(); heads.eval()
        train_single, train_update, train_diag_loss = evaluate(core, heads, train, device, "train_diag")
        val_single, val_update, val_loss = evaluate(core, heads, val, device, "val")
        train_roll = float(np.mean(rolling_losses)) if rolling_losses else None
        row = metric_row(step, train_last, train_roll, train_diag_loss, val_single, val_update, val_loss, task)
        key = score_key(val_single, val_update, val_loss)
        row["selected_by_validation_priority"] = False
        row["selection_rank_update_latest_correct"] = key[0]
        row["selection_rank_N4_correct"] = key[1]
        row["selection_rank_overall_correct"] = key[2]
        row["selection_rank_lower_validation_loss"] = key[3]
        curves.append(row)
        single_rows.append({"step": step, **{f"N{k}": val_single[str(k)]["answer_accuracy"] for k in CAPS},
                            "overall": val_single["overall"]["answer_accuracy"],
                            "overall_correct": val_single["overall"]["correct"], "overall_count": val_single["overall"]["count"]})
        update_rows.append({"step": step, "old_value_accuracy_after_first_write": val_update["old_value_accuracy_after_first_write"],
                            "latest_value_accuracy_after_update": val_update["latest_value_accuracy_after_update"],
                            "pair_count": val_update["pair_count"], "non_target_slot_mutation": val_update["non_target_slot_mutation"],
                            "update_cross_entropy": val_update["loss"]})
        checkpoint_rows.append({"step": step, "validation_priority_tuple": canonical_json(key),
                                "update_latest": val_update["latest_value_accuracy_after_update"],
                                "N4": val_single["4"]["answer_accuracy"], "single_overall": val_single["overall"]["answer_accuracy"],
                                "validation_loss": val_loss, "scheduled_snapshot": step in SNAPSHOT_STEPS,
                                "selected": False})
        if best_key is None or key > best_key:
            best_key, best_step = key, step
            best_metrics = {"single": val_single, "updates": val_update, "validation_loss": val_loss}
            torch.save({"core": core.state_dict(), "value_heads": heads.state_dict(), "step": step,
                        "val_single": val_single, "val_update": val_update, "validation_loss": val_loss,
                        "selection_key": key, "config": frozen}, best_candidate)
        if step in SNAPSHOT_STEPS:
            torch.save({"core": core.state_dict(), "value_heads": heads.state_dict(), "step": step,
                        "val_single": val_single, "val_update": val_update, "validation_loss": val_loss},
                       args.out_root / "snapshots" / f"step_{step:05d}.pt")
        print(json.dumps({"stage": "VGEN2_10K", **row, "best_step_so_far": best_step}, ensure_ascii=False), flush=True)
        core.train(); heads.train()

    run_eval(0, None, "initialization")
    for step in range(1, STEPS + 1):
        if rng.random() < 0.5:
            cap = int(rng.choice(CAPS))
            rr = train_rows[cap]
            rows = rr[rng.integers(0, len(rr), size=min(BATCH_SIZE, len(rr)))]
            facts, targets = batch_episode(train, rows)
            slots = rng.choice(4, size=cap, replace=False).tolist()
            loss = episode_loss(core, heads, facts, targets, slots, device)
            task = "episode"
        else:
            pair_ids = rng.integers(0, len(train.pairs), size=min(BATCH_SIZE, len(train.pairs)))
            loss = update_loss(core, heads, train, pair_ids, rng, device)
            task = "update"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRADIENT_CLIP)
        opt.step()
        rolling_losses.append(float(loss.detach()))
        if step % EVAL_EVERY == 0:
            run_eval(step, float(loss.detach()), task)

    # Revisit the selected VAL checkpoint; the test split is still unopened.
    best = torch.load(best_candidate, map_location="cpu", weights_only=False)
    core.load_state_dict(best["core"], strict=True)
    heads.load_state_dict(best["value_heads"], strict=True)
    core.eval(); heads.eval()
    val_single, val_update, val_loss = evaluate(core, heads, val, device, "val")
    val_pass = (val_single["overall"]["answer_accuracy"] >= 0.95 and
                val_single["4"]["answer_accuracy"] >= 0.95 and
                val_update["latest_value_accuracy_after_update"] >= 0.95)
    for row in curves:
        row["selected_by_validation_priority"] = row["step"] == best_step
    for row in checkpoint_rows:
        row["selected"] = row["step"] == best_step
    write_csv(args.out_root / "training_curve.csv", curves)
    val_fields = ["step", "validation_loss_mixture_50_50", "validation_single_write_overall",
                  "validation_single_write_N1", "validation_single_write_N2", "validation_single_write_N4",
                  "validation_update_old_value", "validation_update_latest_value",
                  "validation_update_non_target_slot_mutation", "selected_by_validation_priority"]
    write_csv(args.out_root / "validation_curve.csv", [{k: r[k] for k in val_fields} for r in curves])
    write_csv(args.out_root / "checkpoint_metrics.csv", checkpoint_rows)
    write_csv(args.out_root / "single_write_breakdown.csv", single_rows)
    write_csv(args.out_root / "update_breakdown.csv", update_rows)
    error_rows = detailed_error_rows(core, heads, val, device, f"best_step_{best_step}", args.out_root)
    last_snapshot = torch.load(args.out_root / "snapshots" / f"step_{STEPS:05d}.pt", map_location="cpu", weights_only=False)
    last_core, last_heads = OracleLocalCore().to(device), MRValueHeads().to(device)
    last_core.load_state_dict(last_snapshot["core"], strict=True)
    last_heads.load_state_dict(last_snapshot["value_heads"], strict=True)
    error_rows += detailed_error_rows(last_core, last_heads, val, device, "step_10000", args.out_root)
    write_csv(args.out_root / "error_breakdown.csv", error_rows)

    selected = {
        "selection_rule": frozen["selection"], "best_step": int(best_step),
        "best_priority_tuple": list(best_key), "best_validation_single": val_single,
        "best_validation_update": val_update, "best_validation_loss": val_loss,
        "validation_gate": {"single_write_overall_ge_0.95": val_single["overall"]["answer_accuracy"] >= 0.95,
                            "N4_ge_0.95": val_single["4"]["answer_accuracy"] >= 0.95,
                            "update_latest_ge_0.95": val_update["latest_value_accuracy_after_update"] >= 0.95,
                            "pass": bool(val_pass)},
        "test_access_before_gate": False,
    }
    dump_json(args.out_root / "best_checkpoint_selection.json", selected)
    test_rows = [{"status": "NOT_RUN", "reason": "validation gate failed; TestA/TestB caches were not loaded"}]
    tests = None
    if val_pass:
        # This is the first point where TestA/TestB paths are opened.
        test_rows = []
        tests = {}
        for split in ("testA", "testB"):
            data = SplitData(split, args.mr_cache, args.j3c_cache)
            single, updates, test_loss = evaluate(core, heads, data, device, split)
            tests[split] = {"single": single, "updates": updates, "loss": test_loss}
            for cap in CAPS:
                test_rows.append({"split": split, "metric": "single_write_answer", "capacity": cap,
                                  "correct": single[str(cap)]["correct"], "count": single[str(cap)]["count"],
                                  "accuracy": single[str(cap)]["answer_accuracy"]})
            test_rows.append({"split": split, "metric": "single_write_answer", "capacity": "all",
                              "correct": single["overall"]["correct"], "count": single["overall"]["count"],
                              "accuracy": single["overall"]["answer_accuracy"]})
            test_rows.append({"split": split, "metric": "update_old", "capacity": "same_slot",
                              "correct": updates["old_correct"], "count": updates["pair_count"],
                              "accuracy": updates["old_value_accuracy_after_first_write"]})
            test_rows.append({"split": split, "metric": "update_latest", "capacity": "same_slot",
                              "correct": updates["latest_correct"], "count": updates["pair_count"],
                              "accuracy": updates["latest_value_accuracy_after_update"]})
        shutil.copy2(best_candidate, args.out_root / "vgen2_rebase_10k_best.pt")
    write_csv(args.out_root / "test_results.csv", test_rows)
    gate = {
        "experiment": "O3-REBASE1-V2", "protocol_frozen": True, "training_steps": STEPS,
        "best_step": int(best_step), "validation_gate_pass": bool(val_pass),
        "validation_metrics": {"single_write_overall": val_single["overall"]["answer_accuracy"],
                               "single_write_N4": val_single["4"]["answer_accuracy"],
                               "update_latest": val_update["latest_value_accuracy_after_update"],
                               "update_old": val_update["old_value_accuracy_after_first_write"],
                               "non_target_slot_mutation": val_update["non_target_slot_mutation"],
                               "validation_loss": val_loss},
        "test_evaluation": "COMPLETED_AFTER_VALIDATION_PASS" if val_pass else "NOT_RUN_VAL_GATE_FAILED",
        "test_metrics": tests,
        "formal_component_frozen": bool(val_pass),
        "o3a_allocator_trained": False, "memory_integration_run": False,
        "candidate_b_modified": False, "rule_a_threshold_modified": False,
        "o21a_modified": False, "existing_checkpoint_modified": False,
        "test_data_access_before_validation_pass": False,
    }
    dump_json(args.out_root / "vgen2_10k_gate.json", gate)

    # Trajectory diagnosis is computed from fixed, identical validation permutations.
    by_step = {int(r["step"]): r for r in curves}
    def at(step, field):
        return float(by_step[step][field])
    trajectory = {}
    for field in ("validation_single_write_overall", "validation_single_write_N4", "validation_update_latest_value"):
        trajectory[field] = {str(s): at(s, field) for s in (3000, 5000, 7000, 8000, 9000, 10000)}
    final_deltas = {k: v["10000"] - v["8000"] for k, v in trajectory.items()}
    single_pass = val_single["overall"]["answer_accuracy"] >= 0.95 and val_single["4"]["answer_accuracy"] >= 0.95
    if val_pass:
        category = "PASS"
        verdict = "O3-REBASE1-V2-P — VGEN2 10K REBUILD PASSES OPERATIONAL VALUE GATE"
    elif single_pass:
        category = "UPDATE-SPECIFIC"
        verdict = "O3-REBASE1-V2-F — VGEN2 UPDATE-SPECIFIC VALUE BOTTLENECK"
    elif any(delta >= 0.01 for delta in final_deltas.values()):
        category = "UNDERTRAINED — CURVE STILL RISING"
        verdict = "O3-REBASE1-V2-F — VGEN2 REMAINS UNDERTRAINED AT 10K"
    elif abs(final_deltas["validation_single_write_overall"]) < 0.005 and abs(final_deltas["validation_single_write_N4"]) < 0.005 and abs(final_deltas["validation_update_latest_value"]) < 0.005:
        category = "PLATEAU — OBJECTIVE/REPRESENTATION BOTTLENECK"
        verdict = "O3-REBASE1-V2-F — VGEN2 OBJECTIVE/REPRESENTATION BOTTLENECK CONFIRMED"
    else:
        category = "GENERAL VALUE-PATH BOTTLENECK"
        verdict = "O3-REBASE1-V2-F — VGEN2 OBJECTIVE/REPRESENTATION BOTTLENECK CONFIRMED"
    (args.out_root / "FINAL_VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")
    test_summary = "NOT RUN" if tests is None else "; ".join(
        f"{s}: overall={tests[s]['single']['overall']['answer_accuracy']:.4f}, N4={tests[s]['single']['4']['answer_accuracy']:.4f}, update={tests[s]['updates']['latest_value_accuracy_after_update']:.4f}"
        for s in ("testA", "testB"))
    report = [
        "# O3-REBASE1-V2 — VGen2 Fixed 10K Rebuild Diagnostic", "", "## Final verdict", "",
        f"**{verdict}**", "", ("Outcome classification: **VALIDATION PASS**." if val_pass else f"Failure category: **{category}**."), "",
        "## Frozen run and integrity", "",
        f"- Training config matches the archived 3K core config: **YES**; fresh initialization at step 0; exactly {STEPS:,} updates.",
        f"- Selected checkpoint step: {best_step}; selection was validation-only with priority UPDATE latest, N4, overall, then lower validation loss.",
        f"- Train/validation source overlap: {len(fact_overlap)} canonical fact IDs; TestA/TestB opened before gate: **NO**.",
        f"- O21A frozen checkpoint verified: `{EXPECTED_O21A_SHA}`; O21A/Candidate B/Rule A unchanged.",
        "- O3a allocator: NOT RUN. Memory/router integration: NOT RUN.", "",
        "## Validation trajectory", "",
        "| Metric | 3K | 5K | 7K | 8K | 9K | 10K | 8K→10K (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"validation_single_write_overall": "Single-write overall", "validation_single_write_N4": "Single-write N=4",
              "validation_update_latest_value": "UPDATE latest value"}
    for field, vals in trajectory.items():
        report.append("| " + labels[field] + " | " + " | ".join(f"{vals[str(s)]:.2%}" for s in (3000, 5000, 7000, 8000, 9000, 10000)) + f" | {final_deltas[field] * 100:+.2f}pp |")
    report += ["", "## Selected validation checkpoint", "",
               f"- Single-write overall: {val_single['overall']['answer_accuracy']:.2%}",
               f"- N=1 / N=2 / N=4: {val_single['1']['answer_accuracy']:.2%} / {val_single['2']['answer_accuracy']:.2%} / {val_single['4']['answer_accuracy']:.2%}",
               f"- UPDATE old-value retention: {val_update['old_value_accuracy_after_first_write']:.2%}",
               f"- UPDATE latest-value: {val_update['latest_value_accuracy_after_update']:.2%}",
               f"- Validation loss (50/50 episode/update): {val_loss:.6f}",
               f"- Validation gate (overall≥95%, N4≥95%, update latest≥95%): {'PASS' if val_pass else 'FAIL'}.",
               f"- Validation TestA/TestB status: {test_summary}.", "",
               "## Diagnostic interpretation", "",
               f"The 8K→10K changes are {final_deltas['validation_single_write_overall'] * 100:+.2f}pp overall, {final_deltas['validation_single_write_N4'] * 100:+.2f}pp N=4, and {final_deltas['validation_update_latest_value'] * 100:+.2f}pp UPDATE latest. Classification follows these frozen trajectory values; it does not imply an information-theoretic claim.",
               "No architecture, optimizer, learning rate, loss, preprocessing, feature source, backbone, or checkpoint was changed.", "",
               "## Test policy and next step", "",
               f"TestA/TestB: **{test_summary}**.",
               ("One next experiment: Freeze VGen2 and train/rebuild the O3a free-slot allocator under the frozen O3 Rebase contract." if val_pass else
                ("One next experiment: run one fixed longer-budget VGen2 continuation/retrain without architecture changes." if category.startswith("UNDERTRAINED") else
                 ("One next experiment: run a minimal diagnostic of the same-slot value replacement objective only; do not alter query/address." if category == "UPDATE-SPECIFIC" else
                  "One next experiment: run one minimal diagnostic of the VGen2 value objective; do not continue adding steps or alter architecture."))),
               "", "## Artifact index", "",
               "- `vgen2_10k_frozen_config.json` — frozen rebuild config.",
               "- `training_curve.csv`, `validation_curve.csv`, `checkpoint_metrics.csv` — full per-100-step trajectory.",
               "- `single_write_breakdown.csv`, `update_breakdown.csv`, `error_breakdown.csv` — capacity/update/value-class breakdowns.",
               "- `best_checkpoint_selection.json`, `vgen2_10k_gate.json`, `FINAL_VERDICT.txt` — machine-readable selection and gate.",
               f"- Formal best component checkpoint: {'`vgen2_rebase_10k_best.pt` (written and SHA-manifested)' if val_pass else 'NOT WRITTEN; validation gate failed'}.",
               "- `test_results.csv` — post-gate TestA/TestB results (or NOT_RUN if validation fails).", ""]
    (args.out_root / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    dump_json(args.out_root / "runtime.json", {"device": str(device), "device_name": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "wall_clock_s": time.perf_counter() - start_time,
               "training_steps": STEPS, "eval_count": len(curves), "best_step": best_step,
               "gpu_name": torch.cuda.get_device_name(0), "cuda_device_count": torch.cuda.device_count()})
    manifest = []
    for path in sorted(args.out_root.rglob("*")):
        if path.is_file() and path.name != "artifact_sha256_manifest.json":
            manifest.append({"path": str(path.relative_to(args.out_root)).replace("\\", "/"),
                             "bytes": path.stat().st_size, "sha256": sha256(path)})
    dump_json(args.out_root / "artifact_sha256_manifest.json", {"algorithm": "SHA-256", "manifest_excludes_itself": True,
               "file_count": len(manifest), "files": manifest})
    print(json.dumps({"stage": "FINAL", "verdict": verdict, "best_step": best_step,
                      "val_pass": val_pass, "category": category, "test": test_summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
