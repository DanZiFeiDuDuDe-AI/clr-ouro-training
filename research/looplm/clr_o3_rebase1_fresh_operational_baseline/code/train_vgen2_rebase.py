from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CAPS = (1, 2, 4)
SPLITS = ("train", "val", "testA", "testB")
SEED = 42
STEPS = 3000
EVAL_EVERY = 100
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.01


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


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


class OracleLocalCore(nn.Module):
    def __init__(self, hidden_dim=2048, slots=4, dim=64):
        super().__init__()
        self.hidden_dim, self.slots, self.dim = hidden_dim, slots, dim
        self.content_read = nn.Linear(hidden_dim, dim)
        self.shared_gru = nn.GRUCell(dim, dim)

    def zero_state(self, batch_size, device=None, dtype=None):
        p = next(self.parameters())
        return torch.zeros(batch_size, self.slots, self.dim, device=device or p.device, dtype=dtype or p.dtype)

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
    for key, rows in groups.items():
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
        Hrel = np.load(j3c_root / f"H_base_I0_{split}.npy", mmap_mode="r", allow_pickle=False)
        Hmention = np.load(j3c_root / f"H_mention_fresh_{split}.npy", mmap_mode="r", allow_pickle=False)
        n = len(self.d["fact_H"])
        if self.d["fact_H"].shape != Hrel.shape or not np.array_equal(self.d["fact_H"], Hrel):
            raise ValueError(f"{split}: MR fact_H and certified relation carrier rows do not match exactly")
        if Hmention.shape != Hrel.shape or len(self.labels) != n or len(self.sidecar) != n:
            raise ValueError(f"{split}: feature/label/sidecar row count mismatch")
        if any(a.get("canonical_fact_id") != b.get("canonical_fact_id") for a, b in zip(self.labels, self.sidecar)):
            raise ValueError(f"{split}: canonical fact ID mismatch between labels and feature sidecar")
        ids = [x.get("canonical_fact_id", "") for x in self.labels]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{split}: canonical fact IDs are not unique")
        self.index = {fid: i for i, fid in enumerate(ids)}
        self.pairs = build_update_pairs(self.labels, self.sidecar, self.d["fact_Y"])
        if len(self.pairs) == 0:
            raise ValueError(f"{split}: no source-separated same-key changed-value update pairs")
        for cap in CAPS:
            self.rows_by_cap = getattr(self, "rows_by_cap", {})
            self.rows_by_cap[cap] = np.flatnonzero(self.d["N"] == cap).astype(np.int64)
        self.audit = {"split": split, "episodes": int(len(self.d["N"])), "fact_rows": n, "relation_carrier_exact_to_fact_H": True, "entity_carrier_shape": list(Hmention.shape), "label_sidecar_ids_exact": True, "canonical_fact_id_unique": True, "changed_value_update_pair_count": int(len(self.pairs))}


def batch_episode(data: SplitData, rows):
    rows = np.asarray(rows, dtype=np.int64)
    d = data.d
    off = d["fact_offsets"]
    hs = np.stack([d["fact_H"][int(off[i]):int(off[i + 1])] for i in rows]).astype(np.float32, copy=False)
    ys = np.stack([d["fact_Y"][int(off[i]):int(off[i + 1])] for i in rows]).astype(np.int64, copy=False)
    return hs, ys


def episode_loss(core, heads, facts, targets, slot_ids, device):
    b, n, _ = facts.shape
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
def eval_single(core, heads, data: SplitData, device, seed):
    rng = np.random.default_rng(seed)
    out = {}
    total_good = total_n = 0
    for cap in CAPS:
        good = n = 0
        rows = data.rows_by_cap[cap]
        for start in range(0, len(rows), 64):
            rb = rows[start:start + 64]
            facts, targets = batch_episode(data, rb)
            slots = rng.choice(4, size=cap, replace=False).tolist()
            state = core.zero_state(len(rb), device=device)
            for pos, slot in enumerate(slots):
                h = torch.as_tensor(facts[:, pos], dtype=torch.float32, device=device)
                state = core.write_to_slot(state, h, int(slot))
            for pos, slot in enumerate(slots):
                logits = heads.logits(state, int(slot))
                y = torch.as_tensor(targets[:, pos], dtype=torch.long, device=device)
                good += int((logits.argmax(-1) == y).sum())
                n += len(rb)
        out[str(cap)] = {"correct": good, "count": n, "answer_accuracy": good / max(n, 1)}
        total_good += good; total_n += n
    out["overall"] = {"correct": total_good, "count": total_n, "answer_accuracy": total_good / max(total_n, 1)}
    return out


@torch.no_grad()
def eval_updates(core, heads, data: SplitData, device, batch_size=64):
    old_ok = new_ok = n = 0
    non_target_changed = False
    target_ids = np.arange(len(data.pairs), dtype=np.int64)
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
        old_ok += int((heads.logits(state1, slot).argmax(-1) == y1).sum())
        new_ok += int((heads.logits(state2, slot).argmax(-1) == y2).sum())
        others = [j for j in range(4) if j != slot]
        if not torch.equal(state0[:, others], state1[:, others]) or not torch.equal(state1[:, others], state2[:, others]):
            non_target_changed = True
        n += len(p)
    return {"pair_count": n, "old_value_accuracy_after_first_write": old_ok / max(n, 1), "latest_value_accuracy_after_update": new_ok / max(n, 1), "non_target_slot_mutation": non_target_changed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mr-cache", type=Path, required=True)
    ap.add_argument("--j3c-cache", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--o21a-root", type=Path, required=True)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    o21a_root = args.o21a_root
    o21a_gate = json.loads((o21a_root / "o21a_rebase_gate.json").read_text(encoding="utf-8"))
    o21a_checkpoint = o21a_root / "o21a_rebase.pt"
    if not o21a_gate.get("gate_pass") or not o21a_checkpoint.is_file():
        raise RuntimeError("O21A standalone gate/checkpoint is not PASS; stop before VGen2")
    o21a_sha = sha256(o21a_checkpoint)
    config = {"experiment": "CLR-O3-REBASE-1/VGen2", "seed": SEED, "device": str(device), "architecture": "OracleLocalCore: Linear(2048,64) + GRUCell(64,64), plus four Linear(64,64) slot decoders", "initialization": "fresh random initialization; no missing R2/VGen state substituted", "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY, "gradient_clip": 1.0, "batch_size": BATCH_SIZE, "steps": STEPS, "eval_every": EVAL_EVERY, "training_mixture": "50% source-separated multi-slot episodes; 50% source-separated same-key changed-value update pairs", "episode_objective": "mean per-written-slot 64-class cross entropy", "update_objective": "mean CE after first write and after same-slot second write; latest value is supervised", "selection": "best VAL score = 0.5*single-write overall accuracy + 0.5*update latest-value accuracy", "mr_cache": str(args.mr_cache), "j3c_cache": str(args.j3c_cache), "test_policy": "TestA/TestB are not loaded until both VAL standalone gates pass"}
    config["frozen_dependency"] = {"o21a_checkpoint": str(o21a_checkpoint), "o21a_checkpoint_sha256": o21a_sha, "o21a_gate_pass": True}
    dump_json(args.out_root / "vgen2_rebase_config.json", config)
    train = SplitData("train", args.mr_cache, args.j3c_cache)
    val = SplitData("val", args.mr_cache, args.j3c_cache)
    input_audit = {"train": train.audit, "val": val.audit, "cross_split_canonical_fact_id_overlap": len(set(x["canonical_fact_id"] for x in train.labels) & set(x["canonical_fact_id"] for x in val.labels))}
    if input_audit["cross_split_canonical_fact_id_overlap"]:
        raise ValueError("train/validation source facts overlap")
    dump_json(args.out_root / "vgen2_input_join_audit.json", input_audit)
    core, heads = OracleLocalCore().to(device), MRValueHeads().to(device)
    params = list(core.parameters()) + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    rng = np.random.default_rng(SEED + 2400)
    train_rows = {c: train.rows_by_cap[c] for c in CAPS}
    best_score, best_step = -1.0, 0
    curves = []
    started = time.perf_counter()
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
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == STEPS:
            core.eval(); heads.eval()
            single = eval_single(core, heads, val, device, SEED + step)
            updates = eval_updates(core, heads, val, device)
            score = 0.5 * single["overall"]["answer_accuracy"] + 0.5 * updates["latest_value_accuracy_after_update"]
            row = {"step": step, "train_loss": float(loss.detach()), "train_task": task, "val_single_overall": single["overall"]["answer_accuracy"], "val_single_N1": single["1"]["answer_accuracy"], "val_single_N2": single["2"]["answer_accuracy"], "val_single_N4": single["4"]["answer_accuracy"], "val_update_old": updates["old_value_accuracy_after_first_write"], "val_update_latest": updates["latest_value_accuracy_after_update"], "val_score": score}
            curves.append(row)
            print(json.dumps({"stage": "VGEN2_REBASE", **row}, ensure_ascii=False), flush=True)
            if score > best_score:
                best_score, best_step = score, step
                torch.save({"core": core.state_dict(), "value_heads": heads.state_dict(), "step": step, "single": single, "updates": updates, "config": config}, args.out_root / "vgen2_rebase.pt")
            core.train(); heads.train()
    write_csv(args.out_root / "vgen2_rebase_learning_curve.csv", curves)
    best = torch.load(args.out_root / "vgen2_rebase.pt", map_location="cpu", weights_only=False)
    core.load_state_dict(best["core"], strict=True); heads.load_state_dict(best["value_heads"], strict=True)
    core.eval(); heads.eval()
    val_single = eval_single(core, heads, val, device, SEED + 70000)
    val_update = eval_updates(core, heads, val, device)
    val_pass = val_single["overall"]["answer_accuracy"] >= 0.95 and val_update["latest_value_accuracy_after_update"] >= 0.95 and not val_update["non_target_slot_mutation"]
    result_rows = []
    for cap in CAPS:
        result_rows.append({"split": "val", "metric": "single_write_answer", "capacity": cap, **val_single[str(cap)]})
    result_rows.append({"split": "val", "metric": "single_write_answer", "capacity": "all", **val_single["overall"]})
    result_rows.append({"split": "val", "metric": "update", "capacity": "same_slot", **val_update})
    gate = {"experiment": "CLR-O3-REBASE-1/VGen2", "protocol_frozen": True, "best_step": best_step, "best_val_score": best_score, "val_single": val_single, "val_update": val_update, "val_gate_pass": val_pass, "gate_pass": False, "test_evaluation": "NOT_RUN_VAL_GATE_FAILED" if not val_pass else "PENDING_FINAL_FROZEN_EVALUATION"}
    if val_pass:
        # Test caches and changed-value pairs are first loaded here, after the VAL checkpoint is frozen.
        tests = {}
        for split in ("testA", "testB"):
            data = SplitData(split, args.mr_cache, args.j3c_cache)
            tests[split] = {"single": eval_single(core, heads, data, device, SEED + (80000 if split == "testA" else 90000)), "updates": eval_updates(core, heads, data, device)}
            for cap in CAPS:
                result_rows.append({"split": split, "metric": "single_write_answer", "capacity": cap, **tests[split]["single"][str(cap)]})
            result_rows.append({"split": split, "metric": "single_write_answer", "capacity": "all", **tests[split]["single"]["overall"]})
            result_rows.append({"split": split, "metric": "update", "capacity": "same_slot", **tests[split]["updates"]})
        tests_pass = all(tests[s]["single"]["overall"]["answer_accuracy"] >= 0.95 and tests[s]["updates"]["latest_value_accuracy_after_update"] >= 0.95 and not tests[s]["updates"]["non_target_slot_mutation"] for s in ("testA", "testB"))
        gate.update({"test_metrics": tests, "test_gate_pass": tests_pass, "gate_pass": tests_pass, "test_evaluation": "COMPLETED_FROZEN_EVALUATION"})
    write_csv(args.out_root / "vgen2_rebase_results.csv", result_rows)
    dump_json(args.out_root / "vgen2_rebase_gate.json", gate)
    dump_json(args.out_root / "vgen2_rebase_runtime.json", {"device": str(device), "torch": torch.__version__, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "wall_clock_s": time.perf_counter() - started, "steps": STEPS, "best_step": best_step})
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
