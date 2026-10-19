import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from a1_v22_s0_closure_common import (  # noqa: F401
    LinearProbe, SingleSlotBaseline, accuracy, load_representations, train_classifier,
)
from a1_v22_storage_common import DEFAULT_CACHE, FactCache, get_device, seed_all
from clr_v23_oracle_r2 import OracleLocalCore

DEFAULT_OUT = r"D:\training\checkpoints\clr_v23_oracle_r2"
CITY_CLASSES = 64
CAPACITIES = (1, 2, 4)


def make_heads(device, n=4):
    return nn.ModuleList([nn.Linear(64, 64) for _ in range(n)]).to(device=device, dtype=torch.float32)


def rows_for(cache, split, cap, limit=None):
    rows = cache.rows_for_capacity(split, cap)
    return rows if limit is None else rows[:min(len(rows), int(limit))]


@torch.no_grad()
def eval_capacity(core, heads, cache, split, cap, rows=None, batch_size=128, fixed_position=None):
    core.eval(); heads.eval()
    device = next(core.parameters()).device
    rows = rows_for(cache, split, cap) if rows is None else np.asarray(rows, dtype=np.int64)
    correct = total = 0
    by = [0] * cap
    counts = [0] * cap
    for i in range(0, len(rows), batch_size):
        rb = rows[i:i + batch_size]
        facts, labels, positions, _ = cache.batch(split, rb, cap, device)
        state, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
        if fixed_position is not None:
            positions = torch.full_like(positions, int(fixed_position))
        for p in range(cap):
            mask = positions == p
            if not mask.any():
                continue
            # A fixed-position evaluator uses the city stored at that position.
            y = labels[mask, p]
            pred = heads[p](state[mask, p, :]).argmax(1)
            good = int((pred == y).sum())
            correct += good; total += int(mask.sum())
            by[p] += good; counts[p] += int(mask.sum())
    return {"acc": correct / max(total, 1), "correct": correct, "total": total,
            "by_position": [by[p] / max(counts[p], 1) for p in range(cap)],
            "position_total": counts}


def position_matrix(core, heads, cache, split, cap):
    return [eval_capacity(core, heads, cache, split, cap, fixed_position=p) for p in range(cap)]


def collect_states(core, cache, split, cap, rows=None, batch_size=128):
    core.eval(); device = next(core.parameters()).device
    rows = rows_for(cache, split, cap) if rows is None else np.asarray(rows, dtype=np.int64)
    states = []; labels = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            rb = rows[i:i + batch_size]
            facts, y, _, _ = cache.batch(split, rb, cap, device)
            s, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
            states.append(s.cpu().numpy().astype(np.float32)); labels.append(y.cpu().numpy().astype(np.int64))
    return np.concatenate(states), np.concatenate(labels)


def save_checkpoint(path, core, heads, step, stage, metrics, seed=42, extra=None):
    payload = {
        "model": core.state_dict(), "heads": heads.state_dict(), "step": int(step),
        "stage": stage, "seed": int(seed), "metrics": metrics,
        "config": {"slots": 4, "dim": 64, "allocation": "oracle_position",
                   "init": "zero_nontrainable", "readout": "local_64", "architecture": "v2.3-OA-r2"},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device, n_heads=4):
    ck = torch.load(path, map_location=device)
    core = OracleLocalCore(device=device, dtype=torch.float32)
    heads = make_heads(device, n_heads)
    core.load_state_dict(ck["model"], strict=True)
    heads.load_state_dict(ck["heads"], strict=True)
    return core, heads, ck


def json_dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json), encoding="utf-8")


def _json(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)
