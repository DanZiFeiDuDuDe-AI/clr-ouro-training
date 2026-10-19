import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_SLOTS = 4
HIDDEN_DIM = 2048
LATENT_DIM = 64
KEY_CONTENT_DIM = 31
OCCUPANCY_DIM = 1
VALUE_DIM = 32
TEMP = 0.5
ALPHA = 1.0
BETA = 1.0
CITY_CLASSES = 64
DEFAULT_CACHE = r"D:\training\checkpoints\clr_v2_a1_ls2\cache"
DEFAULT_OUT = r"D:\training\checkpoints\clr_v22_storage"


def seed_all(seed=2237):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_dtype(device):
    return torch.bfloat16 if device.type == "cuda" else torch.float32


class FactorizedStorage(nn.Module):
    """CLR-v2.2-S storage-only core.

    Each slot is [key_content(31), occupancy(1), value(32)].  The key
    projection emits 31 content coordinates because the remaining key
    coordinate is the persistent occupancy signal.  No query path exists.
    """

    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.n_slots = N_SLOTS
        self.latent_dim = LATENT_DIM
        self.key_content_dim = KEY_CONTENT_DIM
        self.value_dim = VALUE_DIM
        if device is None:
            device = get_device()
        if dtype is None:
            dtype = model_dtype(device)
        self.device_ref = device
        self.dtype_ref = dtype
        self.key_read = nn.Linear(HIDDEN_DIM, KEY_CONTENT_DIM)
        self.value_read = nn.Linear(HIDDEN_DIM, VALUE_DIM)
        self.key_gru = nn.GRUCell(KEY_CONTENT_DIM, KEY_CONTENT_DIM)
        self.value_gru = nn.GRUCell(VALUE_DIM, VALUE_DIM)
        init = torch.zeros(1, N_SLOTS, LATENT_DIM, dtype=dtype)
        init[:, :, :KEY_CONTENT_DIM] = 0.02 * torch.randn(1, N_SLOTS, KEY_CONTENT_DIM, dtype=dtype)
        init[:, :, KEY_CONTENT_DIM] = 0.0
        init[:, :, KEY_CONTENT_DIM + 1:] = 0.02 * torch.randn(1, N_SLOTS, VALUE_DIM, dtype=dtype)
        self.init_s = nn.Parameter(init)
        self.to(device=device, dtype=dtype)

    def init_state(self, batch_size, device=None):
        if device is None:
            device = self.device_ref
        return self.init_s.expand(batch_size, -1, -1).to(device=device, dtype=self.dtype_ref).clone()

    def update(self, state, h_last, return_stats=False):
        h = h_last.to(device=state.device, dtype=state.dtype)
        key_fact = self.key_read(h)
        value_fact = self.value_read(h)
        key_content = state[:, :, :KEY_CONTENT_DIM]
        occupancy = state[:, :, KEY_CONTENT_DIM].clamp(0.0, 1.0)
        value = state[:, :, KEY_CONTENT_DIM + 1:]

        key_norm = F.normalize(key_content, dim=-1, eps=1e-6)
        fact_norm = F.normalize(key_fact, dim=-1, eps=1e-6).unsqueeze(1)
        match = (key_norm * fact_norm).sum(dim=-1)
        score = ALPHA * match - BETA * occupancy
        route = F.softmax(score / TEMP, dim=1)

        key_input = key_fact.unsqueeze(1).expand(-1, N_SLOTS, -1).reshape(-1, KEY_CONTENT_DIM)
        key_hidden = key_content.reshape(-1, KEY_CONTENT_DIM)
        key_candidate = self.key_gru(key_input, key_hidden).view(-1, N_SLOTS, KEY_CONTENT_DIM)

        value_input = value_fact.unsqueeze(1).expand(-1, N_SLOTS, -1).reshape(-1, VALUE_DIM)
        value_hidden = value.reshape(-1, VALUE_DIM)
        value_candidate = self.value_gru(value_input, value_hidden).view(-1, N_SLOTS, VALUE_DIM)

        write = route.unsqueeze(-1)
        new_key = (1.0 - write) * key_content + write * key_candidate
        new_value = (1.0 - write) * value + write * value_candidate
        new_occupancy = (1.0 - route) * occupancy + route
        new_state = torch.cat([new_key, new_occupancy.unsqueeze(-1), new_value], dim=-1)
        if not return_stats:
            return new_state
        stats = {
            "route": route,
            "score": score,
            "match": match,
            "occupancy_before": occupancy,
            "occupancy_after": new_occupancy,
            "key_fact": key_fact,
            "value_fact": value_fact,
            "argmax": route.argmax(dim=1),
        }
        return new_state, stats

    def forward_facts(self, facts, keep_states=False, keep_stats=False):
        state = self.init_state(facts.shape[0], facts.device)
        states = [state] if keep_states else None
        stats = [] if keep_stats else None
        for t in range(facts.shape[1]):
            if keep_stats:
                state, st = self.update(state, facts[:, t], return_stats=True)
                stats.append(st)
            else:
                state = self.update(state, facts[:, t])
            if keep_states:
                states.append(state)
        return state, states, stats

    def parameter_report(self):
        return {
            "total": int(sum(p.numel() for p in self.parameters())),
            "trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
            "key_read": int(sum(p.numel() for p in self.key_read.parameters())),
            "value_read": int(sum(p.numel() for p in self.value_read.parameters())),
            "key_gru": int(sum(p.numel() for p in self.key_gru.parameters())),
            "value_gru": int(sum(p.numel() for p in self.value_gru.parameters())),
            "init_s": int(self.init_s.numel()),
        }


def make_heads(device=None, dtype=None):
    if device is None:
        device = get_device()
    if dtype is None:
        dtype = model_dtype(device)
    heads = nn.ModuleList([nn.Linear(N_SLOTS * LATENT_DIM, CITY_CLASSES) for _ in range(N_SLOTS)])
    return heads.to(device=device, dtype=dtype)


class FactCache:
    def __init__(self, root=DEFAULT_CACHE):
        self.root = Path(root)
        self.manifest = {}
        mp = self.root / "manifest.json"
        if mp.exists():
            self.manifest = json.loads(mp.read_text(encoding="utf-8"))
        self.data = {}
        for split in ("train", "val", "testA", "testB"):
            npz_path = self.root / f"{split}_data.npz"
            if not npz_path.exists():
                npz_path = self.root / split / "split_data.npz"
            if not npz_path.exists():
                npz_path = self.root / split / "data.npz"
            if not npz_path.exists():
                continue
            z = np.load(npz_path, allow_pickle=False)
            self.data[split] = {k: z[k] for k in z.files}
        if not self.data:
            raise FileNotFoundError("No LS2 fact-only cache split_data.npz files found")
        self.persons = self._load_persons()

    def _load_persons(self):
        candidates = list(self.root.glob("*.jsonl")) + list(self.root.glob("**/*.jsonl"))
        out = {}
        for path in candidates:
            try:
                with path.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        eid = int(obj.get("episode_id", idx))
                        persons = obj.get("persons", obj.get("fact_persons", obj.get("people", [])))
                        if isinstance(persons, list):
                            out[eid] = [str(x) for x in persons]
            except Exception:
                continue
        return out

    def split_size(self, split):
        return int(len(self.data[split]["N"]))

    def rows_for_capacity(self, split, capacity):
        return np.flatnonzero(self.data[split]["N"] == capacity).astype(np.int64)

    def target_positions(self, split, rows):
        d = self.data[split]
        if "query_idx" in d:
            return d["query_idx"][rows].astype(np.int64)
        rng = np.random.default_rng(2237)
        pos = rng.integers(0, np.maximum(d["N"], 1))
        return pos[rows].astype(np.int64)

    def batch(self, split, rows, capacity, device):
        d = self.data[split]
        rows = np.asarray(rows, dtype=np.int64)
        offsets = d["fact_offsets"]
        starts = offsets[rows]
        # fact_offsets has one terminal entry in the existing LS2 cache.
        ends = offsets[rows + 1]
        facts = np.stack([d["fact_H"][int(s):int(e)] for s, e in zip(starts, ends)], axis=0)
        labels = np.stack([d["fact_Y"][int(s):int(e)] for s, e in zip(starts, ends)], axis=0)
        if facts.shape[1] != capacity:
            raise ValueError("capacity mismatch in cache batch")
        positions = self.target_positions(split, rows)
        y = labels[np.arange(len(rows)), positions]
        return (
            torch.from_numpy(facts).to(device=device),
            torch.from_numpy(labels.astype(np.int64)).to(device=device),
            torch.from_numpy(positions.astype(np.int64)).to(device=device),
            torch.from_numpy(y.astype(np.int64)).to(device=device),
        )


def select_head_logits(heads, state, positions):
    flat = state.flatten(1)
    out = []
    for p in range(N_SLOTS):
        mask = positions == p
        if mask.any():
            out.append((mask, heads[p](flat[mask])))
    return out


@torch.no_grad()
def eval_capacity(model, heads, cache, split, capacity, batch_size=128, fixed_position=None, rows=None):
    model.eval()
    if rows is None:
        rows = cache.rows_for_capacity(split, capacity)
    rows = np.asarray(rows, dtype=np.int64)
    if fixed_position is not None:
        rows = rows[cache.target_positions(split, rows) >= 0]
    correct = 0
    total = 0
    by_pos = [0 for _ in range(capacity)]
    pos_total = [0 for _ in range(capacity)]
    for i in range(0, len(rows), batch_size):
        rb = rows[i:i + batch_size]
        facts, labels, positions, y = cache.batch(split, rb, capacity, next(model.parameters()).device)
        state, _, _ = model.forward_facts(facts)
        if fixed_position is not None:
            positions = torch.full_like(positions, int(fixed_position))
            y = labels[:, int(fixed_position)]
        for mask, logits in select_head_logits(heads, state, positions):
            pred = logits.argmax(dim=1)
            yy = y[mask]
            pp = positions[mask]
            correct += int((pred == yy).sum())
            total += int(mask.sum())
            for p in range(capacity):
                pm = pp == p
                pos_total[p] += int(pm.sum())
                by_pos[p] += int((pred[pm] == yy[pm]).sum())
    acc = correct / max(total, 1)
    return {"acc": acc, "correct": correct, "total": total,
            "by_position": [by_pos[i] / max(pos_total[i], 1) for i in range(capacity)],
            "position_total": pos_total}


@torch.no_grad()
def fixed_position_matrix(model, heads, cache, split, capacity, batch_size=128, rows=None):
    return [eval_capacity(model, heads, cache, split, capacity, batch_size=batch_size,
                          fixed_position=p, rows=rows) for p in range(capacity)]


def macro_capacity(metrics, capacities=(1, 2, 4)):
    vals = [metrics[str(c)]["acc"] for c in capacities if str(c) in metrics]
    return float(np.mean(vals)) if vals else 0.0


def save_checkpoint(path, model, heads, optimizer=None, step=0, stage="", metrics=None, extra=None):
    payload = {
        "model": model.state_dict(),
        "heads": heads.state_dict(),
        "step": int(step),
        "stage": stage,
        "metrics": metrics or {},
        "config": {
            "slots": N_SLOTS, "latent_dim": LATENT_DIM, "key_content_dim": KEY_CONTENT_DIM,
            "occupancy_dim": OCCUPANCY_DIM, "value_dim": VALUE_DIM, "temperature": TEMP,
            "alpha": ALPHA, "beta": BETA, "architecture": "v2.2-S-factorized-storage",
        },
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device=None):
    if device is None:
        device = get_device()
    ckpt = torch.load(path, map_location=device)
    model = FactorizedStorage(device=device)
    heads = make_heads(device=device, dtype=model_dtype(device))
    model.load_state_dict(ckpt["model"])
    heads.load_state_dict(ckpt["heads"])
    return model, heads, ckpt


def chance_rate():
    return 1.0 / CITY_CLASSES


def normalized_capacity(acc, acc1):
    ch = chance_rate()
    return (acc - ch) / max(acc1 - ch, 1e-8)


def json_dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(x):
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    raise TypeError(type(x).__name__)


def now():
    return time.perf_counter()
