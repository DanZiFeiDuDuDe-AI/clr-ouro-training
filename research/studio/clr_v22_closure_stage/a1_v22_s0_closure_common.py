import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v22_storage_common import (
    DEFAULT_CACHE, DEFAULT_OUT, FactCache, load_checkpoint, get_device,
    json_dump, seed_all,
)


DEFAULT_CLOSURE_OUT = r"D:\training\checkpoints\clr_v22_s0_closure"
CITY_CLASSES = 64
KEY_DIM = 31
VALUE_START = 32


def closure_paths():
    return Path(os.environ.get("V22_CLOSURE_OUT", DEFAULT_CLOSURE_OUT))


def load_v22(cache_root=None, checkpoint=None):
    if cache_root is None:
        cache_root = os.environ.get("V22_CACHE_DIR", DEFAULT_CACHE)
    if checkpoint is None:
        out = Path(os.environ.get("V22_OUT_DIR", r"D:\training\checkpoints\clr_v22_storage"))
        checkpoint = out / "best_s0.pt"
    device = get_device()
    model, heads, ckpt = load_checkpoint(checkpoint, device=device)
    model.eval(); heads.eval()
    for p in model.parameters(): p.requires_grad_(False)
    for p in heads.parameters(): p.requires_grad_(False)
    return FactCache(cache_root), model, heads, ckpt, device


def target_rows(cache, split):
    return cache.rows_for_capacity(split, 1)


def person_subset(cache, limit=64):
    train_names = [p for vals in split_persons(cache, "train").values() for p in vals]
    counts = {}
    for p in train_names: counts[p] = counts.get(p, 0) + 1
    names = sorted(counts, key=lambda p: (-counts[p], p))
    if len(names) < limit:
        names = sorted({p for split in ("train", "val", "testA", "testB")
                        for vals in split_persons(cache, split).values() for p in vals})
    return names[:limit], {name: i for i, name in enumerate(names[:limit])}


def person_label_array(cache, split, rows, subset_map, cap=1):
    vals_by_row = split_persons(cache, split)
    y = np.full((len(rows), cap), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        vals = vals_by_row.get(int(row), [])
        for p in range(min(cap, len(vals))):
            if vals[p] in subset_map:
                y[i, p] = subset_map[vals[p]]
    return y


def split_persons(cache, split):
    path = cache.root / (split + "_meta.jsonl")
    out = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    obj = json.loads(line)
                    vals = obj.get("persons", obj.get("fact_persons", []))
                    if isinstance(vals, list): out[idx] = [str(x) for x in vals]
                except Exception:
                    pass
    if not out:
        for idx, vals in cache.persons.items():
            out[int(idx)] = vals
    return out


def transition(model, init_state, h, route_mode="normal", value_mode="normal"):
    """Reproduce v2.2 one-fact update with frozen parameters and interventions."""
    h = h.to(device=init_state.device, dtype=init_state.dtype)
    kf = model.key_read(h)
    vf = model.value_read(h)
    key = init_state[:, :, :KEY_DIM]
    occ = init_state[:, :, KEY_DIM].clamp(0.0, 1.0)
    val = init_state[:, :, VALUE_START:]
    match = (F.normalize(key, dim=-1, eps=1e-6) * F.normalize(kf, dim=-1, eps=1e-6).unsqueeze(1)).sum(-1)
    score = match - occ
    normal_route = F.softmax(score / 0.5, dim=1)
    if route_mode == "slot0":
        route = torch.zeros_like(normal_route); route[:, 0] = 1.0
    elif route_mode == "uniform":
        route = torch.full_like(normal_route, 0.25)
    else:
        route = normal_route
    key_in = kf.unsqueeze(1).expand(-1, 4, -1).reshape(-1, KEY_DIM)
    val_in = vf.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 32)
    key_candidate = model.key_gru(key_in, key.reshape(-1, KEY_DIM)).view(-1, 4, KEY_DIM)
    value_candidate = model.value_gru(val_in, val.reshape(-1, 32)).view(-1, 4, 32)
    new_key = (1.0 - route.unsqueeze(-1)) * key + route.unsqueeze(-1) * key_candidate
    new_val = (1.0 - route.unsqueeze(-1)) * val + route.unsqueeze(-1) * value_candidate
    if value_mode == "bypass":
        bypass = torch.zeros_like(new_val)
        bypass[:, 0] = vf
        new_val = bypass
    new_occ = (1.0 - route) * occ + route
    state = torch.cat([new_key, new_occ.unsqueeze(-1), new_val], dim=-1)
    return {
        "state": state, "key_read": kf, "value_read": vf,
        "key_candidate": key_candidate, "value_candidate": value_candidate,
        "route": route, "normal_route": normal_route, "match": match,
        "occupancy": new_occ,
    }


@torch.no_grad()
def collect_split(cache, model, split, rows, device, batch_size=256):
    all_out = {k: [] for k in ["H", "key_read", "value_read", "key_candidate", "value_candidate",
                               "state", "slot0", "route", "force_slot0", "force_uniform",
                               "bypass", "zero_init", "random_init", "labels", "persons"]}
    for i in range(0, len(rows), batch_size):
        rb = rows[i:i + batch_size]
        facts, labels, _, _ = cache.batch(split, rb, 1, device)
        h = facts[:, 0]
        init = model.init_state(len(rb), device)
        normal = transition(model, init, h)
        slot0 = normal["state"][:, 0]
        zero = transition(model, torch.zeros_like(init), h)
        # Norm-matched random init, fixed seed comes from the outer seed.
        rnd = torch.randn_like(init)
        ref_norm = init.norm(dim=-1, keepdim=True)
        rnd = rnd / (rnd.norm(dim=-1, keepdim=True) + 1e-6) * ref_norm
        random_init = transition(model, rnd, h)
        force0 = transition(model, init, h, route_mode="slot0")
        uniform = transition(model, init, h, route_mode="uniform")
        bypass = transition(model, init, h, value_mode="bypass")
        all_out["H"].append(h.float().cpu().numpy())
        for k in ["key_read", "value_read", "key_candidate", "value_candidate"]:
            all_out[k].append(normal[k].float().cpu().numpy())
        all_out["state"].append(normal["state"].float().cpu().numpy())
        all_out["slot0"].append(slot0.float().cpu().numpy())
        all_out["route"].append(normal["route"].float().cpu().numpy())
        for name, obj in [("force_slot0", force0), ("force_uniform", uniform), ("bypass", bypass),
                          ("zero_init", zero), ("random_init", random_init)]:
            all_out[name].append(obj["state"].float().cpu().numpy())
        all_out["labels"].append(labels[:, 0].cpu().numpy())
        # Filled after concatenation by person_label_array.
    for k in all_out:
        if all_out[k]: all_out[k] = np.concatenate(all_out[k], axis=0)
    return all_out


def collect_all(cache, model, device):
    names, subset = person_subset(cache)
    result = {"person_subset": names, "splits": {}}
    for split in ("train", "val", "testA", "testB"):
        rows = target_rows(cache, split)
        rep = collect_split(cache, model, split, rows, device)
        rep["persons"] = person_label_array(cache, split, rows, subset)
        result["splits"][split] = rep
    return result


def save_representations(path, data):
    arrays = {"person_subset": np.asarray(data["person_subset"], dtype="U128")}
    for split, rep in data["splits"].items():
        for k, v in rep.items():
            arrays[split + "__" + k] = np.asarray(v)
    np.savez_compressed(path, **arrays)


def load_representations(path):
    z = np.load(path, allow_pickle=False)
    out = {"person_subset": z["person_subset"].tolist(), "splits": {}}
    for key in z.files:
        if key == "person_subset": continue
        split, name = key.split("__", 1)
        out["splits"].setdefault(split, {})[name] = z[key]
    return out


class LinearProbe(nn.Module):
    def __init__(self, in_dim, n_classes=64):
        super().__init__(); self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x): return self.fc(x)


class BottleneckBaseline(nn.Module):
    def __init__(self, in_dim, bottleneck, n_classes=64):
        super().__init__(); self.down = nn.Linear(in_dim, bottleneck); self.out = nn.Linear(bottleneck, n_classes)
    def forward(self, x): return self.out(self.down(x))


class SingleSlotBaseline(nn.Module):
    def __init__(self):
        super().__init__(); self.read = nn.Linear(2048, 64); self.gru = nn.GRUCell(64, 64); self.head = nn.Linear(64, 64)
    def forward(self, x):
        h = self.read(x); s = self.gru(h, torch.zeros_like(h)); return self.head(s), s


def accuracy(model, x, y, device, batch_size=128):
    model.eval(); good = total = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).float().to(device)
            yb = torch.from_numpy(y[i:i + batch_size]).long().to(device)
            logits = model(xb)
            if isinstance(logits, tuple): logits = logits[0]
            good += int((logits.argmax(1) == yb).sum()); total += len(yb)
    return good / max(total, 1)


def train_classifier(model, x_train, y_train, x_val, y_val, device, lr=1e-3,
                     max_epochs=100, patience=10, seed=42, save_path=None):
    seed_all(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    best = -1.0; best_state = None; stale = 0
    xtr = torch.from_numpy(x_train).float().to(device); ytr = torch.from_numpy(y_train).long().to(device)
    for epoch in range(1, max_epochs + 1):
        model.train(); perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]; opt.zero_grad(set_to_none=True)
            out = model(xtr[idx]); logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.float(), ytr[idx]); loss.backward(); opt.step()
        val = accuracy(model, x_val, y_val, device)
        if val > best:
            best = val; stale = 0; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    if best_state is not None: model.load_state_dict(best_state)
    if save_path is not None: torch.save({"model": model.state_dict(), "val_acc": best}, save_path)
    return model, {"best_val_acc": best, "epochs": epoch}


def probe_one(data, name, in_dim, n_classes=64, person=False, device=None, out_path=None):
    if device is None: device = get_device()
    tr, va, ta, tb = [data["splits"][s] for s in ("train", "val", "testA", "testB")]
    ykey = "persons" if person else "labels"
    mask = {s: ((rep[ykey][:, 0] if rep[ykey].ndim > 1 else rep[ykey]) >= 0)
            for s, rep in [("train", tr), ("val", va), ("testA", ta), ("testB", tb)]}
    if person:
        # Keep the exact same feature/label subset across splits where available.
        tr_x, va_x, ta_x, tb_x = [rep[name][mask[s]] for s, rep in [("train", tr), ("val", va), ("testA", ta), ("testB", tb)]]
        tr_y, va_y, ta_y, tb_y = [rep[ykey][mask[s], 0] for s, rep in [("train", tr), ("val", va), ("testA", ta), ("testB", tb)]]
    else:
        tr_x, va_x, ta_x, tb_x = tr[name], va[name], ta[name], tb[name]
        tr_y, va_y, ta_y, tb_y = tr[ykey], va[ykey], ta[ykey], tb[ykey]
    model, info = train_classifier(LinearProbe(in_dim, n_classes), tr_x, tr_y, va_x, va_y, device, save_path=out_path)
    return {"name": name, "kind": "person" if person else "city", "dim": in_dim,
            "train_acc": accuracy(model, tr_x, tr_y, device), "val_acc": info["best_val_acc"],
            "testA_acc": accuracy(model, ta_x, ta_y, device), "testB_acc": accuracy(model, tb_x, tb_y, device),
            "epochs": info["epochs"], "n_train": len(tr_y), "n_testA": len(ta_y), "n_testB": len(tb_y)}


def geometry_for(rep, labels):
    x = rep.astype(np.float32); n = len(x); rng = np.random.default_rng(42)
    if n > 2000: idx = rng.choice(n, size=2000, replace=False); x = x[idx]; labels = labels[idx]; n = len(x)
    norms = np.linalg.norm(x, axis=1)
    same = []; diff = []
    for _ in range(min(100000, n * 20)):
        i = int(rng.integers(n)); j = int(rng.integers(n));
        c = float(np.dot(x[i], x[j]) / (norms[i] * norms[j] + 1e-8))
        (same if labels[i] == labels[j] else diff).append(c)
    return {"mean_norm": float(norms.mean()), "inter_sample_cosine": float(np.mean(diff + same)),
            "same_city_cosine": float(np.mean(same)) if same else None,
            "different_city_cosine": float(np.mean(diff)) if diff else None,
            "margin_same_minus_diff": (float(np.mean(same) - np.mean(diff)) if same and diff else None)}
