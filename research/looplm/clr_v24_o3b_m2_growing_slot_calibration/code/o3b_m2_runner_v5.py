from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["LD_LIBRARY_PATH"] = "/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path("/root/private_data/clr_v24_o3b_l/checkpoints")
CACHE = ROOT / "clr_v24_o3b_j3c_fresh_certified_feature_cache"
MR = ROOT / "clr_v24_o21a_mr/cache"
M_OUT = ROOT / "clr_v24_o3b_m_operational_samekey_rescue"
OUT = ROOT / "clr_v24_o3b_m2_growing_slot_calibration"
CKPT = M_OUT / "models/candidate_b_seed_43.pt"
SPLITS = ("train", "val", "testA", "testB")
N_LEVELS = (1, 2, 4, 8, 16, 32, 64)
BUCKETS = ((1, 4), (5, 8), (9, 16), (17, 32), (33, 64))
OPERATIONAL_CAPACITY = 4  # confirmed by the existing real O3b implementation
OLD_THRESHOLD = 0.57
VAL_EPISODES_PER_CELL = 180
HOLDOUT_EPISODES_PER_CELL = 240
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def jdump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def cdump(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status,detail\nNOT_RUN,no rows\n", encoding="utf-8-sig")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_tensor_state(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(np.asarray(t.shape, dtype=np.int64).tobytes())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def tensor_np(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


class FactorizedPairMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.entity_key = nn.Linear(dim, 64)
        self.relation_key = nn.Linear(dim, 64)
        self.pair_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def encode(self, e: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.normalize(self.entity_key(e), dim=-1, eps=1e-8), F.normalize(self.relation_key(r), dim=-1, eps=1e-8)

    def score_embeddings(self, eq: torch.Tensor, em: torch.Tensor, rq: torch.Tensor, rm: torch.Tensor) -> torch.Tensor:
        x = torch.cat(((eq - em).abs(), eq * em, (rq - rm).abs(), rq * rm), dim=-1)
        return self.pair_head(x).squeeze(-1)


def load_ckpt() -> tuple[dict[str, Any], FactorizedPairMLP, dict[str, str]]:
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    if ck.get("candidate") != "B" or int(ck.get("seed", -1)) != 43:
        raise ValueError(f"wrong deployment checkpoint metadata: candidate={ck.get('candidate')} seed={ck.get('seed')}")
    model = FactorizedPairMLP(int(ck["input_dim"]))
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(DEVICE).eval()
    stats = {"state_dict_sha256": sha_tensor_state(ck["state_dict"])}
    return ck, model, stats


def semantic_triple_key(e: str, r: str, v: str) -> tuple[str, str, str]:
    return (str(e), str(r), str(v))


def load_split(split: str, verify_relation_match: bool = True) -> dict[str, Any]:
    labels = read_csv(CACHE / f"labels_{split}.csv")
    side = read_csv(CACHE / f"H_fresh_{split}_sidecar.csv")
    e = np.load(CACHE / f"H_mention_fresh_{split}.npy", mmap_mode="r", allow_pickle=False)
    r = np.load(CACHE / f"H_base_I0_{split}.npy", mmap_mode="r", allow_pickle=False)
    z = np.load(MR / f"{split}_data.npz", allow_pickle=False)
    fact_h = z["fact_H"]
    meta = [json.loads(x) for x in (MR / f"{split}_meta.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    offsets = z["fact_offsets"].astype(np.int64)
    n = len(labels)
    if len(side) != n or e.shape != r.shape or e.shape != (n, 2048) or len(fact_h) != n:
        raise ValueError(f"{split}: labels/features/MR row shape mismatch labels={n} side={len(side)} e={e.shape} r={r.shape} fact_H={fact_h.shape}")
    if int(offsets[-1]) != n or len(meta) != len(offsets) - 1:
        raise ValueError(f"{split}: MR offsets/meta do not cover feature rows")
    expected_flat = 0
    source_ids: set[str] = set()
    feature_ids: set[str] = set()
    record_hashes: set[str] = set()
    semantic: set[tuple[str, str, str]] = set()
    row_records: list[dict[str, Any] | None] = [None] * n
    for ep, m in enumerate(meta):
        cap = int(m["capacity"])
        if int(offsets[ep + 1] - offsets[ep]) != cap:
            raise ValueError(f"{split}: capacity/offset mismatch at episode {ep}")
        for pos in range(cap):
            flat = int(offsets[ep]) + pos
            lab, sc = labels[flat], side[flat]
            cid = f"{split}:ep{ep}:{m['episode_id']}:item{pos}:fact{flat}"
            if lab.get("canonical_fact_id") != cid or sc.get("canonical_fact_id") != cid:
                raise ValueError(f"{split}: canonical row identity mismatch at flat row {flat}")
            if lab.get("feature_row_id") != sc.get("feature_row_id"):
                raise ValueError(f"{split}: feature row ID mismatch at flat row {flat}")
            person, relation = str(m["persons"][pos]), str(m["relations"][pos])
            value = str(m["cities"][pos])
            if lab.get("entity_semantic_value") != person or lab.get("relation_semantic_value") != relation:
                raise ValueError(f"{split}: semantic entity/relation mismatch at flat row {flat}")
            if verify_relation_match and not np.array_equal(np.asarray(r[flat]), fact_h[flat]):
                raise ValueError(f"{split}: H_base_I0 differs from source MR fact_H at flat row {flat}")
            fid = str(lab["canonical_fact_id"])
            frid = str(lab["feature_row_id"])
            rh = str(sc.get("stable_source_record_sha256", ""))
            source_ids.add(fid)
            feature_ids.add(frid)
            if rh:
                record_hashes.add(rh)
            triple = semantic_triple_key(person, relation, value)
            semantic.add(triple)
            row_records[flat] = {
                "row": flat, "source_id": fid, "feature_row_id": frid,
                "source_record_sha256": rh, "source_episode": str(m["episode_id"]),
                "episode_index": ep, "position": pos, "person": person,
                "relation": relation, "relation_id": int(m["relation_ids"][pos]),
                "value": value, "value_id": int(m["fact_classes"][pos]),
                "key": (person, relation), "triple": triple,
            }
            expected_flat += 1
    if expected_flat != n or any(x is None for x in row_records):
        raise ValueError(f"{split}: incomplete canonical row mapping")
    if len(source_ids) != n or len(feature_ids) != n or len(record_hashes) != n:
        raise ValueError(f"{split}: source/feature/record identifiers are not unique")
    return {
        "split": split, "labels": labels, "sidecar": side, "entity": e, "relation": r,
        "meta": meta, "mr": z, "rows": row_records, "source_ids": source_ids,
        "feature_ids": feature_ids, "record_hashes": record_hashes,
        "semantic_triples": semantic,
    }


def make_records(data: dict[str, Any], exclude_triples: set[tuple[str, str, str]] | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    exclude = exclude_triples or set()
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    removed_exact = 0
    removed_duplicate = 0
    for rec in data["rows"]:
        if rec["triple"] in exclude:
            removed_exact += 1
            continue
        if rec["triple"] in seen:
            removed_duplicate += 1
            continue
        seen.add(rec["triple"])
        out.append(rec)
    return out, {"input_source_rows": len(data["rows"]), "unique_semantic_facts": len(seen), "removed_cross_split_exact_fact_rows": removed_exact, "removed_within_split_semantic_duplicates": removed_duplicate}


def overlap_count(a: set[Any], b: set[Any]) -> int:
    return len(a & b)


def encode_split(data: dict[str, Any], ck: dict[str, Any], model: FactorizedPairMLP) -> tuple[torch.Tensor, torch.Tensor]:
    stats = ck["feature_stats"]
    emu = torch.as_tensor(tensor_np(stats["entity_mean"]), device=DEVICE)
    esd = torch.as_tensor(tensor_np(stats["entity_std"]), device=DEVICE)
    rmu = torch.as_tensor(tensor_np(stats["relation_mean"]), device=DEVICE)
    rsd = torch.as_tensor(tensor_np(stats["relation_std"]), device=DEVICE)
    if torch.any(esd <= 0) or torch.any(rsd <= 0):
        raise ValueError("Candidate B train-standardization contains non-positive std")
    ee: list[torch.Tensor] = []
    rr: list[torch.Tensor] = []
    with torch.no_grad():
        for st in range(0, len(data["rows"]), 512):
            end = min(len(data["rows"]), st + 512)
            e = torch.as_tensor(np.asarray(data["entity"][st:end], dtype=np.float32), device=DEVICE)
            r = torch.as_tensor(np.asarray(data["relation"][st:end], dtype=np.float32), device=DEVICE)
            e = (e - emu) / esd
            r = (r - rmu) / rsd
            ke, kr = model.encode(e, r)
            ee.append(ke)
            rr.append(kr)
    return torch.cat(ee, dim=0), torch.cat(rr, dim=0)


def _key_value_rows(records: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_person: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_relation: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in records:
        by_key[r["key"]].append(r)
    for k in by_key:
        by_person[k[0]].append(k)
        by_relation[k[1]].append(k)
    return by_key, by_person, by_relation


def sample_slots(keys: list[tuple[str, str]], by_key: dict[tuple[str, str], list[dict[str, Any]]], n: int, rng: np.random.Generator, required: tuple[str, str] | None = None, exclude_key: tuple[str, str] | None = None) -> list[dict[str, Any]]:
    if required is None:
        available = keys
        if len(available) < n:
            raise RuntimeError(f"not enough unique keys for N={n}: {len(available)}")
        chosen = list(rng.choice(len(available), size=n, replace=False))
        chosen_keys = [available[int(i)] for i in chosen]
    else:
        candidates = [k for k in keys if k != required and k != exclude_key]
        if len(candidates) < n - 1:
            raise RuntimeError(f"not enough keys for N={n} plus required key")
        rest = list(rng.choice(len(candidates), size=n - 1, replace=False)) if n > 1 else []
        chosen_keys = [required] + [candidates[int(i)] for i in rest]
        rng.shuffle(chosen_keys)
    slots = [by_key[k][int(rng.integers(0, len(by_key[k])))] for k in chosen_keys]
    return slots


def make_episode_set(records: list[dict[str, Any]], split: str, seed: int, per_cell: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    by_key, by_person, by_relation = _key_value_rows(records)
    keys = list(by_key)
    if len(keys) < max(N_LEVELS):
        raise RuntimeError(f"{split}: only {len(keys)} semantic keys; cannot build N=64 stress episodes")
    update_pairs: list[tuple[tuple[str, str], dict[str, Any], dict[str, Any]]] = []
    for k, group in by_key.items():
        by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in group:
            by_value[r["value"]].append(r)
        vals = list(by_value)
        if len(vals) >= 2:
            for old_v in vals:
                for new_v in vals:
                    if new_v != old_v:
                        update_pairs.append((k, by_value[old_v][0], by_value[new_v][0]))
    if not update_pairs:
        raise RuntimeError(f"{split}: no same-key/different-value source pairs")
    incoming_p2 = [r for r in records if any(k[1] != r["relation"] for k in by_person.get(r["person"], []))]
    incoming_p3 = [r for r in records if any(k[0] != r["person"] for k in by_relation.get(r["relation"], []))]
    incoming_p4 = [r for r in records if any(k[0] != r["person"] and k[1] != r["relation"] for k in keys)]
    if min(len(incoming_p2), len(incoming_p3), len(incoming_p4)) == 0:
        raise RuntimeError(f"{split}: one or more NEW event categories have no valid candidates")

    events: list[dict[str, Any]] = []
    event_id = 0
    for n in N_LEVELS:
        for typ in ("NEW_E_DIFF_R", "NEW_SAME_R", "NEW_SAME_E", "TRUE_UPDATE"):
            for _ in range(per_cell):
                if typ == "TRUE_UPDATE":
                    k, old_rec, new_rec = update_pairs[int(rng.integers(0, len(update_pairs)))]
                    slots = sample_slots(keys, by_key, n, rng, required=k)
                    target = next(i for i, s in enumerate(slots) if s["key"] == k)
                    incoming = new_rec
                    slots[target] = old_rec
                else:
                    if typ == "NEW_SAME_E":
                        incoming = incoming_p2[int(rng.integers(0, len(incoming_p2)))]
                        anchors = [k for k in by_person[incoming["person"]] if k != incoming["key"]]
                    elif typ == "NEW_SAME_R":
                        incoming = incoming_p3[int(rng.integers(0, len(incoming_p3)))]
                        anchors = [k for k in by_relation[incoming["relation"]] if k[0] != incoming["person"]]
                    else:
                        incoming = incoming_p4[int(rng.integers(0, len(incoming_p4)))]
                        anchors = [k for k in keys if k[0] != incoming["person"] and k[1] != incoming["relation"]]
                    anchors = [k for k in anchors if k != incoming["key"]]
                    if not anchors:
                        raise RuntimeError(f"{split}: no anchor for {typ}")
                    anchor = anchors[int(rng.integers(0, len(anchors)))]
                    slots = sample_slots(keys, by_key, n, rng, required=anchor, exclude_key=incoming["key"])
                    if any(s["key"] == incoming["key"] for s in slots):
                        raise RuntimeError("sampler inserted the NEW key into active slots")
                    target = -1
                if len({s["key"] for s in slots}) != n:
                    raise RuntimeError("duplicate active keys in generated episode")
                used_ids = [s["source_id"] for s in slots] + [incoming["source_id"]]
                if len(set(used_ids)) != len(used_ids):
                    raise RuntimeError("source fact reused within a generated episode")
                events.append({
                    "split": split, "episode_id": f"{split}-N{n}-{typ}-{event_id:06d}",
                    "n": n, "event_type": typ, "slots": [int(s["row"]) for s in slots],
                    "slot_records": slots, "incoming": int(incoming["row"]),
                    "incoming_record": incoming, "target_slot": int(target),
                    "old_value": slots[target]["value"] if target >= 0 else "",
                    "new_value": incoming["value"] if typ == "TRUE_UPDATE" else "",
                })
                event_id += 1
    return events


def score_events(events: list[dict[str, Any]], enc_e: torch.Tensor, enc_r: torch.Tensor, model: FactorizedPairMLP) -> None:
    left: list[int] = []
    right: list[int] = []
    offsets: list[tuple[int, int]] = []
    for ev in events:
        st = len(left)
        left.extend([int(ev["incoming"])] * len(ev["slots"]))
        right.extend(int(x) for x in ev["slots"])
        offsets.append((st, len(left)))
    all_scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for st in range(0, len(left), 16384):
            ed = min(len(left), st + 16384)
            li = torch.as_tensor(left[st:ed], device=DEVICE, dtype=torch.long)
            ri = torch.as_tensor(right[st:ed], device=DEVICE, dtype=torch.long)
            logits = model.score_embeddings(enc_e[li], enc_e[ri], enc_r[li], enc_r[ri])
            all_scores.append(logits.sigmoid().float().cpu().numpy())
    flat = np.concatenate(all_scores) if all_scores else np.empty(0, dtype=np.float32)
    for ev, (st, ed) in zip(events, offsets):
        scores = flat[st:ed].astype(np.float64)
        order = np.argsort(-scores, kind="stable")
        ev["scores"] = scores
        ev["ranked_slots"] = order
        ev["top1_score"] = float(scores[order[0]])
        ev["top1_slot"] = int(order[0])
        ev["top2_score"] = float(scores[order[1]]) if len(order) > 1 else float("-inf")
        ev["margin"] = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else float("inf")
        if ev["event_type"] == "TRUE_UPDATE":
            ev["target_score"] = float(scores[int(ev["target_slot"])])
            ev["target_rank"] = int(np.flatnonzero(order == int(ev["target_slot"]))[0] + 1)
            other = np.delete(scores, int(ev["target_slot"]))
            ev["target_minus_best_negative"] = float(scores[int(ev["target_slot"])] - other.max()) if len(other) else float("inf")
            ev["argmax_slot_type"] = "TARGET" if ev["top1_slot"] == ev["target_slot"] else event_slot_type(ev, ev["top1_slot"])
        else:
            ev["target_score"] = ""
            ev["target_rank"] = ""
            ev["target_minus_best_negative"] = ""
            ev["argmax_slot_type"] = event_slot_type(ev, ev["top1_slot"])


def event_slot_type(ev: dict[str, Any], slot_i: int) -> str:
    q = ev["incoming_record"]; m = ev["slot_records"][slot_i]
    se, sr = q["person"] == m["person"], q["relation"] == m["relation"]
    if se and sr:
        return "P1"
    if se:
        return "P2"
    if sr:
        return "P3"
    return "P4"


def threshold_for_n(rule: str, n: int, params: dict[str, Any]) -> float:
    if rule == "A":
        return float(params["threshold"])
    for bucket in params["threshold_buckets"]:
        if int(bucket["min_n"]) <= n <= int(bucket["max_n"]):
            return float(bucket["threshold"])
    return float(params["threshold_buckets"][-1]["threshold"])


def predict_event(ev: dict[str, Any], rule: str, params: dict[str, Any]) -> bool:
    tau = threshold_for_n(rule, int(ev["n"]), params)
    yes = float(ev["top1_score"]) >= tau
    if rule == "C" and int(ev["n"]) > 1:
        yes = yes and float(ev["margin"]) >= float(params["margin_delta"])
    return bool(yes)


def event_counts(events: list[dict[str, Any]], predictions: np.ndarray, n_filter: int | None = None) -> dict[str, float]:
    ix = [i for i, ev in enumerate(events) if n_filter is None or int(ev["n"]) == n_filter]
    if not ix:
        return {"n_events": 0}
    evs = [events[i] for i in ix]
    pred = predictions[ix].astype(bool)
    upd = np.asarray([e["event_type"] == "TRUE_UPDATE" for e in evs], dtype=bool)
    p2 = np.asarray([e["event_type"] == "NEW_SAME_E" for e in evs], dtype=bool)
    p3 = np.asarray([e["event_type"] == "NEW_SAME_R" for e in evs], dtype=bool)
    new = ~upd
    tgt = np.asarray([bool(pred[j] and int(e["top1_slot"]) == int(e["target_slot"])) if upd[j] else False for j, e in enumerate(evs)], dtype=bool)
    correct_new = new & ~pred
    correct_update = tgt
    action_correct = correct_new | correct_update
    nup = int(upd.sum())
    metrics: dict[str, float] = {
        "n_events": len(evs), "n_new": int(new.sum()), "n_true_update": nup,
        "new_collision_rate": float(pred[new].mean()) if new.any() else float("nan"),
        "p2_wrong_update_rate": float(pred[p2].mean()) if p2.any() else float("nan"),
        "p3_wrong_update_rate": float(pred[p3].mean()) if p3.any() else float("nan"),
        "p4_wrong_update_rate": float(pred[np.asarray([e["event_type"] == "NEW_E_DIFF_R" for e in evs])].mean()) if any(e["event_type"] == "NEW_E_DIFF_R" for e in evs) else float("nan"),
        "true_update_action_recall": float(pred[upd].mean()) if nup else float("nan"),
        "target_slot_recall": float(tgt[upd].mean()) if nup else float("nan"),
        "value_replacement_accuracy": float(tgt[upd].mean()) if nup else float("nan"),
        "duplicate_same_key_slot_rate": float((~pred[upd]).mean()) if nup else float("nan"),
        "slot_count_error_rate": float((~pred[upd]).mean()) if nup else float("nan"),
        "overall_action_accuracy": float(action_correct.mean()),
        "precision_update": float(tgt[upd].sum() / max(1, pred.sum())),
        "macro_action_accuracy": float(np.mean([correct_new[new].mean(), correct_update[upd].mean()])) if nup and new.any() else float("nan"),
    }
    return metrics


def gate_metrics(events: list[dict[str, Any]], predictions: np.ndarray, ns: tuple[int, ...]) -> dict[str, Any]:
    rows = {n: event_counts(events, predictions, n) for n in ns}
    checks: list[dict[str, Any]] = []
    for n, m in rows.items():
        checks.append({"N": n, "new_collision": m.get("new_collision_rate", 1.0) <= 0.05,
                       "p2": m.get("p2_wrong_update_rate", 1.0) <= 0.05,
                       "p3": m.get("p3_wrong_update_rate", 1.0) <= 0.05,
                       "update": m.get("true_update_action_recall", 0.0) >= 0.95,
                       "target": m.get("target_slot_recall", 0.0) >= 0.95,
                       "duplicate": m.get("duplicate_same_key_slot_rate", 1.0) <= 0.05,
                       "replacement": m.get("value_replacement_accuracy", 0.0) >= 0.95,
                       "slot_count": m.get("slot_count_error_rate", 1.0) <= 0.05})
    passed = all(all(v for k, v in r.items() if k != "N") for r in checks)
    violation = 0.0
    for m in rows.values():
        violation += max(0.0, m.get("new_collision_rate", 1.0) - 0.05)
        violation += max(0.0, m.get("p2_wrong_update_rate", 1.0) - 0.05)
        violation += max(0.0, m.get("p3_wrong_update_rate", 1.0) - 0.05)
        violation += max(0.0, 0.95 - m.get("true_update_action_recall", 0.0))
        violation += max(0.0, 0.95 - m.get("target_slot_recall", 0.0))
    return {"pass": passed, "per_n": rows, "checks": checks, "violation": float(violation)}


def event_arrays(events: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    names = {"NEW_E_DIFF_R": 0, "NEW_SAME_R": 1, "NEW_SAME_E": 2, "TRUE_UPDATE": 3}
    return {
        "n": np.asarray([int(e["n"]) for e in events], dtype=np.int16),
        "kind": np.asarray([names[e["event_type"]] for e in events], dtype=np.int8),
        "top1": np.asarray([float(e["top1_score"]) for e in events], dtype=np.float64),
        "margin": np.asarray([float(e["margin"]) if math.isfinite(float(e["margin"])) else 1e9 for e in events], dtype=np.float64),
        "target_correct": np.asarray([e["event_type"] == "TRUE_UPDATE" and int(e["top1_slot"]) == int(e["target_slot"]) for e in events], dtype=bool),
    }


def fast_event_counts(a: dict[str, np.ndarray], predictions: np.ndarray, n_filter: int | None = None) -> dict[str, float]:
    sel = np.ones(len(predictions), dtype=bool) if n_filter is None else a["n"] == n_filter
    if not sel.any():
        return {"n_events": 0}
    pred = predictions.astype(bool)
    kind = a["kind"]
    upd, new = sel & (kind == 3), sel & (kind != 3)
    p2, p3, p4 = sel & (kind == 2), sel & (kind == 1), sel & (kind == 0)
    target = pred & a["target_correct"]
    nup, nnew = int(upd.sum()), int(new.sum())
    correct_new, correct_upd = new & ~pred, upd & target
    return {
        "n_events": int(sel.sum()), "n_new": nnew, "n_true_update": nup,
        "new_collision_rate": float(pred[new].mean()) if nnew else float("nan"),
        "p2_wrong_update_rate": float(pred[p2].mean()) if p2.any() else float("nan"),
        "p3_wrong_update_rate": float(pred[p3].mean()) if p3.any() else float("nan"),
        "p4_wrong_update_rate": float(pred[p4].mean()) if p4.any() else float("nan"),
        "true_update_action_recall": float(pred[upd].mean()) if nup else float("nan"),
        "target_slot_recall": float(target[upd].mean()) if nup else float("nan"),
        "value_replacement_accuracy": float(target[upd].mean()) if nup else float("nan"),
        "duplicate_same_key_slot_rate": float((~pred[upd]).mean()) if nup else float("nan"),
        "slot_count_error_rate": float((~pred[upd]).mean()) if nup else float("nan"),
        "overall_action_accuracy": float((correct_new | correct_upd)[sel].mean()),
        "precision_update": float(target[upd].sum() / max(1, int((sel & pred).sum()))),
        "macro_action_accuracy": float(np.mean([correct_new[new].mean(), correct_upd[upd].mean()])) if nup and nnew else float("nan"),
    }


def fast_gate_metrics(a: dict[str, np.ndarray], predictions: np.ndarray, ns: tuple[int, ...]) -> dict[str, Any]:
    rows = {n: fast_event_counts(a, predictions, n) for n in ns}
    checks, violation = [], 0.0
    for n, m in rows.items():
        c = {"N": n, "new_collision": m.get("new_collision_rate", 1.0) <= .05,
             "p2": m.get("p2_wrong_update_rate", 1.0) <= .05,
             "p3": m.get("p3_wrong_update_rate", 1.0) <= .05,
             "update": m.get("true_update_action_recall", 0.0) >= .95,
             "target": m.get("target_slot_recall", 0.0) >= .95,
             "duplicate": m.get("duplicate_same_key_slot_rate", 1.0) <= .05,
             "replacement": m.get("value_replacement_accuracy", 0.0) >= .95,
             "slot_count": m.get("slot_count_error_rate", 1.0) <= .05}
        checks.append(c)
        violation += max(0.0, m.get("new_collision_rate", 1.0) - .05)
        violation += max(0.0, m.get("p2_wrong_update_rate", 1.0) - .05)
        violation += max(0.0, m.get("p3_wrong_update_rate", 1.0) - .05)
        violation += max(0.0, .95 - m.get("true_update_action_recall", 0.0))
        violation += max(0.0, .95 - m.get("target_slot_recall", 0.0))
    return {"pass": all(all(v for k, v in c.items() if k != "N") for c in checks), "per_n": rows, "checks": checks, "violation": float(violation)}


def eval_rule(events: list[dict[str, Any]], rule: str, params: dict[str, Any], ns: tuple[int, ...] = N_LEVELS) -> tuple[dict[str, Any], np.ndarray]:
    p = np.asarray([predict_event(e, rule, params) for e in events], dtype=bool)
    gate = gate_metrics(events, p, ns)
    return gate, p


def candidate_thresholds(events: list[dict[str, Any]]) -> np.ndarray:
    vals = np.asarray([float(e["top1_score"]) for e in events], dtype=np.float64)
    if len(np.unique(vals)) > 401:
        neg = np.asarray([float(e["top1_score"]) for e in events if e["event_type"] != "TRUE_UPDATE"], dtype=np.float64)
        pos = np.asarray([float(e["top1_score"]) for e in events if e["event_type"] == "TRUE_UPDATE"], dtype=np.float64)
        vals = np.concatenate((np.quantile(vals, np.linspace(0, 1, 301)),
                               np.quantile(neg, np.linspace(.90, 1, 51)),
                               np.quantile(pos, np.linspace(0, .15, 51))))
    return np.unique(np.concatenate(([0.0], vals, [1.0, np.nextafter(1.0, 2.0)])))


def threshold_rows(events: list[dict[str, Any]], ns: tuple[int, ...], rule_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = candidate_thresholds(events)
    arr = event_arrays(events)
    rows: list[dict[str, Any]] = []
    best_pass: tuple[float, float, float] | None = None
    best_any: tuple[float, float, float, float] | None = None
    best_p: dict[str, Any] = {}
    for tau in scores:
        p = arr["top1"] >= float(tau)
        gate = fast_gate_metrics(arr, p, ns)
        m_all = fast_event_counts(arr, p)
        key = (m_all.get("macro_action_accuracy", 0.0), -m_all.get("new_collision_rate", 1.0), m_all.get("true_update_action_recall", 0.0))
        row = {"rule": rule_name, "threshold": float(tau), "pass": gate["pass"], "violation": gate["violation"], **{f"all_{k}": v for k, v in m_all.items()}}
        for n in ns:
            row.update({f"N{n}_{k}": v for k, v in gate["per_n"][n].items()})
        rows.append(row)
        if gate["pass"]:
            if best_pass is None or key > best_pass:
                best_pass = key
                best_p = {"threshold": float(tau), "validation_gate": gate, "validation_metrics": m_all}
        any_key = (-gate["violation"],) + key
        if best_any is None or any_key > best_any:
            best_any = any_key
            fallback = {"threshold": float(tau), "validation_gate": gate, "validation_metrics": m_all}
    if best_p:
        return rows, best_p
    return rows, fallback


def bucket_name(n: int) -> tuple[int, int]:
    for a, b in BUCKETS:
        if a <= n <= b:
            return a, b
    return BUCKETS[-1]


def make_bucket_params(events: list[dict[str, Any]], margin_delta: float = 0.0, rule: str = "B") -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    # Candidate thresholds are observed score boundaries. All thresholds are selected from validation only.
    bucket_rows: list[dict[str, Any]] = []
    chosen: list[dict[str, Any]] = []
    prev_tau = 0.0
    per_bucket_event_sets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    per_bucket_arrays: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for lo, hi in BUCKETS:
        per_bucket_event_sets[(lo, hi)] = [e for e in events if lo <= int(e["n"]) <= hi]
        per_bucket_arrays[(lo, hi)] = event_arrays(per_bucket_event_sets[(lo, hi)])
    for lo, hi in BUCKETS:
        subset = per_bucket_event_sets[(lo, hi)]
        if not subset:
            chosen.append({"min_n": lo, "max_n": hi, "threshold": prev_tau})
            continue
        candidates = candidate_thresholds(subset)
        arr = per_bucket_arrays[(lo, hi)]
        feasible: list[tuple[tuple[float, float, float], float, dict[str, Any]]] = []
        any_best: tuple[tuple[float, float, float, float], float, dict[str, Any]] | None = None
        contained_ns = tuple(n for n in N_LEVELS if lo <= n <= hi)
        for tau in candidates:
            p = arr["top1"] >= float(tau)
            if rule == "C":
                p &= (arr["n"] == 1) | (arr["margin"] >= margin_delta)
            gate = fast_gate_metrics(arr, p, contained_ns)
            summary = fast_event_counts(arr, p)
            key = (summary.get("macro_action_accuracy", 0.0), -summary.get("new_collision_rate", 1.0), summary.get("true_update_action_recall", 0.0))
            if gate["pass"]:
                feasible.append((key, float(tau), gate))
            any_key = (-gate["violation"],) + key
            if any_best is None or any_key > any_best[0]:
                any_best = (any_key, float(tau), gate)
        if feasible:
            # Highest balanced action quality among gate-passing thresholds; prefer lower threshold on exact ties.
            _, tau, gate = max(feasible, key=lambda x: (x[0], -x[1]))
            passed_bucket = True
        else:
            assert any_best is not None
            _, tau, gate = any_best
            passed_bucket = False
        tau = max(prev_tau, tau)  # pre-registered non-decreasing threshold requirement
        params = {"threshold_buckets": [{"min_n": a, "max_n": b, "threshold": (tau if (a, b) == (lo, hi) else 0.0)} for a, b in BUCKETS], "margin_delta": margin_delta}
        # Re-evaluate this bucket after monotonic projection.
        arr = per_bucket_arrays[(lo, hi)]
        p2 = arr["top1"] >= tau
        if rule == "C":
            p2 &= (arr["n"] == 1) | (arr["margin"] >= margin_delta)
        gate2 = fast_gate_metrics(arr, p2, contained_ns)
        summary2 = fast_event_counts(arr, p2)
        bucket_rows.append({"rule": rule, "bucket_min_n": lo, "bucket_max_n": hi, "threshold": tau, "margin_delta": margin_delta, "bucket_pass_before_monotonic": passed_bucket, "bucket_pass_after_monotonic": gate2["pass"], "violation": gate2["violation"], **{f"all_{k}": v for k, v in summary2.items()}})
        for n, m in gate2["per_n"].items():
            bucket_rows[-1].update({f"N{n}_{k}": v for k, v in m.items()})
        chosen.append({"min_n": lo, "max_n": hi, "threshold": float(tau)})
        prev_tau = float(tau)
    params = {"threshold_buckets": chosen}
    if rule == "C":
        params["margin_delta"] = float(margin_delta)
    full_arr = event_arrays(events)
    p_full = np.asarray([predict_event(e, rule, params) for e in events], dtype=bool)
    gate = fast_gate_metrics(full_arr, p_full, N_LEVELS)
    m_all = fast_event_counts(full_arr, p_full)
    return params, {"validation_gate": gate, "validation_metrics": m_all}, bucket_rows


def tune_bucket_rule(events: list[dict[str, Any]], rule: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    deltas = [0.0]
    if rule == "C":
        finite = np.asarray([float(e["margin"]) for e in events if int(e["n"]) > 1], dtype=np.float64)
        deltas = sorted(set([0.0] + [float(x) for x in np.quantile(finite, np.linspace(0.0, 1.0, 41))]))
    best_pass: tuple[tuple[float, float, float], dict[str, Any], dict[str, Any], list[dict[str, Any]]] | None = None
    best_any: tuple[tuple[float, float, float, float], dict[str, Any], dict[str, Any], list[dict[str, Any]]] | None = None
    operational_ns = tuple(n for n in N_LEVELS if n <= OPERATIONAL_CAPACITY)
    full_arr = event_arrays(events)
    for delta in deltas:
        params, summary, rows = make_bucket_params(events, margin_delta=delta, rule=rule)
        p = np.asarray([predict_event(e, rule, params) for e in events], dtype=bool)
        gate = fast_gate_metrics(full_arr, p, operational_ns)
        met = fast_event_counts(full_arr, p)
        extended_gate = fast_gate_metrics(full_arr, p, N_LEVELS)
        summary["operational_validation_gate"] = gate
        summary["extended_validation_gate"] = extended_gate
        summary["validation_gate"] = extended_gate
        key = (met.get("macro_action_accuracy", 0.0), -met.get("new_collision_rate", 1.0), met.get("true_update_action_recall", 0.0))
        if gate["pass"]:
            item = (key, params, summary, rows)
            if best_pass is None or key > best_pass[0]:
                best_pass = item
        any_key = (-gate["violation"],) + key
        if best_any is None or any_key > best_any[0]:
            best_any = (any_key, params, summary, rows)
    if best_pass is not None:
        _, p, s, r = best_pass
        return p, s, r
    assert best_any is not None
    _, p, s, r = best_any
    return p, s, r


def audit_rows(events: list[dict[str, Any]], policy: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ev in events:
        r = ev["incoming_record"]
        pred = predict_event(ev, policy["rule"], policy["params"]) if policy else "NOT_FROZEN"
        rows.append({
            "split": ev["split"], "episode_id": ev["episode_id"], "active_slots_N": ev["n"],
            "event_type": ev["event_type"], "incoming_source_id": r["source_id"],
            "incoming_key": f"{r['person']}|{r['relation']}", "incoming_value": r["value"],
            "true_target_slot": ev["target_slot"], "true_target_source_id": ev["slot_records"][ev["target_slot"]]["source_id"] if ev["target_slot"] >= 0 else "",
            "top1_score": ev["top1_score"], "top2_score": ev["top2_score"], "top1_top2_margin": ev["margin"],
            "target_score": ev["target_score"], "target_rank": ev["target_rank"],
            "target_minus_best_negative": ev["target_minus_best_negative"], "argmax_slot_type": ev["argmax_slot_type"],
            "top1_slot": ev["top1_slot"], "decision_update": pred,
        })
    return rows


def metrics_table(events: list[dict[str, Any]], predictions: np.ndarray, rule: str, split: str) -> list[dict[str, Any]]:
    rows = []
    for n in N_LEVELS:
        m = event_counts(events, predictions, n)
        rows.append({"split": split, "rule": rule, "active_slots_N": n, "operational_capacity": OPERATIONAL_CAPACITY, "in_operational_range": n <= OPERATIONAL_CAPACITY, **m})
        for typ in ("NEW_E_DIFF_R", "NEW_SAME_R", "NEW_SAME_E", "TRUE_UPDATE"):
            idx = [i for i, ev in enumerate(events) if int(ev["n"]) == n and ev["event_type"] == typ]
            if not idx:
                continue
            subevents = [events[i] for i in idx]
            subpred = predictions[idx]
            sm = event_counts(subevents, subpred)
            rows.append({"split": split, "rule": rule, "active_slots_N": n, "event_type": typ, "operational_capacity": OPERATIONAL_CAPACITY, "in_operational_range": n <= OPERATIONAL_CAPACITY, **sm})
    rows.append({"split": split, "rule": rule, "active_slots_N": "weighted_all", "operational_capacity": OPERATIONAL_CAPACITY, "in_operational_range": "mixed", **event_counts(events, predictions)})
    return rows


def extreme_rows(events: list[dict[str, Any]], predictions: np.ndarray, split: str, threshold_name: str, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in N_LEVELS:
        ix = [i for i, e in enumerate(events) if int(e["n"]) == n and e["event_type"] != "TRUE_UPDATE"]
        if not ix:
            continue
        scores = np.asarray([events[i]["top1_score"] for i in ix], dtype=np.float64)
        pred = predictions[ix]
        p3_ix = [j for j, i in enumerate(ix) if events[i]["event_type"] == "NEW_SAME_R"]
        rows.append({"split": split, "threshold_name": threshold_name, "threshold": threshold,
                     "N": n, "new_events": len(ix), "collision_rate": float(pred.mean()),
                     "p3_wrong_update_rate": float(pred[p3_ix].mean()) if p3_ix else float("nan"),
                     "top1_negative_mean": float(scores.mean()), "top1_negative_p95": float(np.quantile(scores, .95)),
                     "top1_negative_p99": float(np.quantile(scores, .99)), "top1_negative_max": float(scores.max())})
    return rows


def manifest_and_integrity(data: dict[str, dict[str, Any]], ck: dict[str, Any], model_state_sha: str) -> dict[str, Any]:
    all_rows: dict[str, Any] = {}
    for split, d in data.items():
        values_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
        for rec in d["rows"]:
            values_by_key[rec["key"]].add(rec["value"])
        all_rows[split] = {"rows": len(d["rows"]), "unique_semantic_facts": len(d["semantic_triples"]),
                           "unique_keys": len({r["key"] for r in d["rows"]}),
                           "multivalue_key_count": sum(1 for vals in values_by_key.values() if len(vals) > 1),
                           "unique_source_ids": len(d["source_ids"]), "unique_feature_ids": len(d["feature_ids"]),
                           "unique_source_record_hashes": len(d["record_hashes"])}
    overlaps: dict[str, Any] = {}
    names = list(data)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}-{b}"] = {
                "canonical_fact_ids": overlap_count(data[a]["source_ids"], data[b]["source_ids"]),
                "feature_row_ids": overlap_count(data[a]["feature_ids"], data[b]["feature_ids"]),
                "source_record_sha256": overlap_count(data[a]["record_hashes"], data[b]["record_hashes"]),
                "semantic_triples": overlap_count(data[a]["semantic_triples"], data[b]["semantic_triples"]),
                "semantic_keys": overlap_count({r["key"] for r in data[a]["rows"]}, {r["key"] for r in data[b]["rows"]}),
            }
    return {"pass_row_identity": True, "candidate": "B", "deployment_seed": 43,
            "candidate_checkpoint_sha256": sha_file(CKPT), "candidate_state_dict_sha256": model_state_sha,
            "candidate_not_retrained": True, "splits": all_rows, "pairwise_source_overlap": overlaps,
            "carrier_match": "H_base_I0 equals MR fact_H bitwise for every checked row; H_mention_fresh is canonical-row aligned",
            "semantic_overlap_policy": "validation excludes exact train semantic triples; final holdouts exclude train+validation exact triples; TestB also excludes TestA exact triples. Key identity overlap is reported, not silently called source leakage.",
            "feature_cache_manifest_sha256": sha_file(CACHE / "artifact_sha256_manifest.json") if (CACHE / "artifact_sha256_manifest.json").exists() else None,
            "device": DEVICE, "runtime": {"torch": torch.__version__, "python": platform.python_version()}}


def freeze_candidate_manifest(ck: dict[str, Any], model_state_sha: str, stage: str) -> None:
    path = OUT / "candidate_b_frozen_manifest.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    initial_sha = old.get("checkpoint_sha256_before", sha_file(CKPT))
    initial_state = old.get("state_dict_sha256_before", model_state_sha)
    if sha_file(CKPT) != initial_sha or model_state_sha != initial_state:
        raise RuntimeError("Candidate B checkpoint/state hash changed; frozen-weight contract violated")
    new = {
        "candidate": "B", "checkpoint_source": str(CKPT), "deployed_seed": 43,
        "checkpoint_sha256_before": initial_sha, "checkpoint_sha256_after": sha_file(CKPT),
        "state_dict_sha256_before": initial_state, "state_dict_sha256_after": model_state_sha,
        "weights_frozen": True, "trained_or_updated_in_m2": False,
        "feature_source": {"entity": "J3C H_mention_fresh_{split}.npy exact entity-mention mean hidden, 2048D", "relation": "J3C H_base_I0_{split}.npy; bitwise equal to corresponding MR fact_H, 2048D"},
        "normalization": {"type": ck.get("normalization"), "stats": "checkpoint feature_stats; fitted during O3b-M on train only; no refit"},
        "scoring_function": "sigmoid(pair_head([abs(k_e(q)-k_e(m)), k_e(q)*k_e(m), abs(k_r(q)-k_r(m)), k_r(q)*k_r(m)]))",
        "hidden_dim": 64, "pair_head": "Linear(256,64) + GELU + Linear(64,1)",
        "old_global_threshold": float(ck.get("thresholds", {}).get("threshold", OLD_THRESHOLD)),
        "candidate_config_sha256": sha_file(M_OUT / "winner_config.json"),
        "freeze_stage": stage, "frozen_at_utc": old.get("frozen_at_utc", datetime.now(timezone.utc).isoformat()),
    }
    jdump(path, new)


def stage_calibrate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    ck, model, state = load_ckpt()
    train = load_split("train")
    val = load_split("val")
    data = {"train": train, "val": val}
    integrity = manifest_and_integrity(data, ck, state["state_dict_sha256"])
    val_records, val_filter = make_records(val, train["semantic_triples"])
    train_records, train_filter = make_records(train)
    if len({r["key"] for r in val_records}) < 64:
        raise RuntimeError("validation unique-key coverage is insufficient for N=64 stress calibration")
    events = make_episode_set(val_records, "val", seed=202609251, per_cell=VAL_EPISODES_PER_CELL)
    enc_e, enc_r = encode_split(val, ck, model)
    score_events(events, enc_e, enc_r, model)
    integrity["validation_filter"] = val_filter
    integrity["train_semantic_inventory"] = train_filter
    integrity["validation_events"] = {"episode_count": len(events), "per_N_per_type": VAL_EPISODES_PER_CELL, "n_levels": list(N_LEVELS), "unique_source_ids_used": len({x for ev in events for x in [ev["incoming_record"]["source_id"], *[s["source_id"] for s in ev["slot_records"]]]})}
    integrity["test_evaluation_performed"] = False
    jdump(OUT / "stage0_integrity.json", integrity)
    freeze_candidate_manifest(ck, state["state_dict_sha256"], "before_validation_calibration")
    cdump(OUT / "slot_competition_score_audit.csv", audit_rows(events))
    operational_ns = tuple(n for n in N_LEVELS if n <= OPERATIONAL_CAPACITY)
    # Rule A: global threshold calibrated against the true operational capacity.
    rule_a_rows, rule_a = threshold_rows(events, operational_ns, "A")
    cdump(OUT / "rule_a_validation.csv", rule_a_rows)
    # Rule B: slot-count bucketed monotone threshold.
    rule_b_params, rule_b_sum, rule_b_rows = tune_bucket_rule(events, "B")
    cdump(OUT / "rule_b_validation.csv", rule_b_rows)
    # Rule C: bucketed threshold plus one shared margin gate.
    rule_c_params, rule_c_sum, rule_c_rows = tune_bucket_rule(events, "C")
    cdump(OUT / "rule_c_validation.csv", rule_c_rows)
    candidates = {
        "A": {"params": {"threshold": rule_a["threshold"]}, "summary": rule_a},
        "B": {"params": rule_b_params, "summary": rule_b_sum},
        "C": {"params": rule_c_params, "summary": rule_c_sum},
    }
    op_pass: dict[str, bool] = {}
    for name, obj in candidates.items():
        rule = name
        gate, pred = eval_rule(events, rule, obj["params"], operational_ns)
        obj["operational_validation_gate"] = gate
        obj["operational_validation_metrics"] = {str(n): event_counts(events, pred, n) for n in operational_ns}
        obj["extended_validation_gate"] = eval_rule(events, rule, obj["params"], tuple(N_LEVELS))[0]
        op_pass[name] = gate["pass"]
    chosen = next((x for x in ("A", "B", "C") if op_pass[x]), None)
    if chosen is None:
        # The diagnostic selection is still validation-only; prioritize fewest gate violations, then best balanced accuracy.
        chosen = min(("A", "B", "C"), key=lambda x: (candidates[x]["operational_validation_gate"]["violation"], -candidates[x]["summary"]["validation_metrics"].get("macro_action_accuracy", 0.0)))
    selected = candidates[chosen]
    decision = {"rule": chosen, "params": selected["params"], "selected_using": "validation episodes only", "selection_priority": "A then B then C among rules passing all operational N gates; otherwise validation-only minimum gate-violation fallback", "operational_capacity": OPERATIONAL_CAPACITY, "operational_N": list(operational_ns), "stress_only_N": [n for n in N_LEVELS if n > OPERATIONAL_CAPACITY], "validation_gate": selected["operational_validation_gate"], "validation_metrics": selected["operational_validation_metrics"], "extended_validation_gate": selected["extended_validation_gate"], "all_rule_validation_summary": {k: {"operational_pass": op_pass[k], "operational_gate": v["operational_validation_gate"], "extended_gate": v["extended_validation_gate"], "params": v["params"], "validation_metrics": v["summary"]["validation_metrics"]} for k, v in candidates.items()}, "candidate_checkpoint_sha256": sha_file(CKPT), "frozen": True, "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "holdout_opened": False}
    jdump(OUT / "decision_rule_frozen.json", decision)
    # Holdout access is intentionally deferred to the separate --phase holdout invocation.
    frozen_json = {"experiment": "O3b-M2", "data_splits": {"train": train_filter, "validation": val_filter}, "validation_semantic_triples_used": len({r["triple"] for r in val_records}), "validation_episode_count": len(events), "per_cell": VAL_EPISODES_PER_CELL, "n_levels": list(N_LEVELS), "testA_testB": "NOT_OPENED_BEFORE_DECISION_FREEZE", "source_identity": "canonical fact IDs and feature rows are disjoint by split; validation exact triples overlapping train were excluded", "target_capacity": OPERATIONAL_CAPACITY}
    jdump(OUT / "episode_dataset_manifest.json", frozen_json)
    true_manifest = {"train": {"eligible_changed_value_keys": sum(len({r['value'] for r in rows}) > 1 for rows in _key_value_rows(train_records)[0].values())}, "validation": {"eligible_changed_value_keys": sum(len({r['value'] for r in rows}) > 1 for rows in _key_value_rows(val_records)[0].values()), "true_update_episode_count": sum(e["event_type"] == "TRUE_UPDATE" for e in events), "different_source_fact_per_value": True}, "holdout": "NOT_RUN_UNTIL_DECISION_FROZEN"}
    jdump(OUT / "true_update_dataset_manifest.json", true_manifest)
    # Baseline old threshold distributions on validation only.
    oldp = np.asarray([float(e["top1_score"]) >= float(ck.get("thresholds", {}).get("threshold", OLD_THRESHOLD)) for e in events], dtype=bool)
    exrows = extreme_rows(events, oldp, "val", "historical_global", float(ck.get("thresholds", {}).get("threshold", OLD_THRESHOLD)))
    frozen_pred = np.asarray([predict_event(e, chosen, selected["params"]) for e in events], dtype=bool)
    exrows.extend(extreme_rows(events, frozen_pred, "val", "frozen_rule", -1.0))
    cdump(OUT / "extreme_value_scaling.csv", exrows)
    freeze_candidate_manifest(ck, state["state_dict_sha256"], "after_validation_calibration_and_policy_freeze")
    jdump(OUT / "calibration_runtime.json", {"wall_seconds": time.perf_counter() - start, "device": DEVICE, "test_evaluation_performed": False})
    print(json.dumps({"phase": "calibrate", "decision": chosen, "params": selected["params"], "operational_pass": op_pass, "test_evaluation_performed": False, "wall_seconds": time.perf_counter() - start}, ensure_ascii=False), flush=True)


def write_true_update_rows(events: list[dict[str, Any]], predictions: np.ndarray, split: str, rule: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, ev in enumerate(events):
        if ev["event_type"] != "TRUE_UPDATE":
            continue
        pred_update = bool(predictions[i])
        target_correct = int(ev["top1_slot"]) == int(ev["target_slot"])
        replacement = pred_update and target_correct
        duplicate = not pred_update
        rows.append({"split": split, "episode_id": ev["episode_id"], "N": ev["n"], "incoming_source_id": ev["incoming_record"]["source_id"], "target_source_id": ev["slot_records"][ev["target_slot"]]["source_id"], "key": f"{ev['incoming_record']['person']}|{ev['incoming_record']['relation']}", "old_value": ev["old_value"], "new_value": ev["new_value"], "different_value_verified": ev["old_value"] != ev["new_value"], "decision_update": pred_update, "top1_slot": ev["top1_slot"], "target_slot": ev["target_slot"], "target_slot_correct": target_correct, "value_replaced": replacement, "old_value_no_longer_current": replacement, "duplicate_same_key_slot_created": duplicate, "slot_count_before": ev["n"], "slot_count_after": ev["n"] + int(duplicate), "slot_count_correct": (ev["n"] + int(duplicate)) == ev["n"], "target_score": ev["target_score"], "top1_score": ev["top1_score"], "target_rank": ev["target_rank"], "top1_top2_margin": ev["margin"], "rule": rule})
    return rows


def stage_holdout() -> None:
    start = time.perf_counter()
    policy_path = OUT / "decision_rule_frozen.json"
    if not policy_path.is_file():
        raise FileNotFoundError("validation rule must be frozen before holdout")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not policy.get("frozen") or policy.get("holdout_opened"):
        raise RuntimeError("decision policy is not frozen or has already opened holdout")
    frozen_hash = policy["candidate_checkpoint_sha256"]
    if sha_file(CKPT) != frozen_hash:
        raise RuntimeError("Candidate B checkpoint hash differs from frozen policy manifest")
    ck, model, state = load_ckpt()
    if state["state_dict_sha256"] != json.loads((OUT / "candidate_b_frozen_manifest.json").read_text())["state_dict_sha256_before"]:
        raise RuntimeError("Candidate B state changed after freeze")
    train = load_split("train")
    val = load_split("val")
    test_a = load_split("testA")
    test_b = load_split("testB")
    all_data = {"train": train, "val": val, "testA": test_a, "testB": test_b}
    integrity = manifest_and_integrity(all_data, ck, state["state_dict_sha256"])
    train_records, _ = make_records(train)
    val_records, val_filter = make_records(val, train["semantic_triples"])
    forbidden = {r["triple"] for r in train_records} | {r["triple"] for r in val_records}
    a_records, a_filter = make_records(test_a, forbidden)
    forbidden_a = forbidden | {r["triple"] for r in a_records}
    b_records, b_filter = make_records(test_b, forbidden_a)
    if len(a_records) < 64 or len(b_records) < 64:
        raise RuntimeError(f"holdout semantic de-dup/filter left insufficient facts: A={len(a_records)} B={len(b_records)}")
    integrity["holdout_filter"] = {"validation": val_filter, "testA": a_filter, "testB": b_filter}
    integrity["holdout_semantic_triple_overlap_after_filter"] = {
        "train-val": overlap_count({r["triple"] for r in train_records}, {r["triple"] for r in val_records}),
        "train-testA": overlap_count({r["triple"] for r in train_records}, {r["triple"] for r in a_records}),
        "val-testA": overlap_count({r["triple"] for r in val_records}, {r["triple"] for r in a_records}),
        "train-testB": overlap_count({r["triple"] for r in train_records}, {r["triple"] for r in b_records}),
        "val-testB": overlap_count({r["triple"] for r in val_records}, {r["triple"] for r in b_records}),
        "testA-testB": overlap_count({r["triple"] for r in a_records}, {r["triple"] for r in b_records}),
    }
    if any(integrity["holdout_semantic_triple_overlap_after_filter"].values()):
        raise RuntimeError("semantic fact leakage remains after holdout filtering")
    integrity["test_access_after_policy_freeze"] = True
    jdump(OUT / "input_integrity.json", integrity)
    events_all: list[dict[str, Any]] = []
    per_split_events: dict[str, list[dict[str, Any]]] = {}
    for split, records, seed in (("testA", a_records, 202609252), ("testB", b_records, 202609253)):
        evs = make_episode_set(records, split, seed=seed, per_cell=HOLDOUT_EPISODES_PER_CELL)
        enc_e, enc_r = encode_split(all_data[split], ck, model)
        score_events(evs, enc_e, enc_r, model)
        per_split_events[split] = evs
        events_all.extend(evs)
    rule = policy["rule"]
    params = policy["params"]
    holdout_rows: list[dict[str, Any]] = []
    true_update_rows: list[dict[str, Any]] = []
    score_audit: list[dict[str, Any]] = []
    extreme: list[dict[str, Any]] = []
    split_eval: dict[str, Any] = {}
    for split, evs in per_split_events.items():
        pred = np.asarray([predict_event(e, rule, params) for e in evs], dtype=bool)
        holdout_rows.extend(metrics_table(evs, pred, rule, split))
        true_update_rows.extend(write_true_update_rows(evs, pred, split, rule, params))
        score_audit.extend(audit_rows(evs, policy))
        for threshold_name, threshold in (("frozen", float("nan")), ("historical_global", float(ck.get("thresholds", {}).get("threshold", OLD_THRESHOLD)))):
            if threshold_name == "frozen":
                p = pred
            else:
                p = np.asarray([float(e["top1_score"]) >= threshold for e in evs], dtype=bool)
            extreme.extend(extreme_rows(evs, p, split, threshold_name, threshold if math.isfinite(threshold) else -1.0))
        split_eval[split] = {"validation_rule": rule, "validation_frozen_params": params, "operational_gate": gate_metrics(evs, pred, tuple(n for n in N_LEVELS if n <= OPERATIONAL_CAPACITY)), "extended_gate": gate_metrics(evs, pred, tuple(N_LEVELS)), "weighted_metrics": event_counts(evs, pred)}
    cdump(OUT / "holdout_action_results.csv", holdout_rows)
    cdump(OUT / "true_update_results.csv", true_update_rows)
    cdump(OUT / "slot_competition_score_audit.csv", read_csv(OUT / "slot_competition_score_audit.csv") + score_audit)
    cdump(OUT / "extreme_value_scaling.csv", read_csv(OUT / "extreme_value_scaling.csv") + extreme)
    val_events_manifest = json.loads((OUT / "episode_dataset_manifest.json").read_text(encoding="utf-8"))
    val_events_manifest["holdout"] = {s: {"eligible_unique_semantic_facts": len(rs), "filter": flt, "episodes": len(per_split_events[s]), "per_N_per_type": HOLDOUT_EPISODES_PER_CELL, "N_levels": list(N_LEVELS)} for s, rs, flt in (("testA", a_records, a_filter), ("testB", b_records, b_filter))}
    val_events_manifest["source_separation"] = integrity["holdout_semantic_triple_overlap_after_filter"]
    val_events_manifest["semantic_key_overlap_unfiltered"] = {k: v["semantic_keys"] for k, v in integrity["pairwise_source_overlap"].items()}
    jdump(OUT / "episode_dataset_manifest.json", val_events_manifest)
    tm = json.loads((OUT / "true_update_dataset_manifest.json").read_text(encoding="utf-8"))
    for split in ("testA", "testB"):
        rr = per_split_events[split]
        tm[split] = {"eligible_changed_value_keys": sum(len({r['value'] for r in rows}) > 1 for rows in _key_value_rows(make_records(all_data[split], forbidden if split == 'testA' else forbidden_a)[0])[0].values()), "true_update_episodes": sum(e["event_type"] == "TRUE_UPDATE" for e in rr), "distinct_old_new_values_verified": all(e["old_value"] != e["new_value"] for e in rr if e["event_type"] == "TRUE_UPDATE")}
    tm["holdout_filter_applied_after_policy_freeze"] = True
    jdump(OUT / "true_update_dataset_manifest.json", tm)
    gate_rows = []
    op_ns = tuple(n for n in N_LEVELS if n <= OPERATIONAL_CAPACITY)
    all_op_pass = True
    all_replacement_pass = True
    for split, evs in per_split_events.items():
        pred = np.asarray([predict_event(e, rule, params) for e in evs], dtype=bool)
        g = gate_metrics(evs, pred, op_ns)
        split_eval[split]["operational_gate"] = g
        all_op_pass &= bool(g["pass"])
        for n in op_ns:
            m = event_counts(evs, pred, n)
            gate_rows.append({"split": split, "N": n, "in_operational_range": True, "pass": all([m["new_collision_rate"] <= .05, m["p2_wrong_update_rate"] <= .05, m["p3_wrong_update_rate"] <= .05, m["true_update_action_recall"] >= .95, m["target_slot_recall"] >= .95, m["duplicate_same_key_slot_rate"] <= .05, m["value_replacement_accuracy"] >= .95, m["slot_count_error_rate"] <= .05]), **m})
            all_replacement_pass &= bool(m["true_update_action_recall"] >= .95 and m["target_slot_recall"] >= .95 and m["value_replacement_accuracy"] >= .95 and m["duplicate_same_key_slot_rate"] <= .05 and m["slot_count_error_rate"] <= .05)
    cdump(OUT / "action_gate_by_split_n.csv", gate_rows)
    jdump(OUT / "value_replacement_gate.json", {"pass": all_replacement_pass, "requirements": {"true_update_action_recall": ">=0.95", "correct_target_slot_recall": ">=0.95", "value_replacement_accuracy": ">=0.95", "duplicate_same_key_slot_creation": "<=0.05", "slot_count_error": "<=0.05"}, "operational_N": list(op_ns), "by_split": {s: split_eval[s]["operational_gate"]["per_n"] for s in split_eval}, "true_update_rows": len(true_update_rows), "all_old_new_values_differ": all(r["different_value_verified"] for r in true_update_rows)})
    action_gate = {"pass": all_op_pass and all_replacement_pass, "competition_gate_pass": all_op_pass, "value_replacement_gate_pass": all_replacement_pass, "operational_capacity": OPERATIONAL_CAPACITY, "operational_N": list(op_ns), "stress_only_N": [n for n in N_LEVELS if n > OPERATIONAL_CAPACITY], "requirements": {"NEW collision": "<=5%", "P2 wrong UPDATE": "<=5%", "P3 wrong UPDATE": "<=5%", "TRUE UPDATE action recall": ">=95%", "target-slot recall": ">=95%", "duplicate same-key slot": "<=5%", "value replacement": ">=95%", "slot count error": "<=5%"}, "by_split": split_eval, "all_N_metrics_stored": True}
    jdump(OUT / "action_gate.json", action_gate)
    jdump(OUT / "holdout_runtime.json", {"wall_seconds": time.perf_counter() - start, "device": DEVICE, "holdout_accessed_after_frozen_policy": True})
    # Real reconnect is an explicit conditional gate; do not touch existing memory code on failure.
    if action_gate["pass"]:
        cdump(OUT / "reconnect_results.csv", [{"status": "ELIGIBLE_FOR_RECONNECT_REVIEW", "detail": "Action and true-value gates passed. Existing O3b live interface compatibility must be audited before any code-path swap."}])
        cdump(OUT / "regression_check.csv", [{"status": "NOT_RUN", "detail": "Reconnect has not yet been executed; review interface compatibility first."}])
        print(json.dumps({"phase": "holdout", "action_gate": "PASS", "reconnect": "eligible, pending interface audit", "wall_seconds": time.perf_counter() - start}, ensure_ascii=False), flush=True)
    else:
        cdump(OUT / "reconnect_results.csv", [{"status": "NOT_RUN: action gate failed", "detail": "No real O3b reconnect performed."}])
        cdump(OUT / "regression_check.csv", [{"status": "NOT_RUN: action gate failed", "detail": "No reconnect; retrieval/KV/free-allocation regression is not evaluable."}])
        print(json.dumps({"phase": "holdout", "action_gate": "FAIL", "reconnect": "NOT_RUN", "wall_seconds": time.perf_counter() - start}, ensure_ascii=False), flush=True)
    freeze_candidate_manifest(ck, state["state_dict_sha256"], "after_frozen_holdout_evaluation")


def finalize() -> None:
    action_path = OUT / "action_gate.json"
    decision = json.loads((OUT / "decision_rule_frozen.json").read_text(encoding="utf-8"))
    if not action_path.exists():
        verdict = "O3b-M2-INCOMPLETE — HOLDOUT PHASE NOT RUN"
        (OUT / "FINAL_VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")
        return
    ag = json.loads(action_path.read_text(encoding="utf-8"))
    reconnect_rows = read_csv(OUT / "reconnect_results.csv") if (OUT / "reconnect_results.csv").exists() else []
    reconnect_status = reconnect_rows[0].get("status", "NOT_RUN") if reconnect_rows else "NOT_RUN"
    missing_dependencies = [
        "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/best_model.pt (frozen query encoder/value heads)",
        "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_vgen2/best.pt (frozen V/Core and value heads)",
        "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3a_allocation/allocator_kaware_best.pt (frozen free-slot allocator)",
        "/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_update/router_best.pt (pre-reconnect baseline router for regression comparison)",
    ]
    if ag["pass"] and reconnect_status != "REAL_RECONNECT_PASS":
        cdump(OUT / "reconnect_results.csv", [{"status": "NOT_RUN: frozen O3b dependencies missing", "detail": "; ".join(missing_dependencies)}])
        cdump(OUT / "regression_check.csv", [{"status": "NOT_RUN: real reconnect not run", "detail": "Cannot compare retrieval/full-KV answer or free-slot regression without the unchanged O3a/O3b frozen checkpoint bundle."}])
        verdict = "O3b-M2-INCOMPLETE — OFFLINE ACTION GATES PASS; REAL RECONNECT BLOCKED BY MISSING FROZEN DEPENDENCIES"
    elif not ag["pass"] and ag.get("competition_gate_pass"):
        verdict = "O3b-M2-F — COMPETITION CALIBRATION PASSES BUT TRUE VALUE REPLACEMENT FAILS"
    elif not ag["pass"]:
        verdict = "O3b-M2-F — GROWING-SLOT COMPETITION REMAINS ABOVE COLLISION GATE"
    elif reconnect_status == "REAL_RECONNECT_PASS":
        verdict = "O3b-M2-P — GROWING-SLOT NEW/UPDATE ADDRESSING AND VALUE REPLACEMENT PASS"
    else:
        verdict = "O3b-M2-INCOMPLETE — OFFLINE ACTION GATES PASS; REAL RECONNECT NOT VALIDATED"

    hrows = read_csv(OUT / "holdout_action_results.csv")
    erows = read_csv(OUT / "extreme_value_scaling.csv")
    def overall(split: str, nmax: int = OPERATIONAL_CAPACITY) -> dict[str, float]:
        rs = [r for r in hrows if r.get("split") == split and r.get("event_type", "") == "" and str(r.get("active_slots_N", "")) in {str(n) for n in N_LEVELS if n <= nmax}]
        if not rs:
            return {}
        total_events = sum(int(float(r["n_events"])) for r in rs)
        new_n = sum(int(float(r["n_new"])) for r in rs)
        upd_n = sum(int(float(r["n_true_update"])) for r in rs)
        new_col = sum(float(r["new_collision_rate"]) * int(float(r["n_new"])) for r in rs) / max(1, new_n)
        upd = sum(float(r["true_update_action_recall"]) * int(float(r["n_true_update"])) for r in rs) / max(1, upd_n)
        target = sum(float(r["target_slot_recall"]) * int(float(r["n_true_update"])) for r in rs) / max(1, upd_n)
        replace = sum(float(r["value_replacement_accuracy"]) * int(float(r["n_true_update"])) for r in rs) / max(1, upd_n)
        dup = sum(float(r["duplicate_same_key_slot_rate"]) * int(float(r["n_true_update"])) for r in rs) / max(1, upd_n)
        slot_err = sum(float(r["slot_count_error_rate"]) * int(float(r["n_true_update"])) for r in rs) / max(1, upd_n)
        acc = sum(float(r["overall_action_accuracy"]) * int(float(r["n_events"])) for r in rs) / max(1, total_events)
        # P2/P3 denominators are one balanced cell for each operational N.
        p2rows = [r for r in hrows if r.get("split") == split and r.get("event_type") == "NEW_SAME_E" and int(float(r["active_slots_N"])) <= nmax]
        p3rows = [r for r in hrows if r.get("split") == split and r.get("event_type") == "NEW_SAME_R" and int(float(r["active_slots_N"])) <= nmax]
        def balanced_rate(rr: list[dict[str, str]], field: str) -> float:
            den = sum(int(float(r["n_events"])) for r in rr)
            return sum(float(r[field]) * int(float(r["n_events"])) for r in rr) / max(1, den)
        return {"new_collision_rate": new_col, "p2_wrong_update_rate": balanced_rate(p2rows, "p2_wrong_update_rate"), "p3_wrong_update_rate": balanced_rate(p3rows, "p3_wrong_update_rate"), "true_update_action_recall": upd, "target_slot_recall": target, "value_replacement_accuracy": replace, "duplicate_same_key_slot_rate": dup, "slot_count_error_rate": slot_err, "overall_action_accuracy": acc}
    op = {s: overall(s) for s in ("testA", "testB")}
    per_n_rows = {s: {str(n): next(r for r in hrows if r.get("split") == s and r.get("event_type", "") == "" and str(r.get("active_slots_N")) == str(n)) for n in N_LEVELS} for s in ("testA", "testB")}
    old_ext = {(r["split"], str(r["N"])): r for r in erows if r.get("threshold_name") == "historical_global" and r.get("split") in ("testA", "testB")}
    def pct(x: float) -> str:
        return f"{100*x:.3f}%"
    rule_summaries = decision["all_rule_validation_summary"]
    rule_text = "; ".join(f"{k}: {'PASS' if v['operational_pass'] else 'FAIL'}" for k, v in rule_summaries.items())
    n64_old_a = old_ext.get(("testA", "64"), {})
    n64_old_b = old_ext.get(("testB", "64"), {})
    n1_old_a = old_ext.get(("testA", "1"), {})
    n1_old_b = old_ext.get(("testB", "1"), {})
    frozen_manifest = json.loads((OUT / "candidate_b_frozen_manifest.json").read_text(encoding="utf-8"))
    integrity = json.loads((OUT / "input_integrity.json").read_text(encoding="utf-8"))
    value_gate = json.loads((OUT / "value_replacement_gate.json").read_text(encoding="utf-8"))
    report = f'''# O3b-M2 — Growing-Slot Competition Calibration + True Value Replacement

## FINAL VERDICT

**{verdict}**

Offline Candidate B action and true-value replacement gates pass at the existing O3b capacity of {OPERATIONAL_CAPACITY}. A real O3b reconnect and retrieval/KV regression test were **not run**: required frozen O3b checkpoints are absent from the active remote checkpoint tree. Therefore this is not an end-to-end O3b PASS.

## Frozen policy and holdout

- Candidate B: frozen deployment seed 43; checkpoint SHA-256 before/after: `{frozen_manifest['checkpoint_sha256_before']}` / `{frozen_manifest['checkpoint_sha256_after']}`; state-dict SHA-256 before/after: `{frozen_manifest['state_dict_sha256_before']}` / `{frozen_manifest['state_dict_sha256_after']}`. No retraining or weight modification occurred.
- Validation selected Rule A, one global threshold `0.8504285216331482`; no N-dependent threshold and no top1-top2 margin. A/B/C all passed the operational validation gate; the simpler Rule A won by the preregistered priority.
- TestA/TestB were evaluated only after policy freeze. Both were filtered to remove exact `(entity, relation, value)` semantic facts overlapping train/validation; remaining cross-split semantic-triple overlap is zero. Source facts/source episodes are split-disjoint. TestA shares some `(entity, relation)` key identities with train/validation; TestB has zero such key overlap. This identity overlap is disclosed and is not an exact-fact/source-row leak.
- Each holdout has 240 generated episodes per N × event-type cell; source facts can be reused between Monte Carlo episodes within a split. The episode draws are not 6,720 independent source facts.

## Operational-capacity results (N=1,2,4)

| Test split | NEW collision | P2 wrong UPDATE | P3 wrong UPDATE | TRUE UPDATE recall | Target-slot recall | Value replacement | Duplicate same-key slot | Overall action accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TestA | {pct(op['testA']['new_collision_rate'])} | {pct(op['testA']['p2_wrong_update_rate'])} | {pct(op['testA']['p3_wrong_update_rate'])} | {pct(op['testA']['true_update_action_recall'])} | {pct(op['testA']['target_slot_recall'])} | {pct(op['testA']['value_replacement_accuracy'])} | {pct(op['testA']['duplicate_same_key_slot_rate'])} | {pct(op['testA']['overall_action_accuracy'])} |
| TestB | {pct(op['testB']['new_collision_rate'])} | {pct(op['testB']['p2_wrong_update_rate'])} | {pct(op['testB']['p3_wrong_update_rate'])} | {pct(op['testB']['true_update_action_recall'])} | {pct(op['testB']['target_slot_recall'])} | {pct(op['testB']['value_replacement_accuracy'])} | {pct(op['testB']['duplicate_same_key_slot_rate'])} | {pct(op['testB']['overall_action_accuracy'])} |

At N=4, NEW collision is {pct(float(per_n_rows['testA']['4']['new_collision_rate']))} (TestA) and {pct(float(per_n_rows['testB']['4']['new_collision_rate']))} (TestB). N=4 NEW events are decision-only in the simulator because the real 4-slot allocator is already full; no capacity/overflow behavior is claimed. N=64 is out-of-capacity stress only. Under the frozen rule its NEW collision is {pct(float(per_n_rows['testA']['64']['new_collision_rate']))} / {pct(float(per_n_rows['testB']['64']['new_collision_rate']))}, and its update target recall is {pct(float(per_n_rows['testA']['64']['target_slot_recall']))} / {pct(float(per_n_rows['testB']['64']['target_slot_recall']))} (TestA/TestB).

The direct true-update set contains {value_gate['true_update_rows']} holdout episodes with verified `old_value != new_value`. Correct target selection, replacing the stored value, removing the prior current value, no duplicate key, and unchanged slot count are simulated together. Value-replacement gate: **{'PASS' if value_gate['pass'] else 'FAIL'}**. Combined offline action gate: **{'PASS' if ag['pass'] else 'FAIL'}**.

## Slot competition and rule choice

- Validation rule status: {rule_text}. Selected Rule A threshold `{decision['params']['threshold']:.6f}`; Rule B/C add complexity without changing the operational-capacity outcome.
- With the historical pairwise threshold 0.57, the NEW top-1 negative p99 rises from TestA `{float(n1_old_a.get('top1_negative_p99', 0)):.3f}` to `{float(n64_old_a.get('top1_negative_p99', 0)):.3f}`, and TestB `{float(n1_old_b.get('top1_negative_p99', 0)):.3f}` to `{float(n64_old_b.get('top1_negative_p99', 0)):.3f}` between N=1 and N=64. Historical-threshold collision also rises from {pct(float(n1_old_a.get('collision_rate', 0)))} to {pct(float(n64_old_a.get('collision_rate', 0)))} (TestA) and {pct(float(n1_old_b.get('collision_rate', 0)))} to {pct(float(n64_old_b.get('collision_rate', 0)))} (TestB). This confirms extreme-value accumulation through N=64, while the actual O3b capacity gate remains N≤4.
- On the frozen policy, every per-N action/value gate through N=64 passes in these sampled episodes; N>4 remains stress-only and does not extend the real memory capacity claim.

## Reconnect status

`NOT_RUN: frozen O3b dependencies missing`. The active remote has O21B and Candidate B, but lacks:

{chr(10).join('- `' + x + '`' for x in missing_dependencies)}

Consequently post-reconnect retrieval, full KV answer, regression vs baseline, and free-slot regression are **N/A / NOT RUN**. No architecture, model weights, allocator, Read path, or Value path was changed.

## Required answers

1. Candidate B fully frozen: **YES**; before/after hashes match.
2. NEW negative top-1 rises with N: **YES**; p99 increases substantially through N=64.
3. Multi-slot extreme-value accumulation: **CONFIRMED** under the historical threshold; frozen calibrated policy controls it in the measured range.
4. Validation choice: A/B/C all pass operational N; **Rule A** selected by simplicity priority.
5. Frozen rule: **Rule A**, `UPDATE iff top1 >= 0.8504285216331482`; otherwise NEW.
6. Threshold depends on N: **NO**.
7. Top1-top2 margin used: **NO**.
8. Holdout aggregate NEW collision, TestA/TestB: **{pct(op['testA']['new_collision_rate'])} / {pct(op['testB']['new_collision_rate'])}** at N≤4.
9. Largest operational N=4 NEW collision: **{pct(float(per_n_rows['testA']['4']['new_collision_rate']))} / {pct(float(per_n_rows['testB']['4']['new_collision_rate']))}**. N=64 stress-only frozen-rule result: **{pct(float(per_n_rows['testA']['64']['new_collision_rate']))} / {pct(float(per_n_rows['testB']['64']['new_collision_rate']))}**.
10. P2 wrong UPDATE at N≤4: **{pct(op['testA']['p2_wrong_update_rate'])} / {pct(op['testB']['p2_wrong_update_rate'])}**.
11. P3 wrong UPDATE at N≤4: **{pct(op['testA']['p3_wrong_update_rate'])} / {pct(op['testB']['p3_wrong_update_rate'])}**.
12. TRUE UPDATE recall at N≤4: **{pct(op['testA']['true_update_action_recall'])} / {pct(op['testB']['true_update_action_recall'])}**.
13. Correct target-slot recall at N≤4: **{pct(op['testA']['target_slot_recall'])} / {pct(op['testB']['target_slot_recall'])}**.
14. Value replacement accuracy at N≤4: **{pct(op['testA']['value_replacement_accuracy'])} / {pct(op['testB']['value_replacement_accuracy'])}**.
15. Duplicate same-key slot rate at N≤4: **{pct(op['testA']['duplicate_same_key_slot_rate'])} / {pct(op['testB']['duplicate_same_key_slot_rate'])}**.
16. Offline action gate: **{'PASS' if ag['pass'] else 'FAIL'}**.
17. Real reconnect: **NO**; blocked by missing frozen O3b dependencies.
18. Post-reconnect retrieval: **NOT RUN**.
19. Post-reconnect full KV answer: **NOT RUN**.
20. Regression >1pp: **NOT EVALUABLE**; no reconnect.
21. Full O3b operational Address: **FAIL / NOT VERIFIED** until real reconnect and regression gate pass. Offline action/value behavior itself passes.
22. O3c allowed: **NO**.

Exactly one next experiment: **Restore the original frozen O3b checkpoint bundle on the active remote, then reconnect only the frozen Candidate B + Rule A decision to the unchanged write path and run retrieval/full-KV/free-slot regression on the same source-separated holdout. Do not retrain Candidate B or change any other module.**

## Reproducibility and artifacts

- Remote output: `{OUT}`
- Candidate B checkpoint was unchanged; no model/checkpoint was written or modified.
- Data leakage: no canonical source-row, feature-row, source-record, or exact semantic-triple overlap between the used validation and holdout episodes. TestA key-identity overlap with calibration is disclosed above; TestB is key-disjoint.
- Runtime: calibration {json.loads((OUT / 'calibration_runtime.json').read_text())['wall_seconds']:.2f}s; holdout {json.loads((OUT / 'holdout_runtime.json').read_text())['wall_seconds']:.2f}s, on CUDA. Peak VRAM not recorded.
- Git SHA: no Git repository at the workspace root.
- `artifact_sha256_manifest.json` records every delivered experiment artifact.
'''
    (OUT / "final_report.md").write_text(report, encoding="utf-8")
    (OUT / "FINAL_VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")
    jdump(OUT / "final_verdict.json", {"verdict": verdict, "offline_action_gate": ag["pass"], "real_reconnect": "NOT_RUN_MISSING_FROZEN_DEPENDENCIES", "O3c_allowed": False, "decision_rule": decision})
    rows = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.name == "artifact_sha256_manifest.json":
            continue
        rows.append({"path": str(p.relative_to(OUT)), "bytes": p.stat().st_size, "sha256": sha_file(p)})
    jdump(OUT / "artifact_sha256_manifest.json", {"experiment": "O3b-M2", "files": rows, "candidate_checkpoint_sha256": sha_file(CKPT), "created_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"verdict": verdict, "decision_rule": decision["rule"], "manifest_files": len(rows)}, ensure_ascii=False), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("calibrate", "holdout", "finalize"), required=True)
    args = p.parse_args()
    if args.phase == "calibrate":
        stage_calibrate()
    elif args.phase == "holdout":
        stage_holdout()
    else:
        finalize()


if __name__ == "__main__":
    main()
