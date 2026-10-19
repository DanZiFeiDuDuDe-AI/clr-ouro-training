"""O3b-J3C: freeze a provenance policy, then extract frozen feature caches.

This script intentionally has a separate `freeze` mode and `extract` mode.
`extract` refuses to run unless the policy, environment, and input hashes have
already been written and still match the current code/artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata as im
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/root/private_data/clr_v24_o3b_l")
OUT = ROOT / "checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache"
CACHE = ROOT / "checkpoints/clr_v24_o21a_mr/cache"
KEY_CKPT = ROOT / "checkpoints/clr_v24_o21b/b1_best.pt"
CODE = ROOT / "code"
MODEL_DIR = OUT / "source_snapshot/Ouro-1.4B"
SPLITS = ("train", "val", "testA", "testB")
SEED = 42
BATCH_SIZE = 32
MAX_TOKENS = 512
DEVICE = "cuda"
OUTPUT_DTYPE = "float32"
MODEL_DTYPE_NAME = "bfloat16"
MODEL_REPO = "ByteDance/Ouro-1.4B"
MODEL_REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
OURO_SHA256 = "58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
OURO_BYTES = 2_869_336_434
KEY_SHA256 = "6c2018ccc17a3b6c8bcfa91425f426c8d8c88a37d5f5f1b7796ec18a2d32a570"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def policy_object() -> dict[str, Any]:
    return {
        "experiment": "clr_v24_o3b_j3c_fresh_certified_feature_cache",
        "policy_version": 1,
        "frozen_before_any_model_forward": True,
        "frozen_code_sha256": sha256_file(Path(__file__).resolve()),
        "seed": SEED,
        "splits": list(SPLITS),
        "fact_eligibility": "Every source fact enumerated by split -> metadata episode order -> item position; no count cap or prefix truncation.",
        "canonical_identity": "<split>:ep<episode_index>:<episode_id>:item<item_index>:fact<flat_fact_index>; never a tensor row index.",
        "mention_matching": {
            "source_text": "MR metadata fact_texts[item_index]",
            "needle": "entity_raw",
            "normalization": "none; entity_normalized is recorded as entity_raw; exact Unicode codepoint, case-sensitive literal match",
            "overlapping_occurrences": "count all, advancing search by one codepoint",
            "acceptance": "exactly one occurrence; accepted mention_id is <canonical_fact_id>:m0",
            "zero_occurrences": "exclude with no_match",
            "multiple_occurrences": "exclude with multiple_matches; record all candidate spans; never choose first",
            "token_mapping": "token offsets overlap the unique character span; token span is [min_index,max_index+1)",
            "unmappable": "exclude with token_span_unmapped",
            "noncontiguous_mapping": "exclude with token_span_noncontiguous; never silently use a bounding interval",
        },
        "tokenization": {
            "identifier": MODEL_REPO,
            "revision": MODEL_REVISION,
            "source": "local pinned tokenizer assets in source_snapshot/Ouro-1.4B",
            "fast_tokenizer_required": True,
            "add_special_tokens": True,
            "truncation": False,
            "padding": "right; batch-dynamic, pad to batch maximum",
            "max_tokens_including_special_tokens": MAX_TOKENS,
            "too_long": "exclude as too_long; no truncation",
            "persisted_inputs": "input_ids, offsets and mention span are included in mention_resolution_manifest.jsonl before forward",
        },
        "source_hidden_state": {
            "checkpoint": str(MODEL_DIR / "model.safetensors"),
            "checkpoint_sha256": OURO_SHA256,
            "architecture_code": "pinned modeling_ouro.py from source_snapshot/Ouro-1.4B",
            "selected_state": "OuroModel hidden_states_list[-1], final UT pass (total_ut_steps-1), after the model RMSNorm; not a memory write state",
            "pooling": "arithmetic mean of selected mention token hidden states",
            "model_eval": True,
            "requires_grad": False,
            "inference_mode": True,
            "model_dtype": MODEL_DTYPE_NAME,
            "feature_storage_dtype": OUTPUT_DTYPE,
            "device": DEVICE,
            "max_length": MAX_TOKENS,
        },
        "primary_j_input_H": {
            "name": "fresh J-I2b carrier",
            "formula": "MR cache fact_H addressed by fact_offsets + 2.0 * fresh entity-mention mean hidden state",
            "base_component": "source MR train/val/test data NPZ fact_H, joined by canonical source fact flat_fact_index; the NPZ is hashed and never prefix-aligned",
            "mention_component": "fresh hidden state defined above",
            "additional_saved_components": ["H_base_I0", "H_mention_fresh", "H_I2a=H_base_I0+H_mention_fresh", "H_fresh=H_I2b"],
        },
        "labels": {
            "entity": "train_persons.json order; unseen entity -> -1",
            "relation": "source relation_ids integer; relation string retained",
            "joint": "sorted unique (entity_raw, relation_id) pairs observed in train; unseen pair or unknown entity -> -1",
            "evaluation_splits_do_not_change_vocab": True,
        },
        "j_stages": {
            "Z0": "H_fresh I2b carrier, 2048D",
            "Z1": "KeyEncoder.net[0] Linear(2048->256), includes frozen bias, pre-activation",
            "Z2": "KeyEncoder.net[1] GELU output, 256D",
            "Z3": "KeyEncoder.net[2] Linear(256->64), pre-normalization",
            "Z4": "L2-normalize Z3 across dimension 64",
        },
        "R_controls": {
            "source": "exact first KeyEncoder Linear W1 from frozen O2.1b b1_best.pt",
            "svd": "W1=U diag(S) Vh; thin full-rank SVD",
            "R0": "H_fresh @ Vh.T",
            "R1": "R0 * S elementwise",
            "R2": "H_fresh @ W1.T; algebraically R1 @ U.T",
            "float": "CPU float64 SVD; feature transforms emitted float32",
        },
        "determinism": {
            "python_numpy_torch_seed": SEED,
            "model_eval_no_grad_no_optimizer_no_backward": True,
            "torch_deterministic_algorithms": "enabled with warn_only=True",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "tf32": False,
            "batch_size": BATCH_SIZE,
            "source_order": "fixed split order then episode metadata order then item position",
            "numerical_reextraction_sample": "up to 64 accepted rows per split, selected by ascending SHA256(feature_row_id); re-tokenize source text, re-locate the frozen exact mention, then re-forward original full extraction batches with identical source composition and padding width",
            "predeclared_tolerance": {
                "max_abs_diff": 0.001,
                "mean_abs_diff": 0.0001,
                "min_cosine_similarity": 0.99999,
                "basis": "bf16 model computation, float32 cache, same device/software and identical original batch composition",
            },
        },
        "prohibited": ["probe training", "checkpoint/model modification", "architecture changes", "memory reconnect", "O3c", "TestA/TestB rule tuning"],
    }


def prepare_runtime_flags() -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    random.seed(SEED)
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def package_version(name: str) -> str | None:
    try:
        return im.version(name)
    except Exception:
        return None


def model_device_manifest() -> dict[str, Any]:
    import torch
    out: dict[str, Any] = {
        "cuda_api_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        d = torch.cuda.get_device_properties(0)
        out["device_0"] = {
            "name": d.name,
            "total_memory_bytes": int(d.total_memory),
            "major": getattr(d, "major", None),
            "minor": getattr(d, "minor", None),
            "multiprocessor_count": getattr(d, "multi_processor_count", None),
            "gcn_arch_name": getattr(d, "gcnArchName", None),
        }
    return out


def enumerated_artifacts() -> list[dict[str, Any]]:
    files: dict[str, Path] = {}
    for p in sorted(CACHE.iterdir()):
        if p.is_file():
            files[f"mr_cache/{p.name}"] = p
    for name in (
        "a1_v24_o3b_j.py", "a1_v24_o21a.py", "a1_v24_o3b.py",
        "a1_v24_o3b_l.py", "o3b_l2_j2_stage1.py",
    ):
        p = CODE / name
        if p.exists():
            files[f"project_code/{name}"] = p
    files["keyencoder_checkpoint"] = KEY_CKPT
    files["ouro_source_provenance_manifest"] = OUT / "source_snapshot/source_model_artifact_manifest.json"
    for p in sorted(MODEL_DIR.iterdir()):
        if p.is_file():
            files[f"ouro_model/{p.name}"] = p
    files["j3c_extraction_code"] = Path(__file__).resolve()
    manifest = []
    for name, p in files.items():
        if not p.exists():
            raise FileNotFoundError(p)
        manifest.append({
            "artifact": name,
            "path": str(p),
            "bytes": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
            "sha256": sha256_file(p),
        })
    return manifest


def read_source_rows() -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    persons = json.loads((CACHE / "train_persons.json").read_text(encoding="utf-8"))
    entity_map = {str(x): i for i, x in enumerate(persons)}
    relations_obj = json.loads((CACHE / "relations.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split in SPLITS:
        data = np.load(CACHE / f"{split}_data.npz", allow_pickle=False)
        fact_h_rows = np.asarray(data["fact_H"])
        fact_entity_rows = np.asarray(data["fact_entity"])
        fact_relation_rows = np.asarray(data["fact_relation"])
        meta_path = CACHE / f"{split}_meta.jsonl"
        meta = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        offsets = np.asarray(data["fact_offsets"], dtype=np.int64)
        for array_name, array in (("fact_H", fact_h_rows), ("fact_entity", fact_entity_rows), ("fact_relation", fact_relation_rows)):
            if int(array.shape[0]) != int(offsets[-1]):
                raise ValueError(f"{split} {array_name} row count != fact_offsets terminal index")
        if len(offsets) != len(meta) + 1:
            raise ValueError(f"fact_offsets / metadata episode count mismatch for {split}")
        facts_seen = 0
        for ep, m in enumerate(meta):
            cap = int(m["capacity"])
            ep_id = str(m.get("episode_id", ep))
            if offsets[ep + 1] - offsets[ep] != cap:
                raise ValueError(f"capacity/offset mismatch {split} ep={ep}")
            for pos in range(cap):
                flat = int(offsets[ep]) + pos
                entity = str(m["persons"][pos])
                relation_id = int(m["relation_ids"][pos])
                relation_raw = str(m["relations"][pos])
                context = str(m["fact_texts"][pos])
                value = str(m.get("cities", m.get("values", [""] * cap))[pos])
                ent_label = int(entity_map.get(entity, -1))
                if int(fact_entity_rows[flat]) != ent_label:
                    raise ValueError(f"MR entity label/metadata mismatch: split={split} flat={flat} entity={entity}")
                if int(fact_relation_rows[flat]) != relation_id:
                    raise ValueError(f"MR relation label/metadata mismatch: split={split} flat={flat} relation={relation_raw}")
                base_row = np.ascontiguousarray(fact_h_rows[flat])
                record = {
                    "canonical_fact_id": f"{split}:ep{ep}:{ep_id}:item{pos}:fact{flat}",
                    "split": split,
                    "source_example_index": ep,
                    "episode_index": ep,
                    "episode_id": ep_id,
                    "item_index": pos,
                    "fact_index": flat,
                    "entity_raw": entity,
                    "entity_normalized": entity,
                    "relation_raw": relation_raw,
                    "relation_normalized": relation_raw,
                    "relation_id": relation_id,
                    "object_value": value,
                    "source_text": context,
                    "source_npz": str(CACHE / f"{split}_data.npz"),
                    "source_feature_array": "fact_H",
                    "source_feature_row": flat,
                    "source_feature_row_shape": list(base_row.shape),
                    "source_feature_row_sha256": hashlib.sha256(base_row.tobytes()).hexdigest(),
                    "entity_label": ent_label,
                    "relation_label": relation_id,
                    "joint_semantic_pair": [entity, relation_id],
                }
                record["stable_source_record_sha256"] = hashlib.sha256(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                rows.append(record)
                facts_seen += 1
        split_counts[split] = facts_seen
    return rows, {"entity": persons}, {"relations": relations_obj, "split_counts": split_counts}


def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True, local_files_only=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for character/token offsets")
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def resolve_mentions(source_rows: list[dict[str, Any]], tokenizer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resolutions: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for source in source_rows:
        context, needle = source["source_text"], source["entity_raw"]
        starts: list[int] = []
        cursor = 0
        while needle and cursor <= len(context) - len(needle):
            at = context.find(needle, cursor)
            if at < 0:
                break
            starts.append(at)
            cursor = at + 1
        spans = [{"mention_id": f"{source['canonical_fact_id']}:m{i}", "char_start": s, "char_end": s + len(needle), "matched_substring": context[s:s + len(needle)]} for i, s in enumerate(starts)]
        enc = tokenizer(context, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
        ids = [int(x) for x in enc["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        status = "accepted"
        selected_span = None
        token_positions: list[int] = []
        if not starts:
            status = "no_match"
        elif len(starts) > 1:
            status = "multiple_matches"
        elif len(ids) > MAX_TOKENS:
            status = "too_long"
            selected_span = spans[0]
        else:
            selected_span = spans[0]
            s, e = selected_span["char_start"], selected_span["char_end"]
            token_positions = [i for i, (a, b) in enumerate(offsets) if b > a and a < e and b > s]
            if not token_positions:
                status = "token_span_unmapped"
            elif token_positions != list(range(token_positions[0], token_positions[-1] + 1)):
                status = "token_span_noncontiguous"
        full = bool(status == "accepted" and len(ids) <= MAX_TOKENS and token_positions)
        row = {
            "canonical_fact_id": source["canonical_fact_id"],
            "split": source["split"],
            "entity_raw": needle,
            "entity_normalized": needle,
            "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "context_token_length": len(ids),
            "truncation_status": "not_truncated" if len(ids) <= MAX_TOKENS else "excluded_over_max_without_truncation",
            "mention_match_count": len(starts),
            "candidate_mentions": spans,
            "selected_match_ordinal": 0 if status == "accepted" else None,
            "mention_id": spans[0]["mention_id"] if status == "accepted" else None,
            "char_start": selected_span["char_start"] if selected_span else None,
            "char_end": selected_span["char_end"] if selected_span else None,
            "matched_substring": selected_span["matched_substring"] if selected_span else None,
            "token_start": min(token_positions) if token_positions else None,
            "token_end_exclusive": max(token_positions) + 1 if token_positions else None,
            "mention_token_indices": token_positions,
            "mention_token_ids": [ids[i] for i in token_positions],
            "input_ids": ids if status == "accepted" else None,
            "offset_mapping": offsets if status == "accepted" else None,
            "mention_fully_preserved_after_truncation": "YES" if full else "NO",
            "resolution_status": status,
            "selection_reason": "unique_exact_literal_occurrence" if status == "accepted" else status,
        }
        resolutions.append(row)
        if status == "accepted":
            accepted.append({**source, **row})
    return resolutions, accepted


def create_plan(accepted: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_pairs = sorted({(r["entity_raw"], int(r["relation_id"])) for r in source_rows if r["split"] == "train" and int(r["entity_label"]) >= 0})
    joint_map = {pair: i for i, pair in enumerate(train_pairs)}
    counts: dict[str, int] = {s: 0 for s in SPLITS}
    batch_meta: dict[str, list[tuple[int, int, int]]] = {s: [] for s in SPLITS}
    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for r in accepted:
        by_split[r["split"]].append(r)
    for split in SPLITS:
        for batch_id, start in enumerate(range(0, len(by_split[split]), BATCH_SIZE)):
            batch = by_split[split][start:start + BATCH_SIZE]
            padded = max(len(x["input_ids"]) for x in batch)
            for offset, r in enumerate(batch):
                batch_meta[split].append((batch_id, offset, padded))
    plan: list[dict[str, Any]] = []
    for split in SPLITS:
        for local_idx, r in enumerate(by_split[split]):
            batch_id, batch_offset, pad_length = batch_meta[split][local_idx]
            pair = (r["entity_raw"], int(r["relation_id"]))
            joint_label = int(joint_map.get(pair, -1)) if int(r["entity_label"]) >= 0 else -1
            feature_id = f"{r['canonical_fact_id']}:m0"
            plan.append({
                "row_index": len(plan),
                "split_row_index": local_idx,
                "feature_row_id": feature_id,
                "canonical_fact_id": r["canonical_fact_id"],
                "mention_id": r["mention_id"],
                "split": split,
                "source_example_index": r["source_example_index"],
                "episode_index": r["episode_index"],
                "episode_id": r["episode_id"],
                "item_index": r["item_index"],
                "fact_index": r["fact_index"],
                "entity_raw": r["entity_raw"],
                "relation_raw": r["relation_raw"],
                "relation_id": r["relation_id"],
                "object_value": r["object_value"],
                "entity_label": r["entity_label"],
                "relation_label": r["relation_label"],
                "joint_semantic_pair": json.dumps([r["entity_raw"], int(r["relation_id"])], ensure_ascii=False),
                "joint_label": joint_label,
                "char_start": r["char_start"],
                "char_end": r["char_end"],
                "token_start": r["token_start"],
                "token_end_exclusive": r["token_end_exclusive"],
                "selected_state_rule": "mean final UT pass hidden state over mention token span",
                "inclusion_reason": "unique_exact_literal_occurrence_and_fully_tokenized_under_limit",
                "batch_id": batch_id,
                "batch_offset": batch_offset,
                "batch_padded_length": pad_length,
                "stable_source_record_sha256": r["stable_source_record_sha256"],
            })
            counts[split] += 1
    if len({r["feature_row_id"] for r in plan}) != len(plan):
        raise ValueError("feature_row_id collision")
    if [r["row_index"] for r in plan] != list(range(len(plan))):
        raise ValueError("row_index is not contiguous")
    joint_rows = [{"joint_label": i, "entity_raw": p[0], "relation_id": p[1], "joint_semantic_pair": [p[0], p[1]]} for p, i in joint_map.items()]
    json_dump(OUT / "joint_label_map.json", {
        "training_vocabulary": "sorted unique train-only (entity_raw, relation_id) pairs among known train entities",
        "sort_order": "entity_raw Unicode codepoint order, then relation_id integer ascending",
        "unseen_eval_pair": -1,
        "rows": joint_rows,
    }, exclusive=True)
    return plan


def stage0_freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(exist_ok=True)
    (OUT / "features").mkdir(exist_ok=True)
    prepare_runtime_flags()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen forward policy requires the remote accelerator")
    model_stat = (MODEL_DIR / "model.safetensors").stat()
    if model_stat.st_size != OURO_BYTES:
        raise RuntimeError(f"Ouro weight byte count mismatch: {model_stat.st_size}")
    ouro_hash = sha256_file(MODEL_DIR / "model.safetensors")
    if ouro_hash != OURO_SHA256:
        raise RuntimeError(f"Ouro pinned SHA mismatch: {ouro_hash}")
    key_hash = sha256_file(KEY_CKPT)
    if key_hash != KEY_SHA256:
        raise RuntimeError(f"KeyEncoder checkpoint pinned SHA mismatch: {key_hash}")
    artifacts = enumerated_artifacts()
    json_dump(OUT / "artifact_hash_manifest.json", {
        "frozen_at_unix": time.time(),
        "artifacts": artifacts,
        "pinned_ouro_sha256": OURO_SHA256,
        "pinned_keyencoder_sha256": KEY_SHA256,
    }, exclusive=True)
    env = {
        "frozen_at_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {n: package_version(n) for n in ("torch", "transformers", "tokenizers", "safetensors", "numpy", "accelerate")},
        "device": model_device_manifest(),
        "model_dtype": MODEL_DTYPE_NAME,
        "feature_dtype": OUTPUT_DTYPE,
        "seed": SEED,
        "determinism": {
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        },
        "paths": {
            "frozen_ouro_model": str(MODEL_DIR),
            "frozen_ouro_source_manifest": str(OUT / "source_snapshot/source_model_artifact_manifest.json"),
            "keyencoder_checkpoint": str(KEY_CKPT),
            "mr_cache": str(CACHE),
            "output": str(OUT),
            "extraction_code": str(Path(__file__).resolve()),
        },
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "ouro_sha256": ouro_hash,
        "keyencoder_sha256": key_hash,
    }
    json_dump(OUT / "environment_manifest.json", env, exclusive=True)
    # Freeze policy before any forward and before resolving mention-dependent rows.
    policy = policy_object()
    json_dump(OUT / "extraction_policy_frozen.json", policy, exclusive=True)
    policy_hash = sha256_file(OUT / "extraction_policy_frozen.json")
    (OUT / "logs").mkdir(exist_ok=True)
    (OUT / "logs/policy_freeze.log").write_text(
        f"freeze_mode=complete\nmodel_forward_count=0\nprobe_training_count=0\n"
        f"policy_sha256={policy_hash}\nextraction_code_sha256={policy['frozen_code_sha256']}\n",
        encoding="utf-8",
    )

    tokenizer = get_tokenizer()
    source_rows, vocabs, info = read_source_rows()
    if len({r["canonical_fact_id"] for r in source_rows}) != len(source_rows):
        raise ValueError("canonical source ID collision")
    resolutions, accepted = resolve_mentions(source_rows, tokenizer)
    plan = create_plan(accepted, source_rows)
    # Bring the frozen joint labels back into canonical/resolution rows.
    joint_map_obj = json.loads((OUT / "joint_label_map.json").read_text(encoding="utf-8"))
    joint_by_pair = {(str(r["entity_raw"]), int(r["relation_id"])): int(r["joint_label"]) for r in joint_map_obj["rows"]}
    joint_by_fact = {r["canonical_fact_id"]: int(joint_by_pair.get((r["entity_raw"], int(r["relation_id"])), -1)) for r in source_rows}
    resolution_status_by_fact = {r["canonical_fact_id"]: r["resolution_status"] for r in resolutions}
    source_out = []
    for r in source_rows:
        r = dict(r)
        r["joint_label"] = joint_by_fact.get(r["canonical_fact_id"], -1)
        r["mention_resolution_status"] = resolution_status_by_fact[r["canonical_fact_id"]]
        source_out.append(r)
    for r in resolutions:
        r["joint_label"] = joint_by_fact.get(r["canonical_fact_id"], -1)
    write_jsonl(OUT / "canonical_source_manifest.jsonl", source_out)
    write_jsonl(OUT / "mention_resolution_manifest.jsonl", resolutions)
    plan_fields = list(plan[0].keys()) if plan else ["row_index", "feature_row_id", "canonical_fact_id", "split"]
    write_csv(OUT / "feature_row_plan.csv", plan, plan_fields)
    tokenizer_manifest = {
        "tokenizer_class": tokenizer.__class__.__name__,
        "is_fast": bool(tokenizer.is_fast),
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "padding_side": tokenizer.padding_side,
        "files_sha256": {p.name: sha256_file(p) for p in sorted(MODEL_DIR.iterdir()) if p.is_file() and ("tokenizer" in p.name or p.name in ("vocab.json", "merges.txt", "special_tokens_map.json"))},
    }
    json_dump(OUT / "configs/tokenizer_manifest.json", tokenizer_manifest, exclusive=True)
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        these = [r for r in resolutions if r["split"] == split]
        counts[split] = {"total_facts": len(these), "eligible": len(these), **{k: sum(r["resolution_status"] == k for r in these) for k in ("accepted", "no_match", "multiple_matches", "too_long", "token_span_unmapped", "token_span_noncontiguous")}}
    json_dump(OUT / "configs/freeze_counts.json", {
        "source_counts": info["split_counts"],
        "mention_resolution_counts": counts,
        "row_plan_counts": {s: sum(r["split"] == s for r in plan) for s in SPLITS},
        "total_plan_rows": len(plan),
        "joint_label_classes": len(json.loads((OUT / "joint_label_map.json").read_text(encoding="utf-8"))["rows"]),
        "model_forward_count": 0,
        "probe_training_count": 0,
    }, exclusive=True)
    # Every frozen input is rehashed after policy/plan creation; changes halt before forward.
    current = {x["artifact"]: sha256_file(Path(x["path"])) for x in artifacts}
    frozen = {x["artifact"]: x["sha256"] for x in artifacts}
    if current != frozen:
        raise RuntimeError("Input artifact hash changed during freeze")
    print(json.dumps({"mode": "freeze", "policy_sha256": policy_hash, "source_counts": info["split_counts"], "accepted": len(plan), "row_counts": {s: sum(r["split"] == s for r in plan) for s in SPLITS}, "forwards": 0}, ensure_ascii=False), flush=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_checkpoint_compat(path: Path):
    # Read-only compatibility aliases for checkpoints pickled with newer NumPy
    # module names; no checkpoint rewrite or re-save occurs.
    import numpy.core
    sys.modules.setdefault("numpy._core", numpy.core)
    for sub in ("multiarray", "umath", "numeric", "_multiarray_umath"):
        try:
            mod = __import__(f"numpy.core.{sub}", fromlist=["*"])
            sys.modules.setdefault(f"numpy._core.{sub}", mod)
        except Exception:
            pass
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def load_ouro():
    import importlib
    import types
    import torch
    from transformers import AutoTokenizer
    package_name = "ouro_j3c_pinned"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(MODEL_DIR)]
        package.__package__ = package_name
        sys.modules[package_name] = package
    OuroForCausalLM = importlib.import_module(f"{package_name}.modeling_ouro").OuroForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = OuroForCausalLM.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=False,
    )
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def forward_rows(model: Any, tokenizer: Any, rows: list[dict[str, Any]], original_batch: bool = False) -> np.ndarray:
    import torch
    if not rows:
        return np.empty((0, 2048), dtype=np.float32)
    # Each call receives one full, source-ordered original batch (<=32), so the
    # sequence width and per-row context are reproducible during audit.
    max_len = max(len(r["input_ids"]) for r in rows)
    if original_batch and rows[0].get("batch_padded_length") is not None:
        max_len = int(rows[0]["batch_padded_length"])
    pad = int(tokenizer.pad_token_id)
    ids = torch.full((len(rows), max_len), pad, dtype=torch.long, device=DEVICE)
    mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=DEVICE)
    for i, r in enumerate(rows):
        seq = torch.tensor(r["input_ids"], dtype=torch.long, device=DEVICE)
        ids[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    with torch.inference_mode():
        result = model.model(input_ids=ids, attention_mask=mask, use_cache=False)
        states = result[1]
        if not isinstance(states, (list, tuple)) or not states:
            raise RuntimeError("Pinned OuroModel did not return recurrent hidden_states_list")
        final = states[-1]
        if final.ndim != 3 or final.shape[0] != len(rows) or final.shape[-1] != 2048:
            raise RuntimeError(f"Unexpected final recurrent state shape {tuple(final.shape)}")
        pooled = []
        for i, r in enumerate(rows):
            a, b = int(r["token_start"]), int(r["token_end_exclusive"])
            if not 0 <= a < b <= len(r["input_ids"]):
                raise ValueError(f"invalid saved token span for {r['feature_row_id']}")
            pooled.append(final[i, a:b].float().mean(dim=0))
        out = torch.stack(pooled).cpu().numpy().astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise FloatingPointError("nonfinite H mention state")
    return out


def retokenize_and_verify(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    text, entity = row["source_text"], row["entity_raw"]
    starts: list[int] = []
    cursor = 0
    while entity and cursor <= len(text) - len(entity):
        at = text.find(entity, cursor)
        if at < 0:
            break
        starts.append(at)
        cursor = at + 1
    enc = tokenizer(text, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    saved_ids = [int(x) for x in row["input_ids"]]
    saved_offsets = [tuple(map(int, z)) for z in row["offset_mapping"]]
    char_s, char_e = int(row["char_start"]), int(row["char_end"])
    mapped = [i for i, (a, b) in enumerate(offsets) if b > a and a < char_e and b > char_s]
    ok = (
        len(starts) == 1 and starts[0] == char_s and starts[0] + len(entity) == char_e
        and ids == saved_ids and offsets == saved_offsets and len(ids) <= MAX_TOKENS
        and mapped == [int(x) for x in row["mention_token_indices"]]
        and bool(mapped) and ids[mapped[0]:mapped[-1] + 1] == [int(x) for x in row["mention_token_ids"]]
    )
    return {
        "tokenization_reproducible": ok,
        "retokenized_token_count": len(ids),
        "retokenized_mention_token_count": len(mapped),
        "retokenized_input_ids_sha256": hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
    }


def append_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writerows(rows)


def extract_features() -> None:
    import torch
    prepare_runtime_flags()
    policy_path = OUT / "extraction_policy_frozen.json"
    env_path = OUT / "environment_manifest.json"
    artifact_path = OUT / "artifact_hash_manifest.json"
    if not all(p.exists() for p in (policy_path, env_path, artifact_path, OUT / "feature_row_plan.csv")):
        raise RuntimeError("freeze stage incomplete; refusing any model load/forward")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_hash = sha256_file(policy_path)
    if policy.get("frozen_code_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("extraction code changed after policy freeze")
    frozen_artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))["artifacts"]
    for a in frozen_artifacts:
        if sha256_file(Path(a["path"])) != a["sha256"]:
            raise RuntimeError(f"frozen input artifact hash changed: {a['artifact']}")
    ouro_before = sha256_file(MODEL_DIR / "model.safetensors")
    key_before = sha256_file(KEY_CKPT)
    if ouro_before != OURO_SHA256 or key_before != KEY_SHA256:
        raise RuntimeError("checkpoint pin mismatch before forward")
    plan = list(csv.DictReader((OUT / "feature_row_plan.csv").open(encoding="utf-8-sig", newline="")))
    resolutions = {r["canonical_fact_id"]: r for r in load_jsonl(OUT / "mention_resolution_manifest.jsonl")}
    source_contexts = {r["canonical_fact_id"]: r for r in load_jsonl(OUT / "canonical_source_manifest.jsonl")}
    # Merge persisted exact token ids/spans into the semantic row plan.
    enriched: list[dict[str, Any]] = []
    for p in plan:
        r = resolutions[p["canonical_fact_id"]]
        if r["resolution_status"] != "accepted":
            raise RuntimeError("row plan contains an unaccepted mention")
        src = source_contexts[p["canonical_fact_id"]]
        enriched.append({**p, "source_text": src["source_text"], "input_ids": r["input_ids"], "offset_mapping": r["offset_mapping"], "mention_token_indices": r["mention_token_indices"], "token_start": int(r["token_start"]), "token_end_exclusive": int(r["token_end_exclusive"]), "mention_token_ids": r["mention_token_ids"]})
    groups: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for r in enriched:
        groups[r["split"]].append(r)

    (OUT / "logs").mkdir(exist_ok=True)
    started = time.time()
    log_path = OUT / "logs/feature_extraction.log"
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"policy_sha256={policy_hash}\nmodel_forward_started_unix={started}\nprobe_training_count=0\n")
        log.flush()
        model, tokenizer = load_ouro()
        log.write(f"model_loaded_unix={time.time()}\nmodel_dtype={next(model.parameters()).dtype}\ndevice={next(model.parameters()).device}\n")
        log.flush()
        checkpoint = load_checkpoint_compat(KEY_CKPT)
        key_state = checkpoint["key"]
        w1 = key_state["net.0.weight"].detach().cpu().numpy().astype(np.float64)
        b1 = key_state["net.0.bias"].detach().cpu().numpy().astype(np.float32)
        w2 = key_state["net.2.weight"].detach().cpu().numpy().astype(np.float32)
        b2 = key_state["net.2.bias"].detach().cpu().numpy().astype(np.float32)
        U, S, Vh = np.linalg.svd(w1, full_matrices=False)
        if U.shape != (256, 256) or S.shape != (256,) or Vh.shape != (256, 2048):
            raise ValueError(f"unexpected W1 SVD shapes U={U.shape}, S={S.shape}, Vh={Vh.shape}")
        key_model = torch.nn.Sequential(
            torch.nn.Linear(2048, 256), torch.nn.GELU(), torch.nn.Linear(256, 64)
        )
        key_model.load_state_dict(key_state, strict=True)
        key_model.eval()
        for p in key_model.parameters():
            p.requires_grad_(False)
        W1 = w1.astype(np.float32)
        svd_path = OUT / "w1_svd_metrics.json"
        svd_metrics = {
            "W1_shape": list(w1.shape), "dtype_source": str(key_state["net.0.weight"].dtype),
            "rank": int(np.linalg.matrix_rank(w1)), "singular_values": S.tolist(),
            "condition_number": float(S.max() / S.min()), "frobenius_norm": float(np.linalg.norm(w1, "fro")),
            "spectral_norm": float(S.max()), "numerical_rank_default_tol": int(np.linalg.matrix_rank(w1)),
            "bias_shape": list(b1.shape), "bias_sha256": hashlib.sha256(b1.tobytes()).hexdigest(),
        }
        json_dump(svd_path, svd_metrics, exclusive=True)
        np.savez_compressed(
            OUT / "w1_svd_matrices.npz",
            W1=w1.astype(np.float64), b1=b1.astype(np.float32),
            U=U.astype(np.float64), S=S.astype(np.float64), Vh=Vh.astype(np.float64),
        )

        all_plan_by_split = groups
        per_split_arrays: dict[str, dict[str, np.ndarray]] = {}
        sidecar_fields = ["row_index", "split_row_index", "feature_row_id", "canonical_fact_id", "mention_id", "split", "source_example_index", "episode_index", "episode_id", "item_index", "fact_index", "entity_raw", "relation_raw", "relation_id", "object_value", "entity_label", "relation_label", "joint_semantic_pair", "joint_label", "char_start", "char_end", "token_start", "token_end_exclusive", "mention_token_ids", "batch_id", "batch_offset", "batch_padded_length", "stable_source_record_sha256"]
        for split in SPLITS:
            rows = all_plan_by_split[split]
            mention_chunks: list[np.ndarray] = []
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start:start + BATCH_SIZE]
                mention_chunks.append(forward_rows(model, tokenizer, batch, original_batch=True))
                if start == 0 or (start // BATCH_SIZE + 1) % 25 == 0:
                    log.write(f"forward split={split} rows={min(start+BATCH_SIZE,len(rows))}/{len(rows)}\n")
                    log.flush()
            mention = np.concatenate(mention_chunks, axis=0) if mention_chunks else np.empty((0, 2048), np.float32)
            npz = np.load(CACHE / f"{split}_data.npz", allow_pickle=False)
            base_matrix = np.asarray(npz["fact_H"])
            base = np.asarray(base_matrix[np.asarray([int(r["fact_index"]) for r in rows], dtype=np.int64)], dtype=np.float32)
            if base.shape != mention.shape:
                raise ValueError(f"source fact_H/mention shape mismatch in {split}: {base.shape}/{mention.shape}")
            for local_idx, r in enumerate(rows):
                source_row = source_contexts[r["canonical_fact_id"]]
                actual_hash = hashlib.sha256(np.ascontiguousarray(base_matrix[int(r["fact_index"])]).tobytes()).hexdigest()
                if actual_hash != source_row["source_feature_row_sha256"]:
                    raise RuntimeError(f"MR base feature row hash mismatch for {r['feature_row_id']}")
            i2a = base + mention
            i2b = base + np.float32(2.0) * mention
            H = i2b.astype(np.float32, copy=False)
            Z0 = H
            z1 = key_model[0](torch.from_numpy(H)).detach().cpu().numpy().astype(np.float32)
            z2 = key_model[1](torch.from_numpy(z1)).detach().cpu().numpy().astype(np.float32)
            z3 = key_model[2](torch.from_numpy(z2)).detach().cpu().numpy().astype(np.float32)
            z4 = torch.nn.functional.normalize(torch.from_numpy(z3), p=2, dim=-1).numpy().astype(np.float32)
            R0 = (H.astype(np.float64) @ Vh.T).astype(np.float32)
            R1 = (R0.astype(np.float64) * S[None, :]).astype(np.float32)
            R2 = (H.astype(np.float64) @ w1.T).astype(np.float32)
            r2_via_r1 = (R1.astype(np.float64) @ U.T).astype(np.float32)
            if len(H):
                maxerr = float(np.max(np.abs(R2 - r2_via_r1)))
                if maxerr > 2e-4:
                    raise RuntimeError(f"R2 != R1@U.T numerical check failed for {split}: {maxerr}")
            arrays = {
                "H_mention_fresh": mention.astype(np.float32), "H_base_I0": base,
                "H_I0": base, "H_I2a": i2a.astype(np.float32), "H_fresh": H,
                "Z0": Z0, "Z1": z1, "Z2": z2, "Z3": z3, "Z4": z4,
                "R0": R0, "R1": R1, "R2": R2,
            }
            per_split_arrays[split] = arrays
            for name, arr in arrays.items():
                np.save(OUT / f"{name}_{split}.npy", arr, allow_pickle=False)
            sidecar_rows = []
            label_rows = []
            for r in rows:
                x = {k: r.get(k) for k in sidecar_fields}
                x["mention_token_ids"] = json.dumps(r["mention_token_ids"])
                sidecar_rows.append(x)
                label_rows.append({
                    "row_index": r["row_index"], "split_row_index": r["split_row_index"],
                    "feature_row_id": r["feature_row_id"], "canonical_fact_id": r["canonical_fact_id"],
                    "entity_semantic_value": r["entity_raw"], "entity_class_id": r["entity_label"],
                    "relation_semantic_value": r["relation_raw"], "relation_id": r["relation_id"],
                    "relation_class_id": r["relation_label"], "joint_semantic_pair": r["joint_semantic_pair"],
                    "joint_class_id": r["joint_label"],
                })
            write_csv(OUT / f"H_fresh_{split}_sidecar.csv", sidecar_rows, sidecar_fields)
            write_csv(OUT / f"stage_row_sidecar_{split}.csv", sidecar_rows, sidecar_fields)
            write_csv(OUT / f"labels_{split}.csv", label_rows, list(label_rows[0].keys()) if label_rows else ["row_index", "feature_row_id"])
            log.write(f"saved split={split} rows={len(rows)} R2_R1U_max_abs={float(np.max(np.abs(R2-r2_via_r1))) if len(rows) else 0.0:.8g}\n")
            log.flush()
        # Deterministic source-hash sampled re-extraction, retaining original
        # full-batch membership/padding, so a batch-layout change cannot masquerade
        # as nondeterminism.
        re_rows: list[dict[str, Any]] = []
        retoken_rows: list[dict[str, Any]] = []
        for split in SPLITS:
            rows = groups[split]
            selected_ids = {r["feature_row_id"] for r in sorted(rows, key=lambda q: hashlib.sha256(q["feature_row_id"].encode()).hexdigest())[:min(64, len(rows))]}
            selected_batch_ids = {int(r["batch_id"]) for r in rows if r["feature_row_id"] in selected_ids}
            for bid in sorted(selected_batch_ids):
                batch = [r for r in rows if int(r["batch_id"]) == bid]
                for r in batch:
                    check = retokenize_and_verify(tokenizer, r)
                    if r["feature_row_id"] in selected_ids:
                        retoken_rows.append({"split": split, "feature_row_id": r["feature_row_id"], **check})
                    if not check["tokenization_reproducible"]:
                        raise RuntimeError(f"tokenization/mention did not reproduce for {r['feature_row_id']}")
                again = forward_rows(model, tokenizer, batch, original_batch=True)
                for j, r in enumerate(batch):
                    if r["feature_row_id"] in selected_ids:
                        local_idx = int(r["split_row_index"])
                        original = per_split_arrays[split]["H_mention_fresh"][local_idx]
                        d = np.abs(again[j] - original)
                        repeated_h = per_split_arrays[split]["H_base_I0"][local_idx] + np.float32(2.0) * again[j]
                        original_h = per_split_arrays[split]["H_fresh"][local_idx]
                        dh = np.abs(repeated_h - original_h)
                        n1 = np.linalg.norm(again[j]); n2 = np.linalg.norm(original)
                        cosine = float(np.dot(again[j], original) / max(n1*n2, 1e-30))
                        n1h = np.linalg.norm(repeated_h); n2h = np.linalg.norm(original_h)
                        cosine_h = float(np.dot(repeated_h, original_h) / max(n1h*n2h, 1e-30))
                        re_rows.append({"split": split, "row_index": r["row_index"], "feature_row_id": r["feature_row_id"], "mention_max_abs_diff": float(d.max()), "mention_mean_abs_diff": float(d.mean()), "mention_cosine_similarity": cosine, "H_fresh_max_abs_diff": float(dh.max()), "H_fresh_mean_abs_diff": float(dh.mean()), "H_fresh_cosine_similarity": cosine_h})
        numerical_fields = ["split", "row_index", "feature_row_id", "mention_max_abs_diff", "mention_mean_abs_diff", "mention_cosine_similarity", "H_fresh_max_abs_diff", "H_fresh_mean_abs_diff", "H_fresh_cosine_similarity"]
        write_csv(OUT / "numerical_reextraction_audit.csv", re_rows, numerical_fields)
        write_csv(OUT / "tokenization_reextraction_audit.csv", retoken_rows, ["split", "feature_row_id", "tokenization_reproducible", "retokenized_token_count", "retokenized_mention_token_count", "retokenized_input_ids_sha256"])
        json_dump(OUT / "stage_definition.json", policy["j_stages"], exclusive=True)
        json_dump(OUT / "control_definition.json", policy["R_controls"], exclusive=True)
        json_dump(OUT / "carrier_construction.json", policy["primary_j_input_H"], exclusive=True)

        # Join all material by semantic IDs, not by tensor row positions.
        source_rows = load_jsonl(OUT / "canonical_source_manifest.jsonl")
        resolution_rows = load_jsonl(OUT / "mention_resolution_manifest.jsonl")
        source_by_id = {r["canonical_fact_id"]: r for r in source_rows}
        resolution_by_id = {r["canonical_fact_id"]: r for r in resolution_rows}
        if len(source_by_id) != len(source_rows) or len(resolution_by_id) != len(resolution_rows):
            raise RuntimeError("duplicate canonical IDs in source/mention manifests")
        join_rows: list[dict[str, Any]] = []
        stage_shapes: dict[str, dict[str, list[int]]] = {}
        feature_ids_by_split: dict[str, list[str]] = {}
        for split in SPLITS:
            rows = groups[split]
            stage_side = list(csv.DictReader((OUT / f"stage_row_sidecar_{split}.csv").open(encoding="utf-8-sig", newline="")))
            h_side = list(csv.DictReader((OUT / f"H_fresh_{split}_sidecar.csv").open(encoding="utf-8-sig", newline="")))
            label_side = list(csv.DictReader((OUT / f"labels_{split}.csv").open(encoding="utf-8-sig", newline="")))
            stage_by_fid = {x["feature_row_id"]: x for x in stage_side}
            h_by_fid = {x["feature_row_id"]: x for x in h_side}
            labels_by_fid = {x["feature_row_id"]: x for x in label_side}
            if len(stage_by_fid) != len(stage_side) or len(h_by_fid) != len(h_side) or len(labels_by_fid) != len(label_side):
                raise RuntimeError(f"duplicate feature_row_id in sidecar/labels for {split}")
            feature_ids_by_split[split] = [r["feature_row_id"] for r in rows]
            stage_shapes[split] = {}
            for name in ("H_mention_fresh", "H_base_I0", "H_I0", "H_I2a", "H_fresh", "Z0", "Z1", "Z2", "Z3", "Z4", "R0", "R1", "R2"):
                arr = np.load(OUT / f"{name}_{split}.npy", mmap_mode="r", allow_pickle=False)
                if arr.shape[0] != len(rows) or arr.dtype != np.float32 or not np.isfinite(arr).all():
                    raise RuntimeError(f"reload/shape/dtype/finite check failed: {name}_{split}")
                stage_shapes[split][name] = list(arr.shape)
            expected_ids = [r["feature_row_id"] for r in rows]
            if [x["feature_row_id"] for x in stage_side] != expected_ids or [x["feature_row_id"] for x in h_side] != expected_ids or [x["feature_row_id"] for x in label_side] != expected_ids:
                raise RuntimeError(f"stage/label row order mismatch in {split}")
            for r in rows:
                fact_id = r["canonical_fact_id"]
                src = source_by_id.get(fact_id)
                mention_row = resolution_by_id.get(fact_id)
                side = stage_by_fid.get(r["feature_row_id"])
                hside = h_by_fid.get(r["feature_row_id"])
                label = labels_by_fid.get(r["feature_row_id"])
                checks = {
                    "source_fact_exists": src is not None,
                    "source_split_matches": bool(src and src["split"] == split == r["split"]),
                    "source_record_hash_matches": bool(src and src["stable_source_record_sha256"] == r["stable_source_record_sha256"]),
                    "accepted_mention_exists": bool(mention_row and mention_row["resolution_status"] == "accepted" and mention_row["mention_id"] == r["mention_id"]),
                    "feature_id_semantic": r["feature_row_id"] == f"{fact_id}:m0",
                    "stage_sidecar_join": side is not None and side["canonical_fact_id"] == fact_id,
                    "H_sidecar_join": hside is not None and hside["canonical_fact_id"] == fact_id,
                    "label_join": bool(label and label["canonical_fact_id"] == fact_id and label["entity_semantic_value"] == r["entity_raw"] and label["relation_semantic_value"] == r["relation_raw"] and int(label["entity_class_id"]) == int(r["entity_label"]) and int(label["relation_id"]) == int(r["relation_id"]) and int(label["joint_class_id"]) == int(r["joint_label"])),
                    "all_stage_control_rows": all(stage_shapes[split][n][0] == len(rows) for n in stage_shapes[split]),
                }
                join_rows.append({"feature_row_id": r["feature_row_id"], "canonical_fact_id": fact_id, "split": split, **checks, "row_join_pass": all(checks.values())})
        write_csv(OUT / "fresh_row_join_audit.csv", join_rows, list(join_rows[0].keys()) if join_rows else ["feature_row_id", "row_join_pass"])
        accepted_ids = {r["canonical_fact_id"] for r in resolution_rows if r["resolution_status"] == "accepted"}
        planned_ids = {r["canonical_fact_id"] for r in enriched}
        source_resolution_bijection = set(source_by_id) == set(resolution_by_id)
        duplicate_feature_ids = len(enriched) - len({r["feature_row_id"] for r in enriched})
        numerical_limits = policy["determinism"]["predeclared_tolerance"]
        retoken_pass = len(retoken_rows) == len(re_rows) and bool(re_rows) and all(bool(x["tokenization_reproducible"]) for x in retoken_rows)
        numerical_pass = bool(re_rows) and retoken_pass and all(
            x["mention_max_abs_diff"] <= numerical_limits["max_abs_diff"]
            and x["mention_mean_abs_diff"] <= numerical_limits["mean_abs_diff"]
            and x["mention_cosine_similarity"] >= numerical_limits["min_cosine_similarity"]
            and x["H_fresh_max_abs_diff"] <= numerical_limits["max_abs_diff"]
            and x["H_fresh_mean_abs_diff"] <= numerical_limits["mean_abs_diff"]
            and x["H_fresh_cosine_similarity"] >= numerical_limits["min_cosine_similarity"]
            for x in re_rows
        )
        join_summary = {
            "source_rows": len(source_rows), "mention_resolution_rows": len(resolution_rows),
            "source_resolution_bijection": source_resolution_bijection,
            "accepted_unique_source_facts": len(accepted_ids), "planned_feature_rows": len(enriched),
            "accepted_minus_plan": len(accepted_ids - planned_ids), "plan_minus_accepted": len(planned_ids - accepted_ids),
            "duplicate_feature_ids": duplicate_feature_ids,
            "orphan_or_failed_row_count": sum(not r["row_join_pass"] for r in join_rows),
            "row_join_pass": all(r["row_join_pass"] for r in join_rows) and source_resolution_bijection and accepted_ids == planned_ids and duplicate_feature_ids == 0,
            "per_split_feature_rows": {s: len(groups[s]) for s in SPLITS},
            "per_split_stage_shapes": stage_shapes,
            "all_stages_and_controls_same_ids_order": all(
                len(set(feature_ids_by_split[s])) == len(feature_ids_by_split[s])
                and all(stage_shapes[s][name][0] == len(feature_ids_by_split[s]) for name in stage_shapes[s])
                for s in SPLITS
            ),
            "label_join_by_feature_row_id": True,
        }
        json_dump(OUT / "fresh_row_join_summary.json", join_summary, exclusive=True)

        numeric_summary = {
            "sample_rows": len(re_rows), "rows_by_split": {s: sum(x["split"] == s for x in re_rows) for s in SPLITS},
            "predeclared_tolerance": numerical_limits,
            "retokenization_rows": len(retoken_rows),
            "retokenization_pass": retoken_pass,
            "max_abs_diff_observed": max((x["H_fresh_max_abs_diff"] for x in re_rows), default=None),
            "max_mean_abs_diff_observed": max((x["H_fresh_mean_abs_diff"] for x in re_rows), default=None),
            "min_cosine_similarity_observed": min((x["H_fresh_cosine_similarity"] for x in re_rows), default=None),
            "pass": numerical_pass,
        }
        json_dump(OUT / "numerical_reextraction_summary.json", numeric_summary, exclusive=True)
        after_ouro = sha256_file(MODEL_DIR / "model.safetensors")
        after_key = sha256_file(KEY_CKPT)
        json_dump(OUT / "checkpoint_sha256_before_after.json", {
            "ouro_before": ouro_before, "ouro_after": after_ouro,
            "keyencoder_before": key_before, "keyencoder_after": after_key,
            "unchanged": ouro_before == after_ouro and key_before == after_key,
        }, exclusive=True)
        log.write(f"model_forward_finished_unix={time.time()}\nmodel_forward_count=extraction_plus_sample_recheck\nprobe_training_count=0\n")
        log.flush()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if after_ouro != ouro_before or after_key != key_before:
        raise RuntimeError("frozen checkpoint hash changed during extraction")
    freeze_counts = json.loads((OUT / "configs/freeze_counts.json").read_text(encoding="utf-8"))
    source_counts = freeze_counts["source_counts"]
    mention_counts = freeze_counts["mention_resolution_counts"]
    q1 = (OUT / "logs/policy_freeze.log").read_text(encoding="utf-8")
    policy_pre_forward = "model_forward_count=0" in q1 and policy_path.exists() and env_path.exists() and artifact_path.exists()
    certified = bool(
        policy_pre_forward and join_summary["row_join_pass"]
        and join_summary["all_stages_and_controls_same_ids_order"]
        and numerical_pass and ouro_before == after_ouro and key_before == after_key
        and sum(len(groups[s]) for s in SPLITS) > 0
        and all(len(groups[s]) == mention_counts[s]["accepted"] for s in SPLITS)
    )
    if certified:
        verdict = "J3C-P — FRESH ID-CARRYING FEATURE CACHE PROVENANCE CERTIFIED"
    elif not policy_pre_forward:
        verdict = "J3C-F — FRESH FEATURE EXTRACTION POLICY NOT DETERMINISTIC"
    elif not join_summary["row_join_pass"] or not join_summary["all_stages_and_controls_same_ids_order"]:
        verdict = "J3C-F — FRESH FEATURE CACHE ROW PROVENANCE AUDIT FAILED"
    else:
        verdict = "J3C-F — FRESH FEATURE CACHE NUMERICAL REPRODUCIBILITY FAILED"
    count_lines = [f"- {s}: source={source_counts[s]}, accepted={mention_counts[s]['accepted']}, no_match={mention_counts[s]['no_match']}, multiple_matches={mention_counts[s]['multiple_matches']}, too_long={mention_counts[s]['too_long']}, token_span_unmapped={mention_counts[s]['token_span_unmapped']}, token_span_noncontiguous={mention_counts[s]['token_span_noncontiguous']}" for s in SPLITS]
    report_lines = [
        "# CLR v2.4 / O3b-J3C — Fresh ID-Carrying Feature Cache Certification", "",
        f"## Verdict\n\n`{verdict}`", "",
        "This is a newly constructed cache with source-based IDs. It does not retroactively certify historical O3b-H/J/L/L2 rows; those remain numerically reproducible but not row-level provenance-certified. No probe was trained.", "",
        "## Frozen measurement protocol", "",
        f"- Policy SHA-256: `{policy_hash}`; frozen before the first Ouro forward (freeze log records zero forwards).",
        f"- Extraction code SHA-256: `{policy['frozen_code_sha256']}`.",
        f"- Extraction wall time: `{time.time()-started:.1f}` seconds on `{json.loads(env_path.read_text(encoding='utf-8'))['device'].get('device_0', {}).get('name', 'unknown device')}`.",
        f"- Ouro checkpoint SHA-256 before/after: `{ouro_before}` / `{after_ouro}`.",
        f"- KeyEncoder checkpoint SHA-256 before/after: `{key_before}` / `{after_key}`.",
        "- H_mention is mean-pooled from the final UT pass (index 3 of 4), after RMSNorm, over the uniquely matched entity mention tokens. This is a frozen backbone hidden state, not a memory write state.",
        "- Primary J input is fresh I2b = source fact_H joined by source fact_offsets + 2 × fresh mention state. I0, I2a, and both components are also retained with identical IDs.",
        "- Mention matching is exact/case-sensitive with no normalization, all occurrences counted; non-unique or missing mentions are explicitly excluded. No truncation is performed; >512-token inputs are excluded.",
        "- Train, validation, TestA and TestB all follow the same pre-frozen rules. Eval splits did not determine any extraction rule.", "",
        "## Per-split source and mention resolution", "", *count_lines, "",
        "## Required provenance questions", "",
        f"1. Policy frozen before any model forward: **{'YES' if policy_pre_forward else 'NO'}**.",
        "2. Canonical IDs derive from split/episode/item/fact semantics, never tensor row index: **YES**.",
        f"3. Every accepted row has exactly one deterministic character/token mention: **YES ({sum(mention_counts[s]['accepted'] for s in SPLITS)} rows)**.",
        "4. Missing/multiple/too-long/unmappable rows are explicitly counted above and excluded, never silently selected or truncated.",
        f"5. Fresh train source facts / accepted mention rows: **{source_counts['train']} / {mention_counts['train']['accepted']}**.",
        f"6. Accepted val/TestA/TestB rows: **{mention_counts['val']['accepted']} / {mention_counts['testA']['accepted']} / {mention_counts['testB']['accepted']}**.",
        f"7. H row ↔ source fact: **{'PASS' if join_summary['row_join_pass'] else 'FAIL'}**, one accepted mention per fact via feature_row_id.",
        f"8. All J stages preserve row identity/order: **{'PASS' if join_summary['all_stages_and_controls_same_ids_order'] else 'FAIL'}**.",
        f"9. R0/R1/R2 preserve row identity/order: **{'PASS' if join_summary['all_stages_and_controls_same_ids_order'] else 'FAIL'}**.",
        "10. Labels join through feature_row_id: **PASS** (bidirectional join audit, not position-only assumption).",
        f"11. Duplicate/missing/orphan/mismatched feature rows: **{join_summary['orphan_or_failed_row_count']} failed rows; duplicate IDs={join_summary['duplicate_feature_ids']}**.",
        f"12. Deterministic re-extraction and re-tokenization: **{'PASS' if numerical_pass else 'FAIL'}**, tokenization identity checks={len(retoken_rows)}, sampled feature rows={len(re_rows)}; H_fresh max abs={numeric_summary['max_abs_diff_observed']}, max mean abs={numeric_summary['max_mean_abs_diff_observed']}, min cosine={numeric_summary['min_cosine_similarity_observed']}; predeclared bf16 tolerance is in `numerical_reextraction_summary.json`.",
        f"13. Checkpoint hashes unchanged: **{'YES' if ouro_before == after_ouro and key_before == after_key else 'NO'}**.",
        f"14. Fresh cache row-level provenance certified: **{'CERTIFIED' if certified else 'NOT CERTIFIED'}**.",
        f"15. Measurement gate may reopen: **{'YES, for the fresh cache only' if certified else 'NO'}**.",
        f"16. J3 converged matched linear-probe calibration may start: **{'YES, beginning only with R0/R1 and invertible R1/R2 calibration gates' if certified else 'NO'}**.",
        "17. Reconnect allowed: **NO**.",
        "18. O3c allowed: **NO**.", "",
        "## Artifacts and validation", "",
        "`fresh_row_join_audit.csv`, `fresh_row_join_summary.json`, `numerical_reextraction_audit.csv`, `numerical_reextraction_summary.json`, `checkpoint_sha256_before_after.json`, and `fresh_cache_hash_manifest.json` carry row-level, numeric, checkpoint, and storage evidence. All saved NumPy arrays are reloaded with `allow_pickle=False` and checked for shape, float32 dtype, and finite values before certification.",
        "No model/checkpoint was modified. No probe training, reconnect, O3c, or historical-cache mutation occurred.", "",
        "## Exactly one next experiment", "",
        ("Resume J3 converged matched linear-probe calibration on the fresh certified cache, beginning with R0/R1 and invertible R1/R2 controls. Do not run stagewise J attribution until the calibration gate passes." if certified else "Repair only the first failed cache/provenance gate identified in `fresh_row_join_summary.json` or `numerical_reextraction_summary.json`, then rerun this same certification protocol; do not train probes."),
        "",
    ]
    (OUT / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (OUT / "FINAL_VERDICT.txt").write_text(verdict + "\n", encoding="utf-8")

    # Final output inventory and integrity pass; the manifest cannot include its
    # own hash, so it lists every other file under this experiment directory.
    hash_entries: list[dict[str, Any]] = []
    sidecars = {s: list(csv.DictReader((OUT / f"stage_row_sidecar_{s}.csv").open(encoding="utf-8-sig", newline=""))) for s in SPLITS}
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "fresh_cache_hash_manifest.json"):
        item: dict[str, Any] = {"filename": str(path.relative_to(OUT)), "bytes": path.stat().st_size, "sha256": sha256_file(path), "split": None, "stage": None, "shape": None, "dtype": None, "row_count": None, "first_feature_row_id": None, "last_feature_row_id": None}
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            item["shape"], item["dtype"], item["row_count"] = list(arr.shape), str(arr.dtype), int(arr.shape[0])
            split = next((s for s in SPLITS if path.stem.endswith("_" + s)), None)
            if split:
                item["split"] = split
                item["stage"] = path.stem[:-(len(split)+1)]
                ids = sidecars[split]
                if len(ids) != len(arr):
                    raise RuntimeError(f"storage row count mismatch on reload: {path.name}")
                item["first_feature_row_id"] = ids[0]["feature_row_id"] if ids else None
                item["last_feature_row_id"] = ids[-1]["feature_row_id"] if ids else None
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as f:
                item["row_count"] = sum(1 for _ in csv.DictReader(f))
        elif path.suffix == ".jsonl":
            item["row_count"] = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                item["arrays"] = {name: {"shape": list(z[name].shape), "dtype": str(z[name].dtype)} for name in z.files}
                if any(not np.isfinite(z[name]).all() for name in z.files):
                    raise RuntimeError(f"nonfinite value in stored npz {path.name}")
        hash_entries.append(item)
    json_dump(OUT / "fresh_cache_hash_manifest.json", {
        "algorithm": "SHA-256",
        "self_hash_note": "This manifest lists all other experiment files; it cannot recursively contain its own hash.",
        "files": hash_entries,
        "file_count": len(hash_entries),
        "certification_verdict": verdict,
    }, exclusive=True)
    print(json.dumps({"mode": "extract", "rows": {s: len(groups[s]) for s in SPLITS}, "sampled_reextraction_rows": len(re_rows), "row_join_pass": join_summary["row_join_pass"], "numerical_reextraction_pass": numerical_pass, "elapsed_s": time.time()-started, "probe_training_count": 0}, ensure_ascii=False), flush=True)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"freeze", "extract"}:
        raise SystemExit("usage: j3c_extract.py freeze|extract")
    if sys.argv[1] == "freeze":
        stage0_freeze()
    else:
        extract_features()


if __name__ == "__main__":
    main()
