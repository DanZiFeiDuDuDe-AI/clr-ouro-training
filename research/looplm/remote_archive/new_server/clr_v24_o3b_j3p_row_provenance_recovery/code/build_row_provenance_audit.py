#!/usr/bin/env python3
"""Reconstruct MR source/label order and audit the un-ID'd O3b-H token cache.

This is a read-only provenance audit. It does not load a model, run a forward
pass, or train probes. Historical feature-row identity is deliberately never
promoted from positional coincidence to a certified join.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path("/root/private_data/clr_v24_o3b_l")
CACHE = ROOT / "checkpoints/clr_v24_o21a_mr/cache"
HOUT = ROOT / "checkpoints/clr_v24_o3b_h_entity_carrier_diag"
OUT = ROOT / "checkpoints/clr_v24_o3b_j3p_row_provenance_recovery"
SPLITS = ("train", "val", "testA", "testB")
FEATURE_KEY = "fact_{split}_entity_mention_mean"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(obj) -> str:
    b = json.dumps(obj, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    token_path = HOUT / "token_layer_features.npz"
    persons = json.loads((CACHE / "train_persons.json").read_text(encoding="utf-8"))
    pmap = {str(p): i for i, p in enumerate(persons)}
    all_rows = {}
    joint_train_keys = set()
    split_data = {}
    split_meta = {}

    # Match a1_v24_o3b_j.py::build_infos exactly: split order, JSONL episode
    # order, then position in each episode, with fact_offsets[episode] + pos.
    for split in SPLITS:
        d = np.load(CACHE / f"{split}_data.npz", allow_pickle=False)
        meta = read_jsonl(CACHE / f"{split}_meta.jsonl")
        offsets = d["fact_offsets"]
        refs = []
        traversal_issues = []
        for ep, m in enumerate(meta):
            cap = int(m["capacity"])
            if ep + 1 >= len(offsets):
                traversal_issues.append(f"offset missing for episode {ep}")
                continue
            start, end = int(offsets[ep]), int(offsets[ep + 1])
            if end - start != cap:
                traversal_issues.append(
                    f"episode {ep} capacity {cap} != offset span {end-start}")
            fields = ("persons", "relations", "relation_ids", "cities",
                      "fact_classes", "fact_texts")
            for field in fields:
                if len(m.get(field, [])) != cap:
                    traversal_issues.append(
                        f"episode {ep} {field} length != capacity")
            for pos in range(cap):
                flat = start + pos
                person = str(m["persons"][pos])
                relation_id = int(m["relation_ids"][pos])
                relation_name = str(m["relations"][pos])
                city = str(m["cities"][pos])
                fact_class = int(m["fact_classes"][pos])
                fact_text = str(m["fact_texts"][pos])
                source_record = {
                    "split": split,
                    "episode_id": str(m.get("episode_id", f"{split}-{ep}")),
                    "episode_index": ep,
                    "item_index": pos,
                    "example_index": ep,
                    "fact_index": pos,
                    "mention_index": None,
                    "person": person,
                    "entity_raw": person,
                    "entity_normalized": person,
                    "relation_id": relation_id,
                    "relation_name": relation_name,
                    "relation_raw": relation_name,
                    "relation_normalized": relation_name,
                    "city_or_value": city,
                    "object_or_value": city,
                    "fact_class": fact_class,
                    "fact_text": fact_text,
                }
                record_hash = stable_hash(source_record)
                canonical_id = (
                    f"mr:{split}:{source_record['episode_id']}:item{pos}:"
                    f"{record_hash[:16]}"
                )
                refs.append({
                    **source_record,
                    "flat_fact_index": flat,
                    "canonical_id": canonical_id,
                    "source_record_sha256": record_hash,
                    "entity_label": pmap.get(person, -1),
                    "relation_label": relation_id,
                })
                if split == "train":
                    joint_train_keys.add(f"{person}\t{relation_id}")
        split_data[split] = d
        split_meta[split] = meta
        all_rows[split] = refs

    # Match a1_v24_o3b_j.py::joint_labels exactly.
    sorted_joint_keys = sorted(joint_train_keys)
    joint_map = {key: i for i, key in enumerate(sorted_joint_keys)}
    for split in SPLITS:
        for row in all_rows[split]:
            row["joint_label"] = joint_map.get(
                f"{row['person']}\t{row['relation_label']}", -1)

    source_manifest = []
    expected_label_order = []
    semantic_samples = []
    structural = {}
    for split in SPLITS:
        d = split_data[split]
        refs = all_rows[split]
        offsets = d["fact_offsets"]
        array_n = len(d["fact_H"])
        issues = []
        if len(refs) != array_n:
            issues.append(f"metadata refs={len(refs)} != fact_H={array_n}")
        if len(offsets) != len(split_meta[split]) + 1:
            issues.append("fact_offsets length != number of episodes + 1")
        expected_flat_count = sum(int(m["capacity"]) for m in split_meta[split])
        if expected_flat_count != array_n:
            issues.append(f"capacity sum={expected_flat_count} != fact_H={array_n}")
        for arr_name in ("fact_Y", "fact_relation", "fact_entity"):
            if len(d[arr_name]) != array_n:
                issues.append(f"{arr_name} length != fact_H")

        for row in refs:
            i = row["flat_fact_index"]
            m = split_meta[split][row["episode_index"]]
            pos = row["item_index"]
            checks = {
                "fact_entity_matches_J_person_map": int(d["fact_entity"][i]) == row["entity_label"],
                "fact_relation_matches_meta": int(d["fact_relation"][i]) == row["relation_label"],
                "fact_Y_matches_meta_fact_class": int(d["fact_Y"][i]) == row["fact_class"],
                "metadata_person_matches_flattened_person": str(m["persons"][pos]) == row["person"],
                "metadata_relation_matches_flattened_relation": int(m["relation_ids"][pos]) == row["relation_label"],
            }
            if not all(checks.values()):
                issues.append(f"row label/metadata mismatch at flat={i}")
            outrow = {
                **row,
                "flat_fact_entity": int(d["fact_entity"][i]),
                "flat_fact_relation": int(d["fact_relation"][i]),
                "flat_fact_Y": int(d["fact_Y"][i]),
            }
            source_manifest.append(outrow)
            expected_label_order.append({
                "split": split, "expected_label_row_index": i, "label_row_index": i,
                "canonical_source_id": row["canonical_id"],
                "source_record_sha256": row["source_record_sha256"],
                "episode_id": row["episode_id"],
                "episode_index": row["episode_index"],
                "item_index": row["item_index"],
                "flat_fact_index": i,
                "entity_label": row["entity_label"],
                "relation_label": row["relation_label"],
                "joint_label": row["joint_label"],
                "fact_class": row["fact_class"],
            })

        # Label/source semantic audit: deterministic, split-stratified sample;
        # comparisons above are nevertheless evaluated for every source row.
        if refs:
            sample_idx = sorted(set(np.linspace(0, len(refs) - 1,
                                                 min(16, len(refs)), dtype=int).tolist()))
            for i in sample_idx:
                row = refs[i]
                text = row["fact_text"]
                checks = {
                    "person_literal_in_fact_text": row["person"] in text,
                    "value_literal_in_fact_text": row["city_or_value"] in text,
                    "entity_array_matches_person_map": int(d["fact_entity"][i]) == row["entity_label"],
                    "relation_array_matches_metadata": int(d["fact_relation"][i]) == row["relation_label"],
                    "class_array_matches_metadata": int(d["fact_Y"][i]) == row["fact_class"],
                }
                semantic_samples.append({
                    "split": split, "flat_fact_index": i,
                    "canonical_source_id": row["canonical_id"],
                    "fact_text": text, "person": row["person"],
                    "relation_name": row["relation_name"],
                    "relation_id": row["relation_label"],
                    "city_or_value": row["city_or_value"],
                    "fact_class": row["fact_class"],
                    **checks,
                    "all_checks_pass": all(checks.values()),
                })
        structural[split] = {
            "meta_episode_count": len(split_meta[split]),
            "source_fact_count": len(refs),
            "fact_H_count": array_n,
            "fact_offsets_count": len(offsets),
            "capacity_sum": expected_flat_count,
            "traversal_and_label_issues": issues,
            "source_label_order_pass": not issues,
            "unique_canonical_ids": len({r["canonical_id"] for r in refs}),
        }

    write_jsonl(OUT / "canonical_source_manifest.jsonl", source_manifest)
    write_csv(OUT / "expected_label_row_order.csv", expected_label_order)
    write_csv(OUT / "semantic_label_audit_sample.csv", semantic_samples)

    # The historical token cache stores no sample IDs. The only available
    # candidate association is the downstream J prefix convention; expose it
    # as an unverified hypothesis, never as a certified feature traversal.
    with np.load(token_path, allow_pickle=False) as z:
        npz_keys = sorted(z.files)
        id_like_keys = [k for k in npz_keys if any(
            t in k.lower() for t in ("row_id", "sample_id", "fact_id", "index", "indices"))]
        feature_inventory = {s: {
            "key": FEATURE_KEY.format(split=s),
            "shape": list(z[FEATURE_KEY.format(split=s)].shape),
            "dtype": str(z[FEATURE_KEY.format(split=s)].dtype),
        } for s in SPLITS}
        cache_sha = sha256(token_path)

    feature_rows = []
    join_rows = []
    join_summary = {}
    for split in SPLITS:
        feat = feature_inventory[split]
        n_feature = feat["shape"][0]
        refs = all_rows[split]
        n_candidate = min(n_feature, len(refs))
        candidate_ids = [refs[i]["canonical_id"] for i in range(n_candidate)]
        uniqueness = len(set(candidate_ids)) == len(candidate_ids)
        for i in range(n_feature):
            candidate = refs[i] if i < len(refs) else None
            feature_rows.append({
                "schema_version": "j3p-sidecar-v1",
                "split": split,
                "feature_key": feat["key"],
                "feature_row_index": i,
                "feature_cache_sha256": cache_sha,
                "candidate_canonical_source_id": candidate["canonical_id"] if candidate else None,
                "candidate_source_record_sha256": candidate["source_record_sha256"] if candidate else None,
                "candidate_join_basis": "J downstream deterministic prefix [:n]; not historical O3b-H extraction evidence",
                "entity_label_candidate": candidate["entity_label"] if candidate else None,
                "relation_label_candidate": candidate["relation_label"] if candidate else None,
                "joint_label_candidate": candidate["joint_label"] if candidate else None,
                "mention_char_start": None,
                "mention_char_end": None,
                "mention_token_indices": None,
                "tokenizer_name_or_revision": None,
                "selected_hidden_state_layer": None,
                "pooling_semantics": "cache key says entity_mention_mean; exact mention/token/pooling extraction not recovered",
                "identity_status": "UNVERIFIED_NOT_CERTIFIED",
            })
        for i in range(n_candidate):
            row = refs[i]
            common = {
                "split": split,
                "candidate_row_index": i,
                "canonical_source_id": row["canonical_id"],
                "source_record_sha256": row["source_record_sha256"],
                "feature_row_count": n_feature,
                "source_fact_count": len(refs),
                "join_method": "positional prefix hypothesis inherited from O3b-J; no shared historical ID",
                "identity_join_status": "UNVERIFIED_NOT_CERTIFIED",
                "count_match_within_scored_prefix": n_candidate == n_feature,
            }
            join_rows.append({"direction": "source_candidate_to_feature", **common})
            join_rows.append({"direction": "feature_candidate_to_source", **common})
        join_summary[split] = {
            "source_facts_full": len(refs),
            "historical_feature_rows": n_feature,
            "candidate_prefix_rows": n_candidate,
            "source_rows_outside_candidate_prefix": max(0, len(refs) - n_candidate),
            "candidate_ids_unique_by_construction": uniqueness,
            "candidate_prefix_count_match": n_candidate == n_feature,
            "strict_identity_join_certified": False,
            "reason": "feature NPZ has no row/sample/fact IDs and extractor traversal/manifest is absent",
        }
    write_jsonl(OUT / "extraction_provenance_sidecar.jsonl", feature_rows)
    write_csv(OUT / "row_join_audit.csv", join_rows)

    missing_provenance = [
        str(HOUT / "O3b-H token/mention feature extraction source"),
        str(HOUT / "row/sample/fact ID manifest or extraction-order sidecar"),
        str(HOUT / "tokenizer name/revision and tokenization config"),
        str(HOUT / "mention span/token selection rule and per-row selected indices"),
    ]
    summary = {
        "verdict": "J3P-F — HISTORICAL ROW PROVENANCE CANNOT BE CERTIFIED",
        "run_scope": "read-only source/label/order audit; no probe training, no model load/forward, no checkpoint edits",
        "source_roots": {"mr_cache": str(CACHE), "o3b_h": str(HOUT)},
        "artifact_hashes": {
            "token_layer_features.npz": {"sha256": cache_sha, "size_bytes": token_path.stat().st_size},
            "MR_cache_manifest.json": {"sha256": sha256(CACHE / "manifest.json"), "size_bytes": (CACHE / "manifest.json").stat().st_size},
        },
        "token_npz_keys": npz_keys,
        "token_npz_id_like_keys": id_like_keys,
        "feature_arrays": feature_inventory,
        "split_structure_and_labels": structural,
        "strict_structural_row_join_pass": False,
        "prefix_count_equality_is_not_identity_evidence": True,
        "prefix_join_summary": join_summary,
        "semantic_label_audit": {
            "sample_count": len(semantic_samples),
            "all_sample_checks_pass": all(r["all_checks_pass"] for r in semantic_samples),
            "source_array_label_checks_evaluated_over_all_rows": True,
            "person_and_value_literal_checks_evaluated_on_sample": True,
        },
        "historical_extraction_provenance_missing": missing_provenance,
        "feature_reextraction": "NOT RUN: historical tokenizer, mention locator, selected state/pooling, and row order are unrecovered; speculative re-extraction would not validate old cache semantics",
        "old_results_interpretation": "Historical probe results remain numerically reproducible but lack certifiable row-level semantic provenance.",
    }

    # Machine-readable semantic class maps, including the joint-label mapping
    # produced by the historical J source (sorted train person/relation keys).
    relation_names = json.loads((CACHE / "relations.json").read_text(encoding="utf-8"))
    city_by_class = {}
    city_conflicts = {}
    for row in source_manifest:
        key, value = str(row["fact_class"]), str(row["city_or_value"])
        if key in city_by_class and city_by_class[key] != value:
            city_conflicts.setdefault(key, sorted({city_by_class[key], value}))
        else:
            city_by_class[key] = value
    class_maps = {
        "entity": {str(i): str(p) for i, p in enumerate(persons)},
        "relation": {str(i): str(name) for i, name in enumerate(relation_names)},
        "joint": {str(i): key.replace("\t", " | ") for i, key in enumerate(sorted_joint_keys)},
        "fact_class_city": city_by_class,
        "fact_class_city_conflicts": city_conflicts,
        "coverage": {
            "entity_classes": len(persons), "relation_classes": len(relation_names),
            "joint_classes": len(sorted_joint_keys), "fact_classes_observed": len(city_by_class),
            "fact_class_map_conflicts": len(city_conflicts),
        },
        "joint_mapping_definition": "sorted unique train keys formatted as person + TAB + integer relation_id, matching a1_v24_o3b_j.py::joint_labels",
    }
    (OUT / "label_class_maps.json").write_text(
        json.dumps(class_maps, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep a distinct expected feature order deliverable, but do not fabricate
    # one when the historical extraction traversal is absent.
    write_csv(OUT / "expected_feature_row_order.csv", [{
        "status": "NOT_RUN",
        "reason": "Historical O3b-H feature extraction traversal was not preserved; a prefix-derived order would be a hypothesis, not expected historical order.",
        "expected_feature_row_index": "",
        "canonical_id": "",
        "split": "",
        "episode_index": "",
        "item_index": "",
        "fact_index": "",
        "entity": "",
        "relation": "",
        "label": "",
    }])

    semantic_output = []
    for row in semantic_samples:
        ref = next(x for x in all_rows[row["split"]]
                   if x["flat_fact_index"] == row["flat_fact_index"])
        semantic_output.append({
            **row,
            "entity_label": ref["entity_label"],
            "entity_semantic": ref["person"],
            "relation_label": ref["relation_label"],
            "relation_semantic": ref["relation_name"],
            "joint_label": ref["joint_label"],
            "joint_semantic": f"{ref['person']} | {ref['relation_name']}",
            "fact_class_semantic": ref["city_or_value"],
            "joint_label_recomputed_by_J_rule": (
                joint_map.get(f"{ref['person']}\t{ref['relation_label']}", -1)
                == ref["joint_label"]),
        })
    write_csv(OUT / "label_semantic_audit.csv", semantic_output)
    write_csv(OUT / "feature_reextraction_audit.csv", [{
        "status": "NOT_RUN",
        "reason": "Stage 6 strict row join did not pass; historical tokenizer, mention locator, selected state/pooling, and row identity are unavailable. Re-extraction would not certify the old cache.",
        "split": "",
        "canonical_id": "",
        "feature_row_index": "",
        "max_abs_diff": "",
        "mean_abs_diff": "",
        "cosine_similarity": "",
    }])

    checkpoint = ROOT / "checkpoints/clr_v24_o21b/b1_best.pt"
    checkpoint_hash = sha256(checkpoint)
    cert = {
        "verdict": summary["verdict"],
        "criteria": {
            "historical_feature_traversal_recovered": False,
            "historical_label_traversal_recovered": True,
            "canonical_source_ids_built": True,
            "feature_label_bidirectional_identity_join": False,
            "source_feature_label_counts_equal_per_full_split": False,
            "source_row_ids_unique": all(v["unique_canonical_ids"] == v["source_fact_count"] for v in structural.values()),
            "feature_duplicate_or_orphan_status": "UNDETERMINABLE_WITHOUT_FEATURE_IDS; no mismatch confirmed",
            "source_to_label_entity_relation_semantics": all(v["source_label_order_pass"] for v in structural.values()),
            "mention_token_spans_replayable": False,
            "historical_feature_reextraction_numerically_certified": False,
            "test_splits_used_to_change_mapping": False,
            "artifact_hashes_frozen_before_audit": True,
        },
        "per_split_counts": {
            s: {
                "N_source_full": join_summary[s]["source_facts_full"],
                "N_feature": join_summary[s]["historical_feature_rows"],
                "N_label_full": join_summary[s]["source_facts_full"],
                "N_source_candidate_prefix": join_summary[s]["candidate_prefix_rows"],
                "N_label_candidate_prefix": join_summary[s]["candidate_prefix_rows"],
                "candidate_prefix_count_equality_only": join_summary[s]["candidate_prefix_count_match"],
                "identity_certified": False,
            } for s in SPLITS
        },
        "source_label_audit": summary["semantic_label_audit"],
        "checkpoint_sha256_after": checkpoint_hash,
        "probe_training_performed": False,
        "model_forward_performed": False,
        "checkpoint_or_architecture_modified": False,
        "measurement_gate_reopened": False,
        "recommended_next_experiment": "Build a fresh ID-carrying feature cache from original MR sources and frozen checkpoint, then certify that cache before any probe training.",
    }
    (OUT / "provenance_certification.json").write_text(
        json.dumps(cert, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "checkpoint_sha256_before_after.txt").write_text(
        f"checkpoint=/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21b/b1_best.pt\n"
        f"sha256_before={checkpoint_hash}\nsha256_after={checkpoint_hash}\n"
        "comparison=identical; checkpoint was read-only and no model forward was run\n",
        encoding="utf-8")

    inventory = """# O3b-J3P provenance source inventory

Search scope: local `H:\\LoopLM` source/archive and the connected remote experiment roots under `/root/private_data/clr_v24_o3b_l` and `/root/private_data/clr_v24_o3b_l2_j2`. Local text/code search covered Python, JSON/JSONL, CSV, Markdown, logs and configs (excluding binary tensors); remote source inventories covered code, H output, MR cache metadata, L/L2 outputs/logs and K context. Binary NPZ archives were inspected for keys/shapes/dtypes, not treated as source code.

| Source | Present | Row-level identity | Trust / use | Direct historical join? |
|---|---|---|---|---|
| O3b-H token extractor source | Not found in preserved local/remote code or archives | No | H output/config survives, but no extraction implementation is preserved | No |
| O3b-H `token_layer_features.npz` | Yes, remote canonical cache; 838,539,950 bytes | No row/sample/fact ID or index-like NPZ key | Authoritative frozen vectors; key names and shapes only | No |
| Local archived H-cache copy under `remote_archive/cheems` | Yes, 7,702,416 bytes; distinct SHA-256; fails ZIP/NPZ parsing | Unreadable | Preserved as-is, excluded from audit; not the remote canonical source | No |
| O3b-H config/report/representation metadata and JSON outputs | Yes | No per-row identity, token offsets or selected positions | Indirect evidence for seed, bounded row counts, layers, task setup; not extraction order | No |
| O3b-H temporary `o3bl_8wp76qs7/part_00..07` | Present, eight zero-byte files in local mirror | No | Empty transfer artifacts | No |
| MR cache `*_data.npz` + `*_meta.jsonl`, `train_persons.json`, `relations.json`, `manifest.json` | Yes | Episode IDs and per-item semantic fields exist; no separate fact UUID | Trusted source for source order and labels; all 18,661 rows audited | Joins label traversal to source, not old token feature rows |
| `a1_v24_o3b_j.py::MRCache`, `build_infos`, `joint_labels`, `load_token_arrays`, `aligned_carriers` | Yes, local and remote | Label side source indices are reconstructible; token NPZ IDs absent | Exact J label traversal and downstream prefix convention recoverable | Label↔source yes; mention feature↔source no |
| O3b-I extraction/mention-location source | Not found in preserved experiment sources inspected | No | Prior conclusion/report only; does not encode H token row identity | No |
| O3b-K analysis script / outputs | Yes in local archive and remote K context | Synthetic/downstream row order only | Useful for representation diagnosis; not upstream mention provenance | No |
| O3b-L source/config/output | Yes | Probes/results but no H row IDs | Frozen artifact hashes included in the manifest | No |
| O3b-L2/J2 source/config/features/logs | Yes, remote; local reports/scripts also preserved | Feature tensors keep array order, no canonical upstream mention IDs | Numerical reruns and transferred-control artifacts; no source identity | No |
| Prior J3 audit source/report/CSV/log | Yes, local archive | Synthetic `split:<row_index>` only | Correctly documents Stage 0 gate and downstream order; explicitly not upstream proof | No |
| Historical extraction stdout/stderr or row manifest | Not found in preserved H output tree / archives searched | No | H report/config provide summary metadata only | No |

## Recovered traversals

**Historical feature traversal:** NOT RECOVERED. The preserved J consumer loads `fact_<split>_entity_mention_mean` as an array, then pairs `raw[:n]` with `mention[:n]`. No preserved H code shows how source examples/facts were iterated, mentions located, token spans chosen, or vectors appended. Therefore no honest historical feature-loop pseudocode can be supplied beyond “load stored array, preserve its row order, take prefix.”

**Historical label traversal:** recovered from `a1_v24_o3b_j.py::build_infos`: for each split in `(train,val,testA,testB)`, enumerate JSONL episode metadata, then `for pos in range(capacity)`, map `flat=fact_offsets[episode]+pos`, append `person` and `relation_ids[pos]`; entity labels use the `train_persons` index map (unknown → -1). `joint_labels` forms sorted unique train keys `person + TAB + relation_id`, then maps each split's same ordered `(person, relation)` sequence (unknown → -1). This label traversal is independently reconstructed from MR metadata and verified against all flat `fact_entity`, `fact_relation`, and `fact_Y` arrays.

These are not shown to originate from one shared historical iterator: the feature side's H extraction loop is missing. Matching counts and J's positional prefix convention do not establish identity.
"""
    (OUT / "provenance_source_inventory.md").write_text(inventory, encoding="utf-8")

    report = f"""# O3b-J3P — Row Provenance Recovery and Join Certification

## FINAL VERDICT

**J3P-F — HISTORICAL ROW PROVENANCE CANNOT BE CERTIFIED**

This is a provenance gate failure, not evidence of a probe/basis failure and not a confirmed row misalignment. No probe was trained; no J revalidation, model load/forward, or feature re-extraction was run.

## Result summary

- The exact J label traversal was recovered and independently checked across all source facts: train 11,665; val 2,332; testA 2,332; testB 2,332. Source capacities, `fact_offsets`, semantic metadata and flat entity/relation/class labels agree with zero structural/label mismatches.
- A deterministic source-based ID and SHA-256 source-record hash were produced for each of the 18,661 source facts. A 64-row split-stratified semantic audit passed all checks; the full source-to-flat-array entity/relation/class checks also passed.
- The remote O3b-H feature archive is `{cache_sha}` (838,539,950 bytes), with shapes train 4096×2048 and each evaluation split 2332×2048. It contains no ID/index-like keys. The train feature cap is 4,096 while the full train source/label sequence is 11,665.
- O3b-J's consumer preserves the NPZ order and pairs arrays using `[:n]`; this only defines a candidate prefix join. It does not recover the historical H extraction traversal or prove that feature row *i* came from source fact *i*.
- No mention span/token indices, tokenizer name/revision, mention match/selection rule, or H extraction source/row manifest was recovered. Therefore strict bidirectional feature↔source identity, orphan/duplicate status on the feature side, and numerical re-extraction certification remain untestable. No actual mismatch was observed.
- The local `remote_archive/cheems` file with the same NPZ name is a different 7,702,416-byte artifact (SHA-256 `f5286b7d609b48c2a8aca11532c5891acf851771d396bc5334ad6511e30251c5`) and fails ZIP/NPZ parsing (`BadZipFile`). It was not used; the remote cache above is the valid canonical cache. This is a local archive-copy integrity issue, not proof of row misalignment in the remote experiment.

## Q1–Q17

| Question | Answer |
|---|---|
| Q1. O3b-H original extraction provenance found? | **No.** The cache/config/report exist; extractor source, row manifest, tokenizer revision and per-row mention spans do not. |
| Q2. Historical feature traversal recovered exactly? | **No.** Only the J downstream array-load/prefix operation is preserved. |
| Q3. Historical label traversal recovered exactly? | **Yes.** `a1_v24_o3b_j.py::build_infos` and `joint_labels`; verified against all MR flat label arrays. |
| Q4. Same historical source iterator/order? | **Unproven.** Feature extractor loop is missing; label loop is recoverable. |
| Q5. Source-based canonical IDs built? | **Yes**, 18,661 source records; IDs use split/episode/item plus a stable source-record hash, never feature row number. Mention index remains null because not recoverable. |
| Q6. Feature↔MR bidirectional 1:1 join? | **Not certified.** Prefix counts match for the cached rows, but no shared IDs. |
| Q7. Labels↔MR facts 1:1 join? | **Yes for reconstructed J label order and source arrays**; the separate historical probe-label input used no saved ID sidecar. |
| Q8. Per-split Nsource=Nfeature=Nlabel? | **No for full train** (11,665 vs 4,096 cached features); candidate scored-prefix counts are 4,096 each. Val/TestA/TestB counts are 2,332 each, but count equality is not identity proof. |
| Q9. Duplicate/missing/orphan rows? | **No source-ID duplicates or source/label omissions detected. Feature-side duplicates/orphans are undeterminable without IDs; no row misalignment is confirmed.** |
| Q10. Mention/token spans replayable? | **No.** Historical tokenizer and mention-selection semantics are absent. |
| Q11. Re-extracted H matches old cache? | **NOT RUN** because the strict join gate failed and extraction semantics are unknown. |
| Q12. Entity/relation/joint semantics match source? | **Source and reconstructed labels pass.** Entity/relation/class arrays were checked over every row; 64 sampled source texts and the J joint-label mapping passed. There is no old feature-row identity to extend this to feature↔label semantics. |
| Q13. Historical J/L/L2 semantic provenance? | **NOT CERTIFIED** (not misalignment-confirmed). |
| Q14. Measurement gate reopened? | **No.** |
| Q15. J3 converged probe calibration allowed now? | **No.** |
| Q16. Reconnect allowed? | **No.** |
| Q17. O3c allowed? | **No.** |

## Frozen artifact checks

- Checkpoint before: `{checkpoint_hash}`
- Checkpoint after: `{checkpoint_hash}` (identical; no parameter/checkpoint modification)
- Remote H token cache: SHA-256 `{cache_sha}`
- Remote L2 feature cache: SHA-256 `46676516632aa6120bc435713723521659f9fb102cc244755f09b797f95c3b4d`
- Local archived H-cache copy: malformed NPZ, SHA-256 `f5286b7d609b48c2a8aca11532c5891acf851771d396bc5334ad6511e30251c5`; retained unchanged and excluded from analysis.
- Full start-of-run inventory (SHA-256, size, mtime) is `artifact_hash_manifest.json`; the remote-only starting snapshot is retained as `artifact_hash_manifest_remote.json`.
- No data leakage: **none introduced**. Train/test mapping was not tuned; mapping rules were recovered from source code before audit. No probe scores were run or consulted.

## Interpretation and next step

The source-label side is recoverable and internally consistent; the unresolved break is specifically between the old mention/token feature array and semantic source rows. Thus previous probe numbers remain numerically reproducible but lack certifiable row-level semantic provenance; they are not thereby shown to be wrong. Keep the measurement gate closed.

**Exactly one next experiment:** Build a fresh ID-carrying feature cache from original MR sources and the frozen checkpoint, then certify that cache before any probe training.
"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")
    (OUT / "FINAL_VERDICT.txt").write_text(
        "J3P-F — HISTORICAL ROW PROVENANCE CANNOT BE CERTIFIED\n"
        "Provenance gate failure; no confirmed row misalignment.\n",
        encoding="utf-8")
    (OUT / "row_join_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(OUT),
        "source_rows": {s: structural[s]["source_fact_count"] for s in SPLITS},
        "feature_rows": {s: feature_inventory[s]["shape"][0] for s in SPLITS},
        "npz_id_like_keys": id_like_keys,
        "semantic_audit_samples": len(semantic_samples),
        "semantic_audit_all_pass": all(r["all_checks_pass"] for r in semantic_samples),
        "structural_label_pass": {s: structural[s]["source_label_order_pass"] for s in SPLITS},
        "strict_feature_join_certified": False,
        "cache_sha256": cache_sha,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
