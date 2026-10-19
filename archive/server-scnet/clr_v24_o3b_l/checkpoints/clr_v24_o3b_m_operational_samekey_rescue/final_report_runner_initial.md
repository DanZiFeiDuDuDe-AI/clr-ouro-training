# O3b-M — Operational Same-Key Addressing Rescue

**O3b-M-F — SAME-KEY CLASSIFICATION PASSES BUT ACTION-LEVEL ADDRESSING FAILS**

## Scope and data

J/K/L/J3 attribution is archived as unresolved and was not resumed. The J3C provenance-certified frozen cache was reused; no backbone forward, model/checkpoint edit, read-path change, value-path change, or O3c run occurred.

Entity-side input: `H_mention_fresh_{split}.npy` (2048D, frozen). Relation-side input: `H_base_I0_{split}.npy` (2048D, frozen). Pair datasets were built within each fixed source split; canonical fact, feature-row, and stable source-record IDs had zero cross-split overlap.

## Candidate comparison

Validation selected Candidate B, deployed seed 43, without ensembling. Validation operational scores by candidate: A=0.7125, B=0.9997, C=0.7056. TestA/TestB rows in `candidate_comparison.csv` are evaluation-only and were not used for selection.

## Offline behavior gate

```json
{
  "pass": true,
  "winner": "B",
  "seed": 43,
  "testA": {
    "same_key_f1": 0.9987107864202837,
    "precision": 0.9974248927038627,
    "update_recall": 1.0,
    "p1_accuracy": 1.0,
    "p2_accuracy": 0.9982788296041308,
    "p3_accuracy": 0.9991394148020654,
    "p4_accuracy": 1.0,
    "new_false_update_rate": 0.0008605851979345956
  },
  "testB": {
    "same_key_f1": 0.9977708426214892,
    "precision": 0.9973262032085561,
    "update_recall": 0.9982158786797503,
    "p1_accuracy": 0.9982158786797503,
    "p2_accuracy": 0.9991079393398751,
    "p3_accuracy": 0.9982158786797503,
    "p4_accuracy": 1.0,
    "new_false_update_rate": 0.0008920606601248885
  },
  "conditions": {
    "testA_same_key_f1": true,
    "testB_same_key_f1": true,
    "testA_update_recall": true,
    "testB_update_recall": true,
    "testA_p2_accuracy": true,
    "testB_p2_accuracy": true,
    "testA_new_false_update": true,
    "testB_new_false_update": true
  }
}
```

## Action simulation and reconnect

```json
{
  "action_simulation": "FAIL",
  "reconnect": "NOT_RUN: real O3b reconnect is locked unless both offline behavior and action-simulation gates PASS.",
  "O3c": "NO"
}
```

## Guardrails

- No TestA/TestB threshold tuning or candidate selection.
- No key/projection attribution or representation-mechanism claim.
- No modification of Ouro, KeyEncoder checkpoint, backbone, Value/Read path, memory capacity, allocator, or original O3b code.
- Real reconnect is conditional and is not executed by this offline/action harness; if both gates pass, it is authorized as the single next scoped stage, not O3c.

## Artifacts

See `input_integrity.json`, `pair_dataset_manifest.json`, `pairs_*.csv`, `candidate_*_results.csv`, `candidate_comparison.csv`, `threshold_sweep.csv`, `offline_behavior_gate.json`, `action_simulation_results.csv`, `action_gate.json`, `reconnect_results.csv`, `regression_check.csv`, and `artifact_sha256_manifest.json`.
