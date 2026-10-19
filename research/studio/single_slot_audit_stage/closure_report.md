# CLR-v2.2-S0 Closure — Locate the Single-Fact Write Collapse

All v2.2-S0 parameters were frozen. No Ouro rerun, no S1, no architecture change.

## Verdict context

The v2.2-S0 oracle head was 41.75% on TestA and 37.50% on TestB; the original LS0 reference was about 91%/90%.

## Main representation probes

| Representation | TestA | TestB |
|---|---:|---:|
| H | 0.800 | 0.800 |
| value_read | 0.535 | 0.512 |
| value_candidate | 0.470 | 0.440 |
| slot0 | 0.585 | 0.565 |
| fulls | 0.593 | 0.542 |
| H direct (reported LS0 reference) | 0.910 | 0.900 |

## Bottleneck baselines

| H -> d -> city | TestA | TestB |
|---|---:|---:|
| 32 | 0.795 | 0.750 |
| 64 | 0.802 | 0.760 |
| 128 | 0.802 | 0.787 |

## Single-slot baseline

| Baseline | TestA | TestB |
|---|---:|---:|
| SS-64 fixed slot0 | 0.848 | 0.838 |

## Key/value probes

```json
{
  "key_read_to_city": {
    "name": "key_read",
    "kind": "city",
    "dim": 31,
    "train_acc": 0.6905,
    "val_acc": 0.49,
    "testA_acc": 0.48,
    "testB_acc": 0.4675,
    "epochs": 100,
    "n_train": 2000,
    "n_testA": 400,
    "n_testB": 400
  },
  "key_read_to_person": {
    "name": "key_read",
    "kind": "person",
    "dim": 31,
    "train_acc": 0.016666666666666666,
    "val_acc": 0.08333333333333333,
    "testA_acc": 0.0,
    "testB_acc": 0.0,
    "epochs": 11,
    "n_train": 180,
    "n_testA": 10,
    "n_testB": 0
  },
  "value_read_to_city": {
    "name": "value_read",
    "kind": "city",
    "dim": 32,
    "train_acc": 0.751,
    "val_acc": 0.5575,
    "testA_acc": 0.535,
    "testB_acc": 0.5125,
    "epochs": 100,
    "n_train": 2000,
    "n_testA": 400,
    "n_testB": 400
  },
  "value_read_to_person": {
    "name": "value_read",
    "kind": "person",
    "dim": 32,
    "train_acc": 0.005555555555555556,
    "val_acc": 0.0,
    "testA_acc": 0.0,
    "testB_acc": 0.0,
    "epochs": 11,
    "n_train": 180,
    "n_testA": 10,
    "n_testB": 0
  }
}
```

## Routing and GRU bypass

```json
{
  "routing": {
    "fulls": {
      "name": "fulls",
      "kind": "city",
      "dim": 256,
      "train_acc": 0.8625,
      "val_acc": 0.68,
      "testA_acc": 0.6175,
      "testB_acc": 0.585,
      "epochs": 100,
      "n_train": 2000,
      "n_testA": 400,
      "n_testB": 400
    },
    "force_slot0_full": {
      "name": "force_slot0_full",
      "kind": "city",
      "dim": 256,
      "train_acc": 0.8325,
      "val_acc": 0.67,
      "testA_acc": 0.5975,
      "testB_acc": 0.535,
      "epochs": 66,
      "n_train": 2000,
      "n_testA": 400,
      "n_testB": 400
    },
    "force_uniform_full": {
      "name": "force_uniform_full",
      "kind": "city",
      "dim": 256,
      "train_acc": 0.8475,
      "val_acc": 0.675,
      "testA_acc": 0.585,
      "testB_acc": 0.565,
      "epochs": 83,
      "n_train": 2000,
      "n_testA": 400,
      "n_testB": 400
    }
  },
  "gru_bypass": {
    "bypass_full": {
      "name": "bypass_full",
      "kind": "city",
      "dim": 256,
      "train_acc": 0.825,
      "val_acc": 0.64,
      "testA_acc": 0.595,
      "testB_acc": 0.555,
      "epochs": 57,
      "n_train": 2000,
      "n_testA": 400,
      "n_testB": 400
    },
    "fulls": {
      "name": "fulls",
      "kind": "city",
      "dim": 256,
      "train_acc": 0.8625,
      "val_acc": 0.68,
      "testA_acc": 0.6175,
      "testB_acc": 0.585,
      "epochs": 100,
      "n_train": 2000,
      "n_testA": 400,
      "n_testB": 400
    }
  }
}
```

## Init-state controls

```json
{
  "fulls": {
    "name": "fulls",
    "kind": "city",
    "dim": 256,
    "train_acc": 0.8365,
    "val_acc": 0.6725,
    "testA_acc": 0.5925,
    "testB_acc": 0.5425,
    "epochs": 71,
    "n_train": 2000,
    "n_testA": 400,
    "n_testB": 400
  },
  "zero_init_full": {
    "name": "zero_init_full",
    "kind": "city",
    "dim": 256,
    "train_acc": 0.828,
    "val_acc": 0.66,
    "testA_acc": 0.575,
    "testB_acc": 0.535,
    "epochs": 63,
    "n_train": 2000,
    "n_testA": 400,
    "n_testB": 400
  },
  "random_init_full": {
    "name": "random_init_full",
    "kind": "city",
    "dim": 256,
    "train_acc": 0.8825,
    "val_acc": 0.69,
    "testA_acc": 0.59,
    "testB_acc": 0.5725,
    "epochs": 100,
    "n_train": 2000,
    "n_testA": 400,
    "n_testB": 400
  }
}
```

## Geometry

```json
{
  "H": {
    "mean_norm": 35.47036361694336,
    "inter_sample_cosine": 0.8230257462419569,
    "same_city_cosine": 0.844678932231509,
    "different_city_cosine": 0.8226675103508492,
    "margin_same_minus_diff": 0.022011421880659854
  },
  "value_read": {
    "mean_norm": 33.370262145996094,
    "inter_sample_cosine": 0.955971409137547,
    "same_city_cosine": 0.9839425842333499,
    "different_city_cosine": 0.9555086468059155,
    "margin_same_minus_diff": 0.028433937427434386
  },
  "value_candidate": {
    "mean_norm": 3.9868834018707275,
    "inter_sample_cosine": 0.04769072828060139,
    "same_city_cosine": 0.7547402179323583,
    "different_city_cosine": 0.0359931192495385,
    "margin_same_minus_diff": 0.7187470986828198
  },
  "slot0": {
    "mean_norm": 5.450508117675781,
    "inter_sample_cosine": 0.15256712099688186,
    "same_city_cosine": 0.7842267095676398,
    "different_city_cosine": 0.1421167819244896,
    "margin_same_minus_diff": 0.6421099276431502
  },
  "fulls": {
    "mean_norm": 5.463028430938721,
    "inter_sample_cosine": 0.15564387986602338,
    "same_city_cosine": 0.7850418067243975,
    "different_city_cosine": 0.14523095830804728,
    "margin_same_minus_diff": 0.6398108484163503
  }
}
```

## Q1-Q8

- q1_2048_to_32_sufficient: False
- q2_valueread_lost_information: True
- q3_valuegru_further_loss: True
- q4_routing_main_at_n1: False
- q5_fixed_single_slot_recovers_ls0: False
- q6_ls0_distributed_dependency: inferred only if SS-64 is low while the reported LS0 distributed reference is high
- q7_primary_failure: closure measurement; see bottleneck, ValueRead, ValueGRU, routing, and SS-64 rows
- q8_next_direction: diagnostic recommendation only; no architecture change implemented

## Decision boundary

This is a closure diagnosis only. It does not authorize v2.3, S1, query retrieval, or A2.

## Stop

Stopped after CLR-v2.2-S0 Closure.