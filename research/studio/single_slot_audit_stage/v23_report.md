# CLR-v2.3-OA — Oracle Allocation Sanity

Oracle allocation only: fact position k always writes slot k. No learned router, query, or A2.

## Verdict

O0 FAIL

## Capacity

testA
- N=1: 0.690

testB
- N=1: 0.740

## Position matrix
```json
{
  "train": {
    "1": [
      {
        "acc": 0.973,
        "correct": 1946,
        "total": 2000,
        "by_position": [
          0.973
        ],
        "position_total": [
          2000
        ]
      }
    ]
  },
  "val": {
    "1": [
      {
        "acc": 0.735,
        "correct": 294,
        "total": 400,
        "by_position": [
          0.735
        ],
        "position_total": [
          400
        ]
      }
    ]
  },
  "testA": {
    "1": [
      {
        "acc": 0.69,
        "correct": 276,
        "total": 400,
        "by_position": [
          0.69
        ],
        "position_total": [
          400
        ]
      }
    ]
  },
  "testB": {
    "1": [
      {
        "acc": 0.74,
        "correct": 296,
        "total": 400,
        "by_position": [
          0.74
        ],
        "position_total": [
          400
        ]
      }
    ]
  }
}
```

## Fresh full-state probes
```json
{
  "1": [
    {
      "train_acc": 0.9740000367164612,
      "val_acc": 0.7674999833106995,
      "testA_acc": 0.6949999928474426,
      "testB_acc": 0.7299999594688416,
      "epochs": 31
    }
  ]
}
```

## Slot purity/local probes
```json
{
  "skipped": "O0 gate failed"
}
```

## Write survival and exact preservation
```json
{
  "skipped": "O0 gate failed"
}
```

## Baselines
```json
{
  "original_LS2_N4": {
    "testA": 0.165,
    "testB": 0.1925
  },
  "v21a_N4": {
    "testA": 0.185,
    "testB": 0.19
  },
  "v21b_N4": {
    "testA": 0.1275,
    "testB": 0.155
  },
  "v22_S0_N1": {
    "testA": 0.4175,
    "testB": 0.375
  }
}
```

## Interpretation

- Oracle allocation isolates storage from learned allocation.
- N>4 is only evaluated zero-shot after an O1 pass.
- This run stops after CLR-v2.3-OA.