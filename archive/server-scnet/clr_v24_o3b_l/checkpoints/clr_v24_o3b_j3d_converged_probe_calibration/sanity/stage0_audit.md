# O3b-J3D Stage 0 — Certified cache integrity recheck

Verdict: **PASS**. Selected-file SHA-256/byte checks: 59/59.

All four splits were checked for the certified row counts, row-order identity across feature-row IDs / canonical IDs / labels / sidecars / plan, duplicate IDs, label agreement, feature shapes, dtype, and finite values. TestA/TestB were read only for integrity and identity checks; no accuracy/loss was computed and no test result informed selection.

- train: 11665 rows; 11665 unique IDs; PASS
- val: 2332 rows; 2332 unique IDs; PASS
- testA: 2332 rows; 2332 unique IDs; PASS
- testB: 2332 rows; 2332 unique IDs; PASS

No integrity discrepancies were detected. Probe fitting has not started.
