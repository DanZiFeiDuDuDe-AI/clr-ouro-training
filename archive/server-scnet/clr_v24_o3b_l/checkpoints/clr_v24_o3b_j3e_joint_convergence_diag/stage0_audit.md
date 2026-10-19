# O3b-J3E Stage 0 input integrity

Verdict: **PASS**. Joint-only label integrity, certified R1/R2 files, row order, and standardized affine equivalence were audited; no probe was trained.

R1/R2 standardization used train-only means/stds. The induced standardized transform has validation max absolute reconstruction error 2.32e-06; synthetic affine classifier-transfer max logit error 4.348e-05.

Joint classes: 2378; validation known 2307; TestA known 2312; TestB known 0.

No integrity discrepancies detected. Test splits were used only for hashes, row identity, and known-label counts; no Test metric was computed or used for selection.
