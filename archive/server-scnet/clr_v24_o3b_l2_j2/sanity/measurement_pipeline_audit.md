# Measurement pipeline audit — O3b-L2 branch B

No model/checkpoint changes were made. The exact frozen H/W1 artifacts and O3b-L sample ordering were reused.

## Verified

- Label mappings, row counts and flat reference order are checked against `MRCache`; class coverage/imbalance is recorded in `measurement_pipeline_audit.json`.
- Label arrays are sliced to the same deterministic first-4096 train carrier prefix used by O3b-L; validation/test arrays are fully covered.
- R0/R1 train-standardized tensors match to <1.2e-6 max absolute error on every split.
- Each fit creates new parameters and a fresh optimizer; seeds and fixed minibatch RNG schedules match O3b-L.
- Probe tensors and model parameters are float32. No autocast, gradient clipping, batch normalization, dropout or early stopping is used.
- The transformed R1→R2 classifier preserves logits/accuracy on all split/task checks within float precision; see `09_weight_transfer_control.csv`.

## Provenance limitation

The token-feature NPZ has no explicit per-row sample ID array. Alignment follows the same flattened fact order and `j.aligned_carriers` used by O3b-J/L, and the MR reference/label side is independently checked. Therefore there is no detected ordering error, but token-to-MR row identity is inherited from the cache-producing contract rather than independently joined by IDs.

## Interpretation

The probe score is optimization-dependent: 500→1000 trajectories still move materially, and R1/R2 independently retrained scores can diverge despite exact algebraic equivalence. This is a measurement reliability failure for mechanism attribution; it does not imply feature information changed.
