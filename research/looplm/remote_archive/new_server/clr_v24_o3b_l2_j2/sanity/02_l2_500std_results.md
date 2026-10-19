# O3b-L2-500STD — Stage 1 measurement calibration

## Scope

Frozen L-I2b carrier and frozen W1 only. No Ouro/backbone retraining, no checkpoint/model modification, no reconnect/O3c, and no repeated-V update work. All newly generated data are under this new experiment directory.

Exact five seeds were read from the O3b-L `config.json`: `11, 23, 37, 53, 71`. All conditions use the inherited linear probe, AdamW 3e-4 / 0.01, batch ≤1024, same seed/task initialization rule, same task-specific minibatch schedule, and fixed 500-step endpoint; 750/1000 are convergence-only extensions. Standardization uses train-only per-dimension mean/std.

## Paired results (mean ± SD, percent; paired t-CI also shown)

### raw

| task | split | step | R0 | R1 | R2 |
|---|---|---:|---:|---:|---:|
| entity | val | 500 | 95.84 ± 0.47% (paired-seed t95 95.25–96.43) | 92.43 ± 0.55% (paired-seed t95 91.75–93.11) | 92.85 ± 0.67% (paired-seed t95 92.01–93.68) |
| entity | testA | 500 | 96.40 ± 0.40% (paired-seed t95 95.90–96.89) | 92.45 ± 0.56% (paired-seed t95 91.76–93.15) | 92.30 ± 0.44% (paired-seed t95 91.76–92.84) |
| entity | testB | 500 | n/a | n/a | n/a |
| relation | val | 500 | 76.76 ± 1.61% (paired-seed t95 74.76–78.76) | 75.19 ± 3.38% (paired-seed t95 70.99–79.39) | 81.50 ± 1.82% (paired-seed t95 79.24–83.76) |
| relation | testA | 500 | 77.14 ± 1.22% (paired-seed t95 75.63–78.66) | 75.11 ± 3.42% (paired-seed t95 70.86–79.36) | 81.33 ± 1.81% (paired-seed t95 79.08–83.57) |
| relation | testB | 500 | 75.27 ± 1.71% (paired-seed t95 73.15–77.40) | 74.86 ± 3.35% (paired-seed t95 70.70–79.02) | 79.94 ± 1.13% (paired-seed t95 78.54–81.34) |
| joint | val | 500 | 36.18 ± 0.54% (paired-seed t95 35.51–36.85) | 33.86 ± 0.41% (paired-seed t95 33.36–34.37) | 41.45 ± 0.84% (paired-seed t95 40.40–42.49) |
| joint | testA | 500 | 37.65 ± 0.31% (paired-seed t95 37.27–38.03) | 34.75 ± 0.41% (paired-seed t95 34.24–35.26) | 41.52 ± 0.36% (paired-seed t95 41.08–41.97) |
| joint | testB | 500 | n/a | n/a | n/a |

### standardized

| task | split | step | R0 | R1 | R2 |
|---|---|---:|---:|---:|---:|
| entity | val | 500 | 99.90 ± 0.09% (paired-seed t95 99.78–100.01) | 99.88 ± 0.08% (paired-seed t95 99.78–99.98) | 98.38 ± 0.07% (paired-seed t95 98.29–98.47) |
| entity | testA | 500 | 100.00 ± 0.00% (paired-seed t95 100.00–100.00) | 99.99 ± 0.02% (paired-seed t95 99.97–100.02) | 98.38 ± 0.15% (paired-seed t95 98.20–98.56) |
| entity | testB | 500 | n/a | n/a | n/a |
| relation | val | 500 | 86.03 ± 0.74% (paired-seed t95 85.12–86.94) | 86.71 ± 0.60% (paired-seed t95 85.97–87.45) | 87.16 ± 0.27% (paired-seed t95 86.82–87.50) |
| relation | testA | 500 | 86.98 ± 0.66% (paired-seed t95 86.17–87.80) | 86.79 ± 0.50% (paired-seed t95 86.17–87.42) | 87.03 ± 0.34% (paired-seed t95 86.61–87.45) |
| relation | testB | 500 | 82.06 ± 0.44% (paired-seed t95 81.51–82.61) | 82.54 ± 0.58% (paired-seed t95 81.82–83.26) | 84.42 ± 0.25% (paired-seed t95 84.11–84.72) |
| joint | val | 500 | 50.40 ± 0.39% (paired-seed t95 49.92–50.89) | 50.59 ± 0.36% (paired-seed t95 50.14–51.03) | 57.13 ± 0.29% (paired-seed t95 56.77–57.49) |
| joint | testA | 500 | 50.71 ± 0.12% (paired-seed t95 50.56–50.86) | 50.42 ± 0.27% (paired-seed t95 50.07–50.76) | 56.81 ± 0.18% (paired-seed t95 56.58–57.03) |
| joint | testB | 500 | n/a | n/a | n/a |

## Conditioning

Train-only raw and standardized covariance spectra, condition numbers, effective rank, participation ratio, feature norms and per-dimension standard deviations are in `conditioning_raw_standardized.json`. Learning curves at 20/50/100/180/300/500/750/1000 are in `03_probe_convergence.csv`; only validation metrics are used to inspect plateau, never to stop training.

## Algebraic / transfer controls

Train-standardized R0/R1 feature-level differences: `{"train": {"max_abs_R0std_R1std": 1.1920928955078125e-06, "mean_abs_R0std_R1std": 4.978304559430727e-08, "pearson_flattened": 0.9999999999999964}, "val": {"max_abs_R0std_R1std": 7.748603820800781e-07, "mean_abs_R0std_R1std": 4.984127092138649e-08, "pearson_flattened": 0.9999999999999971}, "testA": {"max_abs_R0std_R1std": 1.1920928955078125e-06, "mean_abs_R0std_R1std": 4.9867208321889076e-08, "pearson_flattened": 0.9999999999999958}, "testB": {"max_abs_R0std_R1std": 8.344650268554688e-07, "mean_abs_R0std_R1std": 4.991475535188153e-08, "pearson_flattened": 0.9999999999999921}}`.

R2 is algebraically `R1 @ U.T`; transferred classifiers use the derived inverse-coordinate transform and include an affine transform for independently standardized coordinates. Per-seed max logit errors and accuracy parity are in `09_weight_transfer_control.csv`.

## Gate status

No result is interpreted as information content. Stage 1 gate remains `FAIL-MEASUREMENT` because convergence is not reached by 1000 steps and R1/R2 independently refitted probes remain optimization-sensitive despite transfer invariance. Full raw metrics are in the accompanying CSV/JSON files.

Runtime: not captured on BW. Output root: `/root/private_data/clr_v24_o3b_l2_j2`.
