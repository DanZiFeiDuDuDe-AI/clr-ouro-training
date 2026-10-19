# Optimizer calibration (train/validation only)

Candidate screen: 18 preregistered AdamW/SGD × LR × weight-decay settings, 3,000 steps, seed 11, averaged across R0/R1 raw and train-standardized validation conditions. No TestA/TestB metrics were read.

| task | chosen optimizer | LR | WD | mean last-5 val accuracy | mean last-5 val CE |
|---|---|---:|---:|---:|---:|
| entity | AdamW | 0.003 | 0.0 | 1.000000 | 0.000785 |
| relation | AdamW | 0.003 | 0.0 | 0.972449 | 0.091877 |
| joint | AdamW | 0.003 | 0.0 | 0.853294 | 0.500858 |
