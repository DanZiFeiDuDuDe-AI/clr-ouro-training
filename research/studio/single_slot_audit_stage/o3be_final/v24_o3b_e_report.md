# CLR-v2.4-O3b-E — Relation-Aware Factorized Key

**O3b-E FAIL: RELATION-AWARE FACTORIZATION INSUFFICIENT**

## Executive Summary
Best variant=original, best score=E1_a1_b1; factor head parameters=4292. Original KeyEncoder parameters were frozen; no real write/read reconnect was run.

## Baseline Reproduction
O3b-D raw cosine reproduction: TestA F1=88.44%, TestB F1=86.56%; prior deltas=0.0/0.0.

## Main Results
original: E1_a1_b1: A F1=90.73%, recall=93.00%; B F1=88.97%, recall=89.00%; P2 A/B=85.20%/90.80% | E1_a1_b2: A F1=89.65%, recall=93.53%; B F1=89.20%, recall=92.27%; P2 A/B=88.80%/92.00% | E1_a1_b4: A F1=87.95%, recall=90.47%; B F1=86.59%, recall=87.80%; P2 A/B=92.00%/95.60% | E1_a2_b1: A F1=90.98%, recall=92.13%; B F1=88.10%, recall=87.40%; P2 A/B=78.40%/82.00% | E2_gated_tau1: A F1=89.73%, recall=89.13%; B F1=86.81%, recall=84.67%; P2 A/B=72.40%/74.40% | E2_gated_tau2: A F1=90.20%, recall=90.20%; B F1=87.48%, recall=85.73%; P2 A/B=76.00%/78.80% | E2_gated_tau4: A F1=89.92%, recall=94.60%; B F1=88.31%, recall=92.13%; P2 A/B=70.00%/69.60% | E2_gated_tau8: A F1=89.62%, recall=93.80%; B F1=87.87%, recall=90.80%; P2 A/B=70.00%/70.80% | E2_linear_lambda1: A F1=90.73%, recall=93.00%; B F1=88.97%, recall=89.00%; P2 A/B=85.20%/90.80% | E2_linear_lambda2: A F1=89.65%, recall=93.53%; B F1=89.20%, recall=92.27%; P2 A/B=88.80%/92.00% | E3_full_concat: A F1=90.73%, recall=93.00%; B F1=88.97%, recall=89.00%; P2 A/B=85.20%/90.80%
p2_25: E1_a1_b1: A F1=87.60%, recall=85.27%; B F1=85.87%, recall=81.87%; P2 A/B=94.00%/96.40% | E1_a1_b2: A F1=88.31%, recall=87.40%; B F1=86.60%, recall=84.20%; P2 A/B=96.40%/98.40% | E1_a1_b4: A F1=88.05%, recall=88.20%; B F1=85.84%, recall=84.87%; P2 A/B=97.60%/98.80% | E1_a2_b1: A F1=87.38%, recall=86.07%; B F1=84.86%, recall=81.67%; P2 A/B=87.20%/89.60% | E2_gated_tau1: A F1=84.49%, recall=81.73%; B F1=82.46%, recall=78.53%; P2 A/B=79.60%/80.80% | E2_gated_tau2: A F1=86.01%, recall=83.60%; B F1=83.43%, recall=79.87%; P2 A/B=82.00%/84.40% | E2_gated_tau4: A F1=86.08%, recall=84.07%; B F1=83.92%, recall=80.73%; P2 A/B=83.20%/86.40% | E2_gated_tau8: A F1=85.74%, recall=83.60%; B F1=83.60%, recall=80.40%; P2 A/B=82.00%/85.60% | E2_linear_lambda1: A F1=87.60%, recall=85.27%; B F1=85.87%, recall=81.87%; P2 A/B=94.00%/96.40% | E2_linear_lambda2: A F1=88.31%, recall=87.40%; B F1=86.60%, recall=84.20%; P2 A/B=96.40%/98.40% | E3_full_concat: A F1=87.60%, recall=85.27%; B F1=85.87%, recall=81.87%; P2 A/B=94.00%/96.40%
p2_50: E1_a1_b1: A F1=88.07%, recall=88.07%; B F1=82.37%, recall=83.47%; P2 A/B=97.60%/97.20% | E1_a1_b2: A F1=87.58%, recall=86.53%; B F1=80.98%, recall=80.60%; P2 A/B=99.20%/98.00% | E1_a1_b4: A F1=87.04%, recall=85.27%; B F1=80.62%, recall=79.60%; P2 A/B=100.00%/98.00% | E1_a2_b1: A F1=87.73%, recall=88.67%; B F1=83.63%, recall=86.53%; P2 A/B=92.80%/95.20% | E2_gated_tau1: A F1=87.74%, recall=90.40%; B F1=82.92%, recall=87.20%; P2 A/B=87.60%/92.80% | E2_gated_tau2: A F1=87.77%, recall=89.93%; B F1=83.17%, recall=87.00%; P2 A/B=88.40%/93.60% | E2_gated_tau4: A F1=87.64%, recall=89.80%; B F1=83.14%, recall=87.27%; P2 A/B=88.80%/93.20% | E2_gated_tau8: A F1=87.65%, recall=90.13%; B F1=83.03%, recall=87.40%; P2 A/B=88.40%/92.40% | E2_linear_lambda1: A F1=88.07%, recall=88.07%; B F1=82.37%, recall=83.47%; P2 A/B=97.60%/97.20% | E2_linear_lambda2: A F1=87.58%, recall=86.53%; B F1=80.98%, recall=80.60%; P2 A/B=99.20%/98.00% | E3_full_concat: A F1=88.07%, recall=88.07%; B F1=82.37%, recall=83.47%; P2 A/B=97.60%/97.20%

## Best Variant / Probes
Best=original/E1_a1_b1. TestA F1=90.73%, TestB F1=88.97%; UPDATE recall A/B=93.00%/89.00%; P1 A/B=93.00%/89.00%; P2 A/B=85.20%/90.80%.
K_r→relation TestA/TestB=82.68%/81.90%; full probe details are in probe_results.json.

## Retrieval / Error Audit
Frozen K-only retrieval baseline A/B=99.47%/99.40%; factorized=98.35%/97.45%. Nearest-neighbor error categories are in nearest_neighbor_audit.json.

## Gate
{"baseline_reproduced": true, "testA_f1_ge_95": false, "testB_f1_ge_95": false, "testA_update_recall_ge_95": false, "testB_update_recall_ge_95": false, "P2_testA_ge_95": false, "P2_testB_ge_90": true, "K_r_to_relation_testB_ge_90": false}

## Boundary
Repeated V update contamination remains an independent unresolved issue. This experiment did not modify V, GRU, allocator, memory storage, or answer supervision. No reconnect and no O3c were entered.

## Stop
O3b-E complete; await a separate instruction before any reconnect or O3c.
