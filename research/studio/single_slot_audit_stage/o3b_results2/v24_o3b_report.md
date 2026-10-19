# CLR-v2.4-O3b — Same-Key Update / Semantic Write Routing

**O3b FAIL: SAME-KEY DETECTION**

## Primary result
Learned same-key detection: TestA F1 90.96%, recall 89.60%; TestB F1 90.43%, recall 88.80%.
Update target accuracy: TestA 98.80%, TestB 98.00%.
Full pipeline retrieval: TestA 98.25%, TestB 98.79%; answer: TestA 93.10%, TestB 91.75%.
NEW collision: TestA 10.60%, TestB 8.80%; duplicate-key rate: TestA 4.90%, TestB 5.60%.

## Controls
Oracle update baseline: TestA 0.990 retrieval / 0.935 answer; TestB 0.987 / 0.916.
Full-memory same-key update learned retrieval/answer: 0.976 / 0.920.
Wrong update target intervention: retrieval 0.679, latest answer 0.649.
Slot permutation update-target drop: 3.60pp; fact-order drop: -0.60pp.

## K/V boundary
UpdateRouter reads K_new, existing K, and occupancy only; V is not an input. UPDATE keeps the existing K and writes the new V. NEW uses the frozen O3a allocator.

## Boundary
This phase does not test unseen-new-key overflow, eviction, replacement, capacity-full new-key management, or reasoning.

## Stop
O3b complete; do not automatically implement O3c.
