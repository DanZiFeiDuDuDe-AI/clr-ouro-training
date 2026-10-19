# CLR-v2.4-O3b — Same-Key Update / Semantic Write Routing

**O3b FAIL: SAME-KEY DETECTION**

## Primary result
Learned same-key detection: TestA F1 90.78%, recall 89.60%; TestB F1 90.43%, recall 88.80%.
Update target accuracy: TestA 98.80%, TestB 98.00%.
Full pipeline retrieval: TestA 98.29%, TestB 98.79%; answer: TestA 93.12%, TestB 91.78%.
NEW collision: TestA 11.60%, TestB 9.60%; duplicate-key rate: TestA 4.60%, TestB 5.40%.

## Controls
Oracle update baseline: TestA 0.990 retrieval / 0.935 answer; TestB 0.987 / 0.916.
Full-memory same-key update learned retrieval/answer: 0.967 / 0.911.
Wrong update target intervention: retrieval 0.679, latest answer 0.649.
Slot permutation update-target drop: 3.20pp; fact-order drop: -1.00pp.

## K/V boundary
UpdateRouter reads K_new, existing K, and occupancy only; V is not an input. UPDATE keeps the existing K and writes the new V. NEW uses the frozen O3a allocator.

## Boundary
This phase does not test unseen-new-key overflow, eviction, replacement, capacity-full new-key management, or reasoning.

## Stop
O3b complete; do not automatically implement O3c.
