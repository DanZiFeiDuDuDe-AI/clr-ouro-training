# CLR-v2.4-O3b — Same-Key Update / Semantic Write Routing

**O3b FAIL: SAME-KEY DETECTION**

## Primary result
Learned same-key detection: TestA F1 66.93%, recall 100.00%; TestB F1 66.93%, recall 100.00%.
Update target accuracy: TestA 98.40%, TestB 97.80%.
Full pipeline retrieval: TestA 90.03%, TestB 88.16%; answer: TestA 84.75%, TestB 81.96%.
NEW collision: TestA 98.80%, TestB 98.00%; duplicate-key rate: TestA 0.80%, TestB 1.10%.

## Controls
Oracle update baseline: TestA 0.990 retrieval / 0.935 answer; TestB 0.987 / 0.916.
Full-memory same-key update learned retrieval/answer: 0.918 / 0.865.
Wrong update target intervention: retrieval 0.679, latest answer 0.649.
Slot permutation update-target drop: 0.00pp; fact-order drop: 0.00pp.

## K/V boundary
UpdateRouter reads K_new, existing K, and occupancy only; V is not an input. UPDATE keeps the existing K and writes the new V. NEW uses the frozen O3a allocator.

## Boundary
This phase does not test unseen-new-key overflow, eviction, replacement, capacity-full new-key management, or reasoning.

## Stop
O3b complete; do not automatically implement O3c.
