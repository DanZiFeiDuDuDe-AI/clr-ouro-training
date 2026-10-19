# CLR-v2.4-O3a — Learned Free-Slot Allocation

**O3a PASS: LEARNED FREE-SLOT ALLOCATION**

## Result
Soft K-aware allocation N=4 collision rate: TestA 0.00%, TestB 0.00%.
Full pipeline retrieval N=4: TestA 99.47%, TestB 99.40%; full KV answer: TestA 93.77%, TestB 92.19%.
All-facts-retained: TestA 97.90%, TestB 97.60%; all-facts-answered: TestA 78.38%, TestB 74.47%.

## Controls
Hard-mask N=4 validity is reported separately and is guaranteed by construction. One-empty correct-empty rate: 100.00%.
Slot permutation retrieval drop: 0.00 pp; reversed fact-order retrieval drop: 0.00 pp.
Forced occupied-slot intervention: old fact0 retrieval 20.12%, new fact3 retrieval 98.80%.
Legal empty-slot invariance retrieval difference: 0.00 pp.

## K-aware versus K-blind
The K-blind allocator is an explicit control. If its collision-free rate matches K-aware, this phase supports resource allocation rather than semantic placement; K is not required for choosing an arbitrary free slot.

## Protocol
Only Allocator was trained; K/V/query/router/decoder/Ouro were frozen. K-aware parameters=9348; K-blind parameters=580; steps=3000; seed=42.
Occupancy was an explicit binary state. The primary loss rewarded total probability mass on any empty slot and did not use a slot-ID target.

## Boundary
This phase does not test same-key update, overwrite, replacement, eviction, capacity overflow, semantic placement, or latent reasoning.

## Stop
O3a complete; do not automatically implement O3b.
