# CLR-v2.4-O2.1b — Learned Semantic Key Writing

**O2.1b PASS: LEARNED SEMANTIC KEY WRITING**

## Primary result
B1 was trained from a fresh KeyEncoder using frozen QueryEncoder and retrieval-only supervision; entity/relation/joint labels never entered the key loss.
Supervised K upper bound N=4: TestA/TestB 99.10%/99.10%.
Learned K N=4: TestA/TestB 99.10%/100.00%; delta 0.00/0.90 percentage points.
Learned K full KV answer: TestA/TestB 93.99%/90.69%.

## Retrieval stress
MR1 same entity/different relation: 98.46%; MR2 same relation/different entity: 100.00%; same-value: 100.00%; lexical: 98.53%.
Wrong-K swap: {"retrieval": 0.0, "switch_to_alt": 0.9887378215789795, "answer_original": 0.17434388399124146}; full KV permutation: {"retrieval": 0.9887378215789795, "answer": 0.9263054728507996}.

## Probes and geometry
Fresh K probes are in fresh_semantic_probes.json; K→value is diagnostic only and did not train the writer.

## Protocol
KeyWriter parameters: 540992; trainable: KeyEncoder.net.*; frozen QueryEncoder/V/decoder/router/allocation/Ouro; steps=3000; eval_every=100; seed=42.
Oracle physical slot allocation was retained; allocation learning was not entered.

## Boundary
This result, even if PASS, only supports autonomous fact→K formation under oracle allocation, fixed schema, and frozen query/read paths. It does not establish learned allocation, eviction, long-context memory, KV-cache replacement, or latent reasoning.

## Stop
O2.1b complete; learned allocation is not implemented in this phase.
