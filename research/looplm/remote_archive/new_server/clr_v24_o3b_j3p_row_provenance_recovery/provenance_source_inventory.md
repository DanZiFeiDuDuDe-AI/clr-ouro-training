# O3b-J3P provenance source inventory

Search covered local `H:\LoopLM` sources/archives and remote `/root/private_data/clr_v24_o3b_l` plus `/root/private_data/clr_v24_o3b_l2_j2`: Python, configs, JSON/JSONL, CSV, Markdown, logs, and artifact file listings. Local keyword search covered `row_id`, `fact_id`, `item_id`, `episode_id`, `entity_mention_mean`, `token_layer_features`, mention/token span, hidden state, write state, and flatten/order terms. NPZ archives were inspected for keys/shapes/dtypes. No local pickle/pickle-extension or notebook files were found in the preserved workspace search; empty H transfer parts contain zero bytes.

| Source | Present | Row identity? | Trust/use | Direct join to historical mention features? |
|---|---|---|---|---|
| O3b-H extraction source | Not found in preserved local/remote code/archive | No | H report/config survive but extraction implementation does not | No |
| O3b-H `token_layer_features.npz` | Yes, remote canonical cache, 838,539,950 bytes | No ID/index-like NPZ keys | Frozen vectors are authoritative; array keys/shapes only | No |
| Local archived H-cache copy under `remote_archive/cheems` | Yes, 7,702,416 bytes; different SHA-256; `BadZipFile` on parse | Unreadable | Preserved unchanged; excluded from audit; not the remote canonical source | No |
| O3b-H config/report/representation metadata/JSON outputs | Yes | No per-row IDs/spans/selected positions | Indirect evidence about seed, sample sizes, layers and diagnostics | No |
| O3b-H `o3bl_8wp76qs7/part_00..07` | Eight empty files in local mirror | No | Empty transfer artifacts | No |
| MR `*_data.npz`, `*_meta.jsonl`, `train_persons.json`, `relations.json`, `manifest.json` | Yes | Episode IDs and item semantics; no independent fact UUID | Trusted source and label traversal; all 18,661 source rows audited | Labels↔source only |
| `a1_v24_o3b_j.py::build_infos/joint_labels/load_token_arrays/aligned_carriers` | Yes, local and remote | Label source order recoverable; H IDs absent | Exact J label traversal and downstream prefix convention | Label↔source yes; feature↔source no |
| O3b-I mention extraction source | Not found in preserved sources inspected | No | Prior reports/conclusions are indirect only | No |
| O3b-K analysis script/output | Yes in local archive and remote K context | Downstream/synthetic ordering only | Frozen representation diagnostics | No |
| O3b-L source/config/output | Yes | No upstream row IDs | Experiment results/config; hashes frozen | No |
| O3b-L2/J2 source/config/features/logs | Yes, remote; reports/code local | Arrays preserve their own order, no semantic mention IDs | Numerical measurement artifacts, not provenance | No |
| Prior J3 source/report/alignment CSV/log | Yes locally | Synthetic `split:<row_index>` only | Explicitly documents that downstream order is not upstream proof | No |
| Historical H extraction stdout/stderr, tokenizer/span config, row manifest | Not found in preserved H output/archive | No | Report/config are only indirect metadata | No |

## Recovered traversal descriptions

**Feature side — historical traversal NOT RECOVERED.** Preserved `a1_v24_o3b_j.py::load_token_arrays` loads `fact_<split>_entity_mention_mean`; `aligned_carriers` pairs it with MR `fact_H` using `[:n]`. No preserved H script reveals source-loop nesting, mention location, tokenization, selected state/pooling, or append order. Thus only the downstream consumer's prefix behavior is known.

**Label side — recovered exactly from `a1_v24_o3b_j.py::build_infos`.** Iterate splits `(train,val,testA,testB)`; enumerate each split's JSONL episodes; for each `pos in range(capacity)`, use `flat=fact_offsets[episode]+pos`, append person and `relation_ids[pos]`; map entity through `train_persons` (unknown → -1). `joint_labels` sorts unique train keys `person + TAB + relation_id`, then maps each split's ordered person/relation list (unseen → -1). It agrees with all `fact_entity`, `fact_relation`, and `fact_Y` source arrays.

The recovered feature and label processes are not shown to share one iterator because the original H feature loop is missing. Matching dimensions/counts and downstream prefix pairing are indirect evidence only.
