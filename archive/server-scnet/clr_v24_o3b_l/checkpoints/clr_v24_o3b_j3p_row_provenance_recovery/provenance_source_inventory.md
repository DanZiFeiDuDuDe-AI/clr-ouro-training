# O3b-J3P provenance source inventory

Search scope: local `H:\LoopLM` source/archive and the connected remote experiment roots under `/root/private_data/clr_v24_o3b_l` and `/root/private_data/clr_v24_o3b_l2_j2`. Local text/code search covered Python, JSON/JSONL, CSV, Markdown, logs and configs (excluding binary tensors); remote source inventories covered code, H output, MR cache metadata, L/L2 outputs/logs and K context. Binary NPZ archives were inspected for keys/shapes/dtypes, not treated as source code.

| Source | Present | Row-level identity | Trust / use | Direct historical join? |
|---|---|---|---|---|
| O3b-H token extractor source | Not found in preserved local/remote code or archives | No | H output/config survives, but no extraction implementation is preserved | No |
| O3b-H `token_layer_features.npz` | Yes, remote canonical cache; 838,539,950 bytes | No row/sample/fact ID or index-like NPZ key | Authoritative frozen vectors; key names and shapes only | No |
| Local archived H-cache copy under `remote_archive/cheems` | Yes, 7,702,416 bytes; distinct SHA-256; fails ZIP/NPZ parsing | Unreadable | Preserved as-is, excluded from audit; not the remote canonical source | No |
| O3b-H config/report/representation metadata and JSON outputs | Yes | No per-row identity, token offsets or selected positions | Indirect evidence for seed, bounded row counts, layers, task setup; not extraction order | No |
| O3b-H temporary `o3bl_8wp76qs7/part_00..07` | Present, eight zero-byte files in local mirror | No | Empty transfer artifacts | No |
| MR cache `*_data.npz` + `*_meta.jsonl`, `train_persons.json`, `relations.json`, `manifest.json` | Yes | Episode IDs and per-item semantic fields exist; no separate fact UUID | Trusted source for source order and labels; all 18,661 rows audited | Joins label traversal to source, not old token feature rows |
| `a1_v24_o3b_j.py::MRCache`, `build_infos`, `joint_labels`, `load_token_arrays`, `aligned_carriers` | Yes, local and remote | Label side source indices are reconstructible; token NPZ IDs absent | Exact J label traversal and downstream prefix convention recoverable | Label↔source yes; mention feature↔source no |
| O3b-I extraction/mention-location source | Not found in preserved experiment sources inspected | No | Prior conclusion/report only; does not encode H token row identity | No |
| O3b-K analysis script / outputs | Yes in local archive and remote K context | Synthetic/downstream row order only | Useful for representation diagnosis; not upstream mention provenance | No |
| O3b-L source/config/output | Yes | Probes/results but no H row IDs | Frozen artifact hashes included in the manifest | No |
| O3b-L2/J2 source/config/features/logs | Yes, remote; local reports/scripts also preserved | Feature tensors keep array order, no canonical upstream mention IDs | Numerical reruns and transferred-control artifacts; no source identity | No |
| Prior J3 audit source/report/CSV/log | Yes, local archive | Synthetic `split:<row_index>` only | Correctly documents Stage 0 gate and downstream order; explicitly not upstream proof | No |
| Historical extraction stdout/stderr or row manifest | Not found in preserved H output tree / archives searched | No | H report/config provide summary metadata only | No |

## Recovered traversals

**Historical feature traversal:** NOT RECOVERED. The preserved J consumer loads `fact_<split>_entity_mention_mean` as an array, then pairs `raw[:n]` with `mention[:n]`. No preserved H code shows how source examples/facts were iterated, mentions located, token spans chosen, or vectors appended. Therefore no honest historical feature-loop pseudocode can be supplied beyond “load stored array, preserve its row order, take prefix.”

**Historical label traversal:** recovered from `a1_v24_o3b_j.py::build_infos`: for each split in `(train,val,testA,testB)`, enumerate JSONL episode metadata, then `for pos in range(capacity)`, map `flat=fact_offsets[episode]+pos`, append `person` and `relation_ids[pos]`; entity labels use the `train_persons` index map (unknown → -1). `joint_labels` forms sorted unique train keys `person + TAB + relation_id`, then maps each split's same ordered `(person, relation)` sequence (unknown → -1). This label traversal is independently reconstructed from MR metadata and verified against all flat `fact_entity`, `fact_relation`, and `fact_Y` arrays.

These are not shown to originate from one shared historical iterator: the feature side's H extraction loop is missing. Matching counts and J's positional prefix convention do not establish identity.
