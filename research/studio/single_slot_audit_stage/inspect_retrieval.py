import json
from pathlib import Path

for root in ("o3bf_final", "o3bg_final"):
    p = Path(root) / "retrieval_error_decomposition.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    print(root, list(d.keys()))
    for k, v in d.items():
        print(k, {s: (v[s].get("accuracy"), v[s].get("n"), v[s].get("error_counts")) for s in v})
