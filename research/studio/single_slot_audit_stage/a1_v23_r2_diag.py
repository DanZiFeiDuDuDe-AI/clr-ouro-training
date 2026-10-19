"""Read-only diagnostic entry point for a completed r2 checkpoint.

The main diagnostic collection is intentionally kept in a1_v23_r2_eval.py so
that train/eval use the same core and cache helpers. This wrapper is provided
for explicit provenance and simply prints the saved diagnostic artifacts.
"""
import json
import os
from pathlib import Path


def main():
    out = Path(os.environ.get("V23_R2_OUT_DIR", r"D:\training\checkpoints\clr_v23_oracle_r2"))
    names = ["fresh_local_probes.json", "slot_purity.json", "gradient_interference.json", "overload_capacity.json"]
    result = {name: json.loads((out / name).read_text(encoding="utf-8")) for name in names if (out / name).exists()}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
