import json
import os
import time
from pathlib import Path

from a1_v22_s0_closure_common import (
    closure_paths, collect_all, load_v22, save_representations, seed_all,
)


def main():
    seed_all(42)
    out = closure_paths(); out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cache, model, heads, ckpt, device = load_v22()
    reps = collect_all(cache, model, device)
    save_representations(out / "closure_representations.npz", reps)
    meta = {
        "checkpoint_stage": ckpt.get("stage", ""),
        "checkpoint": str(os.environ.get("V22_OUT_DIR", r"D:\training\checkpoints\clr_v22_storage")) + r"\best_s0.pt",
        "device": str(device), "cache": str(cache.root),
        "splits": {s: {k: list(v.shape) for k, v in rep.items()} for s, rep in reps["splits"].items()},
        "person_subset": reps["person_subset"],
        "wall_clock_s": time.perf_counter() - t0,
        "frozen_main_parameters": True,
        "no_ouro_rerun": True,
    }
    (out / "closure_collect_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__": main()
