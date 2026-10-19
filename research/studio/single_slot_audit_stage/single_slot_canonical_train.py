import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\training")
OUT = ROOT / "checkpoints" / "single_slot_diff_audit"
CLOSURE_OUT = ROOT / "checkpoints" / "clr_v22_s0_closure"
sys.path.insert(0, str(ROOT))

from a1_v22_s0_closure_common import SingleSlotBaseline, accuracy, load_representations, train_classifier  # noqa: E402
from a1_v22_storage_common import get_device, seed_all  # noqa: E402


def run_one(arrays, device, init_seed):
    # The old SS-64 trainer constructed the model before train_classifier reset
    # the optimization RNG. We make that initialization seed explicit here.
    seed_all(init_seed)
    model = SingleSlotBaseline()
    tr, va, ta, tb = [arrays[s] for s in ("train", "val", "testA", "testB")]
    model, info = train_classifier(model, tr["H"], tr["labels"], va["H"], va["labels"],
                                   device, lr=1e-3, max_epochs=100, patience=10, seed=42,
                                   save_path=OUT / (f"canonical_seed{init_seed}.pt"))
    return {
        "init_seed": init_seed,
        "train_acc": accuracy(model, tr["H"], tr["labels"], device),
        "val_acc": info["best_val_acc"],
        "testA_acc": accuracy(model, ta["H"], ta["labels"], device),
        "testB_acc": accuracy(model, tb["H"], tb["labels"], device),
        "epochs": info["epochs"],
        "protocol": {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.0, "batch_size": 128,
                      "max_epochs": 100, "patience": 10, "optimization_seed": 42,
                      "architecture_init_seed": init_seed, "dtype": "float32"},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    data = load_representations(CLOSURE_OUT / "closure_representations.npz")
    arrays = {s: {"H": data["splits"][s]["H"].astype(np.float32),
                  "labels": data["splits"][s]["labels"].astype(np.int64)}
              for s in ("train", "val", "testA", "testB")}
    runs = [run_one(arrays, device, seed) for seed in (42, 43, 44)]
    result = {"device": str(device), "cache": str(ROOT / "checkpoints" / "clr_v2_a1_ls2" / "cache"),
              "n_train": len(arrays["train"]["labels"]), "n_val": len(arrays["val"]["labels"]),
              "n_testA": len(arrays["testA"]["labels"]), "n_testB": len(arrays["testB"]["labels"]),
              "runs": runs, "seed42_pass_80": bool(runs[0]["testA_acc"] >= .80 and runs[0]["testB_acc"] >= .80),
              "protocol_note": "This is the post-audit canonical control only; no v2.3-O1 or N>1 training."}
    (OUT / "canonical_rerun.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
