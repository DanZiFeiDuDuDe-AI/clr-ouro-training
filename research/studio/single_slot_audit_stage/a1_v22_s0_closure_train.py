import json
import time
from pathlib import Path

import numpy as np
import torch

from a1_v22_s0_closure_common import (
    BottleneckBaseline, LinearProbe, SingleSlotBaseline, accuracy, closure_paths,
    get_device, load_representations, probe_one, train_classifier,
)


def prepare(data):
    for rep in data["splits"].values():
        # N=1 diagnostics use the actually routed slot0 candidate.
        rep["key_candidate"] = rep["key_candidate"][:, 0]
        rep["value_candidate"] = rep["value_candidate"][:, 0]
        rep["fulls"] = rep["state"].reshape(len(rep["state"]), -1)
        rep["force_slot0_full"] = rep["force_slot0"].reshape(len(rep["force_slot0"]), -1)
        rep["force_uniform_full"] = rep["force_uniform"].reshape(len(rep["force_uniform"]), -1)
        rep["bypass_full"] = rep["bypass"].reshape(len(rep["bypass"]), -1)
        rep["zero_init_full"] = rep["zero_init"].reshape(len(rep["zero_init"]), -1)
        rep["random_init_full"] = rep["random_init"].reshape(len(rep["random_init"]), -1)
    return data


def train_bottleneck(data, dim, device, out_path):
    tr, va, ta, tb = [data["splits"][s] for s in ("train", "val", "testA", "testB")]
    model, info = train_classifier(BottleneckBaseline(2048, dim), tr["H"], tr["labels"],
                                   va["H"], va["labels"], device, save_path=out_path)
    return {"dim": dim, "train_acc": accuracy(model, tr["H"], tr["labels"], device),
            "val_acc": info["best_val_acc"], "testA_acc": accuracy(model, ta["H"], ta["labels"], device),
            "testB_acc": accuracy(model, tb["H"], tb["labels"], device), "epochs": info["epochs"]}


def train_ss64(data, device, out_path):
    tr, va, ta, tb = [data["splits"][s] for s in ("train", "val", "testA", "testB")]
    model, info = train_classifier(SingleSlotBaseline(), tr["H"], tr["labels"],
                                   va["H"], va["labels"], device, save_path=out_path)
    return {"train_acc": accuracy(model, tr["H"], tr["labels"], device),
            "val_acc": info["best_val_acc"], "testA_acc": accuracy(model, ta["H"], ta["labels"], device),
            "testB_acc": accuracy(model, tb["H"], tb["labels"], device), "epochs": info["epochs"]}


def main():
    out = closure_paths(); out.mkdir(parents=True, exist_ok=True)
    device = get_device(); data = prepare(load_representations(out / "closure_representations.npz"))
    result = {"device": str(device), "probes": {}, "kv": {}, "bottlenecks": {}, "single_slot": {}}
    probes = [
        ("H", 2048, "probe_h.pt"), ("value_read", 32, "probe_valueread.pt"),
        ("value_candidate", 32, "probe_valuecandidate.pt"), ("slot0", 64, "probe_slot0.pt"),
        ("fulls", 256, "probe_fulls.pt"),
    ]
    for name, dim, file_name in probes:
        result["probes"][name] = probe_one(data, name, dim, device=device, out_path=out / file_name)
    for name, dim in [("key_read", 31), ("value_read", 32)]:
        result["kv"][name + "_to_city"] = probe_one(data, name, dim, device=device)
        result["kv"][name + "_to_person"] = probe_one(data, name, dim, n_classes=64, person=True, device=device)
    for dim in (32, 64, 128):
        result["bottlenecks"][str(dim)] = train_bottleneck(data, dim, device, out / ("baseline_h_%d.pt" % dim))
    result["single_slot"] = train_ss64(data, device, out / "single_slot_baseline.pt")
    (out / "kv_factorization_probes.json").write_text(json.dumps(result["kv"], indent=2), encoding="utf-8")
    (out / "bottleneck_baselines.json").write_text(json.dumps(result["bottlenecks"], indent=2), encoding="utf-8")
    (out / "single_slot_baseline.json").write_text(json.dumps(result["single_slot"], indent=2), encoding="utf-8")
    (out / "closure_probe_train_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
