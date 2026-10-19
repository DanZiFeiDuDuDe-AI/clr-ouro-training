import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v22_s0_closure_common import SingleSlotBaseline, accuracy, load_representations, train_classifier
from a1_v22_storage_common import DEFAULT_CACHE, FactCache, get_device, seed_all
from a1_v23_r2_common import (DEFAULT_OUT, CAPACITIES, eval_capacity, json_dump, make_heads,
                              rows_for, save_checkpoint)
from clr_v23_oracle_r2 import OracleLocalCore

ROOT = Path(r"D:\training")
CLOSURE_OUT = ROOT / "checkpoints" / "clr_v22_s0_closure"


def diff_stats(a, b):
    d = (a - b).abs()
    return {"max_abs_diff": float(d.max().item()), "mean_abs_diff": float(d.mean().item()),
            "allclose_1e-6": bool(torch.allclose(a, b, atol=1e-6, rtol=1e-6))}


def forward_equivalence(device):
    torch.manual_seed(7301)
    canonical = SingleSlotBaseline().to(device=device, dtype=torch.float32).eval()
    r2 = OracleLocalCore(device=device, dtype=torch.float32).eval()
    with torch.no_grad():
        r2.content_read.load_state_dict(canonical.read.state_dict())
        r2.shared_gru.load_state_dict(canonical.gru.state_dict())
    head = nn.Linear(64, 64).to(device=device, dtype=torch.float32)
    head.load_state_dict(canonical.head.state_dict())
    h = torch.randn(128, 2048, device=device, dtype=torch.float32)
    z1 = canonical.read(h)
    s1 = canonical.gru(z1, torch.zeros_like(z1))
    logits1 = head(s1)
    state, states = r2.forward_facts(h[:, None, :], slot_indices=[0], keep_states=True)
    z2 = r2.content_read(h)
    s2 = states[1][:, 0, :]
    logits2 = head(s2)
    out = {"dtype": "float32", "batch": 128, "z": diff_stats(z1, z2),
           "gru_output": diff_stats(s1, s2), "slot0": diff_stats(s1, s2),
           "logits": diff_stats(logits1, logits2)}
    out["pass"] = bool(all(out[k]["max_abs_diff"] <= 1e-6 for k in ("z", "gru_output", "logits")))
    out["path"] = "SingleSlotBaseline and OracleLocalCore with shared weights; zero state; local Linear(64,64)"
    return out


def load_arrays():
    data = load_representations(CLOSURE_OUT / "closure_representations.npz")
    return {s: {"H": data["splits"][s]["H"].astype(np.float32),
                "labels": data["splits"][s]["labels"].astype(np.int64)}
            for s in ("train", "val", "testA", "testB")}


def o0_train(arrays, device, out):
    seed_all(42)
    model = SingleSlotBaseline()
    tr, va, ta, tb = [arrays[s] for s in ("train", "val", "testA", "testB")]
    model, info = train_classifier(model, tr["H"], tr["labels"], va["H"], va["labels"],
                                   device, lr=1e-3, max_epochs=100, patience=10, seed=42)
    metrics = {"train_acc": accuracy(model, tr["H"], tr["labels"], device),
               "val_acc": info["best_val_acc"], "testA_acc": accuracy(model, ta["H"], ta["labels"], device),
               "testB_acc": accuracy(model, tb["H"], tb["labels"], device), "epochs": info["epochs"]}
    core = OracleLocalCore(device=device, dtype=torch.float32)
    heads = nn.ModuleList([nn.Linear(64, 64)]).to(device=device, dtype=torch.float32)
    core.content_read.load_state_dict(model.read.state_dict())
    core.shared_gru.load_state_dict(model.gru.state_dict())
    heads[0].load_state_dict(model.head.state_dict())
    ck_metrics = {**metrics, "gate": {"val_ge_78": metrics["val_acc"] >= .78,
                                       "testA_ge_80": metrics["testA_acc"] >= .80,
                                       "testB_ge_80": metrics["testB_acc"] >= .80,
                                       "stop_below_75": metrics["testA_acc"] < .75 or metrics["testB_acc"] < .75}}
    ck_metrics["gate"]["pass"] = bool(ck_metrics["gate"]["val_ge_78"] and
                                        ck_metrics["gate"]["testA_ge_80"] and
                                        ck_metrics["gate"]["testB_ge_80"] and
                                        not ck_metrics["gate"]["stop_below_75"])
    extra = {"protocol": {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.0,
                           "batch_size": 128, "max_epochs": 100, "patience": 10,
                           "optimization_seed": 42, "architecture_init": "zero hidden canonical"}}
    save_checkpoint(out / "best_o0.pt", core, heads, 0, "O0", ck_metrics, seed=42, extra=extra)
    save_checkpoint(out / "final_o0.pt", core, heads, 0, "O0", ck_metrics, seed=42, extra=extra)
    return metrics, ck_metrics["gate"], core, heads


def o1_train(core, heads, cache, device, out, steps=3000, smoke=False):
    rng = np.random.default_rng(42)
    batch = 32 if smoke else 128
    caps = list(CAPACITIES)
    train_rows = {c: rows_for(cache, "train", c) for c in caps}
    val_rows = {c: rows_for(cache, "val", c) for c in caps}
    opt = torch.optim.AdamW(list(core.parameters()) + list(heads.parameters()), lr=1e-3, weight_decay=0.0)
    logs = []; best_macro = -1.0; best_worst = -1.0; best_metrics = {}; position_counts = {c: [0] * c for c in caps}
    first_grad = None; t0 = time.perf_counter()
    for step in range(1, steps + 1):
        cap = int(caps[rng.integers(len(caps))]); rows = train_rows[cap]
        rb = rows if len(rows) <= batch else rng.choice(rows, size=batch, replace=False)
        facts, labels, _, _ = cache.batch("train", rb, cap, device)
        pos_np = rng.integers(0, cap, size=len(rb), dtype=np.int64)
        for p in pos_np: position_counts[cap][int(p)] += 1
        target_pos = torch.from_numpy(pos_np).to(device)
        opt.zero_grad(set_to_none=True)
        state, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
        losses = []
        for p in range(cap):
            mask = target_pos == p
            if mask.any():
                logits = heads[p](state[mask, p, :])
                losses.append(F.cross_entropy(logits.float(), labels[mask, p]))
        loss = sum(losses) / len(losses)
        loss.backward()
        if first_grad is None:
            first_grad = {"content_read": bool(core.content_read.weight.grad is not None),
                          "shared_gru": bool(core.shared_gru.weight_ih.grad is not None),
                          "selected_head": bool(any(h.weight.grad is not None for h in heads)),
                          "has_init_s": False}
        opt.step()
        if step == 1 or step % (5 if smoke else 100) == 0 or step == steps:
            val = {str(c): eval_capacity(core, heads, cache, "val", c, rows=val_rows[c]) for c in caps}
            macro = float(np.mean([val[str(c)]["acc"] for c in caps]))
            worst = float(min(val["4"]["by_position"]))
            rec = {"step": step, "loss": float(loss.detach().cpu()), "macro_val": macro,
                   "val": val, "elapsed_s": time.perf_counter() - t0}
            logs.append(rec)
            if macro > best_macro or (abs(macro - best_macro) < 1e-12 and worst > best_worst):
                best_macro = macro; best_worst = worst; best_metrics = val
                save_checkpoint(out / "best_model.pt", core, heads, step, "O1", val, seed=42,
                                extra={"gradient_sanity": first_grad, "position_counts": position_counts,
                                       "protocol": {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.0,
                                                     "batch_size": batch, "steps": steps, "grad_clip": None,
                                                     "sampling": "uniform N in {1,2,4}, uniform target position"}})
    save_checkpoint(out / "final_model.pt", core, heads, steps, "O1-final", best_metrics, seed=42,
                    extra={"gradient_sanity": first_grad, "position_counts": position_counts})
    return {"steps": steps, "wall_clock_s": time.perf_counter() - t0, "best_macro_val": best_macro,
            "best_n4_worst_position": best_worst, "best_metrics": best_metrics,
            "position_counts": position_counts, "gradient_sanity": first_grad, "logs": logs}


def smoke(out, device):
    out.mkdir(parents=True, exist_ok=True)
    eq = forward_equivalence(device); json_dump(out / "forward_equivalence.json", eq)
    if not eq["pass"]:
        raise RuntimeError("r2 smoke forward equivalence failed")
    core = OracleLocalCore(device=device, dtype=torch.float32)
    heads = make_heads(device, 4)
    facts = torch.randn(8, 4, 2048, device=device)
    state, states = core.forward_facts(facts, slot_indices=[0, 1, 2, 3], keep_states=True)
    loss = sum(heads[p](state[:, p, :]).float().sum() for p in range(4))
    loss.backward()
    grads = {"content_read": core.content_read.weight.grad is not None,
             "shared_gru": core.shared_gru.weight_ih.grad is not None,
             "has_init_s": False}
    max_untouched = 0.0
    for t in range(1, len(states)):
        target = t - 1
        other = [j for j in range(4) if j != target]
        max_untouched = max(max_untouched, float((states[t][:, other] - states[t - 1][:, other]).abs().max().item()))
    torch.save({"model": core.state_dict(), "heads": heads.state_dict(), "config": {"local": True}}, out / "smoke.pt")
    loaded = OracleLocalCore(device=device, dtype=torch.float32); loaded.load_state_dict(torch.load(out / "smoke.pt", map_location=device)["model"])
    result = {"forward_equivalence": eq, "zero_hidden": True, "local_head_dim": 64,
              "no_init_s": "init_s" not in core.state_dict(), "no_flatten_readout": True,
              "full_bptt_grads": grads, "non_target_max_abs_diff": max_untouched,
              "checkpoint_roundtrip": True, "pass": bool(grads["content_read"] and grads["shared_gru"] and max_untouched == 0.0)}
    json_dump(out / "smoke_results.json", result); print(json.dumps(result, indent=2)); return result


def main():
    smoke_mode = os.environ.get("CLR_V23_R2_SMOKE", "0") == "1"
    out = Path(os.environ.get("V23_R2_OUT_DIR", DEFAULT_OUT))
    device = get_device()
    if smoke_mode:
        smoke(out, device); return
    out.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    eq = forward_equivalence(device); json_dump(out / "forward_equivalence.json", eq)
    if not eq["pass"]:
        raise RuntimeError("r2 O0 forward equivalence failed; stopping before training")
    cache = FactCache(os.environ.get("V22_CACHE_DIR", DEFAULT_CACHE))
    arrays = load_arrays()
    o0_metrics, gate, core, head0 = o0_train(arrays, device, out)
    result = {"device": str(device), "forward_equivalence": eq, "o0": o0_metrics, "o0_gate": gate,
              "o1_ran": False, "protocol": {"o0": {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.0,
                                                      "batch_size": 128, "max_epochs": 100, "patience": 10, "seed": 42},
                                              "o1": {"optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.0,
                                                      "batch_size": 128, "steps": 3000, "seed": 42}},
              "parameter_report_o0": {"content_read": int(sum(p.numel() for p in core.content_read.parameters())),
                                      "shared_gru": int(sum(p.numel() for p in core.shared_gru.parameters())),
                                      "head0": int(sum(p.numel() for p in head0[0].parameters())),
                                      "has_init_s": False}}
    if gate["pass"] and not gate["stop_below_75"]:
        torch.manual_seed(42)
        heads = make_heads(device, 4)
        heads[0].load_state_dict(head0[0].state_dict())
        o1 = o1_train(core, heads, cache, device, out, steps=3000)
        result["o1"] = o1; result["o1_ran"] = True
    result["peak_vram_gb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0
    json_dump(out / "train_log.json", result)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
