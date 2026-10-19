import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from a1_v22_storage_common import (
    DEFAULT_CACHE, DEFAULT_OUT, FactCache, FactorizedStorage, eval_capacity,
    get_device, json_dump, macro_capacity, make_heads, model_dtype, save_checkpoint,
    seed_all,
)


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def rows_for(cache, split, cap, limit=None):
    rows = cache.rows_for_capacity(split, cap)
    if limit is not None:
        rows = rows[:min(len(rows), int(limit))]
    return rows


def gradient_sanity(model, loss, stage):
    loss.backward()
    names = {
        "key_read": model.key_read.weight.grad,
        "value_read": model.value_read.weight.grad,
        "key_gru": model.key_gru.weight_ih.grad,
        "value_gru": model.value_gru.weight_ih.grad,
        "init_s": model.init_s.grad,
    }
    report = {k: (v is not None and bool(torch.isfinite(v).all())) for k, v in names.items()}
    report["all_required"] = all(report.values())
    if not report["all_required"]:
        raise RuntimeError("v2.2-S gradient sanity failed: " + repr(report))
    return report


def train_stage(model, heads, cache, out_dir, stage, steps, capacities, eval_caps,
                train_limit=None, val_limit=None, batch_size=64, eval_every=100, seed=2237,
                optimizer=None):
    device = get_device()
    model.train(); heads.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(heads.parameters()),
            lr=3e-4, weight_decay=0.01,
        )
    rng = np.random.default_rng(seed + (0 if stage == "S0" else 1))
    train_rows = {c: rows_for(cache, "train", c, train_limit) for c in capacities}
    val_rows = {c: rows_for(cache, "val", c, val_limit) for c in eval_caps}
    if any(len(v) == 0 for v in train_rows.values()):
        raise RuntimeError("missing training rows for one or more capacities")
    logs = []
    best_score = -1.0
    best_metrics = {}
    best_path = Path(out_dir) / ("best_s0.pt" if stage == "S0" else "best_model.pt")
    first_grad = None
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        cap = int(capacities[rng.integers(0, len(capacities))])
        rows = train_rows[cap]
        if len(rows) <= batch_size:
            rb = rows
        else:
            rb = rng.choice(rows, size=batch_size, replace=False)
        facts, labels, positions, y = cache.batch("train", rb, cap, device)
        optimizer.zero_grad(set_to_none=True)
        state, _, _ = model.forward_facts(facts, keep_states=False, keep_stats=False)
        # The only loss is final-state city CE.  Head selection is oracle position
        # selection, not a learned query/retrieval module.
        logits_parts = []
        target_parts = []
        for p in range(4):
            mask = positions == p
            if mask.any():
                logits_parts.append(heads[p](state[mask].flatten(1)))
                target_parts.append(y[mask])
        logits = torch.cat(logits_parts, dim=0)
        target = torch.cat(target_parts, dim=0)
        loss = F.cross_entropy(logits.float(), target)
        if first_grad is None:
            first_grad = gradient_sanity(model, loss, stage)
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
        optimizer.step()
        if step == 1 or step % eval_every == 0 or step == steps:
            val_metrics = {}
            for c in eval_caps:
                val_metrics[str(c)] = eval_capacity(model, heads, cache, "val", c,
                                                     batch_size=128, rows=val_rows[c])
            score = val_metrics.get("macro", {}).get("acc", 0.0)
            if not score:
                score = float(np.mean([val_metrics[str(c)]["acc"] for c in eval_caps]))
            record = {
                "step": step, "stage": stage, "loss": float(loss.detach().float()),
                "grad_norm": float(grad_norm), "val": val_metrics,
                "elapsed_s": time.perf_counter() - t0,
            }
            logs.append(record)
            if score > best_score:
                best_score = score
                best_metrics = val_metrics
                save_checkpoint(best_path, model, heads, optimizer=optimizer, step=step,
                                stage=stage, metrics=best_metrics,
                                extra={"gradient_sanity": first_grad})
    return {
        "stage": stage, "steps": steps, "wall_clock_s": time.perf_counter() - t0,
        "best_score": best_score, "best_metrics": best_metrics,
        "gradient_sanity": first_grad, "logs": logs, "optimizer": optimizer,
        "best_path": str(best_path),
    }


def main():
    seed_all(2237)
    device = get_device()
    smoke = os.environ.get("A1_LS2_SMOKE", "0") == "1" or os.environ.get("A1_V22_S_SMOKE", "0") == "1"
    cache_root = os.environ.get("V22_CACHE_DIR", DEFAULT_CACHE)
    out_dir = Path(os.environ.get("V22_OUT_DIR", DEFAULT_OUT))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = FactCache(cache_root)
    if smoke:
        s0_steps, s1_steps = 20, 20
        train_limit, val_limit = 256, 128
        batch_size = 32
    else:
        s0_steps, s1_steps = 2000, 3000
        train_limit = val_limit = None
        batch_size = 64
    model = FactorizedStorage(device=device)
    heads = make_heads(device=device, dtype=model_dtype(device))
    all_logs = []
    t_all = time.perf_counter()

    s0 = train_stage(model, heads, cache, out_dir, "S0", s0_steps, [1], [1],
                     train_limit=train_limit, val_limit=val_limit, batch_size=batch_size,
                     eval_every=5 if smoke else 100, seed=2237)
    all_logs.append({k: v for k, v in s0.items() if k not in ("optimizer",)})
    # Reload the selected S0 checkpoint before the gate and before any S1 update.
    s0_ckpt = torch.load(out_dir / "best_s0.pt", map_location=device)
    model.load_state_dict(s0_ckpt["model"])
    heads.load_state_dict(s0_ckpt["heads"])
    # S0 gate is evaluated on held-out TestA/TestB before any N>1 training.
    s0_testA = eval_capacity(model, heads, cache, "testA", 1, batch_size=128,
                             rows=rows_for(cache, "testA", 1, None if not smoke else 128))
    s0_testB = eval_capacity(model, heads, cache, "testB", 1, batch_size=128,
                             rows=rows_for(cache, "testB", 1, None if not smoke else 128))
    s0_gate = {
        "testA_n1": s0_testA, "testB_n1": s0_testB,
        "pass": bool(s0_testA["acc"] >= 0.85 and s0_testB["acc"] >= 0.85),
        "stop_below_80": bool(s0_testA["acc"] < 0.80 or s0_testB["acc"] < 0.80),
    }
    json_dump(out_dir / "s0_gate.json", s0_gate)

    run_s1 = s0_gate["pass"] or smoke
    if not run_s1:
        # Preserve a best_model artifact for deterministic downstream evaluation,
        # while marking that S1 did not run.
        save_checkpoint(out_dir / "best_model.pt", model, heads, step=s0_steps,
                        stage="S0-only", metrics={"s0_gate": s0_gate})
        save_checkpoint(out_dir / "final_model.pt", model, heads, step=s0_steps,
                        stage="S0-only", metrics={"s0_gate": s0_gate})
        result = {
            "smoke": smoke, "device": str(device), "cache": cache_root,
            "s0": {k: v for k, v in s0.items() if k not in ("optimizer",)},
            "s0_gate": s0_gate, "s1_ran": False,
            "overall_wall_clock_s": time.perf_counter() - t_all,
            "parameter_report": model.parameter_report(),
            "peak_vram_gb": (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0,
        }
        json_dump(out_dir / "train_log.json", {"stages": all_logs, "result": result})
        print(json.dumps(result, indent=2, default=str))
        return

    # S1 continues from the same fresh v2.2-S S0 model, not from v2.1a/b or LS1.
    s0["optimizer"].load_state_dict(s0_ckpt.get("optimizer", s0["optimizer"].state_dict()))
    s1 = train_stage(model, heads, cache, out_dir, "S1", s1_steps, [1, 2, 4], [1, 2, 4],
                     train_limit=train_limit, val_limit=val_limit, batch_size=batch_size,
                     eval_every=5 if smoke else 100, seed=2237, optimizer=s0["optimizer"])
    all_logs.append({k: v for k, v in s1.items() if k not in ("optimizer",)})
    save_checkpoint(out_dir / "final_model.pt", model, heads, optimizer=s1["optimizer"],
                    step=s1_steps, stage="S1-final", metrics=s1["best_metrics"])
    result = {
        "smoke": smoke, "device": str(device), "cache": cache_root,
        "s0": {k: v for k, v in s0.items() if k not in ("optimizer",)},
        "s0_gate": s0_gate,
        "s1": {k: v for k, v in s1.items() if k not in ("optimizer",)},
        "s1_ran": True, "overall_wall_clock_s": time.perf_counter() - t_all,
        "parameter_report": model.parameter_report(),
        "peak_vram_gb": (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0,
        "protocol": {"lr": 3e-4, "weight_decay": 0.01, "grad_clip": 1.0,
                     "batch_size": batch_size, "full_bptt": True,
                     "no_query_hidden": True, "no_architecture_change": True},
    }
    json_dump(out_dir / "train_log.json", {"stages": all_logs, "result": result})
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
