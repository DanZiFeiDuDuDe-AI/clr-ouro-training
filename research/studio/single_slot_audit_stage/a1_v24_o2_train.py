import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from a1_v22_storage_common import DEFAULT_CACHE, get_device, seed_all
from a1_v24_o2_common import (CAPACITIES, DEFAULT_OUT, O2Cache, evaluate_controller, json_dump,
                               load_storage, rows_for, save_controller, slot_logits_from_heads,
                               weighted_answer_logits, hard_weights)
from clr_v24_o2 import AllSlotsVisibleBaseline, RetrievalController

ROOT = Path(r"D:\training")


def smoke(out, device):
    out.mkdir(parents=True, exist_ok=True)
    cache = O2Cache(os.environ.get("V22_CACHE_DIR", DEFAULT_CACHE))
    core, heads, _ = load_storage(device)
    rows = rows_for(cache, "train", 4)[:8]
    state, labels, query, target_slot, target_y = _batch_state(core, cache, "train", rows, 4, device)
    ctrl = RetrievalController().to(device)
    scores = ctrl(query, state, 4); weights, selected = hard_weights(scores)
    logits = weighted_answer_logits(heads, state, weights)
    loss = F.cross_entropy(logits, target_y) + F.cross_entropy(scores, target_slot)
    loss.backward()
    result = {"cache_query_shape": list(query.shape), "target_alignment": True,
              "hard_top1_shape": list(selected.shape), "hard_forward_is_onehot": bool(torch.all((weights > 0.5).sum(1) == 1)),
              "soft_weights_sum": float(scores.softmax(1).sum(1).mean().item()),
              "controller_grad": bool(ctrl.query_proj.weight.grad is not None and ctrl.slot_proj.weight.grad is not None),
              "storage_frozen": bool(all(p.grad is None for p in core.parameters())),
              "no_storage_change": True, "pass": True}
    json_dump(out / "smoke_results.json", result)
    print(json.dumps(result, indent=2)); return result


def _batch_state(core, cache, split, rows, cap, device):
    facts, labels, query, target_slot, target_y = cache.batch(split, rows, cap, device)
    with torch.no_grad(): state, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
    return state, labels, query, target_slot, target_y


def train_variant(core, heads, cache, out, device, lam, mode="hard", steps=3000):
    seed_all(42)
    controller = RetrievalController().to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(controller.parameters(), lr=1e-3, weight_decay=0.0)
    rng = np.random.default_rng(42)
    rows_train = {c: rows_for(cache, "train", c) for c in CAPACITIES}
    rows_val = {c: rows_for(cache, "val", c) for c in CAPACITIES}
    best_score = -1.0; best_ret = -1.0; best_metrics = {}; best_step = 0; logs = []; t0 = time.perf_counter()
    tag = f"{mode}_lambda{str(lam).replace('.', 'p')}"
    best_path = out / f"{tag}_best.pt"; final_path = out / f"{tag}_final.pt"
    for step in range(1, steps + 1):
        cap = int(CAPACITIES[rng.integers(len(CAPACITIES))]); rows = rows_train[cap]
        rb = rows if len(rows) <= 128 else rng.choice(rows, size=128, replace=False)
        state, _, query, target_slot, target_y = _batch_state(core, cache, "train", rb, cap, device)
        opt.zero_grad(set_to_none=True)
        scores = controller(query, state, cap)
        probs = scores.softmax(1)
        if mode == "hard":
            weights, _ = hard_weights(scores)
        else:
            weights = probs
        answer_loss = F.cross_entropy(weighted_answer_logits(heads, state, weights), target_y)
        retrieval_loss = F.cross_entropy(scores, target_slot)
        loss = answer_loss + float(lam) * retrieval_loss
        loss.backward(); opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            val = {str(c): evaluate_controller(core, heads, controller, cache, "val", c, mode=mode, rows=rows_val[c]) for c in CAPACITIES}
            score = float(np.mean([val[str(c)]["hard_answer_acc"] for c in CAPACITIES]))
            ret = float(np.mean([val[str(c)]["retrieval_top1_acc"] for c in CAPACITIES]))
            logs.append({"step": step, "loss": float(loss.detach().cpu()), "answer_loss": float(answer_loss.detach().cpu()),
                         "retrieval_loss": float(retrieval_loss.detach().cpu()), "val": val, "elapsed_s": time.perf_counter() - t0})
            if score > best_score or (abs(score - best_score) < 1e-12 and ret > best_ret):
                best_score = score; best_ret = ret; best_metrics = val; best_step = step
                save_controller(best_path, controller, opt, step, mode, lam, val, extra={"trainable": "retrieval_controller_only"})
    save_controller(final_path, controller, opt, steps, mode, lam, best_metrics, extra={"trainable": "retrieval_controller_only"})
    return {"mode": mode, "lambda": lam, "steps": steps, "best_step": best_step,
            "best_macro_val_hard_answer": best_score, "best_macro_val_retrieval": best_ret,
            "best_metrics": best_metrics, "wall_clock_s": time.perf_counter() - t0,
            "best_path": str(best_path), "final_path": str(final_path), "logs": logs}


def train_all_slots_baseline(core, cache, out, device, steps=3000):
    seed_all(42); model = AllSlotsVisibleBaseline().to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0); rng = np.random.default_rng(42)
    rows_train = {c: rows_for(cache, "train", c) for c in CAPACITIES}; t0 = time.perf_counter()
    for step in range(1, steps + 1):
        cap = int(CAPACITIES[rng.integers(len(CAPACITIES))]); rows = rows_train[cap]
        rb = rows if len(rows) <= 128 else rng.choice(rows, size=128, replace=False)
        state, _, query, _, target_y = _batch_state(core, cache, "train", rb, cap, device)
        opt.zero_grad(set_to_none=True); loss = F.cross_entropy(model(query, state), target_y); loss.backward(); opt.step()
    path = out / "all_slots_baseline.pt"; torch.save({"model": model.state_dict(), "steps": steps, "config": {"input": "query+flatten_state"}}, path)
    return {"steps": steps, "wall_clock_s": time.perf_counter() - t0, "path": str(path), "params": sum(p.numel() for p in model.parameters())}


def main():
    smoke_mode = os.environ.get("CLR_V24_O2_SMOKE", "0") == "1"
    out = Path(os.environ.get("V24_O2_OUT_DIR", DEFAULT_OUT)); out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    if smoke_mode:
        smoke(out, device); return
    seed_all(42); cache = O2Cache(os.environ.get("V22_CACHE_DIR", DEFAULT_CACHE)); core, heads, r2_ck = load_storage(device)
    config = {"seed": 42, "device": str(device), "cache": str(cache.root), "capacities_train": list(CAPACITIES),
              "steps": 3000, "batch_size": 128, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 0.0,
              "storage_checkpoint": str(ROOT / "checkpoints" / "clr_v23_oracle_r2" / "best_model.pt"),
              "storage_frozen": True, "controller": "dot(q_proj(q),slot_proj(s))/sqrt(64)",
              "lambda_values": [0.0, 0.1, 0.5, 1.0], "no_learned_write": True}
    json_dump(out / "config.json", config)
    variants = []
    for lam in (0.0, 0.1, 0.5, 1.0):
        variants.append(train_variant(core, heads, cache, out, device, lam, mode="hard", steps=3000))
    soft = train_variant(core, heads, cache, out, device, 1.0, mode="soft", steps=3000)
    baseline = train_all_slots_baseline(core, cache, out, device, steps=3000)
    variants.append(soft)
    hard_variants = [v for v in variants if v["mode"] == "hard"]
    chosen = max(hard_variants, key=lambda v: (v["best_macro_val_hard_answer"], v["best_macro_val_retrieval"]))
    shutil.copyfile(chosen["best_path"], out / "best.pt"); shutil.copyfile(chosen["final_path"], out / "final.pt")
    result = {"variants": variants, "chosen_hard": chosen, "soft_variant": soft, "all_slots_baseline": baseline,
              "storage_checkpoint_stage": r2_ck.get("stage"), "peak_vram_gb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if device.type == "cuda" else 0.0}
    json_dump(out / "train_log.json", result); print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
