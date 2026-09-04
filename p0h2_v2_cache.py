# -*- coding: utf-8 -*-
"""
H2-v2 step 1 — offline teacher cache v2 (fixed-seed, P0H5-aligned protocol).

Fixes H2-v1 seed bug: generate_dataset uses the Python stdlib `random`, so
cache generation MUST call random.seed(42) (not only torch.manual_seed).
Protocol (identical RNG stream to the H2-v2 trainer and P0H5 eval):
    random.seed(42); torch.manual_seed(42); numpy.random.seed(42)
    _ = generate_dataset(600)           # replay parity (same as P0H5)
    _ = generate_dataset(150)[:150]     # P0H5 held-out placeholder — NOT cached,
                                        # so the held-out set NEVER enters training
    corpus = generate_dataset(600)      # train cache (600 traj, multi-epoch)

The trainer re-runs the SAME seed stream, so shards pair 1:1 with corpus
trajectories (prompt-level pairing reproducible).
Output : D:/training/teacher_cache_v2  (does NOT touch teacher_cache)
"""
import os, sys, json, time, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import numpy as np
import torch
from clr_ouro import load_ouro
from clutrr_data import generate_dataset

N_CACHE_TRAJ = int(os.environ.get("V2_CACHE_TRAJ", "600"))
MAX_TOKENS = 128
REPLACED = [3, 7, 11, 15, 19, 23]
CACHE_DIR = "D:/training/teacher_cache_v2"
MANIFEST = "D:/training/teacher_cache_v2_manifest.json"
SEED = 42


def main():
    print("=== H2-v2 offline teacher cache (fixed seed) ===", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _train = generate_dataset(600)               # replay RNG stream (parity)
    _heldout = generate_dataset(150)[:150]       # P0H5 held-out placeholder (skip)
    corpus = generate_dataset(N_CACHE_TRAJ)[:N_CACHE_TRAJ]
    print("protocol: seed=%d 600-replay | 150-heldout-skip | %d train" % (
        SEED, len(corpus)), flush=True)

    print("loading teacher (original Ouro)...", flush=True)
    t0 = time.time()
    model, tok = load_ouro()
    print("loaded %.0fs" % (time.time() - t0), flush=True)
    m = model.model

    block_outs = {}
    def mk_hook(idx):
        def hook(module, inp, out):
            block_outs[idx] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [m.layers[li].register_forward_hook(mk_hook(li)) for li in REPLACED]

    cached = 0
    total_turns = 0
    for ti, traj in enumerate(corpus):
        shard = {"traj_idx": ti, "turns": []}
        for turn_i, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt",
                      truncation=True, max_length=MAX_TOKENS).to("cuda")
            with torch.no_grad():
                base_out, _, _ = m(
                    input_ids=inp["input_ids"],
                    attention_mask=inp["attention_mask"], use_cache=False)
                h = base_out.last_hidden_state
                pos = int(inp["attention_mask"].sum(dim=1)[0].item()) - 1
                logits = model.lm_head(h[:, pos])[0]
                final_h = h[0, pos]
            block_outs.clear()
            with torch.no_grad():
                m(input_ids=inp["input_ids"],
                  attention_mask=inp["attention_mask"], use_cache=False)
            shard["turns"].append({
                "turn_idx": turn_i,
                "answer": turn["answer"],
                "block_outs": {str(li): block_outs[li][0, :].float().cpu().clone()
                               for li in REPLACED if li in block_outs},
                "final_h": final_h.float().cpu().clone(),
                "logits": logits.float().cpu().clone(),
                "last_pos": pos,
            })
            total_turns += 1
        torch.save(shard, os.path.join(CACHE_DIR, "traj_%04d.pt" % ti))
        cached += 1
        if ti % 50 == 0 and ti > 0:
            print("  cached %d trajectories (%d turns) %.0fs" % (
                ti, total_turns, time.time() - t0), flush=True)
    for hnd in handles:
        hnd.remove()

    manifest = {
        "n_traj": cached, "n_turns": total_turns,
        "replaced_layers": REPLACED, "max_tokens": MAX_TOKENS,
        "seed": SEED, "seed_protocol": "random.seed(42);torch.manual_seed(42);np.random.seed(42) -> generate_dataset(600) -> generate_dataset(150)[:150] skip -> generate_dataset(600)",
        "heldout_skipped": True,
        "cache_dir": CACHE_DIR,
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print("\ncached %d trajectories, %d turns (%.0fs)" % (
        cached, total_turns, time.time() - t0), flush=True)
    print("manifest: %s" % MANIFEST, flush=True)
    print("H2V2 CACHE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
