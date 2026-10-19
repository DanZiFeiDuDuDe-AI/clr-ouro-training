# -*- coding: utf-8 -*-
"""
Phase 0H step 1 — offline teacher cache (§17).

Original Ouro teacher (no_grad) on CLUTRR distillation corpus. Cache:
  - 6 replaced layers' block output (post-attn residual hidden) — D1 target
  - final hidden at last token — D2 target
  - logits at last token — D3 target
Only cache what distillation needs (§17). Selected layers, selected positions.

Cache is per-trajectory-per-turn (CLUTRR is multi-turn). Saved as .pt shards
to avoid one huge file. Loaded by the distillation trainer.

§34: avoid teacher+student resident simultaneously — teacher runs alone here,
cache to disk, then student trains against cache.
"""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import torch
from clr_ouro import load_ouro, inject_film
from clutrr_data import generate_dataset

N_TRAJ = int(os.environ.get("C_N_TRAJ", "300"))
MAX_TOKENS = 128
REPLACED = [3, 7, 11, 15, 19, 23]
CACHE_DIR = "D:/training/teacher_cache"
MANIFEST = "D:/training/teacher_cache_manifest.json"

def main():
    print("=== Phase 0H step1: offline teacher cache ===", flush=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.manual_seed(42)
    _train = generate_dataset(600)  # replay RNG stream (parity)
    corpus = generate_dataset(N_TRAJ)

    print("loading teacher (original Ouro)...", flush=True)
    t0 = time.time()
    model, tok = load_ouro()
    print("loaded %.0fs" % (time.time()-t0), flush=True)
    m = model.model

    cached = 0
    total_turns = 0
    for ti, traj in enumerate(corpus):
        shard = {"traj_idx": ti, "turns": []}
        for turn_i, turn in enumerate(traj["turns"]):
            inp = tok(turn["prompt"], return_tensors="pt",
                      truncation=True, max_length=MAX_TOKENS).to("cuda")
            with torch.no_grad():
                base_out, hs_list, gate_list = m(
                    input_ids=inp["input_ids"],
                    attention_mask=inp["attention_mask"], use_cache=False)
                h = base_out.last_hidden_state  # [1, L, 2048]
                pos = int(inp["attention_mask"].sum(dim=1)[0].item()) - 1
                logits = model.lm_head(h[:, pos])[0]  # [vocab]
                final_h = h[0, pos]  # [2048]
            # D1 targets: the 6 replaced layers' block output. But OuroModel
            # forward runs ALL layers in a loop; hs_list is per-UT-step, not
            # per-layer. We need per-layer block output — recompute by running
            # the layer stack manually for the replaced layers.
            # Simpler: hook replaced layers' forward output. Since teacher is
            # UNCONVERTED here (original OuroAttention), we hook layers[i] for
            # i in REPLACED to grab their output.
            # We re-run with hooks (cheap, one more forward):
            block_outs = {}
            handles = []
            for li in REPLACED:
                def mk_hook(idx):
                    def hook(module, inp, out):
                        # OuroDecoderLayer.forward returns a tuple? No — returns
                        # a single tensor (audit). But GradientCheckpointingLayer
                        # may wrap. Grab first element if tuple.
                        block_outs[idx] = out[0] if isinstance(out, tuple) else out
                    return hook
                handles.append(m.layers[li].register_forward_hook(mk_hook(li)))
            with torch.no_grad():
                m(input_ids=inp["input_ids"],
                  attention_mask=inp["attention_mask"], use_cache=False)
            for hnd in handles:
                hnd.remove()
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
            print("  cached %d trajectories (%d turns)" % (ti, total_turns), flush=True)

    manifest = {"n_traj": cached, "n_turns": total_turns,
                "replaced_layers": REPLACED, "max_tokens": MAX_TOKENS,
                "n_d1_targets_per_turn": len(REPLACED),
                "cache_dir": CACHE_DIR}
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print("\ncached %d trajectories, %d turns" % (cached, total_turns), flush=True)
    print("manifest: %s" % MANIFEST, flush=True)
    print("PHASE 0H step1 OK", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
