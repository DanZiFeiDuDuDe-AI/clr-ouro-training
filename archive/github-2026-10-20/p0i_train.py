# -*- coding: utf-8 -*-
"""
P0I training: C-reset vs C-persist (turn-boundary state lifetime).

Init: M2_CANONICAL (checkpoints/ouro_retention_m2_d3) — gamma frozen,
      post-norms trainable. Same teacher cache / seed 42 / loss / steps as 0H.
Sole variable: turn-boundary state lifetime.

Usage:
  P0I_MODE=creset  python p0i_train.py   # or cpersist
Env: P0I_MODE, P0I_STEPS (600), P0I_LR (3e-4), P0I_OUT_DIR, P0I_SMOKE=1
"""
import os, sys, json, time, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"

import torch
import torch.nn.functional as F
from p0i_loop_retention import (load_m2_canonical, reset_turn_state,
                                set_persist_state, clear_all_state, REPLACED)
from clutrr_data import generate_dataset

MODE = os.environ.get("P0I_MODE", "creset")
assert MODE in ("creset", "cpersist"), "P0I_MODE must be creset|cpersist"
N_STEPS = int(os.environ.get("P0I_STEPS", "600"))
LR = float(os.environ.get("P0I_LR", "3e-4"))
LAM_BLOCK = float(os.environ.get("P0I_LAM_BLOCK", "1.0"))
LAM_LOGIT = float(os.environ.get("P0I_LAM_LOGIT", "0.5"))
SMOKE = os.environ.get("P0I_SMOKE", "") == "1"
OUT_DIR = os.environ.get("P0I_OUT_DIR",
                         "D:/training/checkpoints/p0i_%s_s42" % MODE)
MAX_TOKENS = 128
CACHE_DIR = "D:/training/teacher_cache"
POST_NORM_NAMES = ["input_layernorm_2", "post_attention_layernorm_2"]
PROGRESS_PATH = os.path.join(OUT_DIR, "progress.json")


@torch.no_grad()
def forward_turn(model, tok, prompt, offset=0):
    """Single-turn forward with explicit position_ids (offset for continuity).
    Returns (block_outs, h, pos, seq_len)."""
    inp = tok(prompt, return_tensors="pt", truncation=True,
              max_length=MAX_TOKENS).to("cuda")
    s_len = int(inp["attention_mask"].sum(dim=1)[0].item())
    if offset > 0:
        pos_ids = (offset + torch.arange(s_len, device="cuda")).unsqueeze(0)
    else:
        pos_ids = torch.arange(s_len, device="cuda").unsqueeze(0)
    base_out, _, _ = model.model(input_ids=inp["input_ids"],
                                 attention_mask=inp["attention_mask"],
                                 use_cache=False, position_ids=pos_ids)
    h = base_out.last_hidden_state
    pos = s_len - 1
    return inp, h, pos, s_len


def main():
    print("=== P0I training mode=%s ===" % MODE, flush=True)
    print("steps=%d lr=%g lam_block=%g lam_logit=%g smoke=%s out=%s" % (
        N_STEPS, LR, LAM_BLOCK, LAM_LOGIT, SMOKE, OUT_DIR), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("loading M2_CANONICAL (d3) + frozen gamma + trainable post-norms...", flush=True)
    model, tok, adapters, rets, post_norm_params = load_m2_canonical()
    n_trainable = sum(p.numel() for p in post_norm_params)
    print("trainable post-norm params: %d" % n_trainable, flush=True)
    opt = torch.optim.Adam(post_norm_params, lr=LR)

    shards = sorted(glob.glob(os.path.join(CACHE_DIR, "traj_*.pt")))
    if SMOKE:
        shards = shards[:10]
    print("cache shards: %d" % len(shards), flush=True)

    block_outs = {}

    def mk_hook(idx):
        def hook(module, inp, out):
            block_outs[idx] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [model.model.layers[li].register_forward_hook(mk_hook(li)) for li in REPLACED]

    print("re-generating dataset to pair prompts with cache...", flush=True)
    torch.manual_seed(42)
    _ = generate_dataset(600)
    corpus = generate_dataset(len(shards))[:len(shards)]

    def save_ckpt(step, final=False):
        ck = {
            "variant": "p0i_%s_s42" % MODE,
            "p0i_mode": MODE,
            "replaced_layers": REPLACED,
            "retention_gamma": {str(ret.layer_idx): ret.gamma_logit.detach().cpu().clone()
                                for ret in rets},
            "gamma_param_mode": "sigmoid_logit_frozen",
            "distill_steps": step, "lr": LR,
            "lam_block": LAM_BLOCK, "lam_logit": LAM_LOGIT,
            "unfroze_post_norms": POST_NORM_NAMES,
            "final_loss": float(losses[-1]) if losses else None,
            "mean_last10_loss": float(sum(losses[-10:]) / max(len(losses[-10:]), 1)),
            "post_norm_weights": {
                str(li): {nm: getattr(model.model.layers[li], nm).weight.detach().cpu().clone()
                          for nm in POST_NORM_NAMES}
                for li in REPLACED},
        }
        path = os.path.join(OUT_DIR, "best_model.pt" if final else "ckpt_step%04d.pt" % step)
        torch.save(ck, path)
        return path

    step = 0
    t0 = time.time()
    losses = []
    turn_stats = {"turns": 0, "max_turns_per_traj": 0}
    while step < N_STEPS:
        for shard_path, traj in zip(shards, corpus):
            if step >= N_STEPS:
                break
            shard = torch.load(shard_path, map_location="cuda", weights_only=False)
            # episode boundary: always reset state
            clear_all_state(adapters)
            offset = 0
            n_turn = len(traj["turns"])
            turn_stats["max_turns_per_traj"] = max(turn_stats["max_turns_per_traj"], n_turn)
            for turn_i, turn in enumerate(traj["turns"]):
                if step >= N_STEPS:
                    break
                if turn_i >= len(shard["turns"]):
                    break
                cached = shard["turns"][turn_i]
                # turn boundary state injection
                if turn_i == 0:
                    reset_turn_state(adapters)   # episode+turn start: zero
                elif MODE == "creset":
                    reset_turn_state(adapters)
                else:
                    set_persist_state(adapters)  # C-persist: carry prev turn end
                inp, h, pos, s_len = forward_turn(model, tok, turn["prompt"], offset=offset)
                if s_len - 1 != cached["last_pos"]:
                    if step == 0:
                        print("WARN: student seq %d vs cache pos %d — skipping turn" % (s_len, cached["last_pos"]), flush=True)
                    continue
                l_block = torch.tensor(0.0, device="cuda")
                for li in REPLACED:
                    s_out = block_outs[li][0, pos].float()
                    t_out = cached["block_outs"][str(li)].to("cuda").float()
                    cos = F.cosine_similarity(s_out.unsqueeze(0), t_out.unsqueeze(0))
                    s_n = F.normalize(s_out, dim=0)
                    t_n = F.normalize(t_out, dim=0)
                    nmse = ((s_n - t_n) ** 2).mean()
                    l_block = l_block + (1 - cos) + nmse
                l_block = (l_block / len(REPLACED)).mean()
                s_logits = model.lm_head(h[:, pos])[0].float()
                t_logits = cached["logits"].to("cuda").float()
                l_logit = F.kl_div(F.log_softmax(s_logits, dim=-1),
                                   F.softmax(t_logits, dim=-1), reduction="batchmean")
                loss = (LAM_BLOCK * l_block + LAM_LOGIT * l_logit).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                turn_stats["turns"] += 1
                step += 1
                # advance position offset for C-persist (continuous RoPE)
                offset += s_len
                if MODE == "creset":
                    offset = 0
                if step % 300 == 0 or step == N_STEPS:
                    p = save_ckpt(step, final=(step == N_STEPS))
                    print("checkpoint saved: %s (step %d, loss %.4f)" % (p, step, loss.item()), flush=True)
                if step % 10 == 0 or step == N_STEPS:
                    el = time.time() - t0
                    print("step %d/%d: loss=%.4f block=%.4f logit=%.4f (%.0fs)" % (
                        step, N_STEPS, loss.item(), l_block.item(), l_logit.item(), el), flush=True)
                    try:
                        with open(PROGRESS_PATH, "w") as f:
                            json.dump({"step": step, "n_steps": N_STEPS,
                                       "loss": loss.item(), "block_loss": l_block.item(),
                                       "logit_loss": l_logit.item(), "elapsed_s": el,
                                       "last_losses": losses[-50:]}, f)
                    except Exception:
                        pass
                block_outs.clear()

    for h in handles:
        h.remove()

    final_ckpt_path = save_ckpt(step, final=True)
    with open(os.path.join(OUT_DIR, "distill_log.json"), "w") as f:
        json.dump({"mode": MODE, "steps": step, "losses": losses,
                   "final_loss": float(losses[-1]) if losses else None,
                   "mean_last10_loss": float(sum(losses[-10:]) / max(len(losses[-10:]), 1)),
                   "lr": LR, "lam_block": LAM_BLOCK, "lam_logit": LAM_LOGIT,
                   "trainable": "post_norm(12)", "gamma": "frozen_sigmoid_logit",
                   "turn_stats": turn_stats, "ckpt_path": final_ckpt_path}, f, indent=2)
    print("\nsaved to %s" % OUT_DIR, flush=True)
    print("final loss %.4f, last10 mean %.4f" % (
        float(losses[-1]) if losses else None,
        float(sum(losses[-10:]) / max(len(losses[-10:]), 1))), flush=True)
    print("P0I %s OK" % MODE, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
