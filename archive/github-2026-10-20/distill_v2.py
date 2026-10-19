# -*- coding: utf-8 -*-
"""
Phase 0H step2b — distillation v2 (§1.4 post-norm unfreeze).

If M2 (gamma-only, 300 steps) is insufficient, unfreeze the 8 post-norms
of the replaced layers (input_layernorm_2 / post_attention_layernorm_2 in
layers {3,7,11,15,19,23}) and re-distill.

Trainable: gamma (96 params) + selected post-norms (6*2*2048 = 24576 params).
The rest of the 4-bit backbone stays frozen (same as teacher, exact_copy).

Saves a new checkpoint with BOTH gamma and post-norm weights so post-distill
eval can restore the exact student state.
"""
import os, sys, json, time, math, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention
from clutrr_data import generate_dataset

REPLACED = [3, 7, 11, 15, 19, 23]
POST_NORM_NAMES = ["input_layernorm_2", "post_attention_layernorm_2"]
MAX_TOKENS = 128
CACHE_DIR = "D:/training/teacher_cache"
M1_CKPT = "D:/training/checkpoints/ouro_retention_m1/best_model.pt"
OUT_DIR = os.environ.get("D_OUT_DIR", "D:/training/checkpoints/ouro_retention_m2_d2")
PROGRESS_PATH = os.path.join(OUT_DIR, "progress.json")

N_STEPS = int(os.environ.get("D_STEPS", "600"))
LR = float(os.environ.get("D_LR", "3e-4"))
LAM_BLOCK = float(os.environ.get("D_LAM_BLOCK", "1.0"))
LAM_LOGIT = float(os.environ.get("D_LAM_LOGIT", "0.5"))
SMOKE = os.environ.get("D_SMOKE", "") == "1"
UNFREEZE_POSTNORM = os.environ.get("D_UNFREEZE_POSTNORM", "1") == "1"


def main():
    print("=== Phase 0H step2b: distillation v2 (post-norm unfreeze) ===", flush=True)
    print("steps=%d lr=%g lam_block=%g lam_logit=%g unfreeze_postnorm=%s smoke=%s" % (
        N_STEPS, LR, LAM_BLOCK, LAM_LOGIT, UNFREEZE_POSTNORM, SMOKE), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("loading student (M1)...", flush=True)
    model, tok = load_ouro()
    converted = convert_ouro_to_retention(model, REPLACED)
    m1 = torch.load(M1_CKPT, map_location="cuda", weights_only=False)
    for ret, idx in converted:
        ret.gamma_logit.data.copy_(m1["retention_gamma"][str(idx)])
    print("loaded M1 gamma", flush=True)

    # trainable: gamma + optionally replaced-layer post-norms
    params = [ret.gamma_logit for ret, _ in converted]
    post_norm_params = []
    if UNFREEZE_POSTNORM:
        for li in REPLACED:
            layer = model.model.layers[li]
            for nm in POST_NORM_NAMES:
                norm = getattr(layer, nm)
                norm.weight.requires_grad = True
                post_norm_params.append(norm.weight)
        params = params + post_norm_params
    print("trainable params: %d (gamma=%d, post_norm=%d)" % (
        sum(p.numel() for p in params),
        sum(ret.gamma_logit.numel() for ret, _ in converted),
        sum(p.numel() for p in post_norm_params)), flush=True)
    opt = torch.optim.Adam(params, lr=LR)

    shards = sorted(glob.glob(os.path.join(CACHE_DIR, "traj_*.pt")))
    if SMOKE:
        shards = shards[:20]
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

    step = 0
    t0 = time.time()
    losses = []
    while step < N_STEPS:
        for shard_path, traj in zip(shards, corpus):
            if step >= N_STEPS:
                break
            shard = torch.load(shard_path, map_location="cuda", weights_only=False)
            for turn_i, turn in enumerate(traj["turns"]):
                if step >= N_STEPS:
                    break
                if turn_i >= len(shard["turns"]):
                    break
                cached = shard["turns"][turn_i]
                inp = tok(turn["prompt"], return_tensors="pt",
                          truncation=True, max_length=MAX_TOKENS).to("cuda")
                pos = cached["last_pos"]
                s_len = int(inp["attention_mask"].sum(dim=1)[0].item())
                if s_len - 1 != pos:
                    if step == 0:
                        print("WARN: student seq %d vs cache pos %d — skipping turn" % (s_len, pos), flush=True)
                    continue
                block_outs.clear()
                base_out, _, _ = model.model(
                    input_ids=inp["input_ids"],
                    attention_mask=inp["attention_mask"], use_cache=False)
                h = base_out.last_hidden_state
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
                step += 1
                if step % 10 == 0 or step == N_STEPS:
                    el = time.time() - t0
                    msg = "step %d/%d: loss=%.4f block=%.4f logit=%.4f (%.0fs)" % (
                        step, N_STEPS, loss.item(), l_block.item(), l_logit.item(), el)
                    print(msg, flush=True)
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

    # save M2-v2 checkpoint (gamma + post-norm weights)
    out_ckpt = {
        "variant": "ouro_retention_m2_d2_postnorm" if UNFREEZE_POSTNORM else "ouro_retention_m2_d2",
        "replaced_layers": REPLACED,
        "retention_gamma": {str(idx): ret.gamma_logit.detach().cpu().clone()
                            for ret, idx in converted},
        "m1_source": M1_CKPT,
        "distill_steps": step, "lr": LR,
        "lam_block": LAM_BLOCK, "lam_logit": LAM_LOGIT,
        "unfroze_post_norms": POST_NORM_NAMES if UNFREEZE_POSTNORM else [],
        "final_loss": float(losses[-1]) if losses else None,
        "mean_last10_loss": float(sum(losses[-10:]) / max(len(losses[-10:]), 1)),
    }
    if UNFREEZE_POSTNORM:
        out_ckpt["post_norm_weights"] = {
            str(li): {nm: getattr(model.model.layers[li], nm).weight.detach().cpu().clone()
                      for nm in POST_NORM_NAMES}
            for li in REPLACED}
    torch.save(out_ckpt, os.path.join(OUT_DIR, "best_model.pt"))
    with open(os.path.join(OUT_DIR, "distill_log.json"), "w") as f:
        json.dump({"steps": step, "losses": losses,
                   "final_loss": out_ckpt["final_loss"],
                   "mean_last10_loss": out_ckpt["mean_last10_loss"],
                   "lr": out_ckpt["lr"], "lam_block": out_ckpt["lam_block"],
                   "lam_logit": out_ckpt["lam_logit"],
                   "unfroze_post_norms": out_ckpt["unfroze_post_norms"]}, f, indent=2)
    print("\nsaved M2-v2 to %s" % OUT_DIR, flush=True)
    print("final loss %.4f, last10 mean %.4f" % (
        out_ckpt["final_loss"], out_ckpt["mean_last10_loss"]), flush=True)
    print("PHASE 0H step2b OK", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
