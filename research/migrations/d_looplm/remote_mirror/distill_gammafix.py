# -*- coding: utf-8 -*-
"""
Phase 0H step2d — distillation with FIXED gamma parameterization.

Technical revision to Amendment 1 (p0h-distill-gamma-vanish):
  gamma_logit stores gamma DIRECTLY; gamma = clamp(theta, 1e-4, 1-1e-4).
No sigmoid saturation, so all 16 heads receive gradient.

Two modes (D_MODE):
  gamma     : train only gamma (96 params). Out: ouro_retention_m2_gammafix
  gammanorm : train gamma + replaced-layer post-norms (24672 params).
              Out: ouro_retention_m2_gammafix_norm

Saves intermediate checkpoints every 300 steps and final best_model.pt.
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

MODE = os.environ.get("D_MODE", "gammanorm")  # "gamma" or "gammanorm"
N_STEPS = int(os.environ.get("D_STEPS", "600"))
LR = float(os.environ.get("D_LR", "3e-4"))
LAM_BLOCK = float(os.environ.get("D_LAM_BLOCK", "1.0"))
LAM_LOGIT = float(os.environ.get("D_LAM_LOGIT", "0.5"))
SMOKE = os.environ.get("D_SMOKE", "") == "1"
OUT_DIR = os.environ.get("D_OUT_DIR",
    "D:/training/checkpoints/ouro_retention_m2_gammafix" if MODE == "gamma"
    else "D:/training/checkpoints/ouro_retention_m2_gammafix_norm")

# argv overrides (safer over SSH than cmd set): mode steps out_dir smoke
if len(sys.argv) > 1:
    MODE = sys.argv[1]
if len(sys.argv) > 2:
    N_STEPS = int(sys.argv[2])
if len(sys.argv) > 3:
    OUT_DIR = sys.argv[3]
if len(sys.argv) > 4:
    SMOKE = sys.argv[4] == "1"
PROGRESS_PATH = os.path.join(OUT_DIR, "progress.json")


def main():
    print("=== Phase 0H step2d: distillation gamma-fixed (%s) ===" % MODE, flush=True)
    print("steps=%d lr=%g lam_block=%g lam_logit=%g smoke=%s" % (
        N_STEPS, LR, LAM_BLOCK, LAM_LOGIT, SMOKE), flush=True)
    print("out dir=%s" % OUT_DIR, flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("loading student (fixed gamma init, backbone exact_copy)...", flush=True)
    model, tok = load_ouro()
    converted = convert_ouro_to_retention(model, REPLACED, gamma_mode="clamped_direct")
    print("converted %d layers with clamped_direct gamma" % len(converted), flush=True)

    # diagnostics: per-head theta/gamma before training
    theta_init = {str(idx): ret.gamma_logit.detach().cpu().clone()
                  for ret, idx in converted}
    gamma_init = {str(idx): ret.gamma.detach().cpu().clone()
                  for ret, idx in converted}
    grad_norm_log = []  # list of {step, layer, grad_norm_per_head}

    params = [ret.gamma_logit for ret, _ in converted]
    post_norm_params = []
    if MODE == "gammanorm":
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

    def save_ckpt(step, final=False):
        ck = {
            "variant": "ouro_retention_m2_gammafix" if MODE == "gamma" else "ouro_retention_m2_gammafix_norm",
            "replaced_layers": REPLACED,
            "retention_gamma": {str(idx): ret.gamma_logit.detach().cpu().clone()
                                for ret, idx in converted},
            "gamma_param_mode": "clamped_direct",
            "distill_steps": step, "lr": LR,
            "lam_block": LAM_BLOCK, "lam_logit": LAM_LOGIT,
            "unfroze_post_norms": POST_NORM_NAMES if MODE == "gammanorm" else [],
            "final_loss": float(losses[-1]) if losses else None,
            "mean_last10_loss": float(sum(losses[-10:]) / max(len(losses[-10:]), 1)),
        }
        if MODE == "gammanorm":
            ck["post_norm_weights"] = {
                str(li): {nm: getattr(model.model.layers[li], nm).weight.detach().cpu().clone()
                          for nm in POST_NORM_NAMES}
                for li in REPLACED}
        path = os.path.join(OUT_DIR, "best_model.pt" if final else "ckpt_step%04d.pt" % step)
        torch.save(ck, path)
        return path

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
                # record per-head gamma gradient norm at key steps (before opt.step)
                if (step + 1) == 1 or (step + 1) % 300 == 0 or (step + 1) == N_STEPS:
                    for ret, idx in converted:
                        g = ret.gamma_logit.grad
                        grad_norm_log.append({
                            "step": step + 1,
                            "layer": str(idx),
                            "grad_norm_per_head": g.detach().cpu().abs().clone().tolist(),
                        })
                opt.step()
                losses.append(float(loss.item()))
                step += 1
                if step % 300 == 0 or step == N_STEPS:
                    p = save_ckpt(step, final=(step == N_STEPS))
                    print("checkpoint saved: %s (step %d, loss %.4f)" % (p, step, loss.item()), flush=True)
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

    # final gamma diagnostics (per-head theta/gamma/boundary/timescale)
    theta_final = {str(idx): ret.gamma_logit.detach().cpu().clone()
                   for ret, idx in converted}
    gamma_final = {str(idx): ret.gamma.detach().cpu().clone()
                   for ret, idx in converted}
    EPS = 1e-6

    def boundary_counts(theta):
        n_lo = int((theta <= EPS).sum().item())
        n_hi = int((theta >= 1.0 - EPS).sum().item())
        return n_lo, n_hi

    def tau_of(gamma):
        g = gamma.clamp(1e-12, 1 - 1e-12)
        return (-1.0 / torch.log(g)).tolist()

    gamma_diag = {
        "mode": MODE,
        "eps": EPS,
        "layers": {},
    }
    for li in REPLACED:
        k = str(li)
        gamma_diag["layers"][k] = {
            "theta_init": theta_init[k].tolist(),
            "theta_final": theta_final[k].tolist(),
            "gamma_init": gamma_init[k].tolist(),
            "gamma_final": gamma_final[k].tolist(),
            "tau_init": tau_of(gamma_init[k]),
            "tau_final": tau_of(gamma_final[k]),
            "boundary_lo_init": boundary_counts(theta_init[k])[0],
            "boundary_hi_init": boundary_counts(theta_init[k])[1],
            "boundary_lo_final": boundary_counts(theta_final[k])[0],
            "boundary_hi_final": boundary_counts(theta_final[k])[1],
        }
    gamma_diag["grad_norm_log"] = grad_norm_log
    with open(os.path.join(OUT_DIR, "gamma_diagnostics.json"), "w") as f:
        json.dump(gamma_diag, f, indent=2)
    print("gamma diagnostics saved to %s" % os.path.join(OUT_DIR, "gamma_diagnostics.json"), flush=True)

    final_ckpt_path = save_ckpt(step, final=True)
    with open(os.path.join(OUT_DIR, "distill_log.json"), "w") as f:
        json.dump({"mode": MODE, "steps": step, "losses": losses,
                   "final_loss": float(losses[-1]) if losses else None,
                   "mean_last10_loss": float(sum(losses[-10:]) / max(len(losses[-10:]), 1)),
                   "lr": LR, "lam_block": LAM_BLOCK,
                   "lam_logit": LAM_LOGIT,
                   "gamma_param_mode": "clamped_direct",
                   "unfroze_post_norms": POST_NORM_NAMES if MODE == "gammanorm" else [],
                   "ckpt_path": final_ckpt_path}, f, indent=2)
    print("\nsaved to %s" % OUT_DIR, flush=True)
    print("final loss %.4f, last10 mean %.4f" % (
        float(losses[-1]) if losses else None,
        float(sum(losses[-10:]) / max(len(losses[-10:]), 1))), flush=True)
    print("PHASE 0H step2d OK", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
