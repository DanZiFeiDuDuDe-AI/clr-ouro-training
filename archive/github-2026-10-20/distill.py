# -*- coding: utf-8 -*-
"""
Phase 0H step 2 — distillation trainer (§18 D1 first).

Trains M1 (retention student) against the offline teacher cache.
Start with D1 block-output matching only (§18 "from minimal to complex"):
  L_block = sum over 6 replaced layers of D(student_block_out, stopgrad(teacher_block_out))
  D = (1 - cosine) + normalized_mse
Trainable: only gamma (96 params) — per P0_SUBSTRATE_DESIGN §9 Amendment.
If gamma alone can't recover, §1.4 allows unfreezing post-norms (future).

Student backbone is frozen 4-bit (same as teacher weights, exact_copy verified).
Only the mixing of 6 layers differs. Distillation asks: can training gamma make
retention's output approximate softmax's output at those 6 layers?

§19 warning: do NOT claim "student learned persistence" — this only restores
teacher function while changing substrate. Persistence is tested in 0I/0J.
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
MAX_TOKENS = 128
CACHE_DIR = "D:/training/teacher_cache"
M1_CKPT = "D:/training/checkpoints/ouro_retention_m1/best_model.pt"
OUT_DIR = "D:/training/checkpoints/ouro_retention_m2_d1"

N_STEPS = int(os.environ.get("D_STEPS", "300"))
LR = float(os.environ.get("D_LR", "3e-4"))
LAM_BLOCK = float(os.environ.get("D_LAM_BLOCK", "1.0"))
LAM_LOGIT = float(os.environ.get("D_LAM_LOGIT", "0.5"))
SMOKE = os.environ.get("D_SMOKE", "") == "1"

def main():
    print("=== Phase 0H step2: distillation (D1 block + D3 logit) ===", flush=True)
    print("steps=%d lr=%g lam_block=%g lam_logit=%g" % (
        N_STEPS, LR, LAM_BLOCK, LAM_LOGIT), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("loading student (M1)...", flush=True)
    model, tok = load_ouro()
    converted = convert_ouro_to_retention(model, REPLACED)
    m1 = torch.load(M1_CKPT, map_location="cuda", weights_only=False)
    for ret, idx in converted:
        ret.gamma_logit.data.copy_(m1["retention_gamma"][str(idx)])
    print("loaded M1 gamma", flush=True)

    # trainable: only gamma_logit
    params = [ret.gamma_logit for ret, _ in converted]
    opt = torch.optim.Adam(params, lr=LR)
    print("trainable params: %d (gamma only)" % sum(p.numel() for p in params), flush=True)

    # load cache shards into memory (small enough)
    shards = sorted(glob.glob(os.path.join(CACHE_DIR, "traj_*.pt")))
    if SMOKE:
        shards = shards[:20]
    print("cache shards: %d" % len(shards), flush=True)

    # hook student's replaced layers' block output during forward
    block_outs = {}
    def mk_hook(idx):
        def hook(module, inp, out):
            block_outs[idx] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [model.model.layers[li].register_forward_hook(mk_hook(li)) for li in REPLACED]

    # Re-generate dataset deterministically to pair prompts with cache shards.
    # Must match offline_teacher_cache.py's RNG stream exactly (seed 42,
    # generate(600) replay, then generate(len(shards))).
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
            # match turns by index
            for turn_i, turn in enumerate(traj["turns"]):
                if step >= N_STEPS:
                    break
                if turn_i >= len(shard["turns"]):
                    break
                cached = shard["turns"][turn_i]
                inp = tok(turn["prompt"], return_tensors="pt",
                          truncation=True, max_length=MAX_TOKENS).to("cuda")
                pos = cached["last_pos"]
                # sanity: student seq length must match teacher cache
                s_len = int(inp["attention_mask"].sum(dim=1)[0].item())
                if s_len - 1 != pos:
                    if step == 0:
                        print("WARN: student seq %d vs cache pos %d (tokenization mismatch) — skipping turn" % (s_len, pos), flush=True)
                    continue
                # student forward (grad on gamma)
                block_outs.clear()
                base_out, _, _ = model.model(
                    input_ids=inp["input_ids"],
                    attention_mask=inp["attention_mask"], use_cache=False)
                h = base_out.last_hidden_state
                # D1: block output matching at replaced layers
                l_block = torch.tensor(0.0, device="cuda")
                for li in REPLACED:
                    s_out = block_outs[li][0, pos].float()  # [2048]
                    t_out = cached["block_outs"][str(li)].to("cuda").float()
                    cos = F.cosine_similarity(s_out.unsqueeze(0), t_out.unsqueeze(0))
                    s_n = F.normalize(s_out, dim=0)
                    t_n = F.normalize(t_out, dim=0)
                    nmse = ((s_n - t_n) ** 2).mean()
                    l_block = l_block + (1 - cos) + nmse
                l_block = (l_block / len(REPLACED)).mean()
                # D3: logit matching (KL teacher || student)
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
                if step % 50 == 0:
                    el = time.time() - t0
                    print("step %d: loss=%.4f block=%.4f logit=%.4f (%.0fs)" % (
                        step, loss.item(), l_block.item(), l_logit.item(), el), flush=True)
                # detach block_outs for next iter
                block_outs.clear()

    for h in handles:
        h.remove()
    # save M2
    out_ckpt = {
        "variant": "ouro_retention_m2_d1",
        "replaced_layers": REPLACED,
        "retention_gamma": {str(idx): ret.gamma_logit.detach().cpu().clone()
                            for ret, idx in converted},
        "m1_source": M1_CKPT,
        "distill_steps": step, "lr": LR,
        "lam_block": LAM_BLOCK, "lam_logit": LAM_LOGIT,
        "final_loss": float(losses[-1]) if losses else None,
        "mean_last10_loss": float(sum(losses[-10:])/max(len(losses[-10:]),1)),
    }
    torch.save(out_ckpt, os.path.join(OUT_DIR, "best_model.pt"))
    with open(os.path.join(OUT_DIR, "distill_log.json"), "w") as f:
        json.dump({"steps": step, "losses": losses[::50],
                  "final_loss": out_ckpt["final_loss"],
                  "mean_last10_loss": out_ckpt["mean_last10_loss"],
                  "lr": out_ckpt["lr"], "lam_block": out_ckpt["lam_block"],
                  "lam_logit": out_ckpt["lam_logit"]}, f, indent=2)
    print("\nsaved M2 to %s" % OUT_DIR, flush=True)
    print("final loss %.4f, last10 mean %.4f" % (
        out_ckpt["final_loss"], out_ckpt["mean_last10_loss"]), flush=True)
    print("PHASE 0H step2 OK", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
