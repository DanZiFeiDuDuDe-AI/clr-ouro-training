# -*- coding: utf-8 -*-
"""
P0H2-v2 — rank-4 LoRA on 6 converted retention layers (Q/K/V/O) + ranking loss.

Fixes vs H2-v1:
  * Seed protocol: random.seed(42)+torch.manual_seed(42)+numpy.random.seed(42)
    BEFORE any generate_dataset call, so train-cache pairing AND the monitor
    eval draw are reproducible and P0H5-aligned.
  * Cache v2 (teacher_cache_v2) generated under the SAME seed stream with the
    P0H5 held-out 150 skipped — held-out NEVER enters training.
  * Multi-epoch loop over the 600-traj cache until N_STEPS reached.
  * Ranking loss on real correct candidates (correct-vs-best-wrong hinge by
    default; candidate CE via V2_RANK_MODE=ce). NO teacher candidate ranking
    supervision.
  * Monitor adds hidden_cos (6 replaced layers, student vs M0 teacher).

Objective: L = KL(logit,1.0) + 0.3*block + lam_rank * L_rank
Start     : norm1800 (ouro_retention_m2_d3/best_model.pt), NOT overwritten.
Monitors  : cand_acc > mean_rank/margin > delta_nll/logit_cos > hidden_cos.
Saves     : ckpt every 300 steps + best_model.pt(final) + monitor_history.json
            + train_log.json + progress.json.
Output    : P0H2_V2_RESULTS.md at end (written by coordinator), then STOP.
"""
import os, sys, json, time, math, glob, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import numpy as np
import torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention
from clutrr_data import generate_dataset, RELATIONS

REPLACED = [3, 7, 11, 15, 19, 23]
POST_NORM_NAMES = ["input_layernorm_2", "post_attention_layernorm_2"]
MAX_TOKENS = 128
CACHE_DIR = "D:/training/teacher_cache_v2"
BASE_CKPT = "D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt"
OUT_DIR = "D:/training/checkpoints/ouro_retention_m2_h2v2"
PROGRESS_PATH = os.path.join(OUT_DIR, "progress.json")
MONITOR_PATH = os.path.join(OUT_DIR, "monitor_history.json")
SEED = 42

N_STEPS = int(os.environ.get("H2_STEPS", "1500"))
LR = float(os.environ.get("H2_LR", "1e-4"))
RANK = int(os.environ.get("H2_RANK", "4"))
ALPHA = float(os.environ.get("H2_ALPHA", "8"))
LAM_BLOCK = float(os.environ.get("H2_LAM_BLOCK", "0.3"))
LAM_LOGIT = float(os.environ.get("H2_LAM_LOGIT", "1.0"))
LAM_RANK = float(os.environ.get("V2_LAM_RANK", "0.1"))
RANK_MARGIN = float(os.environ.get("V2_RANK_MARGIN", "1.0"))
RANK_MODE = os.environ.get("V2_RANK_MODE", "hinge")   # hinge | ce
MAX_EPOCHS = int(os.environ.get("V2_MAX_EPOCHS", "3"))
MONITOR_EVERY = int(os.environ.get("H2_MONITOR_EVERY", "200"))
N_EVAL = int(os.environ.get("H2_N_EVAL", "150"))
LOGIT_N = int(os.environ.get("H2_LOGIT_N", "20"))


class LoRALinear(torch.nn.Module):
    """Frozen original Linear + trainable rank-r update (A@B)."""
    def __init__(self, orig, r, alpha):
        super().__init__()
        self.orig = orig
        self.r, self.alpha = r, alpha
        in_f, out_f = orig.in_features, orig.out_features
        _a = torch.empty(in_f, r)
        torch.nn.init.kaiming_uniform_(_a, a=math.sqrt(5))
        self.A = torch.nn.Parameter(_a.to(torch.bfloat16))
        self.B = torch.nn.Parameter(torch.zeros(r, out_f, dtype=torch.bfloat16))
        self.orig.requires_grad_(False)

    def forward(self, x):
        return self.orig(x) + (x @ self.A @ self.B) * (self.alpha / self.r)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def attach_lora(converted, r, alpha):
    params = []
    n_lora = 0
    for ret, idx in converted:
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            orig = getattr(ret, proj_name)
            lora = LoRALinear(orig, r, alpha)
            lora = lora.to(orig.weight.device)
            setattr(ret, proj_name, lora)
            params += [lora.A, lora.B]
            n_lora += 1
    print("attached LoRA: %d projections, rank=%d, alpha=%g, trainable=%d" % (
        n_lora, r, alpha, sum(p.numel() for p in params)), flush=True)
    return params


def load_base(model, tok, converted):
    ck = torch.load(BASE_CKPT, map_location="cuda", weights_only=False)
    for ret, idx in converted:
        ret.gamma_logit.data.copy_(ck["retention_gamma"][str(idx)])
    pnw = ck.get("post_norm_weights")
    if pnw:
        for li in REPLACED:
            layer = model.model.layers[li]
            for nm, w in pnw[str(li)].items():
                getattr(layer, nm).weight.data.copy_(w.to("cuda"))
    print("base norm1800 loaded: %s" % BASE_CKPT, flush=True)


def save_ckpt(step, params, converted, model, final=False, extra=None):
    ck = {
        "variant": "ouro_retention_m2_h2v2_lora",
        "base_ckpt": BASE_CKPT,
        "replaced_layers": REPLACED,
        "seed": SEED,
        "retention_gamma": {str(idx): ret.gamma_logit.detach().cpu().clone()
                            for ret, idx in converted},
        "post_norm_weights": {
            str(li): {nm: getattr(model.model.layers[li], nm).weight.detach().cpu().clone()
                      for nm in POST_NORM_NAMES}
            for li in REPLACED},
        "lora": {
            str(idx): {
                proj: {
                    "A": getattr(ret, proj).A.detach().cpu().clone(),
                    "B": getattr(ret, proj).B.detach().cpu().clone(),
                    "r": RANK, "alpha": ALPHA,
                } for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
            } for ret, idx in converted},
        "step": step, "lr": LR, "rank": RANK, "alpha": ALPHA,
        "lam_block": LAM_BLOCK, "lam_logit": LAM_LOGIT,
        "lam_rank": LAM_RANK, "rank_mode": RANK_MODE, "rank_margin": RANK_MARGIN,
    }
    if extra:
        ck.update(extra)
    path = os.path.join(OUT_DIR, "best_model.pt" if final else "ckpt_step%04d.pt" % step)
    torch.save(ck, path)
    return path


def candidate_scores(model, tok, prompt_ids, cand_ids, pad_id):
    """Differentiable [n_cand] mean-logprob scores over candidates."""
    n = len(cand_ids)
    P = len(prompt_ids)
    L = P + max(len(c) for c in cand_ids)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for i, c in enumerate(cand_ids):
        seq = prompt_ids + c
        input_ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
    input_ids, attn = input_ids.cuda(), attn.cuda()
    bo, _, _ = model.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logprobs = torch.log_softmax(model.lm_head(bo.last_hidden_state).float(), dim=-1)
    scores = []
    for i, c in enumerate(cand_ids):
        cc = torch.tensor(c, dtype=torch.long, device=input_ids.device)
        pos = torch.arange(len(c), dtype=torch.long, device=input_ids.device) + (P - 1)
        lp = logprobs[i, pos, cc].mean()
        scores.append(lp)
    return torch.stack(scores)


@torch.no_grad()
def candidate_scores_eval(model, tok, prompt_ids, cand_ids, pad_id):
    """Non-grad scores (same math, returns python list)."""
    return candidate_scores(model, tok, prompt_ids, cand_ids, pad_id).tolist()


@torch.no_grad()
def monitor(model, tok, eval_data, cand_ids, pad_id, teacher_model=None,
            teacher_data=None, step=None):
    m = {"step": step, "n_eval": len(eval_data)}
    ranks, margins = [], []
    for ti, traj in enumerate(eval_data):
        turn = traj["turns"][0]
        prompt_ids = tok(turn["prompt"], truncation=True,
                         max_length=MAX_TOKENS)["input_ids"]
        scores = candidate_scores_eval(model, tok, prompt_ids, cand_ids, pad_id)
        correct_i = RELATIONS.index(turn["answer"])
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank = order.index(correct_i) + 1
        wrong = [s for i, s in enumerate(scores) if i != correct_i]
        ranks.append(rank)
        margins.append(scores[correct_i] - max(wrong))
    m["cand_acc"] = sum(1 for r in ranks if r == 1)
    m["mean_rank"] = sum(ranks) / len(ranks)
    m["mean_margin"] = sum(margins) / len(margins)
    m["rank_gt3"] = sum(1 for r in ranks if r > 3)

    if teacher_model is not None:
        cos_acc, nll_delta = 0.0, 0.0
        hidden_cos_acc = 0.0
        nn = min(LOGIT_N, len(teacher_data))
        # hooks: student + teacher replaced layers
        s_block, t_block = {}, {}
        def mk_hook(d, li):
            def hook(module, inp, out):
                d[li] = out[0] if isinstance(out, tuple) else out
            return hook
        s_handles = []
        t_handles = []
        for li in REPLACED:
            s_handles.append(model.model.layers[li].register_forward_hook(mk_hook(s_block, li)))
            t_handles.append(teacher_model.model.layers[li].register_forward_hook(mk_hook(t_block, li)))
        for ti, traj in enumerate(teacher_data[:nn]):
            turn = traj["turns"][0]
            prompt_ids = tok(turn["prompt"], truncation=True,
                             max_length=MAX_TOKENS)["input_ids"]
            ans_ids = tok(turn["answer"], add_special_tokens=False)["input_ids"]
            P = len(prompt_ids)
            inp = torch.tensor([prompt_ids]).cuda()
            attn = torch.ones_like(inp)
            s_block.clear(); t_block.clear()
            bo_s, _, _ = model.model(input_ids=inp, attention_mask=attn, use_cache=False)
            bo_t, _, _ = teacher_model.model(input_ids=inp, attention_mask=attn, use_cache=False)
            ls = model.lm_head(bo_s.last_hidden_state[:, P - 1]).float()[0]
            lt = teacher_model.lm_head(bo_t.last_hidden_state[:, P - 1]).float()[0]
            cos_acc += F.cosine_similarity(ls.unsqueeze(0), lt.unsqueeze(0)).item()
            nll_s = -F.log_softmax(ls, dim=-1)[ans_ids[0]].item()
            nll_t = -F.log_softmax(lt, dim=-1)[ans_ids[0]].item()
            nll_delta += (nll_s - nll_t)
            hc = 0.0
            for li in REPLACED:
                if li in s_block and li in t_block:
                    hc += F.cosine_similarity(
                        s_block[li][0, P - 1].float().unsqueeze(0),
                        t_block[li][0, P - 1].float().unsqueeze(0)).item()
            hidden_cos_acc += hc / len(REPLACED)
        for h in s_handles + t_handles:
            h.remove()
        m["logit_cosine"] = cos_acc / nn
        m["delta_nll"] = nll_delta / nn
        m["hidden_cos"] = hidden_cos_acc / nn
        m["logit_n"] = nn
    return m


def main():
    global model, converted, step
    print("=== P0H2-v2 rank-4 LoRA + ranking loss (base=norm1800) ===", flush=True)
    print("steps=%d lr=%g rank=%d alpha=%g lam_block=%g lam_logit=%g lam_rank=%g mode=%s margin=%g" % (
        N_STEPS, LR, RANK, ALPHA, LAM_BLOCK, LAM_LOGIT, LAM_RANK, RANK_MODE, RANK_MARGIN), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- fixed seed (CRITICAL: before any generate_dataset / LoRA init) ---
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading student base...", flush=True)
    model, tok = load_ouro()
    converted = convert_ouro_to_retention(model, REPLACED, gamma_mode="sigmoid_logit")
    for ret, idx in converted:
        layer = model.model.layers[idx]
        src_attn = layer.self_attn.src_attn
        from retention_ouro import _RetentionAttentionAdapter
        layer.self_attn = _RetentionAttentionAdapter(ret, src_attn)
    freeze_all(model)
    load_base(model, tok, converted)
    params = attach_lora(converted, RANK, ALPHA)
    opt = torch.optim.Adam(params, lr=LR)

    print("loading teacher (M0) for fidelity monitor...", flush=True)
    teacher_model, _ = load_ouro()
    teacher_model.eval()

    shards = sorted(glob.glob(os.path.join(CACHE_DIR, "traj_*.pt")))
    print("cache shards: %d" % len(shards), flush=True)
    if len(shards) == 0:
        print("ERROR: empty cache dir %s — run p0h2_v2_cache.py first" % CACHE_DIR, flush=True)
        return 2

    block_outs = {}
    def mk_hook(idx):
        def hook(module, inp, out):
            block_outs[idx] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [model.model.layers[li].register_forward_hook(mk_hook(li)) for li in REPLACED]

    # --- dataset draw, EXACT same seed stream as cache generation ---
    print("dataset draw (P0H5-aligned, held-out skipped in cache)...", flush=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _ = generate_dataset(600)                # replay parity
    _ = generate_dataset(150)[:150]          # P0H5 held-out placeholder (skip)
    corpus = generate_dataset(len(shards))[:len(shards)]
    # eval draw: FRESH seed stream -> exact P0H5 protocol (600 -> 150[:150]).
    # This is a DIFFERENT segment of the RNG stream than corpus (corpus sits
    # after the 150-placeholder), so eval never overlaps training data.
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _ = generate_dataset(600)                # replay parity
    eval_data = generate_dataset(N_EVAL)[:N_EVAL]
    print("train traj: %d | eval traj: %d" % (len(corpus), len(eval_data)), flush=True)

    cand_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad_id = tok.pad_token_id

    step = 0
    m0 = monitor(model, tok, eval_data, cand_ids, pad_id,
                 teacher_model=teacher_model, teacher_data=eval_data, step=0)
    monitor_hist = [m0]
    with open(MONITOR_PATH, "w") as f:
        json.dump(monitor_hist, f, indent=2)
    print("monitor@0: %s" % json.dumps(
        {k: m0[k] for k in ["cand_acc", "mean_rank", "mean_margin", "logit_cosine", "delta_nll", "hidden_cos"]}), flush=True)

    t0 = time.time()
    losses = []
    matched = 0
    skipped = 0
    epoch = 0
    while step < N_STEPS and epoch < MAX_EPOCHS:
        epoch += 1
        print("epoch %d/%d (step %d)" % (epoch, MAX_EPOCHS, step), flush=True)
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
                    skipped += 1
                    continue
                matched += 1
                block_outs.clear()
                base_out, _, _ = model.model(
                    input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], use_cache=False)
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
                # --- ranking loss on real correct candidate ---
                block_outs.clear()  # keep block_outs for l_block; avoid candidate fwd pollution
                prompt_ids = tok(turn["prompt"], truncation=True,
                                 max_length=MAX_TOKENS)["input_ids"]
                scores = candidate_scores(model, tok, prompt_ids, cand_ids, pad_id)
                correct_i = RELATIONS.index(turn["answer"])
                if RANK_MODE == "ce":
                    l_rank = F.cross_entropy(scores.unsqueeze(0),
                                             torch.tensor([correct_i], device=scores.device))
                else:  # hinge: correct - best_wrong
                    wrong = torch.cat([scores[:correct_i], scores[correct_i + 1:]])
                    l_rank = torch.clamp(RANK_MARGIN - (scores[correct_i] - wrong.max()), min=0.0)
                loss = (LAM_LOGIT * l_logit + LAM_BLOCK * l_block + LAM_RANK * l_rank).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                step += 1
                if step % 300 == 0 or step == N_STEPS:
                    p = save_ckpt(step, params, converted, model, final=(step == N_STEPS),
                                  extra={"final_loss": float(loss.item()),
                                         "mean_last10": float(sum(losses[-10:]) / max(len(losses[-10:]), 1))})
                    print("checkpoint saved: %s (step %d)" % (p, step), flush=True)
                if step % MONITOR_EVERY == 0 or step == N_STEPS:
                    mm = monitor(model, tok, eval_data, cand_ids, pad_id,
                                 teacher_model=teacher_model, teacher_data=eval_data, step=step)
                    monitor_hist.append(mm)
                    with open(MONITOR_PATH, "w") as f:
                        json.dump(monitor_hist, f, indent=2)
                    print("monitor@%d: %s" % (step, json.dumps(
                        {k: mm[k] for k in ["cand_acc", "mean_rank", "mean_margin", "logit_cosine", "delta_nll", "hidden_cos"]})), flush=True)
                if step % 10 == 0 or step == N_STEPS:
                    el = time.time() - t0
                    print("step %d/%d: loss=%.4f block=%.4f logit=%.4f rank=%.4f (%.0fs)" % (
                        step, N_STEPS, loss.item(), l_block.item(), l_logit.item(), l_rank.item(), el), flush=True)
                    try:
                        with open(PROGRESS_PATH, "w") as f:
                            json.dump({"step": step, "n_steps": N_STEPS,
                                       "loss": loss.item(), "block_loss": l_block.item(),
                                       "logit_loss": l_logit.item(), "rank_loss": l_rank.item(),
                                       "elapsed_s": el, "epoch": epoch,
                                       "matched": matched, "skipped": skipped,
                                       "last_losses": losses[-50:]}, f)
                    except Exception:
                        pass
                block_outs.clear()

    for h in handles:
        h.remove()

    if step < N_STEPS:
        print("WARNING: reached %d/%d steps (matched=%d skipped=%d epochs=%d)" % (
            step, N_STEPS, matched, skipped, epoch), flush=True)

    final_path = save_ckpt(step, params, converted, model, final=True,
                           extra={"final_loss": float(losses[-1]) if losses else None,
                                  "mean_last10": float(sum(losses[-10:]) / max(len(losses[-10:]), 1)),
                                  "matched": matched, "skipped": skipped, "epochs": epoch})
    with open(os.path.join(OUT_DIR, "train_log.json"), "w") as f:
        json.dump({"steps": step, "losses": losses,
                   "final_loss": float(losses[-1]) if losses else None,
                   "lr": LR, "rank": RANK, "alpha": ALPHA,
                   "lam_block": LAM_BLOCK, "lam_logit": LAM_LOGIT,
                   "lam_rank": LAM_RANK, "rank_mode": RANK_MODE, "rank_margin": RANK_MARGIN,
                   "seed": SEED, "matched": matched, "skipped": skipped, "epochs": epoch,
                   "base_ckpt": BASE_CKPT, "ckpt_path": final_path}, f, indent=2)
    print("\nsaved H2-v2 to %s" % OUT_DIR, flush=True)
    print("monitor history: %s" % MONITOR_PATH, flush=True)
    print("P0H2-V2 OK (pause, no 0I)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
