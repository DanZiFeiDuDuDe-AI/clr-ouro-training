# -*- coding: utf-8 -*-
"""
P0H2 — rank-4 LoRA on 6 converted retention layers (Q/K/V/O).

Start point : norm1800 (ouro_retention_m2_d3/best_model.pt), NOT overwritten.
Trainable   : LoRA A/B (rank 4) on ret.q_proj/k_proj/v_proj/o_proj only.
              gamma frozen, post-norms frozen, backbone frozen.
Objective priority (eval interpretation):
    candidate behavior > NLL/logit fidelity > hidden cosine.
Loss        : logit KL (weight 1.0) + weak block loss (weight 0.3,
              hidden-cosine part de-emphasized). Do NOT optimize logit
              cosine alone.
Data split  : train = teacher_cache (600 traj, same as 0H);
              validation = final N=150 set (seed42 offset 600) — NEVER
              touched during training, used only for periodic monitor.
Monitors    : candidate accuracy / correct mean rank / correct-vs-best-wrong
              margin / logit cosine (vs M0 teacher) / delta-NLL.
Saves       : intermediate ckpt every 300 steps + monitor_history.json.
Outputs     : P0H2_LORA_RESULTS.md at end, then STOP (no 0I).
"""
import os, sys, json, time, math, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention
from clutrr_data import generate_dataset, RELATIONS

REPLACED = [3, 7, 11, 15, 19, 23]
POST_NORM_NAMES = ["input_layernorm_2", "post_attention_layernorm_2"]
MAX_TOKENS = 128
CACHE_DIR = "D:/training/teacher_cache"
BASE_CKPT = "D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt"
OUT_DIR = "D:/training/checkpoints/ouro_retention_m2_h2"
PROGRESS_PATH = os.path.join(OUT_DIR, "progress.json")
MONITOR_PATH = os.path.join(OUT_DIR, "monitor_history.json")

N_STEPS = int(os.environ.get("H2_STEPS", "1500"))
LR = float(os.environ.get("H2_LR", "1e-4"))
RANK = int(os.environ.get("H2_RANK", "4"))
ALPHA = float(os.environ.get("H2_ALPHA", "8"))
LAM_BLOCK = float(os.environ.get("H2_LAM_BLOCK", "0.3"))
LAM_LOGIT = float(os.environ.get("H2_LAM_LOGIT", "1.0"))
MONITOR_EVERY = int(os.environ.get("H2_MONITOR_EVERY", "200"))
N_EVAL = int(os.environ.get("H2_N_EVAL", "150"))
LOGIT_N = int(os.environ.get("H2_LOGIT_N", "20"))   # teacher-fidelity subset


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
    """Wrap ret.q/k/v/o_proj of every converted layer with LoRA. Returns param list."""
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


def load_base(tok, converted):
    """Restore norm1800 gamma + post-norms (base state, all frozen)."""
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


def save_ckpt(step, params, final=False, extra=None):
    ck = {
        "variant": "ouro_retention_m2_h2_lora",
        "base_ckpt": BASE_CKPT,
        "replaced_layers": REPLACED,
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
    }
    if extra:
        ck.update(extra)
    path = os.path.join(OUT_DIR, "best_model.pt" if final else "ckpt_step%04d.pt" % step)
    torch.save(ck, path)
    return path


@torch.no_grad()
def monitor(model, tok, eval_data, cand_ids, pad_id, teacher_model=None,
            teacher_data=None):
    """Candidate metrics on full eval set + teacher fidelity on subset."""
    m = {"step": step, "n_eval": len(eval_data)}
    # --- candidate behavior ---
    ranks, margins = [], []
    for ti, traj in enumerate(eval_data):
        turn = traj["turns"][0]
        prompt_ids = tok(turn["prompt"], truncation=True,
                         max_length=MAX_TOKENS)["input_ids"]
        scores = candidate_scores(model, tok, prompt_ids, cand_ids, pad_id)
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

    # --- teacher fidelity (logit cosine / delta-NLL) on subset ---
    if teacher_model is not None:
        cos_acc, nll_delta = 0.0, 0.0
        nn = min(LOGIT_N, len(teacher_data))
        for ti, traj in enumerate(teacher_data[:nn]):
            turn = traj["turns"][0]
            prompt_ids = tok(turn["prompt"], truncation=True,
                             max_length=MAX_TOKENS)["input_ids"]
            ans_ids = tok(turn["answer"], add_special_tokens=False)["input_ids"]
            P = len(prompt_ids)
            seq = prompt_ids
            inp = torch.tensor([seq]).cuda()
            attn = torch.ones_like(inp)
            with torch.no_grad():
                bo_s, _, _ = model.model(input_ids=inp, attention_mask=attn, use_cache=False)
                bo_t, _, _ = teacher_model.model(input_ids=inp, attention_mask=attn, use_cache=False)
            ls = model.lm_head(bo_s.last_hidden_state[:, P - 1]).float()[0]
            lt = teacher_model.lm_head(bo_t.last_hidden_state[:, P - 1]).float()[0]
            cos_acc += F.cosine_similarity(ls.unsqueeze(0), lt.unsqueeze(0)).item()
            nll_s = -F.log_softmax(ls, dim=-1)[ans_ids[0]].item()
            nll_t = -F.log_softmax(lt, dim=-1)[ans_ids[0]].item()
            nll_delta += (nll_s - nll_t)
        m["logit_cosine"] = cos_acc / nn
        m["delta_nll"] = nll_delta / nn
        m["logit_n"] = nn
    return m


@torch.no_grad()
def candidate_scores(model, tok, prompt_ids, cand_ids, pad_id):
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
        lp = sum(logprobs[i, P - 1 + j, t].item() for j, t in enumerate(c)) / len(c)
        scores.append(lp)
    return scores


def main():
    global model, converted, step
    print("=== P0H2 rank-4 LoRA (base=norm1800) ===", flush=True)
    print("steps=%d lr=%g rank=%d alpha=%g lam_block=%g lam_logit=%g" % (
        N_STEPS, LR, RANK, ALPHA, LAM_BLOCK, LAM_LOGIT), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("loading student base...", flush=True)
    model, tok = load_ouro()
    converted = convert_ouro_to_retention(model, REPLACED, gamma_mode="sigmoid_logit")
    for ret, idx in converted:
        layer = model.model.layers[idx]
        src_attn = layer.self_attn.src_attn
        from retention_ouro import _RetentionAttentionAdapter
        layer.self_attn = _RetentionAttentionAdapter(ret, src_attn)
    freeze_all(model)
    load_base(tok, converted)
    params = attach_lora(converted, RANK, ALPHA)
    opt = torch.optim.Adam(params, lr=LR)

    print("loading teacher (M0) for fidelity monitor...", flush=True)
    teacher_model, _ = load_ouro()
    teacher_model.eval()

    shards = sorted(glob.glob(os.path.join(CACHE_DIR, "traj_*.pt")))
    print("cache shards: %d" % len(shards), flush=True)

    block_outs = {}
    def mk_hook(idx):
        def hook(module, inp, out):
            block_outs[idx] = out[0] if isinstance(out, tuple) else out
        return hook
    handles = [model.model.layers[li].register_forward_hook(mk_hook(li)) for li in REPLACED]

    print("re-generating dataset (train=600 cache-paired, eval=150 held out)...", flush=True)
    torch.manual_seed(42)
    _ = generate_dataset(600)
    corpus = generate_dataset(len(shards))[:len(shards)]
    eval_data = generate_dataset(N_EVAL)[:N_EVAL]
    print("train traj: %d | eval traj: %d" % (len(corpus), len(eval_data)), flush=True)

    cand_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad_id = tok.pad_token_id

    # baseline monitor at step 0
    step = 0
    m0 = monitor(model, tok, eval_data, cand_ids, pad_id,
                 teacher_model=teacher_model, teacher_data=eval_data)
    m0["step"] = 0
    monitor_hist = [m0]
    with open(MONITOR_PATH, "w") as f:
        json.dump(monitor_hist, f, indent=2)
    print("monitor@0: %s" % json.dumps(
        {k: m0[k] for k in ["cand_acc", "mean_rank", "mean_margin", "logit_cosine", "delta_nll"]}), flush=True)

    t0 = time.time()
    losses = []
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
                continue
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
            loss = (LAM_BLOCK * l_block + LAM_LOGIT * l_logit).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            step += 1
            if step % 300 == 0 or step == N_STEPS:
                p = save_ckpt(step, params, final=(step == N_STEPS),
                              extra={"final_loss": float(loss.item()),
                                     "mean_last10": float(sum(losses[-10:]) / max(len(losses[-10:]), 1))})
                print("checkpoint saved: %s (step %d)" % (p, step), flush=True)
            if step % MONITOR_EVERY == 0 or step == N_STEPS:
                mm = monitor(model, tok, eval_data, cand_ids, pad_id,
                             teacher_model=teacher_model, teacher_data=eval_data)
                monitor_hist.append(mm)
                with open(MONITOR_PATH, "w") as f:
                    json.dump(monitor_hist, f, indent=2)
                print("monitor@%d: %s" % (step, json.dumps(
                    {k: mm[k] for k in ["cand_acc", "mean_rank", "mean_margin", "logit_cosine", "delta_nll"]})), flush=True)
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

    final_path = save_ckpt(step, params, final=True,
                           extra={"final_loss": float(losses[-1]) if losses else None,
                                  "mean_last10": float(sum(losses[-10:]) / max(len(losses[-10:]), 1))})
    with open(os.path.join(OUT_DIR, "train_log.json"), "w") as f:
        json.dump({"steps": step, "losses": losses,
                   "final_loss": float(losses[-1]) if losses else None,
                   "lr": LR, "rank": RANK, "alpha": ALPHA,
                   "lam_block": LAM_BLOCK, "lam_logit": LAM_LOGIT,
                   "base_ckpt": BASE_CKPT, "ckpt_path": final_path}, f, indent=2)
    print("\nsaved H2 to %s" % OUT_DIR, flush=True)
    print("monitor history: %s" % MONITOR_PATH, flush=True)
    print("P0H2 OK (pause, no 0I)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
