# -*- coding: utf-8 -*-
"""
Matched evaluation across M0 teacher / M1 / M2-v1 / M2-v2 / M2-v3 checkpoints.

Same 30-trajectory draw for every model (seed 42, after the 600 replay).
Metrics per model:
  - CLUTRR fast_eval (argmax at last token, turn 0)
  - CLUTRR candidate-scoring (bare, no CLR; 14 relation candidates, turn 0)
  - vs teacher on first 5 prompts: hidden cosine, logit cosine,
    top-1 agreement, NLL diff

Priority (user, 2026-08-24): task behavior > logit distribution >
final hidden similarity. hidden cosine is diagnostic only.

Writes D:/training/matched_eval_results.json
"""
import os, sys, json, time, math, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention
from clutrr_data import generate_dataset, RELATIONS

N_TRAJ = int(os.environ.get("H_N_TRAJ", "30"))
MAX_TOKENS = 128
REPLACED = [3, 7, 11, 15, 19, 23]

# (tag, checkpoint path or None for teacher, gamma_mode)
CKPTS = [
    ("M0_teacher", None, None),
    ("M1_migrated", "D:/training/checkpoints/ouro_retention_m1/best_model.pt", "sigmoid_logit"),
    ("M2_d1_gamma300_old_frozen", "D:/training/checkpoints/ouro_retention_m2_d1/best_model.pt", "sigmoid_logit"),
    ("M2_d2_norm600_gamma_frozen", "D:/training/checkpoints/ouro_retention_m2_d2/best_model.pt", "sigmoid_logit"),
    ("M2_d3_norm1800total_gamma_frozen", "D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt", "sigmoid_logit"),
    ("M2_gammafix_r2_gamma600", "D:/training/checkpoints/ouro_retention_m2_gammafix_r2/best_model.pt", "clamped_direct"),
    ("M2_gammafix_r2_norm600", "D:/training/checkpoints/ouro_retention_m2_gammafix_norm_r2/best_model.pt", "clamped_direct"),
]

def last_non_pad_pos(mask):
    return int(mask.sum(dim=1)[0].item()) - 1

@torch.no_grad()
def forward_get_hidden(model, tok, prompt):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to("cuda")
    base_out, hs, gates = model.model(
        input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], use_cache=False)
    h = base_out.last_hidden_state
    pos = last_non_pad_pos(inp["attention_mask"])
    logits = model.lm_head(h[:, pos])[0]
    return h[0, pos].float().cpu(), logits.float().cpu(), inp

@torch.no_grad()
def eval_clutrr_fast(model, tok, trajs):
    correct = 0; total = 0
    for traj in trajs:
        turn = traj["turns"][0]
        h, logits, inp = forward_get_hidden(model, tok, turn["prompt"])
        ans = tok(turn["answer"], add_special_tokens=False)
        tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
        pred = int(logits.argmax().item())
        correct += int(pred == tgt)
        total += 1
    return correct, total

@torch.no_grad()
def score_candidates_bare(model, prompt_ids, cand_ids, pad_id):
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
    base_out, _, _ = model.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logprobs = torch.log_softmax(model.lm_head(base_out.last_hidden_state).float(), dim=-1)
    best, best_i = -1e30, 0
    for i, c in enumerate(cand_ids):
        lp = sum(logprobs[i, P - 1 + j, t].item() for j, t in enumerate(c)) / len(c)
        if lp > best:
            best, best_i = lp, i
    return best_i

@torch.no_grad()
def eval_clutrr_candidates(model, tok, trajs):
    cand_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad_id = tok.pad_token_id
    correct = 0; total = 0
    for traj in trajs:
        turn = traj["turns"][0]
        prompt_ids = tok(turn["prompt"], truncation=True, max_length=MAX_TOKENS)["input_ids"]
        pred = score_candidates_bare(model, prompt_ids, cand_ids, pad_id)
        correct += int(RELATIONS[pred] == turn["answer"])
        total += 1
    return correct, total

def load_student(tok, ckpt_path, gamma_mode="sigmoid_logit"):
    model, _ = load_ouro()
    converted = convert_ouro_to_retention(model, REPLACED, gamma_mode=gamma_mode)
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    for ret, idx in converted:
        ret.gamma_logit.data.copy_(ck["retention_gamma"][str(idx)])
    pnw = ck.get("post_norm_weights")
    restored = 0
    if pnw:
        for li in REPLACED:
            layer = model.model.layers[li]
            for nm, w in pnw[str(li)].items():
                getattr(layer, nm).weight.data.copy_(w.to("cuda"))
                restored += 1
    return model, restored

def main():
    print("=== Matched evaluation: M0 / M1 / M2-v1 / M2-v2 / M2-v3 ===", flush=True)
    print("N_TRAJ = %d" % N_TRAJ, flush=True)
    torch.manual_seed(42)
    random.seed(42)
    _train = generate_dataset(600)
    eval_data = generate_dataset(N_TRAJ)[:N_TRAJ]
    print("eval trajectories: %d" % len(eval_data), flush=True)

    # ---- Teacher ----
    print("\nloading teacher (M0, original Ouro)...", flush=True)
    t0 = time.time()
    teacher, tok = load_ouro()
    print("loaded %.0fs" % (time.time()-t0), flush=True)
    tf_c, tf_t = eval_clutrr_fast(teacher, tok, eval_data)
    tc_c, tc_t = eval_clutrr_candidates(teacher, tok, eval_data)
    teacher_reps = []
    for traj in eval_data[:5]:
        h, lg, _ = forward_get_hidden(teacher, tok, traj["turns"][0]["prompt"])
        teacher_reps.append((h, lg))
    print("teacher fast=%d/%d=%.3f cand=%d/%d=%.3f" % (
        tf_c, tf_t, tf_c/max(tf_t,1), tc_c, tc_t, tc_c/max(tc_t,1)), flush=True)
    del teacher
    torch.cuda.empty_cache()

    results = {
        "n_traj": N_TRAJ,
        "teacher": {
            "fast_acc": [tf_c, tf_t],
            "cand_acc": [tc_c, tc_t],
        },
        "rows": [],
    }

    for tag, path, gamma_mode in CKPTS:
        if path is None:
            continue
        print("\n--- %s ---" % tag, flush=True)
        print("checkpoint: %s  gamma_mode=%s" % (path, gamma_mode), flush=True)
        if not os.path.exists(path):
            print("MISSING, skipping", flush=True)
            results["rows"].append({"tag": tag, "missing": True})
            continue
        t1 = time.time()
        model, restored = load_student(tok, path, gamma_mode=gamma_mode)
        print("loaded student %.0fs, restored %d post-norm weights" % (time.time()-t1, restored), flush=True)
        sc_fast, st_fast = eval_clutrr_fast(model, tok, eval_data)
        sc_cand, st_cand = eval_clutrr_candidates(model, tok, eval_data)
        hid_cos = []; logit_cos = []; top1_agree = 0; nll_diff = []
        for i, traj in enumerate(eval_data[:5]):
            hs, lgs, _ = forward_get_hidden(model, tok, traj["turns"][0]["prompt"])
            ht, lgt = teacher_reps[i]
            hid_cos.append(float(F.cosine_similarity(hs.unsqueeze(0), ht.unsqueeze(0)).item()))
            logit_cos.append(float(F.cosine_similarity(lgs.unsqueeze(0), lgt.unsqueeze(0)).item()))
            top1_agree += int(lgs.argmax().item() == lgt.argmax().item())
            ans = tok(traj["turns"][0]["answer"], add_special_tokens=False)
            tgt = (ans[0] if isinstance(ans, list) else ans["input_ids"][0])
            nll_s = F.cross_entropy(lgs.unsqueeze(0), torch.tensor([tgt])).item()
            nll_t = F.cross_entropy(lgt.unsqueeze(0), torch.tensor([tgt])).item()
            nll_diff.append(nll_s - nll_t)
        row = {
            "tag": tag,
            "ckpt": path,
            "fast_acc": [sc_fast, st_fast],
            "fast_acc_frac": sc_fast/max(st_fast,1),
            "cand_acc": [sc_cand, st_cand],
            "cand_acc_frac": sc_cand/max(st_cand,1),
            "hidden_cos_mean": sum(hid_cos)/len(hid_cos),
            "logit_cos_mean": sum(logit_cos)/len(logit_cos),
            "top1_agree": [top1_agree, 5],
            "nll_diff_mean": sum(nll_diff)/len(nll_diff),
        }
        results["rows"].append(row)
        print("fast=%d/%d=%.3f cand=%d/%d=%.3f hidden_cos=%.3f logit_cos=%.3f top1=%d/5 NLLdiff=%.3f" % (
            sc_fast, st_fast, sc_fast/max(st_fast,1),
            sc_cand, st_cand, sc_cand/max(st_cand,1),
            row["hidden_cos_mean"], row["logit_cos_mean"], top1_agree, row["nll_diff_mean"]), flush=True)
        del model
        torch.cuda.empty_cache()

    with open("D:/training/matched_eval_results_r2.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote D:/training/matched_eval_results_r2.json", flush=True)
    print("MATCHED EVAL OK", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
