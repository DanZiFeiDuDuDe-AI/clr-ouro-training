# -*- coding: utf-8 -*-
"""
Phase 0H post-distill eval v2: teacher vs distilled student (M2-v2 with
unfrozen post-norms, or any checkpoint containing retention_gamma +
optional post_norm_weights).

Writes p0h_post_distill_eval_v2.json.
"""
import os, sys, json, time, math
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")

import torch
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention
from clutrr_data import generate_dataset

N_TRAJ = int(os.environ.get("H_N_TRAJ", "30"))
MAX_TOKENS = 128
REPLACED = [3, 7, 11, 15, 19, 23]
M2_CKPT = os.environ.get("H_M2_CKPT", "D:/training/checkpoints/ouro_retention_m2_d2/best_model.pt")
OUT_JSON = "D:/training/p0h_post_distill_eval_v2.json"

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
def eval_clutrr(model, tok, trajs):
    correct = 0; total = 0; format_ok = 0
    for traj in trajs:
        turn = traj["turns"][0]
        h, logits, inp = forward_get_hidden(model, tok, turn["prompt"])
        ans = tok(turn["answer"], add_special_tokens=False)
        tgt = ans[0] if isinstance(ans, list) else ans["input_ids"][0]
        pred = int(logits.argmax().item())
        correct += int(pred == tgt)
        total += 1
        format_ok += 1
    return correct, total, format_ok

def main():
    print("=== Phase 0H post-distill eval v2 (teacher vs M2-v2 student) ===", flush=True)
    print("N_TRAJ = %d" % N_TRAJ, flush=True)
    print("M2 ckpt = %s" % M2_CKPT, flush=True)
    torch.manual_seed(42)
    _train = generate_dataset(600)
    eval_data = generate_dataset(N_TRAJ)[:N_TRAJ]

    print("\nloading teacher (original Ouro)...", flush=True)
    t0 = time.time()
    teacher, tok = load_ouro()
    print("loaded %.0fs" % (time.time()-t0), flush=True)
    print("evaluating teacher...", flush=True)
    tc, tt, tf = eval_clutrr(teacher, tok, eval_data)
    teacher_reps = []
    for traj in eval_data[:5]:
        h, lg, _ = forward_get_hidden(teacher, tok, traj["turns"][0]["prompt"])
        teacher_reps.append((h, lg))
    print("teacher: %d/%d = %.1f%% (format %d/%d)" % (tc, tt, 100*tc/max(tt,1), tf, tt), flush=True)
    del teacher
    torch.cuda.empty_cache()

    print("\nloading student (M2-v2)...", flush=True)
    student, tok = load_ouro()
    converted = convert_ouro_to_retention(student, REPLACED)
    print("converted %d layers" % len(converted), flush=True)
    m2 = torch.load(M2_CKPT, map_location="cuda", weights_only=False)
    for ret, idx in converted:
        ret.gamma_logit.data.copy_(m2["retention_gamma"][str(idx)])
    # restore unfrozen post-norm weights if present
    pnw = m2.get("post_norm_weights")
    if pnw:
        restored_norms = 0
        for li in REPLACED:
            layer = student.model.layers[li]
            for nm, w in pnw[str(li)].items():
                getattr(layer, nm).weight.data.copy_(w.to("cuda"))
                restored_norms += 1
        print("restored %d post-norm weights" % restored_norms, flush=True)
    else:
        print("checkpoint has no post_norm_weights (gamma-only)", flush=True)

    print("evaluating student...", flush=True)
    sc, st, sf = eval_clutrr(student, tok, eval_data)

    hid_cos = []; logit_cos = []; top1_agree = 0; nll_diff = []
    for i, traj in enumerate(eval_data[:5]):
        hs, lgs, _ = forward_get_hidden(student, tok, traj["turns"][0]["prompt"])
        ht, lgt = teacher_reps[i]
        hc = torch.nn.functional.cosine_similarity(hs.unsqueeze(0), ht.unsqueeze(0)).item()
        lc = torch.nn.functional.cosine_similarity(lgs.unsqueeze(0), lgt.unsqueeze(0)).item()
        hid_cos.append(hc); logit_cos.append(lc)
        top1_agree += int(lgs.argmax().item() == lgt.argmax().item())
        ans = tok(traj["turns"][0]["answer"], add_special_tokens=False)
        tgt = (ans[0] if isinstance(ans, list) else ans["input_ids"][0])
        nll_s = torch.nn.functional.cross_entropy(lgs.unsqueeze(0), torch.tensor([tgt])).item()
        nll_t = torch.nn.functional.cross_entropy(lgt.unsqueeze(0), torch.tensor([tgt])).item()
        nll_diff.append(nll_s - nll_t)

    print("\n=== RESULTS ===", flush=True)
    print("teacher CLUTRR: %d/%d = %.1f%%" % (tc, tt, 100*tc/max(tt,1)), flush=True)
    print("student M2-v2 CLUTRR: %d/%d = %.1f%%  (format %d/%d)" % (sc, st, 100*sc/max(st,1), sf, st), flush=True)
    print("acc loss: %.1fpp" % (100*(tc-sc)/max(tt,1)), flush=True)
    print("hidden cosine (sim/student): mean %.3f (min %.3f)" % (sum(hid_cos)/len(hid_cos), min(hid_cos)), flush=True)
    print("logit  cosine: mean %.3f (min %.3f)" % (sum(logit_cos)/len(logit_cos), min(logit_cos)), flush=True)
    print("top-1 agreement: %d/5" % top1_agree, flush=True)
    print("NLL diff (student-teacher): mean %.3f (positive = student worse)" % (sum(nll_diff)/len(nll_diff)), flush=True)

    result = {
        "n_traj": N_TRAJ, "replaced_layers": REPLACED,
        "m2_ckpt": M2_CKPT,
        "teacher_acc": [tc, tt], "student_acc": [sc, st],
        "acc_loss_pp": 100*(tc-sc)/max(tt,1),
        "hidden_cos_mean": sum(hid_cos)/len(hid_cos),
        "logit_cos_mean": sum(logit_cos)/len(logit_cos),
        "top1_agree": [top1_agree, 5],
        "nll_diff_mean": sum(nll_diff)/len(nll_diff),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote %s" % OUT_JSON, flush=True)
    print("PHASE 0H POST-DISTILL EVAL V2 OK", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
