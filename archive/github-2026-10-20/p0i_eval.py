# -*- coding: utf-8 -*-
"""
P0I evaluation: multi-turn CLUTRR candidate-scoring on C-reset vs C-persist.

- N=150 trajectories, ALL turns (not just turn 0).
- Reasoning mode mirrors training: C-reset (state resets each turn, pos from 0)
  vs C-persist (state persists across turns, continuous RoPE position).
- Scoring: canonical CLUTRR candidate-scoring (14 relations).
  Each turn's candidates are scored from a copy of the turn-start state;
  the main state chain is advanced on the answer sequence (prompt + answer).
- Outputs: Acc point + 95% CI (bootstrap), Δ_internal, state norm/delta/
  cross-turn cosine, per-layer stats, VRAM, latency.
- Writes p0i_results.json + P0I_RESULTS.md + P0I_RESOURCE_REPORT.md.
"""
import os, sys, json, time, math, random
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"

import torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from p0i_loop_retention import (convert_p0i, reset_turn_state,
                                set_persist_state, clear_all_state, REPLACED)
from clutrr_data import generate_dataset, RELATIONS

N_TRAJ = int(os.environ.get("H_N_TRAJ", "150"))
MAX_TOKENS = 128
BOOT_B = 1000
CKPT_CRESET = "D:/training/checkpoints/p0i_creset_s42/best_model.pt"
CKPT_CPERSIST = "D:/training/checkpoints/p0i_cpersist_s42/best_model.pt"
RESULTS_JSON = "D:/training/p0i_results.json"


def load_ckpt_model(tok, ckpt_path, gamma_mode="sigmoid_logit"):
    model, _ = load_ouro()
    adapters, rets = convert_p0i(model, gamma_mode=gamma_mode)
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    for ret in rets:
        ret.gamma_logit.data.copy_(ck["retention_gamma"][str(ret.layer_idx)])
    pnw = ck.get("post_norm_weights")
    if pnw:
        for li in REPLACED:
            layer = model.model.layers[li]
            for nm, w in pnw[str(li)].items():
                getattr(layer, nm).weight.data.copy_(w.to("cuda"))
    return model, adapters


@torch.no_grad()
def forward_seq(model, adapters, tok, seq_ids, offset=0):
    """Forward one sequence with explicit position_ids. Returns (h, pos)."""
    input_ids = torch.tensor(seq_ids, dtype=torch.long, device="cuda").unsqueeze(0)
    attn = torch.ones_like(input_ids)
    s_len = input_ids.shape[1]
    if offset > 0:
        pos_ids = (offset + torch.arange(s_len, device="cuda")).unsqueeze(0)
    else:
        pos_ids = torch.arange(s_len, device="cuda").unsqueeze(0)
    base_out, _, _ = model.model(input_ids=input_ids, attention_mask=attn,
                                 use_cache=False, position_ids=pos_ids)
    h = base_out.last_hidden_state
    return h, s_len - 1


@torch.no_grad()
def score_turn_candidates(model, adapters, tok, prompt_ids, cand_ids, pad_id):
    """Candidate scoring from the CURRENT injected turn-start state (copy
    semantics: state tensors are broadcast, never mutated in place)."""
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
    base_out, _, _ = model.model(input_ids=input_ids, attention_mask=attn,
                                 use_cache=False)
    logprobs = torch.log_softmax(model.lm_head(base_out.last_hidden_state).float(), dim=-1)
    best, best_i = -1e30, 0
    for i, c in enumerate(cand_ids):
        lp = sum(logprobs[i, P - 1 + j, t].item() for j, t in enumerate(c)) / len(c)
        if lp > best:
            best, best_i = lp, i
    return best_i


@torch.no_grad()
def state_stats(adapters, prev_norms=None):
    """Collect per-layer state stats after a turn. Returns dict + prev norms."""
    norms, deltas, cosines = {}, {}, {}
    for a in adapters:
        li = a.layer_idx
        st = getattr(a, "_turn_state", None)
        if st is None:
            norms[li] = 0.0
            deltas[li] = 0.0
            cosines[li] = 1.0
            continue
        S, z = st
        norm = float(S.detach().float().norm().item())
        norms[li] = norm
        if prev_norms is not None and li in prev_norms and prev_norms[li] is not None:
            d = float((S.detach().float() - prev_norms[li]).norm().item())
            cos = float(F.cosine_similarity(
                S.detach().float().reshape(-1).unsqueeze(0),
                prev_norms[li].reshape(-1).unsqueeze(0)).item())
            deltas[li] = d
            cosines[li] = cos
    return norms, deltas, cosines


def bootstrap_ci(traj_results):
    """traj_results: list of (correct, total) per trajectory. Returns (acc, lo, hi)."""
    accs = [c / max(t, 1) for c, t in traj_results]
    acc = sum(accs) / max(len(accs), 1)
    rng = random.Random(42)
    B = 1000
    n = len(traj_results)
    bs = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            c, t = rng.choice(traj_results)
            s += c / max(t, 1)
        bs.append(s / n)
    bs.sort()
    lo, hi = bs[int(0.025 * B)], bs[int(0.975 * B)]
    return acc, lo, hi


@torch.no_grad()
def evaluate_mode_full(model, adapters, tok, trajs, persist: bool):
    cand_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad_id = tok.pad_token_id
    traj_results = []
    per_turn_correct = {t: [0, 0] for t in range(1, 6)}
    state_log = []
    lat_total = 0.0
    n_fwd = 0
    peak_vram = 0
    for traj in trajs:
        clear_all_state(adapters)
        prev_norms = None
        offset = 0
        traj_log = []
        tr_c = 0
        tr_t = 0
        for turn_i, turn in enumerate(traj["turns"]):
            if turn_i == 0:
                reset_turn_state(adapters)
            elif persist:
                set_persist_state(adapters)
            else:
                reset_turn_state(adapters)
            prompt_ids = tok(turn["prompt"], truncation=True,
                             max_length=MAX_TOKENS)["input_ids"]
            t0 = time.time()
            pred = score_turn_candidates(model, adapters, tok, prompt_ids, cand_ids, pad_id)
            lat_total += time.time() - t0
            n_fwd += 1
            is_correct = int(RELATIONS[pred] == turn["answer"])
            tr_c += is_correct
            tr_t += 1
            per_turn_correct[min(turn_i + 1, 5)][0] += is_correct
            per_turn_correct[min(turn_i + 1, 5)][1] += 1
            ans_ids = tok(turn["answer"], add_special_tokens=False)["input_ids"]
            seq = prompt_ids + ans_ids
            _h, _pos = forward_seq(model, adapters, tok, seq, offset=offset)
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated())
            norms, deltas, cosines = state_stats(adapters, prev_norms)
            traj_log.append({"turn": turn_i + 1, "norms": norms,
                             "deltas": deltas, "cosines": cosines})
            prev_norms = {li: (getattr(a, "_turn_state", None)[0].detach().float().cpu()
                               if getattr(a, "_turn_state", None) is not None else None)
                          for li, a in zip(REPLACED, adapters)}
            offset += len(seq)
        traj_results.append((tr_c, tr_t))
        state_log.append(traj_log)
    acc, lo, hi = bootstrap_ci(traj_results)
    return {
        "acc": acc, "ci95": [lo, hi], "correct": sum(c for c, _ in traj_results),
        "total": sum(t for _, t in traj_results),
        "per_turn_correct": {str(k): v for k, v in per_turn_correct.items()},
        "state_log": state_log,
        "latency_s": lat_total / max(n_fwd, 1),
        "peak_vram_mb": peak_vram / 1024 / 1024,
        "n_forwards": n_fwd,
    }


def main():
    print("=== P0I multi-turn evaluation ===", flush=True)
    print("N_TRAJ=%d BOOT=%d" % (N_TRAJ, BOOT_B), flush=True)
    torch.manual_seed(42)
    random.seed(42)
    _train = generate_dataset(600)
    eval_data = generate_dataset(N_TRAJ)[:N_TRAJ]
    print("eval trajectories: %d, turns: %s" % (
        len(eval_data), sorted(set(len(t["turns"]) for t in eval_data))), flush=True)

    teacher, tok = load_ouro()
    teacher_res = evaluate_mode_full(teacher, [], tok, eval_data, persist=False)
    print("teacher: acc=%.3f (%.3f-%.3f)" % (teacher_res["acc"], *teacher_res["ci95"]), flush=True)
    del teacher
    torch.cuda.empty_cache()

    results = {"n_traj": N_TRAJ, "teacher": teacher_res, "rows": {}}
    for tag, ckpt, persist in [("C-reset", CKPT_CRESET, False),
                               ("C-persist", CKPT_CPERSIST, True)]:
        if not os.path.exists(ckpt):
            print("MISSING %s — skipping" % ckpt, flush=True)
            results["rows"][tag] = {"missing": True}
            continue
        print("\n--- %s (%s) ---" % (tag, ckpt), flush=True)
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        model, adapters = load_ckpt_model(tok, ckpt)
        print("loaded in %.0fs" % (time.time() - t1), flush=True)
        r = evaluate_mode_full(model, adapters, tok, eval_data, persist=persist)
        r["elapsed_s"] = time.time() - t1
        results["rows"][tag] = r
        print("acc=%.3f (95%%CI %.3f-%.3f) per_turn=%s vram=%.0fMB lat=%.3fs" % (
            r["acc"], r["ci95"][0], r["ci95"][1],
            {k: "%d/%d" % tuple(v) for k, v in r["per_turn_correct"].items()},
            r["peak_vram_mb"], r["latency_s"]), flush=True)
        del model
        torch.cuda.empty_cache()

    # aggregate state stats across trajectories
    for tag, r in results["rows"].items():
        if "state_log" not in r:
            continue
        norm_means = {li: [] for li in REPLACED}
        cos_means = {li: [] for li in REPLACED}
        delt_means = {li: [] for li in REPLACED}
        for traj_log in r["state_log"]:
            for tl in traj_log:
                for li in REPLACED:
                    norm_means[li].append(tl["norms"].get(li, 0.0))
                    cos_means[li].append(tl["cosines"].get(li, 1.0))
                    delt_means[li].append(tl["deltas"].get(li, 0.0))
        r["state_agg"] = {
            "norm_mean": {str(li): sum(norm_means[li]) / max(len(norm_means[li]), 1) for li in REPLACED},
            "cos_mean": {str(li): sum(cos_means[li]) / max(len(cos_means[li]), 1) for li in REPLACED},
            "delta_mean": {str(li): sum(delt_means[li]) / max(len(delt_means[li]), 1) for li in REPLACED},
        }

    # Δ_internal
    if "C-persist" in results["rows"] and "C-reset" in results["rows"] and \
       not results["rows"]["C-persist"].get("missing") and not results["rows"]["C-reset"].get("missing"):
        results["delta_internal"] = results["rows"]["C-persist"]["acc"] - results["rows"]["C-reset"]["acc"]
    else:
        results["delta_internal"] = None

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote %s" % RESULTS_JSON, flush=True)
    print("P0I EVAL OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
