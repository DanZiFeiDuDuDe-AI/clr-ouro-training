#!/usr/bin/env python3
"""
Baseline v2 EXPANDED re-evaluation (2026-07-10).

Motivation: the original _run_baseline_v2.py evaluated on only 8 trajectories
(~9 t2p items; all reported accuracies were multiples of 1/9), giving Wilson
CIs as wide as [12%, 65%]. This script re-evaluates the SAME best checkpoint
(step 200, checkpoints/baseline_v2_fixed/best_model.pt) with the SAME protocol
(fast_eval: pre-update state, next-token argmax) on a 150-trajectory eval set.

Safeguards:
  1. Protocol sanity check: reproduce the original 8-trajectory eval first
     (expect t2p with_S=0.3333, without_S=0.0000 up to init-noise jitter).
  2. Verified-weights protocol: print scale_proj norms, abort if zero.
  3. Train-set contamination guard: new eval trajectories whose full prompt
     signature appears in the (replayed) 600-trajectory train set are dropped.

Outputs: H:/CLR/02_results/baseline_v2_rerun_eval.json
         (+ copy to E:/Desktop/CLR_paper_materials/data/)
"""
import torch, torch.nn as nn
import os, sys, json, random, math, shutil

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, "H:/CLR")

from local_train_clutrr_v2 import FiLMWrapper, PlaaCore
from clutrr_data import generate_dataset, bayes_ceiling

WORKSPACE = "H:/CLR"
MODEL_PATH = f"{WORKSPACE}/models/Qwen3.5-0.8B"
CKPT = f"{WORKSPACE}/checkpoints/baseline_v2_fixed/best_model.pt"
DS, DM = 64, 1024
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23]
MAX_TOKENS = 128
N_NEW = 150
NEW_SEED = 20260710
OUT_JSON = f"{WORKSPACE}/02_results/baseline_v2_rerun_eval.json"
PAPER_COPY = "E:/Desktop/CLR_paper_materials/data/baseline_v2_rerun_eval.json"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

Z = 1.959963984540054

def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = (Z / d) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))

def boot_delta(k1, n1, k2, n2, B=20000, seed=20260710):
    rng = random.Random(seed)
    diffs = []
    p1, p2 = k1 / n1, k2 / n2
    for _ in range(B):
        a = sum(1 for _ in range(n1) if rng.random() < p1) / n1
        b = sum(1 for _ in range(n2) if rng.random() < p2) / n2
        diffs.append(a - b)
    diffs.sort()
    return (diffs[int(0.025 * B)], diffs[int(0.975 * B)])

# ── fast_eval clone that also returns raw counts (protocol IDENTICAL to
#    _run_baseline_v2.fast_eval: pre-update state, next-token argmax) ──
@torch.no_grad()
def fast_eval_counts(model, tok, test_data, film_layers, plaa_core, n, ablate_s=False):
    model.eval()
    correct = torch.zeros(4, dtype=torch.long)
    total = torch.zeros(4, dtype=torch.long)
    for traj in test_data[:n]:
        S = plaa_core.init_state(1, "cuda")
        turns = traj["turns"]
        for ti, turn in enumerate(turns):
            inp = tok(turn["prompt"], return_tensors="pt", truncation=True,
                      max_length=MAX_TOKENS).to("cuda")
            s_val = torch.zeros_like(S) if ablate_s else S
            for l in film_layers:
                l._s = s_val
            out = model(**inp, output_hidden_states=True)
            lens = inp["attention_mask"].sum(dim=1) - 1
            logits = out.logits[0, lens[0]]
            hs = out.hidden_states[-1]
            ans_ids = tok(turn["answer"], add_special_tokens=False)
            pred = logits.argmax().item()
            target = ans_ids[0] if isinstance(ans_ids, list) else ans_ids["input_ids"][0]
            ok = int(pred == target)
            h_last = hs[0, lens[0]]
            S = plaa_core(S, h_last.unsqueeze(0))
            total[3] += 1; correct[3] += ok
            if ti == 0:
                total[0] += 1; correct[0] += ok
            else:
                total[1] += 1; correct[1] += ok
            if ti == len(turns) - 1:
                total[2] += 1; correct[2] += ok
            del out, hs, inp, ans_ids
        torch.cuda.empty_cache()
    keys = ["t1", "t2p", "last", "all"]
    return {k: {"correct": int(correct[i]), "total": int(total[i]),
                "acc": correct[i].item() / max(total[i].item(), 1)}
            for i, k in enumerate(keys)}

def main():
    print("=" * 64, flush=True)
    print("Baseline v2 EXPANDED re-eval | ckpt:", CKPT, flush=True)
    print("=" * 64, flush=True)
    torch.manual_seed(42)

    # ── data: replay original stream, then fresh expanded set ──
    random.seed(42)
    train_data = generate_dataset(600)          # replayed only for dedup + stream position
    orig_test = generate_dataset(8 * 3)         # original test set (fast_eval used [:8])
    train_sigs = {tuple(t["prompt"] for t in traj["turns"]) for traj in train_data}

    rng_new = random.Random(NEW_SEED)
    raw_new = generate_dataset(N_NEW, rng=rng_new)
    new_test, dropped = [], 0
    for traj in raw_new:
        sig = tuple(t["prompt"] for t in traj["turns"])
        if sig in train_sigs:
            dropped += 1
        else:
            new_test.append(traj)
    ceiling_new = bayes_ceiling(new_test)
    n_t2p_expected = sum(len(t["turns"]) - 1 for t in new_test)
    print(f"New eval set: {len(new_test)} trajectories ({dropped} dropped as "
          f"train-duplicates) | expected t2p items: {n_t2p_expected} | "
          f"Bayes ceiling (t2p): {ceiling_new:.4f}", flush=True)

    # ── model ──
    print("Loading Qwen3.5-0.8B (4-bit NF4)...", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
        quantization_config=bnb, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else "<|endoftext|>"
    for p in model.parameters():
        p.requires_grad = False

    layers = model.model.layers
    film_layers = []
    for idx in FULL_ATTN_LAYERS:
        wrapper = FiLMWrapper(layers[idx], DM)
        layers[idx] = wrapper
        film_layers.append(wrapper)
    plaa_core = PlaaCore(DM).cuda()

    # ── checkpoint ──
    try:
        ckpt = torch.load(CKPT, map_location="cuda", weights_only=True)
    except Exception:
        ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    plaa_core.load_state_dict(ckpt["plaa_core"])
    for l, sd in zip(film_layers, ckpt["scale_proj"]):
        l.scale_proj.load_state_dict(sd)
    norms = [float(l.scale_proj.weight.float().norm()) for l in film_layers]
    print(f"Checkpoint step: {ckpt.get('step')} | best_delta: {ckpt.get('best_delta')}",
          flush=True)
    print(f"scale_proj norms: {[round(x, 3) for x in norms]}", flush=True)
    assert all(x > 0.1 for x in norms), "ZERO INJECTION WEIGHTS — aborting (checkpoint bug?)"

    # ── 1) protocol sanity check on the ORIGINAL 8 trajectories ──
    print("\n[1/2] Sanity check: original 8-trajectory eval", flush=True)
    sw = fast_eval_counts(model, tok, orig_test, film_layers, plaa_core, 8, False)
    so = fast_eval_counts(model, tok, orig_test, film_layers, plaa_core, 8, True)
    print(f"  with_S  t2p = {sw['t2p']['acc']:.4f} ({sw['t2p']['correct']}/{sw['t2p']['total']})"
          f"  [original: 0.3333]", flush=True)
    print(f"  without_S t2p = {so['t2p']['acc']:.4f} ({so['t2p']['correct']}/{so['t2p']['total']})"
          f"  [original: 0.0000]", flush=True)

    # ── 2) expanded eval ──
    print(f"\n[2/2] Expanded eval: {len(new_test)} trajectories", flush=True)
    ew = fast_eval_counts(model, tok, new_test, film_layers, plaa_core, len(new_test), False)
    eo = fast_eval_counts(model, tok, new_test, film_layers, plaa_core, len(new_test), True)

    kw, nw = ew["t2p"]["correct"], ew["t2p"]["total"]
    ko, no = eo["t2p"]["correct"], eo["t2p"]["total"]
    ci_w = wilson(kw, nw)
    ci_o = wilson(ko, no)
    ci_d = boot_delta(kw, nw, ko, no)
    delta = kw / nw - ko / no

    print("\n" + "=" * 64, flush=True)
    print("RESULTS (t2p, expanded eval)", flush=True)
    print(f"  with_S    : {kw}/{nw} = {kw/nw:.4f}  Wilson95 [{ci_w[0]:.3f}, {ci_w[1]:.3f}]",
          flush=True)
    print(f"  without_S : {ko}/{no} = {ko/no:.4f}  Wilson95 [{ci_o[0]:.3f}, {ci_o[1]:.3f}]",
          flush=True)
    print(f"  delta     : {delta:+.4f}  boot95 [{ci_d[0]:+.3f}, {ci_d[1]:+.3f}]", flush=True)
    print(f"  ceiling   : {ceiling_new:.4f}  (with_S reaches {kw/nw/ceiling_new:.1%})",
          flush=True)
    print(f"  t1: with {ew['t1']['acc']:.3f} / without {eo['t1']['acc']:.3f}", flush=True)

    results = {
        "date": "2026-07-10",
        "script": "_rerun_baseline_v2_eval.py",
        "checkpoint": CKPT,
        "checkpoint_step": int(ckpt.get("step", -1)),
        "scale_proj_norms": norms,
        "protocol": "fast_eval (pre-update S, next-token argmax), identical to _run_baseline_v2.py",
        "new_eval": {
            "seed": NEW_SEED, "n_traj_requested": N_NEW,
            "n_traj_used": len(new_test), "dropped_as_train_dup": dropped,
            "bayes_ceiling_t2p": ceiling_new,
            "with_S": ew, "without_S": eo,
            "t2p_with_wilson95": ci_w, "t2p_without_wilson95": ci_o,
            "t2p_delta": delta, "t2p_delta_boot95": ci_d,
        },
        "sanity_check_original_8traj": {
            "with_S": sw, "without_S": so,
            "original_reported": {"with_S_t2p": 0.3333, "without_S_t2p": 0.0},
            "note": "init_state adds N(0, 0.01) noise per trajectory; exact "
                    "reproduction is not guaranteed, agreement should be close.",
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=2)
    try:
        shutil.copyfile(OUT_JSON, PAPER_COPY)
        print(f"\nSaved: {OUT_JSON}\nCopied to: {PAPER_COPY}", flush=True)
    except Exception as e:
        print(f"\nSaved: {OUT_JSON} (paper copy failed: {e})", flush=True)

if __name__ == "__main__":
    main()
