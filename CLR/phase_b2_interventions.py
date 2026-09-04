#!/usr/bin/env python3
"""
Phase B2 - eval-time causal interventions on trained Phase B checkpoints.

Question decided: are the evolved-S gains cross-turn memory, or per-turn
adaptation that a memoryless predictor could match? Motivated by the fact
that all Phase B with_S t2p accuracies (0.36-0.48) sit BELOW their per-seed
memoryless Bayes ceilings (0.50-0.57), so aggregate accuracy cannot
distinguish the two hypotheses.

Conditions (same checkpoint, same eval set):
  full   : standard update-then-predict chain (replicates training-time with_S)
  reset  : each turn starts from a fresh init state, S_pred = GRU(S_init, h_t);
           cross-turn carry severed, per-turn machinery intact
  zeroed : S ablated at injection and prediction (replicates without_S)

Memoryless-argmax analysis on the 'full' predictions:
  fit argmax P(answer | atomic) on the training set (turns>=2); report its
  eval accuracy (~= Bayes ceiling), the model's agreement with it, and the
  model's accuracy on the subset where the argmax is wrong. Pure memoryless
  play scores ~0 on that subset; well above 0 requires history.

Usage (GPU must be free; ~10 min/checkpoint on the GTX 1650):
  python phase_b2_interventions.py
  CLR_CKPTS="qwen2.5-0.5b_evolved_s42:42" CLR_MODEL_PATH=<path> python phase_b2_interventions.py
"""
import os, json, random
from collections import Counter
import torch

import local_train_clutrr_v2 as T  # FiLMWrapper/PlaaCore/score_candidates; main() is guarded
from clutrr_data import RELATIONS, generate_dataset, bayes_ceiling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CKPTS = os.environ.get(
    "CLR_CKPTS",
    "local_clutrr_v2_nolora:42,seed_1103:1103,seed_130:130,seed_520:520")
CKPTS = [(n, int(s)) for n, s in (x.split(":") for x in CKPTS.split(","))]

WORKSPACE = T.WORKSPACE
MODEL_PATH = os.environ.get("CLR_MODEL_PATH", T.MODEL_PATH)
FILM_LAYERS = T.FULL_ATTN_LAYERS


@torch.no_grad()
def run_condition(model, tok, film_layers, plaa_core, trajs, cand_ids, mode):
    """mode in {'full', 'reset', 'zeroed'}; returns (acc dict, per-turn details)."""
    pad_id = tok.pad_token_id
    counts = {"t1": [0, 0], "t2p": [0, 0]}
    details = []
    for traj in trajs:
        S = plaa_core.init_state(1, "cuda")
        for ti, turn in enumerate(traj["turns"]):
            prompt_ids = tok(turn["prompt"], truncation=True,
                             max_length=T.MAX_TOKENS)["input_ids"]
            S_in = plaa_core.init_state(1, "cuda") if mode == "reset" else S
            s_inj = torch.zeros_like(S_in) if mode == "zeroed" else S_in
            for l in film_layers:
                l._s = s_inj
            pin = torch.tensor([prompt_ids], device="cuda")
            hs = model(input_ids=pin, attention_mask=torch.ones_like(pin),
                       output_hidden_states=True).hidden_states[-1]
            S_next = plaa_core(S_in, hs[0, -1].unsqueeze(0))
            s_pred = torch.zeros_like(S_next) if mode == "zeroed" else S_next
            pred = T.score_candidates(model, film_layers, prompt_ids, s_pred,
                                      cand_ids, pad_id)
            ok = int(RELATIONS[pred] == turn["answer"])
            key = "t1" if ti == 0 else "t2p"
            counts[key][0] += ok
            counts[key][1] += 1
            details.append({"ti": ti, "atomic": turn["atomic"],
                            "answer": turn["answer"], "pred": RELATIONS[pred]})
            S = S_next
        torch.cuda.empty_cache()
    return {k: c / t for k, (c, t) in counts.items()}, details


def fit_argmax(train_trajs):
    by_atomic = {}
    glob = Counter()
    for traj in train_trajs:
        for ti, turn in enumerate(traj["turns"]):
            if ti < 1:
                continue
            by_atomic.setdefault(turn["atomic"], Counter())[turn["answer"]] += 1
            glob[turn["answer"]] += 1
    fallback = glob.most_common(1)[0][0]
    return {a: c.most_common(1)[0][0] for a, c in by_atomic.items()}, fallback


def argmax_analysis(details, mode_by_atomic, fallback):
    t2p = [d for d in details if d["ti"] >= 1]
    agree = hits_wrong = n_wrong = argmax_hits = 0
    for d in t2p:
        m = mode_by_atomic.get(d["atomic"], fallback)
        argmax_hits += (m == d["answer"])
        agree += (d["pred"] == m)
        if m != d["answer"]:
            n_wrong += 1
            hits_wrong += (d["pred"] == d["answer"])
    n = len(t2p)
    return {"argmax_acc": argmax_hits / n, "agreement": agree / n,
            "acc_on_argmax_wrong": (hits_wrong / n_wrong) if n_wrong else None,
            "n_argmax_wrong": n_wrong, "n_t2p": n}


def main():
    print("Loading model:", MODEL_PATH, flush=True)
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
    DM = model.config.hidden_size

    layers = model.model.layers
    film_layers = []
    for idx in FILM_LAYERS:
        layers[idx] = T.FiLMWrapper(layers[idx], DM)
        film_layers.append(layers[idx])
    plaa_core = T.PlaaCore(DM).cuda()
    model.eval()

    cand_ids = [tok(" " + r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    summary = {}

    for name, seed in CKPTS:
        ckpt_dir = f"{WORKSPACE}/checkpoints/{name}"
        ck = torch.load(f"{ckpt_dir}/best_model.pt", map_location="cuda",
                        weights_only=False)
        for idx, fl in zip(FILM_LAYERS, film_layers):
            fl.scale_proj.load_state_dict(ck["scale_proj"][idx])
        plaa_core.load_state_dict(ck["plaa_core"])

        # regenerate this seed's data streams; all conditions share ONE eval set
        rng_train = random.Random(seed)
        rng_eval = random.Random(seed + 1000)
        train_data = generate_dataset(600, (1, 2, 3), (0.2, 0.5, 0.3), rng_train)
        eval_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_eval)
        ceil = bayes_ceiling(eval_data, min_turn=1)
        mode_by_atomic, fb = fit_argmax(train_data)

        res = {"ckpt_step": ck["step"], "bayes_ceiling_t2p": ceil}
        for mode in ("full", "reset", "zeroed"):
            acc, det = run_condition(model, tok, film_layers, plaa_core,
                                     eval_data, cand_ids, mode)
            res[mode] = acc
            if mode == "full":
                res["memoryless_argmax"] = argmax_analysis(det, mode_by_atomic, fb)
        summary[name] = res

        a = res["memoryless_argmax"]
        print(f"\n{name} (step {ck['step']}, ceiling {ceil:.3f})", flush=True)
        print(f"  t2p: full={res['full']['t2p']:.3f} reset={res['reset']['t2p']:.3f} "
              f"zeroed={res['zeroed']['t2p']:.3f}", flush=True)
        print(f"  argmax: acc={a['argmax_acc']:.3f} agree={a['agreement']:.3f} "
              f"acc_on_argmax_wrong={a['acc_on_argmax_wrong']} "
              f"(n={a['n_argmax_wrong']}/{a['n_t2p']})", flush=True)
        with open(f"{ckpt_dir}/interventions.json", "w") as f:
            json.dump(res, f, indent=2)

    print("\nDecision rule:", flush=True)
    print("  cross-turn memory : reset << full AND acc_on_argmax_wrong well above 0",
          flush=True)
    print("  memoryless shortcut: reset ~= full AND acc_on_argmax_wrong ~= 0",
          flush=True)
    with open(f"{WORKSPACE}/02_results/phase_b2_interventions.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
