#!/usr/bin/env python3
"""
CLR Baseline — FiLM correct + PlaaCore with read_proj
Qwen3.5-0.8B (4-bit) × GTX 1650, no LoRA, DS=64, 400 steps.
Expected: t2p accuracy ≈ 0.37 (baseline).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys, json, random

os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, "H:/CLR")
from local_train_clutrr_v2 import FiLMWrapper, PlaaCore
from clutrr_data import generate_dataset, bayes_ceiling, RELATIONS

WORKSPACE = "H:/CLR"
MODEL_PATH = f"{WORKSPACE}/models/Qwen3.5-0.8B"
DS, DM = 64, 1024
FULL_ATTN_LAYERS = [3, 7, 11, 15, 19, 23]
MAX_TOKENS = 128
TOTAL_STEPS = 400
EVAL_EVERY = 100
TRAIN_SAMPLES = 600
EVAL_SAMPLES = 8
LR = 3e-4
OUT_DIR = f"{WORKSPACE}/checkpoints/baseline_v2_fixed"
os.makedirs(OUT_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_scheduler

# ── Fast eval (next-token only, no score_candidates) ──
@torch.no_grad()
def fast_eval(model, tok, test_data, film_layers, plaa_core, n=12, ablate_s=False):
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
            # Update S for next turn (same forward pass)
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
    model.train()
    return {
        "t1": correct[0].item() / max(total[0].item(), 1),
        "t2p": correct[1].item() / max(total[1].item(), 1),
        "last": correct[2].item() / max(total[2].item(), 1),
        "all": correct[3].item() / max(total[3].item(), 1),
    }

def main():
    print("=" * 60, flush=True)
    print("CLR Baseline v2 — FiLM fixed + PlaaCore read_proj", flush=True)
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu} ({vram:.1f}GB) | DS={DS} | Steps={TOTAL_STEPS}", flush=True)
    print("=" * 60, flush=True)

    random.seed(42)
    train_data = generate_dataset(TRAIN_SAMPLES)
    test_data = generate_dataset(EVAL_SAMPLES * 3)
    ceiling_train = bayes_ceiling(train_data)
    ceiling_test = bayes_ceiling(test_data)
    print(f"Train: {len(train_data)} trajectories | ceiling={ceiling_train:.3f}", flush=True)
    print(f"Test:  {len(test_data)} trajectories | ceiling={ceiling_test:.3f}", flush=True)

    print(f"\nLoading Qwen3.5-0.8B (4-bit)...", flush=True)
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
        nn.init.normal_(wrapper.scale_proj.weight, mean=0.0, std=0.01)
        layers[idx] = wrapper
        film_layers.append(wrapper)
    print(f"FiLM injected into {len(film_layers)} layers", flush=True)

    plaa_core = PlaaCore(DM).cuda()
    print(f"PlaaCore: read_proj={plaa_core.read_proj.weight.shape}", flush=True)

    trainable = list(plaa_core.parameters())
    for l in film_layers:
        trainable += list(l.scale_proj.parameters())
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)
    scheduler = get_scheduler("cosine", optimizer=opt, num_warmup_steps=20,
                              num_training_steps=TOTAL_STEPS)
    print(f"Trainable params: {sum(p.numel() for p in trainable)/1e3:.1f}K", flush=True)

    log = []
    step = 0
    best_delta = -float("inf")
    print(f"\nTraining: {TOTAL_STEPS} steps...", flush=True)

    while step < TOTAL_STEPS:
        random.shuffle(train_data)
        for traj in train_data:
            if step >= TOTAL_STEPS:
                break
            turns = traj["turns"]
            B = 1
            S = plaa_core.init_state(B, "cuda")
            b_loss = 0.0

            for ti, turn in enumerate(turns):
                prompt = turn["prompt"]
                answer = turn["answer"]
                inp = tok(prompt, return_tensors="pt", truncation=True,
                          max_length=MAX_TOKENS).to("cuda")

                for l in film_layers:
                    l._s = S
                with torch.no_grad():
                    out = model(**inp, output_hidden_states=True)
                    h = out.hidden_states[-1].detach()
                    lens = inp["attention_mask"].sum(dim=1) - 1
                    h_last = h[0, lens[0]]

                S_next = plaa_core(S, h_last.unsqueeze(0))

                for l in film_layers:
                    l._s = S_next
                gen_out = model(**inp, output_hidden_states=False)
                lens = inp["attention_mask"].sum(dim=1) - 1
                logits = gen_out.logits[0, lens[0]]
                ans_ids = tok(answer, add_special_tokens=False)
                target = torch.tensor(
                    ans_ids[0] if isinstance(ans_ids, list) else ans_ids["input_ids"][0],
                    dtype=torch.long, device="cuda"
                )
                gen_loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

                coh = (S_next.float() - S.float()).norm().mean()
                loss = 0.95 * gen_loss + 0.05 * coh
                loss.backward()
                b_loss += loss.item()

                S = S_next.detach()
                del out, h, h_last, gen_out, logits, ans_ids, inp
                torch.cuda.empty_cache()

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            scheduler.step()
            opt.zero_grad()
            step += 1

            if step % 10 == 0:
                print(f"  step {step:4d}/{TOTAL_STEPS} | L={b_loss:.4f} | "
                      f"|S|={S.norm().item():.3f} | lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

            if step % EVAL_EVERY == 0 or step >= TOTAL_STEPS:
                print(f"\n  Eval at step {step}...", flush=True)
                torch.cuda.empty_cache()

                acc_w = fast_eval(model, tok, test_data, film_layers, plaa_core,
                                  n=EVAL_SAMPLES, ablate_s=False)
                torch.cuda.empty_cache()
                acc_wo = fast_eval(model, tok, test_data, film_layers, plaa_core,
                                   n=EVAL_SAMPLES, ablate_s=True)
                torch.cuda.empty_cache()

                d_t2p = acc_w["t2p"] - acc_wo["t2p"]
                log.append({
                    "step": step, "with_S": acc_w, "without_S": acc_wo, "delta_t2p": d_t2p,
                })
                print(f"  t2p: with_S={acc_w['t2p']:.4f} without_S={acc_wo['t2p']:.4f} "
                      f"delta={d_t2p:+.4f}", flush=True)
                print(f"  last: with={acc_w['last']:.4f} without={acc_wo['last']:.4f}", flush=True)

                if d_t2p > best_delta and step > 50:
                    best_delta = d_t2p
                    ckpt = {
                        "step": step, "plaa_core": plaa_core.state_dict(),
                        "scale_proj": [l.scale_proj.state_dict() for l in film_layers],
                        "eval_log": log, "best_delta": best_delta,
                    }
                    torch.save(ckpt, f"{OUT_DIR}/best_model.pt")
                    print(f"  -> saved best (delta_t2p={d_t2p:+.4f})", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("DONE", flush=True)
    print(f"  Final delta_t2p: {log[-1]['delta_t2p']:+.4f}" if log else "", flush=True)
    print(f"  Best delta_t2p:  {best_delta:+.4f}", flush=True)
    print(f"  Test ceiling: {ceiling_test:.3f}", flush=True)
    print(f"  Checkpoints in {OUT_DIR}/", flush=True)

    results = {
        "final_delta": log[-1]["delta_t2p"] if log else None,
        "best_delta": best_delta,
        "eval_log": log,
        "test_ceiling": ceiling_test,
    }
    json.dump(results, open(f"{OUT_DIR}/results.json", "w"), indent=2)
    print(f"Results saved.", flush=True)


if __name__ == "__main__":
    main()
