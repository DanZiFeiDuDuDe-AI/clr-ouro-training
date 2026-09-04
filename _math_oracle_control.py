#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capability-floor control for the math-chain null result.

Question: can the BASE Ouro-1.4B (4-bit, no PLAA) do the task's arithmetic
when ALL information is restated in the prompt (no memory needed)?
  "Fact: the number is 5.\nFact: add 3 to the number.\nQuestion: ..."

If the base model fails this oracle version, the math-chain null says nothing
about the state channel — the task exceeds backbone capability. If it passes,
the null implicates the channel.

Scored two ways per item (one batched forward, 20 candidates):
digits with and without leading space.
"""
import torch
import os, sys, json, random, math

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, "D:/LoopLM")

from clr_ouro import load_ouro

N = 150
OUT = "D:/LoopLM/checkpoints/math_oracle_control.json"

Z = 1.959963984540054
def wilson(k, n):
    p = k/n; d = 1 + Z*Z/n
    c = (p + Z*Z/(2*n))/d
    h = (Z/d)*math.sqrt(p*(1-p)/n + Z*Z/(4*n*n))
    return (round(max(0.0, c-h), 4), round(min(1.0, c+h), 4))

def main():
    rng = random.Random(20260711)
    items = []
    for _ in range(N):
        v = rng.randint(0, 9)
        opts = [d for d in (-3, -2, -1, 1, 2, 3) if 0 <= v + d <= 9]
        d = rng.choice(opts)
        op = f"add {d} to the number" if d > 0 else f"subtract {-d} from the number"
        prompt = (f"Fact: the number is {v}.\nFact: {op}.\n"
                  f"Question: What is the number now?\nAnswer:")
        items.append((prompt, str(v + d)))

    model, tok = load_ouro()   # base model, NO FiLM injection
    pad_id = tok.pad_token_id
    digits = [str(i) for i in range(10)]
    cands_ns = [tok(s, add_special_tokens=False)["input_ids"] for s in digits]
    cands_sp = [tok(" " + s, add_special_tokens=False)["input_ids"] for s in digits]
    all_cands = cands_ns + cands_sp

    hit_ns = hit_sp = hit_greedy = 0
    with torch.no_grad():
        for prompt, ans in items:
            p_ids = tok(prompt, truncation=True, max_length=128)["input_ids"]
            P = len(p_ids)
            L = P + max(len(c) for c in all_cands)
            n_c = len(all_cands)
            input_ids = torch.full((n_c, L), pad_id, dtype=torch.long)
            attn = torch.zeros((n_c, L), dtype=torch.long)
            for i, c in enumerate(all_cands):
                seq = p_ids + c
                input_ids[i, :len(seq)] = torch.tensor(seq)
                attn[i, :len(seq)] = 1
            base_out, _, _ = model.model(input_ids=input_ids.cuda(),
                                         attention_mask=attn.cuda(),
                                         use_cache=False)
            lp = torch.log_softmax(model.lm_head(base_out.last_hidden_state).float(),
                                   dim=-1)
            scores = []
            for i, c in enumerate(all_cands):
                scores.append(sum(lp[i, P - 1 + j, t].item()
                                  for j, t in enumerate(c)) / len(c))
            best_ns = max(range(10), key=lambda i: scores[i])
            best_sp = max(range(10), key=lambda i: scores[10 + i])
            hit_ns += int(digits[best_ns] == ans)
            hit_sp += int(digits[best_sp] == ans)
            # greedy next token (row 0 is prompt + first cand; position P-1
            # logits depend only on the prompt prefix)
            greedy_tok = int(lp[0, P - 1].argmax())
            hit_greedy += int(tok.decode([greedy_tok]).strip() == ans)
            torch.cuda.empty_cache()

    res = {
        "n": N,
        "oracle_nospace": {"acc": round(hit_ns/N, 4), "w95": wilson(hit_ns, N)},
        "oracle_space": {"acc": round(hit_sp/N, 4), "w95": wilson(hit_sp, N)},
        "oracle_greedy_token": {"acc": round(hit_greedy/N, 4),
                                "w95": wilson(hit_greedy, N)},
        "chance": 0.1,
    }
    print(json.dumps(res, indent=2), flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"saved -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
