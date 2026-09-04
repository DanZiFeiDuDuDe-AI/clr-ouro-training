# -*- coding: utf-8 -*-
"""
Multi-turn running-value arithmetic — a LOW memoryless-ceiling memory task.

Motivation (2026-07-10 night): CLUTRR's memoryless ceiling is ~0.57, so a
model can near-saturate it without using cross-turn content (confirmed by the
swap test: content swaps/erasure had zero effect). This task is built so that
P(answer | current turn) is near-uniform: each turn states only a DELTA, never
the running value. Correct answers require genuine cross-turn state.

Design (mirrors clutrr_data.py's API):
  - Turn 1:  "Fact: the number is 7."            answer: "7"   (copyable, = t1)
  - Turn t:  "Fact: add 3 to the number." /
             "Fact: subtract 5 from the number." answer: running value (t2p)
  - Values stay in 0..9  ->  single-token answers, fast_eval works unchanged.
  - bayes_ceiling(): best memoryless accuracy from the atomic op alone
    (~0.15 by construction, vs CLUTRR's ~0.57).

generate_dataset(n) -> [{"turns": [{"prompt","answer","atomic"}], "answers": [...]}]
"""
import random
from collections import Counter

VALUES = [str(i) for i in range(10)]   # candidate space for scoring evals


def generate_trajectory(n_turns, rng=random):
    v = rng.randint(0, 9)
    turns = [{
        "prompt": (f"Fact: the number is {v}.\n"
                   f"Question: What is the number?\nAnswer:"),
        "answer": str(v),
        "atomic": "set",
    }]
    for _ in range(1, n_turns):
        # |delta| <= 3: large deltas leak the answer (e.g. "add 9" forces 0->9),
        # inflating the memoryless ceiling. Small deltas keep it near-uniform.
        opts = [d for d in (-3, -2, -1, 1, 2, 3) if 0 <= v + d <= 9]
        d = rng.choice(opts)
        v += d
        if d > 0:
            fact = f"Fact: add {d} to the number."
            atom = f"+{d}"
        else:
            fact = f"Fact: subtract {-d} from the number."
            atom = f"-{-d}"
        turns.append({
            "prompt": f"{fact}\nQuestion: What is the number now?\nAnswer:",
            "answer": str(v),
            "atomic": atom,
        })
    return {"turns": turns, "answers": [t["answer"] for t in turns]}


def generate_dataset(n, hop_choices=(2, 3, 4), hop_weights=(0.2, 0.5, 0.3),
                     rng=random):
    """n trajectories; hop = total turn count (1 set-turn + hop-1 delta-turns)."""
    if isinstance(hop_choices, int):
        hop_choices = (hop_choices,)
    kw = {"weights": list(hop_weights)} if hop_weights is not None else {}
    return [generate_trajectory(rng.choices(list(hop_choices), **kw)[0], rng)
            for _ in range(n)]


def bayes_ceiling(trajs, min_turn=1):
    """Best accuracy of a memoryless predictor seeing only the current prompt
    (i.e., only the atomic delta), over turns with index >= min_turn."""
    by_atomic = {}
    total = 0
    for traj in trajs:
        for ti, turn in enumerate(traj["turns"]):
            if ti < min_turn:
                continue
            by_atomic.setdefault(turn["atomic"], Counter())[turn["answer"]] += 1
            total += 1
    if total == 0:
        return 0.0
    return sum(max(c.values()) for c in by_atomic.values()) / total


if __name__ == "__main__":
    rng = random.Random(42)
    data = generate_dataset(1000, rng=rng)
    print("sample trajectory:")
    for t in data[0]["turns"]:
        print("  ", repr(t["prompt"].replace("\n", " | ")), "->", t["answer"])
    print(f"bayes ceiling (t2p): {bayes_ceiling(data):.4f}")
    n_t2p = sum(len(t["turns"]) - 1 for t in data)
    print(f"t2p turns per 1000 trajs: {n_t2p}")
