# -*- coding: utf-8 -*-
"""
Pure-recall multi-turn task — the minimal content-transmission probe.

Rationale (2026-07-11 night): math-chain gave chance-level with_S (11%) while
the ORACLE control proved the backbone can do the arithmetic (95.3%) when the
info is in-prompt. That implicates the state channel — but math-chain requires
store+compute. This task removes computation entirely: turn 1 states a digit,
later turns are irrelevant distractor facts, EVERY turn asks for the digit.

  t1: "Fact: the number is 7."      -> "7"   (copyable)
  t2: "Fact: the color is red."     -> "7"   (pure recall)
  t3: "Fact: the animal is a dog."  -> "7"   (pure recall)

Memoryless ceiling = exactly 0.10 (distractors carry zero information about
the digit). If with_S ~ 0.10: content transmission fails even for verbatim
storage. If high: transmission works; the math failure was store-WHILE-compute.
"""
import random
from collections import Counter

VALUES = [str(i) for i in range(10)]

_DISTRACTORS = [
    ("color", ["red", "blue", "green", "yellow", "white", "black"]),
    ("animal", ["dog", "cat", "bird", "horse", "fish", "sheep"]),
    ("weather", ["sunny", "rainy", "cloudy", "windy", "snowy", "foggy"]),
    ("city", ["Paris", "London", "Tokyo", "Cairo", "Rome", "Oslo"]),
    ("fruit", ["apple", "pear", "grape", "peach", "plum", "mango"]),
]


def generate_trajectory(n_turns, rng=random):
    v = rng.randint(0, 9)
    turns = [{
        "prompt": (f"Fact: the number is {v}.\n"
                   f"Question: What is the number?\nAnswer:"),
        "answer": str(v),
        "atomic": "set",
    }]
    kinds = rng.sample(_DISTRACTORS, k=min(n_turns - 1, len(_DISTRACTORS)))
    for i in range(1, n_turns):
        kind, opts = kinds[(i - 1) % len(kinds)]
        val = rng.choice(opts)
        art = "a " if kind == "animal" else ""
        turns.append({
            "prompt": (f"Fact: the {kind} is {art}{val}.\n"
                       f"Question: What is the number?\nAnswer:"),
            "answer": str(v),
            "atomic": f"distractor_{kind}",
        })
    return {"turns": turns, "answers": [t["answer"] for t in turns]}


def generate_dataset(n, hop_choices=(2, 3, 4), hop_weights=(0.2, 0.5, 0.3),
                     rng=random):
    if isinstance(hop_choices, int):
        hop_choices = (hop_choices,)
    kw = {"weights": list(hop_weights)} if hop_weights is not None else {}
    return [generate_trajectory(rng.choices(list(hop_choices), **kw)[0], rng)
            for _ in range(n)]


def bayes_ceiling(trajs, min_turn=1):
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
    for t in data[0]["turns"]:
        print(" ", repr(t["prompt"].replace("\n", " | ")), "->", t["answer"])
    print(f"bayes ceiling (t2p): {bayes_ceiling(data):.4f}")
