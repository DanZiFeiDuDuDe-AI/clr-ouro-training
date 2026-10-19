# -*- coding: utf-8 -*-
"""How often does the swap donor carry the SAME running relation anyway?
If high, B~A is partially explained by label coincidence rather than
content-unused. Replicates the swap test's eval set and donor mapping."""
import sys, random, json
sys.path.insert(0, "D:/LoopLM")
from clutrr_data import generate_dataset

random.seed(42)
_train = generate_dataset(600)
trajs = generate_dataset(150)
n = len(trajs)

def donor_idx(i, ti):
    for off in range(7, 7 + n):
        c = (i + off) % n
        if c != i and ti < len(trajs[c]["turns"]):
            return c
    return None

match_prev, total = 0, 0          # running relation BEFORE current turn
match_ans = 0                     # would donor's own answer equal ours?
for i, tr in enumerate(trajs):
    for ti in range(1, len(tr["turns"])):     # t2p turns only
        d = donor_idx(i, ti)
        if d is None:
            continue
        total += 1
        if trajs[d]["answers"][ti - 1] == tr["answers"][ti - 1]:
            match_prev += 1
        if ti < len(trajs[d]["answers"]) and trajs[d]["answers"][ti] == tr["answers"][ti]:
            match_ans += 1

out = {
    "n_t2p_turns": total,
    "donor_same_running_relation": round(match_prev / total, 4),
    "donor_same_final_answer": round(match_ans / total, 4),
}
print(json.dumps(out, indent=2))
json.dump(out, open("D:/LoopLM/checkpoints/clr_ouro_v1_s42/donor_coincidence.json", "w"),
          indent=2)
