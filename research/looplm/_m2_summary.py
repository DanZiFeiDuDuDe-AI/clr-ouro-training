# -*- coding: utf-8 -*-
"""Three-seed M2 summary: hierarchical bootstrap CI for the mean zero-ablation
effect of CLR on Ouro-1.4B. Deterministic (seed 20260710)."""
import random, json, math

random.seed(20260710)
B = 20000
# (with_correct, n, without_correct, ceiling)
SEEDS = {
    "s42":  (78, 150, 0, 0.5667),
    "s130": (72, 162, 0, 0.5247),
    "s520": (101, 184, 0, 0.5543),
}

def binom(k, n):
    p = k / n
    return sum(1 for _ in range(n) if random.random() < p) / n

keys = list(SEEDS)
deltas = []
for _ in range(B):
    vals = []
    for _ in range(3):
        w, n, wo, _c = SEEDS[keys[random.randrange(3)]]
        vals.append(binom(w, n) - binom(wo, n))
    deltas.append(sum(vals) / 3)
deltas.sort()
mean = sum(w/n - wo/n for w, n, wo, _ in SEEDS.values()) / 3
lo, hi = deltas[int(0.025*B)], deltas[int(0.975*B)]

pooled_w = sum(v[0] for v in SEEDS.values())
pooled_n = sum(v[1] for v in SEEDS.values())
frac_ceiling = [ (w/n)/c for w, n, _wo, c in SEEDS.values() ]

out = {
    "per_seed": {k: {"with": f"{v[0]}/{v[1]}", "t2p": round(v[0]/v[1], 4),
                     "without": f"{v[2]}/{v[1]}", "ceiling": v[3],
                     "frac_of_ceiling": round((v[0]/v[1])/v[3], 3)}
                 for k, v in SEEDS.items()},
    "mean_delta_pp": round(100*mean, 1),
    "hier_boot95_pp": [round(100*lo, 1), round(100*hi, 1)],
    "pooled_with": f"{pooled_w}/{pooled_n}",
    "pooled_without": f"0/{pooled_n}",
    "mean_frac_of_ceiling": round(sum(frac_ceiling)/3, 3),
}
print(json.dumps(out, indent=2))
json.dump(out, open("D:/LoopLM/checkpoints/m2_three_seed_summary.json", "w"), indent=2)
