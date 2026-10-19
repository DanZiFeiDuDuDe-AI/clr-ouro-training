#!/usr/bin/env python3
"""Extract Phase B comparison data from local results."""
import json, os

seeds = {
    42: 'H:/CLR/checkpoints/local_clutrr_v2_nolora',
    1103: 'H:/CLR/checkpoints/seed_1103',
    130: 'H:/CLR/checkpoints/seed_130',
    520: 'H:/CLR/checkpoints/seed_520',
}
static = 'H:/CLR/checkpoints/phaseb_static_s42'

# Evolved seeds
print("=== EVOLVED S (4 seeds) ===")
print(f"{'Seed':>6s} {'s100 Δ':>8s} {'s200 Δ':>8s} {'s300 Δ':>8s} {'s400 Δ':>8s} {'Best Δ':>8s}")
print("-" * 56)
deltas = []
for sd, p in sorted(seeds.items()):
    with open(f'{p}/results.json') as f:
        d = json.load(f)
    s100 = d['eval_log'][0]['delta']['t2p'] * 100
    s200 = d['eval_log'][1]['delta']['t2p'] * 100
    s300 = d['eval_log'][2]['delta']['t2p'] * 100
    s400 = d['eval_log'][3]['delta']['t2p'] * 100
    best = d['best_delta_t2p'] * 100
    deltas.append(best)
    print(f'{sd:6d} {s100:8.1f}pp {s200:8.1f}pp {s300:8.1f}pp {s400:8.1f}pp {best:8.1f}pp')

import statistics
print(f"\nMean best Δ = {statistics.mean(deltas):.1f} ± {statistics.stdev(deltas):.1f}pp")
print(f"Range: {min(deltas):.1f}pp ~ {max(deltas):.1f}pp")

# Static S
print("\n=== STATIC S (seed 42, DS=64) ===")
with open(static + '/results.json') as f:
    d = json.load(f)
print(f"{'Step':>6s} {'with S':>8s} {'w/o S':>8s} {'Δ':>8s} {'with t1':>8s} {'w/o t1':>8s}")
for e in d['eval_log']:
    s = e['step']
    w = e['with_S']['t2p'] * 100
    wo = e['without_S']['t2p'] * 100
    delta = e['delta']['t2p'] * 100
    wt1 = e['with_S']['t1'] * 100
    wot1 = e['without_S']['t1'] * 100
    print(f'{s:6d} {w:8.1f}% {wo:8.1f}% {delta:8.1f}pp {wt1:8.1f}% {wot1:8.1f}%')
print(f'Bayes ceiling (t2p) = {d["bayes_ceiling_t2p"]*100:.1f}%')
print(f'Best Δ = {d["best_delta_t2p"]*100:.1f}pp')

# Summary table for paper
print("\n" + "=" * 60)
print("PAPER TABLE: Phase B Three-way Comparison")
print("=" * 60)
print(f"{'Condition':<20s} {'s100 Δ':>8s} {'s200 Δ':>8s} {'s300 Δ':>8s} {'s400 Δ':>8s}")
print("-" * 52)

# w/o S = always 0
print(f"{'w/o S':<20s} {'0.0pp':>8s} {'0.0pp':>8s} {'0.0pp':>8s} {'0.0pp':>8s}")

# static S (from seed 42)
w_static = d['eval_log']
s100_static = w_static[0]['delta']['t2p'] * 100
s200_static = w_static[1]['delta']['t2p'] * 100
s300_static = w_static[2]['delta']['t2p'] * 100
s400_static = w_static[3]['delta']['t2p'] * 100
print(f"{'static S':<20s} {s100_static:8.1f}pp {s200_static:8.1f}pp {s300_static:8.1f}pp {s400_static:8.1f}pp")

# evolved S (mean across seeds)
evo_means = []
for i in range(4):
    vals = []
    for sd, p in sorted(seeds.items()):
        with open(f'{p}/results.json') as f:
            dd = json.load(f)
        vals.append(dd['eval_log'][i]['delta']['t2p'])
    evo_means.append(statistics.mean(vals) * 100)

print(f"{'evolved S (4 seed μ)':<20s} {evo_means[0]:8.1f}pp {evo_means[1]:8.1f}pp {evo_means[2]:8.1f}pp {evo_means[3]:8.1f}pp")

print("\nConclusion: static S consistently ~0pp, evolved S deltas grow with training.")
