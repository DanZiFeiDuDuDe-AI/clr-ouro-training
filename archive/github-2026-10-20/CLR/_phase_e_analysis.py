#!/usr/bin/env python3
"""
Phase E: S_t trajectory tracking and probe analysis.
Post-hoc extraction from existing checkpoints + logs.
"""
import json, os, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = "H:/CLR"
CKPT = f"{WORKSPACE}/checkpoints"
OUT = f"{WORKSPACE}/02_results"

# 1. Extract S_t norm from training logs
def extract_s_norms(log_path):
    norms = []
    if not os.path.exists(log_path):
        return norms
    with open(log_path) as f:
        for line in f:
            m = re.search(r'\|\S\|=([\d.]+)', line)
            if m:
                norms.append(float(m.group(1)))
    return norms

def extract_probe_accs(log_path):
    accs = []
    if not os.path.exists(log_path):
        return accs
    with open(log_path) as f:
        for line in f:
            m = re.search(r'probe_acc=(\d+)/(\d+)', line)
            if m:
                accs.append(int(m.group(1)) / max(int(m.group(2)), 1))
    return accs

def extract_steps(log_path):
    steps = []
    if not os.path.exists(log_path):
        return steps
    with open(log_path) as f:
        for line in f:
            m = re.match(r'\s*step\s+(\d+)', line)
            if m:
                steps.append(int(m.group(1)))
    return steps

# Gather tracking data
tracking = {}

runs = [
    ("evolved_s42", f"{CKPT}/local_clutrr_v2_nolora/training.log"),
    ("evolved_s1103", f"{CKPT}/seed_1103/training.log"),
    ("evolved_s130", f"{CKPT}/seed_130/training.log"),
    ("evolved_s520", f"{CKPT}/seed_520/training.log"),
    ("static_s42", f"{CKPT}/phaseb_static_s42/training.log"),
    ("qwen25_evolved_s42", f"{CKPT}/qwen2.5-0.5b_evolved_s42/training.log"),
]

for name, log_path in runs:
    s_norms = extract_s_norms(log_path)
    steps = extract_steps(log_path)
    tracking[name] = {
        "s_norms": s_norms,
        "steps": steps[:len(s_norms)],
        "n_steps": len(steps),
    }
    print(f"  {name}: {len(steps)} steps logged, {len(s_norms)} |S| readings")

# 2. Plot: S_t norm over training
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Evolved seeds |S| comparison
ax = axes[0, 0]
for name in ["evolved_s42", "evolved_s1103", "evolved_s130", "evolved_s520"]:
    d = tracking[name]
    if d["s_norms"]:
        ax.plot(d["steps"][:len(d["s_norms"])], d["s_norms"], label=name.replace("evolved_s", "seed "))
ax.set_xlabel("Training Step")
ax.set_ylabel("|S_t| (norm)")
ax.set_title("S_t Norm — Evolved Seeds (DS=64)")
ax.legend()
ax.grid(True, alpha=0.3)

# (b) Static vs Evolved |S|
ax = axes[0, 1]
for name, color in [("evolved_s42", "blue"), ("static_s42", "red")]:
    d = tracking[name]
    if d["s_norms"]:
        ax.plot(d["steps"][:len(d["s_norms"])], d["s_norms"], label=name, color=color)
ax.set_xlabel("Training Step")
ax.set_ylabel("|S_t| (norm)")
ax.set_title("Static S vs Evolved S (seed 42, DS=64)")
ax.legend()
ax.grid(True, alpha=0.3)

# (c) Qwen3.5 vs Qwen2.5 |S| comparison
ax = axes[1, 0]
for name in ["evolved_s42", "qwen25_evolved_s42"]:
    d = tracking[name]
    if d["s_norms"]:
        label = "Qwen3.5-0.8B" if "evolved_s42" == name else "Qwen2.5-0.5B"
        ax.plot(d["steps"][:len(d["s_norms"])], d["s_norms"], label=label)
ax.set_xlabel("Training Step")
ax.set_ylabel("|S_t| (norm)")
ax.set_title("S_t Norm — Cross Model (seed 42, DS=64)")
ax.legend()
ax.grid(True, alpha=0.3)

# (d) Final S_t norm distribution across seeds
ax = axes[1, 1]
final_norms = {}
for name in ["evolved_s42", "evolved_s1103", "evolved_s130", "evolved_s520", "static_s42"]:
    d = tracking[name]
    if d["s_norms"]:
        final_norms[name.replace("evolved_s", "seed ").replace("static_s", "static ")] = d["s_norms"][-1]
names_list = list(final_norms.keys())
vals_list = list(final_norms.values())
bars = ax.bar(names_list, vals_list, color=["blue"]*4 + ["red"])
ax.set_ylabel("Final |S_t|")
ax.set_title("Final S_t Norm Comparison")
ax.set_xticklabels(names_list, rotation=30)
for bar, val in zip(bars, vals_list):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/phase_e_s_norm_analysis.png", dpi=150)
print(f"\nSaved: {OUT}/phase_e_s_norm_analysis.png")

# 3. Extract eval metrics across seeds (visualize)
fig2, ax = plt.subplots(figsize=(10, 6))

seeds_eval = {}
for name, ckpt_name in [
    ("seed 42", "local_clutrr_v2_nolora"),
    ("seed 1103", "seed_1103"),
    ("seed 130", "seed_130"),
    ("seed 520", "seed_520"),
]:
    rp = f"{CKPT}/{ckpt_name}/results.json"
    if os.path.exists(rp):
        with open(rp) as f:
            d = json.load(f)
        elog = d.get("eval_log", [])
        steps = [e["step"] for e in elog]
        deltas = [e["delta"]["t2p"]*100 for e in elog]
        with_s = [e["with_S"]["t2p"]*100 for e in elog]
        wo_s = [e["without_S"]["t2p"]*100 for e in elog]
        ax.plot(steps, deltas, 'o-', label=f"{name} (Δ)", markersize=6)

# Add static S
rp = f"{CKPT}/phaseb_static_s42/results.json"
if os.path.exists(rp):
    with open(rp) as f:
        d = json.load(f)
    elog = d.get("eval_log", [])
    steps = [e["step"] for e in elog]
    deltas = [e["delta"]["t2p"]*100 for e in elog]
    ax.plot(steps, deltas, 's--', label="static S (Δ)", color='red', markersize=6)

# Add Qwen2.5
rp = f"{CKPT}/qwen2.5-0.5b_evolved_s42/results.json"
if os.path.exists(rp):
    with open(rp) as f:
        d = json.load(f)
    elog = d.get("eval_log", [])
    steps = [e["step"] for e in elog]
    deltas = [e["delta"]["t2p"]*100 for e in elog]
    ax.plot(steps, deltas, 'v--', label="Qwen2.5-0.5B (Δ)", markersize=6, color='green')

ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel("Training Step")
ax.set_ylabel("Δ ACC (with_S - without_S) [pp]")
ax.set_title("Phase B + Cross-Model: Δ ACC over Training")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/phase_b_delta_comparison.png", dpi=150)
print(f"Saved: {OUT}/phase_b_delta_comparison.png")

# 4. Paper-ready summary table
print("\n" + "=" * 60)
print("PAPER TABLE: Phase B — Three-way Ablation")
print("=" * 60)
table = """\\begin{table}[t]
\\centering
\\caption{Three-way ablation: Persistent state alone does not improve multi-hop reasoning; persistent \\emph{evolution} does.}
\\vspace{-2mm}
\\begin{tabular}{lcccc}
\\toprule
Condition & s=100 & s=200 & s=300 & s=400 \\\\
\\midrule
w/o S (baseline) & 0.0pp & 0.0pp & 0.0pp & 0.0pp \\\\
Static S & -1.4pp & 0.0pp & 0.0pp & 0.0pp \\\\
Evolved S (4-seed $\\mu$)"""
# Compute evolved means
import statistics
evolved_means = []
for i in range(4):
    vals = []
    for seed_name, ckpt_name in [
        ("42", "local_clutrr_v2_nolora"), ("1103", "seed_1103"),
        ("130", "seed_130"), ("520", "seed_520")
    ]:
        rp = f"{CKPT}/{ckpt_name}/results.json"
        with open(rp) as f:
            d = json.load(f)
        vals.append(d["eval_log"][i]["delta"]["t2p"] * 100)
    evolved_means.append(statistics.mean(vals))

table += f" & {evolved_means[0]:+.1f}pp & {evolved_means[1]:+.1f}pp & {evolved_means[2]:+.1f}pp & {evolved_means[3]:+.1f}pp \\\\"

# Also cross-model
rp = f"{CKPT}/qwen2.5-0.5b_evolved_s42/results.json"
with open(rp) as f:
    d = json.load(f)
elog = d["eval_log"]
table += "\nQwen2.5-0.5B Evolved"
for e in elog:
    table += f" & {e['delta']['t2p']*100:+.1f}pp"
table += " \\\\"

table += """
\\midrule
Bayes ceiling & \\multicolumn{4}{c}{49.7\\%--56.6\\%} \\\\
\\bottomrule
\\end{tabular}
\\label{tab:phase_b_ablation}
\\end{table}"""

print(table)

# Save
with open(f"{OUT}/paper_table_phase_b.txt", "w") as f:
    f.write(table)
print(f"\nSaved: {OUT}/paper_table_phase_b.txt")

print("\nPhase E analysis complete.")
