"""
Verify the CLUTRR generator against the independent FamilyTree solver.

For every trajectory:
  - every turn's label must be the UNIQUE relation the genealogy implies
  - names are distinct within a trajectory
  - answer gender matches the newest person's gender
  - turn t>=2 prompts never mention persons from turns 1..t-2 (no history leak)
Across the corpus:
  - every atomic relation must admit >=2 distinct answers on turns >=2
    (no deterministic prompt->answer shortcut)
  - prints the memoryless Bayes ceiling: max accuracy achievable from the
    current prompt alone; the training delta must be measured above it
"""
import random
from collections import Counter

from clutrr_data import (
    generate_trajectory, generate_dataset, bayes_ceiling, FamilyTree,
    rel_gender, COMP_TABLE, RELATIONS,
)


def check_table_gender_invariant():
    # answer gender must equal the newest person's gender = gender of atomic rel
    for (a, b), c in COMP_TABLE.items():
        assert rel_gender(c) == rel_gender(b), f"gender broken: ({a},{b})->{c}"
    print(f"COMP_TABLE: {len(COMP_TABLE)} entries, gender invariant OK")


def check_trajectories(n=3000, seed=123):
    rng = random.Random(seed)
    hop_counts = Counter()
    label_counts = Counter()
    answers_by_atomic = {}

    for i in range(n):
        n_hops = rng.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
        traj = generate_trajectory(n_hops, rng)
        persons, genders = traj["persons"], traj["genders"]
        chain, answers, turns = traj["chain"], traj["answers"], traj["turns"]
        hop_counts[n_hops] += 1

        assert len(set(persons)) == len(persons), f"[{i}] duplicate names {persons}"

        ft = FamilyTree()
        ft.add_person(persons[0], genders[0])
        for t in range(n_hops):
            ft.add_person(persons[t + 1], genders[t + 1])
            ft.add_fact(persons[t + 1], chain[t], persons[t])
            derived = ft.relation(persons[t + 1], persons[0])
            assert derived == [answers[t]], (
                f"[{i}] turn {t+1}: label={answers[t]!r} but solver says "
                f"{derived} (chain={chain})")
            assert rel_gender(answers[t]) == genders[t + 1], f"[{i}] gender mismatch"

            label_counts[answers[t]] += 1
            if t >= 1:
                answers_by_atomic.setdefault(chain[t], set()).add(answers[t])
                prompt = turns[t]["prompt"]
                for j in range(1, t):
                    assert persons[j] not in prompt, (
                        f"[{i}] history leak: {persons[j]} in turn-{t+1} prompt")

    print(f"{n} trajectories verified against FamilyTree solver")
    print(f"hops: {dict(sorted(hop_counts.items()))}")
    print("labels (all turns):")
    for r in RELATIONS:
        print(f"  {r:14s} {label_counts[r]:5d}")
    print("answers per atomic on turns >=2 (each needs >=2):")
    for a, ans in sorted(answers_by_atomic.items()):
        print(f"  {a:10s} -> {sorted(ans)}")
        assert len(ans) >= 2, f"deterministic shortcut: atomic {a} -> {ans}"


def check_ceilings(seed=7):
    rng = random.Random(seed)
    train_like = generate_dataset(3000, (1, 2, 3), (0.2, 0.5, 0.3), rng)
    eval_like = generate_dataset(3000, (2, 3), (0.6, 0.4), rng)
    print(f"memoryless Bayes ceiling, turns>=2 "
          f"(train mix): {bayes_ceiling(train_like):.3f}  "
          f"(eval mix): {bayes_ceiling(eval_like):.3f}")


if __name__ == "__main__":
    check_table_gender_invariant()
    check_trajectories()
    check_ceilings()
    print("ALL CHECKS PASSED")
