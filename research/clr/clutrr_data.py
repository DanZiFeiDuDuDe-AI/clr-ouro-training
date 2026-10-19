"""
Synthetic CLUTRR-style multi-turn trajectories with verified labels.

Trajectory = person chain p0..pn. Turn t (1-indexed) introduces exactly ONE
atomic fact "p_t is the r_t of p_{t-1}" and asks "What is p_t to p0?".
The answer is the running composition R_t = comp(R_{t-1}, r_t).

Design constraints (fixes for the v1 generator whose labels were noise/copyable):
  - Atomic facts are degree-1 relations only (father/mother/brother/sister/son/daughter).
  - COMP_TABLE semantics: p_{t-1} is A of p0, p_t is b of p_{t-1} => p_t is comp(A,b) of p0.
    (v1 table mixed two orientations; (father,mother)->grandfather etc. were wrong.)
  - Composition is closed over the 14 RELATIONS, never random noise.
  - Turn t>=2 never restates earlier facts, so the answer is unreachable without
    cross-turn memory.
  - Every atomic relation admits >=2 answers at t>=2 depending on the remembered
    running relation (e.g. atomic "father" -> "father" if the state is a sibling,
    "grandfather" if it is a parent/uncle state), so P(answer | current prompt)
    is never deterministic. bayes_ceiling() computes the exact best memoryless
    accuracy; anything above it requires cross-turn memory.
  - Names are sampled without replacement within a trajectory.
  - FamilyTree is an independent solver (explicit genealogy + first-principles
    kinship definitions) used by test_clutrr_gen.py to verify every label.
"""
import random
from collections import Counter

RELATIONS = [
    "father", "mother", "brother", "sister",
    "uncle", "aunt", "grandfather", "grandmother",
    "son", "daughter", "nephew", "niece",
    "grandson", "granddaughter",
]
REL2ID = {r: i for i, r in enumerate(RELATIONS)}

MALE_RELS = {"father", "brother", "uncle", "grandfather", "son", "nephew", "grandson"}

def rel_gender(r):
    return "M" if r in MALE_RELS else "F"

# comp(A, b): if p_prev is the A of p0 and p_new is the b of p_prev,
# then p_new is the comp(A, b) of p0. Only unambiguous compositions included.
COMP_TABLE = {
    ("father", "father"): "grandfather",
    ("father", "mother"): "grandmother",
    ("mother", "father"): "grandfather",
    ("mother", "mother"): "grandmother",
    ("father", "brother"): "uncle",
    ("father", "sister"): "aunt",
    ("mother", "brother"): "uncle",
    ("mother", "sister"): "aunt",
    ("brother", "father"): "father",
    ("brother", "mother"): "mother",
    ("sister", "father"): "father",
    ("sister", "mother"): "mother",
    ("brother", "son"): "nephew",
    ("brother", "daughter"): "niece",
    ("sister", "son"): "nephew",
    ("sister", "daughter"): "niece",
    ("son", "brother"): "son",
    ("son", "sister"): "daughter",
    ("daughter", "brother"): "son",
    ("daughter", "sister"): "daughter",
    ("uncle", "father"): "grandfather",
    ("uncle", "mother"): "grandmother",
    ("aunt", "father"): "grandfather",
    ("aunt", "mother"): "grandmother",
    ("nephew", "brother"): "nephew",
    ("nephew", "sister"): "niece",
    ("niece", "brother"): "nephew",
    ("niece", "sister"): "niece",
    ("son", "son"): "grandson",
    ("son", "daughter"): "granddaughter",
    ("daughter", "son"): "grandson",
    ("daughter", "daughter"): "granddaughter",
    ("grandson", "brother"): "grandson",
    ("grandson", "sister"): "granddaughter",
    ("granddaughter", "brother"): "grandson",
    ("granddaughter", "sister"): "granddaughter",
}

# Atomic per-turn facts: degree-1 relations only.
ATOMIC_RELATIONS = ["father", "mother", "brother", "sister", "son", "daughter"]

# Valid continuations of a running relation A.
_CONT = {}
for (_a, _b), _c in COMP_TABLE.items():
    _CONT.setdefault(_a, []).append(_b)

def _has_cont(rel):
    return bool(_CONT.get(rel))

NAMES_M = ["Alex", "Ben", "Chris", "David", "Ethan", "Frank", "George", "Henry"]
NAMES_F = ["Anna", "Beth", "Claire", "Diana", "Eva", "Fiona", "Grace", "Hannah"]


def generate_trajectory(n_hops, rng=random):
    for _attempt in range(100):
        chain = [rng.choice(ATOMIC_RELATIONS)]
        answers = [chain[0]]
        ok = True
        for t in range(1, n_hops):
            R = answers[-1]
            opts = [b for b in _CONT.get(R, [])
                    if t == n_hops - 1 or _has_cont(COMP_TABLE[(R, b)])]
            if not opts:
                ok = False
                break
            b = rng.choice(opts)
            chain.append(b)
            answers.append(COMP_TABLE[(R, b)])
        if ok:
            break
    else:
        raise RuntimeError(f"no valid {n_hops}-hop chain found")

    genders = [rng.choice("MF")] + [rel_gender(r) for r in chain]
    used = set()
    persons = []
    for g in genders:
        pool = [n for n in (NAMES_M if g == "M" else NAMES_F) if n not in used]
        name = rng.choice(pool)
        used.add(name)
        persons.append(name)

    turns = []
    for t in range(n_hops):
        a, b = persons[t + 1], persons[t]
        prompt = (f"Fact: {a} is the {chain[t]} of {b}.\n"
                  f"Question: What is {a} to {persons[0]}?\nAnswer:")
        turns.append({"prompt": prompt, "answer": answers[t], "atomic": chain[t]})

    return {"turns": turns, "persons": persons, "genders": genders,
            "chain": chain, "answers": answers}


def generate_dataset(n, hop_choices=(1, 2, 3), hop_weights=(0.2, 0.5, 0.3), rng=random):
    """Generate n trajectories with given hop counts."""
    if isinstance(hop_choices, int):
        hop_choices = (hop_choices,)
    if isinstance(hop_weights, (int, float)):
        hop_weights = None
    if hop_weights is not None:
        kw = dict(weights=list(hop_weights))
    else:
        kw = {}
    return [generate_trajectory(
                rng.choices(list(hop_choices), **kw)[0], rng)
            for _ in range(n)]


def bayes_ceiling(trajs, min_turn=1):
    """Best possible accuracy of a memoryless predictor that sees only the
    current prompt, i.e. picks argmax_a P(answer=a | atomic relation) over
    turns with index >= min_turn (0-based). Accuracy above this ceiling
    requires cross-turn memory."""
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
    hits = sum(max(c.values()) for c in by_atomic.values())
    return hits / total


class FamilyTree:
    """Independent kinship solver used only to verify generated labels.

    Builds an explicit genealogy (parent edges; siblings share both parents,
    materialized as hidden persons and unified with named persons when later
    facts identify them) and derives relations from first-principles
    definitions, with no reference to COMP_TABLE.
    """

    def __init__(self):
        self.gender = {}
        self.parents = {}  # person -> {"M": name|None, "F": name|None}
        self._n = 0

    def add_person(self, name, g):
        assert name not in self.gender, f"duplicate person {name}"
        self.gender[name] = g
        self.parents[name] = {"M": None, "F": None}

    def _hidden(self, g):
        self._n += 1
        h = f"_h{self._n}"
        self.add_person(h, g)
        return h

    def _replace(self, old, new):
        for ps in self.parents.values():
            for g in ("M", "F"):
                if ps[g] == old:
                    ps[g] = new
        oldp = self.parents.pop(old)
        newp = self.parents[new]
        for g in ("M", "F"):
            if oldp[g] is not None:
                assert newp[g] is None or newp[g] == oldp[g], \
                    f"conflicting {g}-parent when unifying {old} -> {new}"
                newp[g] = oldp[g]
        del self.gender[old]

    def _set_parent(self, p, c):
        g = self.gender[p]
        cur = self.parents[c][g]
        if cur is None:
            self.parents[c][g] = p
        elif cur == p:
            pass
        elif cur.startswith("_h"):
            self._replace(cur, p)
        elif p.startswith("_h"):
            self._replace(p, cur)
        else:
            raise AssertionError(f"conflicting {g}-parent for {c}: {cur} vs {p}")

    def _ensure_parents(self, x):
        for g in ("M", "F"):
            if self.parents[x][g] is None:
                self.parents[x][g] = self._hidden(g)
        return self.parents[x]["M"], self.parents[x]["F"]

    def add_fact(self, a, rel, b):
        """a is the <rel> of b. Atomic degree-1 relations only."""
        if rel in ("father", "mother"):
            self._set_parent(a, b)
        elif rel in ("son", "daughter"):
            self._set_parent(b, a)
        elif rel in ("brother", "sister"):
            f, m = self._ensure_parents(b)
            self._set_parent(f, a)
            self._set_parent(m, a)
        else:
            raise ValueError(f"non-atomic fact relation: {rel}")

    def _parents_of(self, x):
        return [p for p in self.parents.get(x, {}).values() if p is not None]

    def _siblings(self, x, y):
        return x != y and bool(set(self._parents_of(x)) & set(self._parents_of(y)))

    def relation(self, a, b):
        """All relations of a to b under standard kinship definitions."""
        g = self.gender[a]
        out = []
        pa, pb = self._parents_of(a), self._parents_of(b)
        if a in pb:
            out.append("father" if g == "M" else "mother")
        if b in pa:
            out.append("son" if g == "M" else "daughter")
        if any(a in self._parents_of(p) for p in pb):
            out.append("grandfather" if g == "M" else "grandmother")
        if any(b in self._parents_of(p) for p in pa):
            out.append("grandson" if g == "M" else "granddaughter")
        if self._siblings(a, b):
            out.append("brother" if g == "M" else "sister")
        if any(self._siblings(a, q) for q in pb):
            out.append("uncle" if g == "M" else "aunt")
        if any(self._siblings(s, b) for s in pa):
            out.append("nephew" if g == "M" else "niece")
        return out
