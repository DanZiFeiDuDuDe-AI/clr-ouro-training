# -*- coding: utf-8 -*-
"""P0I M1-persist smoke test (before full rerun).

Validates on the REAL training path (trainable post-norm weights, Adam,
candidate CE) that:
  1. forward state persists across turns (turn2_start == turn1_end, detached value)
  2. injected state carries NO grad_fn (turn-boundary detach, no retain_graph)
  3. detached backward succeeds across multiple turns (no graph reuse crash)
  4. position policy: offset changes outputs (persist RoPE continuity is real)
  5. episode boundary resets all state
"""
import os, sys, json, random
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")
import numpy as np
import torch
import torch.nn.functional as F
from clutrr_data import RELATIONS
import p0i_pilot_train as T

SEED = 42


def seed_all():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def main():
    seed_all()
    model, tok, adapters, train_params = T.build_model("m1")
    opt = torch.optim.Adam(train_params, lr=3e-4)
    cands = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad = tok.pad_token_id
    turns = ["Alice has a sister named Bob. ",
             "Bob has a mother named Carol. ",
             "Carol has a brother named Dave. "]
    answers = ["sister", "mother", "brother"]
    results = {}

    def step(prompt, ci, offset):
        p = tok(prompt, truncation=True, max_length=128)["input_ids"]
        sc = T.candidate_scores(model, tok, p, cands, pad, offset=offset)
        loss = F.cross_entropy(sc.unsqueeze(0), torch.tensor([ci], device=sc.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    # ---- turn 0: reset start ----
    T.clear_all_state(adapters); T.reset_turn_state(adapters)
    l0 = step(turns[0], RELATIONS.index(answers[0]), 0)
    end0 = [getattr(a, "_turn_state", None) for a in adapters]
    results["turn0_backward_ok"] = True
    results["turn0_end_state_present"] = all(e is not None for e in end0)
    print("turn0 ok loss=%.4f" % l0, flush=True)

    # ---- turn 1: persist injection ----
    T.set_persist_state(adapters)
    eq = all(e is not None and
             a._state is not None and
             torch.equal(a._state[0], e[0]) and
             torch.equal(a._state[1], e[1])
             for a, e in zip(adapters, end0))
    results["turn2_start_eq_turn1_end_detached"] = bool(eq)
    det = all(a._state is not None and a._state[0].grad_fn is None and
              a._state[1].grad_fn is None and not a._state[0].requires_grad
              for a in adapters)
    results["injected_state_no_grad_fn"] = bool(det)
    l1 = step(turns[1], RELATIONS.index(answers[1]), len(tok(turns[0])["input_ids"]))
    results["turn1_detached_backward_ok"] = True
    print("turn1 ok loss=%.4f (inject_eq=%s detach=%s)" % (l1, eq, det), flush=True)

    # ---- turn 2: persist again ----
    T.set_persist_state(adapters)
    off2 = len(tok(turns[0])["input_ids"]) + len(tok(turns[1])["input_ids"])
    l2 = step(turns[2], RELATIONS.index(answers[2]), off2)
    results["turn2_detached_backward_ok"] = True
    print("turn2 ok loss=%.4f" % l2, flush=True)

    # ---- position policy: offset must change outputs ----
    with torch.no_grad():
        p1 = tok(turns[1], truncation=True, max_length=128)["input_ids"]
        sc_a = T.candidate_scores(model, tok, p1, cands, pad, offset=0)
        sc_b = T.candidate_scores(model, tok, p1, cands, pad, offset=137)
    diff = (sc_a - sc_b).abs().max().item()
    results["position_policy_offset_changes_output"] = bool(diff > 1e-4)
    print("position policy: offset0 vs offset137 max_diff=%.2e" % diff, flush=True)

    # ---- episode boundary reset ----
    T.clear_all_state(adapters)
    all_none = all(getattr(a, "_state", None) is None and
                   getattr(a, "_turn_state", None) is None and
                   getattr(a, "_last_state", None) is None for a in adapters)
    results["episode_boundary_reset"] = bool(all_none)
    print("episode boundary reset:", all_none, flush=True)

    ok = all(results.values())
    print(json.dumps(results, indent=2), flush=True)
    print("SMOKE %s" % ("PASS" if ok else "FAIL"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
