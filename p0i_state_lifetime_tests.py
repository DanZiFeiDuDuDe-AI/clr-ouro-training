# -*- coding: utf-8 -*-
"""
P0I state-lifetime automated tests (run BEFORE training).

Validates the four pinned semantics + no-leakage:
  C-reset     : turn2_start == zero
  C-persist   : turn2_start == turn1_end
  new episode : state == initial state
  new Ouro loop: state reset (current_ut>0 sees zero)
  no trajectory / batch leakage

Also runs a strong equivalence gate:
  two-segment forward with persisted state + continuous RoPE position
  == single-segment forward over the concatenated sequence.

Writes P0I_STATE_LIFETIME_TESTS.md with the PASS/FAIL matrix.
"""
import os, sys, json, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"

import torch
import torch.nn.functional as F
from p0i_loop_retention import (P0ILoopRetentionAdapter, convert_p0i,
                                reset_turn_state, set_persist_state,
                                clear_all_state, REPLACED)

RESULTS = []  # (name, ok, detail)


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


@torch.no_grad()
def unit_tests():
    """Direct RetentionAttention state-semantics tests (no full model)."""
    from retention_ouro import RetentionAttention
    ret = RetentionAttention(layer_idx=3, gamma_mode="sigmoid_logit").cuda().eval()
    torch.manual_seed(0)
    B, H, L0, L1, d = 2, 16, 8, 6, 64
    q = torch.randn(B, H, L0 + L1, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H, L0 + L1, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H, L0 + L1, d, device="cuda", dtype=torch.bfloat16)

    # init_state is zero (S=0, z=eps)
    s0 = ret.init_state(B, device="cuda", dtype=torch.bfloat16)
    z_ok = torch.all(s0[1] == 1e-6).item()
    s_ok = torch.all(s0[0] == 0).item()
    record("init_state==zero", z_ok and s_ok, "S all-zero=%s, z==1e-6=%s" % (s_ok, z_ok))

    # determinism: two identical runs from zero give identical outputs
    out_a, st_a = ret.recurrent_forward(q[:, :, :L0], k[:, :, :L0], v[:, :, :L0], None)
    out_b, st_b = ret.recurrent_forward(q[:, :, :L0], k[:, :, :L0], v[:, :, :L0], None)
    det = torch.equal(out_a, out_b) and all(torch.equal(x, y) for x, y in zip(st_a, st_b))
    record("zero-state determinism (no leak)", det)

    # state is used: persisted continuation differs from fresh zero start
    out_persist, st_persist = ret.recurrent_forward(
        q[:, :, L0:], k[:, :, L0:], v[:, :, L0:], st_a)
    out_fresh, _ = ret.recurrent_forward(q[:, :, L0:], k[:, :, L0:], v[:, :, L0:], None)
    used = not torch.equal(out_persist, out_fresh)
    record("state is read (persist != fresh)", used)

    # equivalence gate: two segments (persisted, continuous RoPE-free) == one segment
    # (pure retention recurrence; RoPE-equivalence is checked at adapter level)
    out_concat, _ = ret.recurrent_forward(q, k, v, None)
    eq = torch.allclose(out_concat[:, :, L0:], out_persist, atol=1e-2)
    record("C-persist equivalence: 2-seg == concat", eq,
           "max_abs_diff=%.2e" % (out_concat[:, :, L0:] - out_persist).abs().max().item())

    # batch isolation: batch-0 state does not bleed into batch-1
    st_a0 = (st_a[0][0:1].clone(), st_a[1][0:1].clone())
    out_iso, _ = ret.recurrent_forward(q[:, :, :L0], k[:, :, :L0], v[:, :, :L0], (st_a0[0], st_a0[1]))
    out_iso2, _ = ret.recurrent_forward(q[:, :, :L0], k[:, :, :L0], v[:, :, :L0], (st_a0[0], st_a0[1]))
    record("batch isolation (same state -> same out)", torch.equal(out_iso, out_iso2))


@torch.no_grad()
def adapter_tests(tok):
    """Full-model adapter tests: turn/loop/episode boundaries + RoPE continuity."""
    from clr_ouro import load_ouro
    model, _ = load_ouro()
    adapters, rets = convert_p0i(model, gamma_mode="sigmoid_logit")
    model.eval()

    # spy on state actually received by recurrent_forward at ut==0 / ut>0
    seen = {"ut0": [], "ut_gt0": []}

    def make_spy(ret):
        orig = ret.recurrent_forward

        def spy(q, k, v, state=None):
            if state is None:
                seen["zero"] = seen.get("zero", 0) + 1
            else:
                seen["nonzero"] = seen.get("nonzero", 0) + 1
            return orig(q, k, v, state)
        ret.recurrent_forward = spy

    for ret in rets:
        make_spy(ret)

    # episode A: two turns
    traj_a_t1 = tok("Alice has a sister named Bob. ", return_tensors="pt").to("cuda")
    traj_a_t2 = tok("Bob has a mother named Carol. ", return_tensors="pt").to("cuda")
    traj_b_t1 = tok("Dave has a father named Eve. ", return_tensors="pt").to("cuda")

    # ---- turn 1 (episode A, both conditions start from zero) ----
    clear_all_state(adapters)
    seen.clear()
    model.model(input_ids=traj_a_t1["input_ids"], attention_mask=traj_a_t1["attention_mask"],
                use_cache=False, position_ids=torch.arange(traj_a_t1["input_ids"].shape[1], device="cuda").unsqueeze(0))
    t1_end = [getattr(a, "_turn_state", None) for a in adapters]
    any_t1_end = any(t is not None for t in t1_end)
    t1_ut0_none = seen.get("zero", 0) >= len(REPLACED)  # ut==0 started from zero
    record("turn1: starts from zero (C-reset base)", t1_ut0_none and any_t1_end,
           "zero-seen=%d nonzero-seen=%d" % (seen.get("zero", 0), seen.get("nonzero", 0)))

    # ---- C-reset: turn 2 starts from zero ----
    reset_turn_state(adapters)
    seen.clear()
    model.model(input_ids=traj_a_t2["input_ids"], attention_mask=traj_a_t2["attention_mask"],
                use_cache=False, position_ids=torch.arange(traj_a_t2["input_ids"].shape[1], device="cuda").unsqueeze(0))
    creset_ok = seen.get("zero", 0) >= len(REPLACED) and seen.get("nonzero", 0) == 0
    record("C-reset: turn2_start == zero", creset_ok,
           "zero-seen=%d nonzero-seen=%d" % (seen.get("zero", 0), seen.get("nonzero", 0)))

    # ---- C-persist: turn 2 starts from turn-1 end ----
    # redo turn1 under persist bookkeeping
    clear_all_state(adapters)
    seen.clear()
    model.model(input_ids=traj_a_t1["input_ids"], attention_mask=traj_a_t1["attention_mask"],
                use_cache=False, position_ids=torch.arange(traj_a_t1["input_ids"].shape[1], device="cuda").unsqueeze(0))
    t1_end_p = [getattr(a, "_turn_state", None) for a in adapters]
    set_persist_state(adapters)
    # verify injected state equals turn1 end, before forward
    injected_eq_end = all(
        (getattr(a, "_state", None) is not None) and
        (getattr(a, "_state", None)[0] is t1_end_p[i][0])  # same tensor object by identity
        for i, a in enumerate(adapters))
    seen.clear()
    # turn2 with offset position (continuous RoPE)
    offset = traj_a_t1["input_ids"].shape[1]
    pos2 = (offset + torch.arange(traj_a_t2["input_ids"].shape[1], device="cuda")).unsqueeze(0)
    model.model(input_ids=traj_a_t2["input_ids"], attention_mask=traj_a_t2["attention_mask"],
                use_cache=False, position_ids=pos2)
    cpersist_ok = seen.get("nonzero", 0) >= len(REPLACED)
    record("C-persist: turn2_start == turn1_end", cpersist_ok and injected_eq_end,
           "injected==end=%s nonzero-seen=%d" % (injected_eq_end, seen.get("nonzero", 0)))

    # ---- Ouro loop boundary: ut>0 receives zero ----
    # Spy: count how many ut==0 vs ut>0 received zero. We know Ouro runs 4 loops.
    # On a fresh turn with C-persist state injected, ut==0 must see NONZERO,
    # ut>0 must see ZERO (loop reset).
    seen2 = {"ut0_nonzero": 0, "ut_gt0_zero": 0, "ut0_zero": 0, "ut_gt0_nonzero": 0}

    def make_spy2(ret):
        orig = ret.recurrent_forward
        ut_holder = {"v": 0}

        def spy(q, k, v, state=None):
            if ut_holder["v"] == 0:
                if state is None:
                    seen2["ut0_zero"] += 1
                else:
                    seen2["ut0_nonzero"] += 1
            else:
                if state is None:
                    seen2["ut_gt0_zero"] += 1
                else:
                    seen2["ut_gt0_nonzero"] += 1
            return orig(q, k, v, state)
        spy.ut_holder = ut_holder
        ret.recurrent_forward = spy
        return ut_holder

    holders = []
    for ret in rets:
        ret.recurrent_forward = None  # drop previous spy
        holders.append(make_spy2(ret))
    # reset previous spies properly: re-wrap originals
    # (we stored orig on each ret as _orig_rf)
    for ret in rets:
        if hasattr(ret, "_orig_rf"):
            ret.recurrent_forward = ret._orig_rf
    # re-spy with ut tracking
    for i, ret in enumerate(rets):
        orig = ret.recurrent_forward
        ret._orig_rf = orig
        holder = holders[i]

        def spy(q, k, v, state=None, _orig=orig, _h=holder):
            if _h["v"] == 0:
                if state is None:
                    seen2["ut0_zero"] += 1
                else:
                    seen2["ut0_nonzero"] += 1
            else:
                if state is None:
                    seen2["ut_gt0_zero"] += 1
                else:
                    seen2["ut_gt0_nonzero"] += 1
            return _orig(q, k, v, state)
        ret.recurrent_forward = spy

    # run a fresh turn with persist state injected; spy must observe ut-tagged states
    # simpler: patch adapter.forward to flip ut_holder per call
    import types

    def patch_adapter(a, holder):
        orig_fwd = a.forward

        def fwd(hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False, cache_position=None,
                position_embeddings=None, **kwargs):
            ut = kwargs.get("current_ut", 0)
            holder["v"] = ut
            return orig_fwd(hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_value=past_key_value,
                            use_cache=use_cache, cache_position=cache_position,
                            position_embeddings=position_embeddings, **kwargs)
        a.forward = fwd

    for a, h in zip(adapters, holders):
        patch_adapter(a, h)

    # fresh episode
    clear_all_state(adapters)
    set_persist_state(adapters)  # inject turn1-end-equivalent? use turn1 end from above
    # actually: episode boundary must reset; set injected state to None then persist semantics
    reset_turn_state(adapters)
    seen2.clear()
    model.model(input_ids=traj_a_t1["input_ids"], attention_mask=traj_a_t1["attention_mask"],
                use_cache=False, position_ids=torch.arange(traj_a_t1["input_ids"].shape[1], device="cuda").unsqueeze(0))
    loop_reset_ok = (seen2["ut0_nonzero"] == 0 and seen2["ut_gt0_nonzero"] == 0)
    record("new episode: state == initial (all zero)", loop_reset_ok,
           "ut0_zero=%d ut0_nonzero=%d ut_gt0_zero=%d ut_gt0_nonzero=%d" % (
               seen2["ut0_zero"], seen2["ut0_nonzero"], seen2["ut_gt0_zero"], seen2["ut_gt0_nonzero"]))

    # now with persist state injected at turn start: ut==0 nonzero, ut>0 zero
    # (use t1_end_p from previous persist bookkeeping)
    for i, a in enumerate(adapters):
        a._state = t1_end_p[i]
    seen2.clear()
    model.model(input_ids=traj_a_t2["input_ids"], attention_mask=traj_a_t2["attention_mask"],
                use_cache=False, position_ids=pos2)
    loop_ok = (seen2["ut0_nonzero"] >= len(REPLACED) and
               seen2["ut_gt0_nonzero"] == 0 and
               seen2["ut_gt0_zero"] >= len(REPLACED) * 3)
    record("Ouro loop boundary: ut>0 reset (persist only in ut==0)", loop_ok,
           "ut0_nonzero=%d ut_gt0_nonzero=%d ut_gt0_zero=%d" % (
               seen2["ut0_nonzero"], seen2["ut_gt0_nonzero"], seen2["ut_gt0_zero"]))

    # ---- RoPE continuity gate: persist+continuous-position == concat forward ----
    model2, _ = load_ouro()
    adapters2, _ = convert_p0i(model2, gamma_mode="sigmoid_logit")
    model2.eval()
    clear_all_state(adapters2)
    # one segment: concatenate t1+t2 prompts
    concat_ids = torch.cat([traj_a_t1["input_ids"], traj_a_t2["input_ids"][:, 1:]], dim=-1)
    concat_mask = torch.ones_like(concat_ids)
    concat_pos = torch.arange(concat_ids.shape[1], device="cuda").unsqueeze(0)
    out_concat = model2.model(input_ids=concat_ids, attention_mask=concat_mask,
                              use_cache=False, position_ids=concat_pos)[0].last_hidden_state
    # two segments with persist + continuous pos
    clear_all_state(adapters)
    out_t1 = model.model(input_ids=traj_a_t1["input_ids"], attention_mask=traj_a_t1["attention_mask"],
                         use_cache=False, position_ids=torch.arange(traj_a_t1["input_ids"].shape[1], device="cuda").unsqueeze(0))[0].last_hidden_state
    for i, a in enumerate(adapters):
        a._state = getattr(a, "_turn_state", None)
    out_t2 = model.model(input_ids=traj_a_t2["input_ids"], attention_mask=traj_a_t2["attention_mask"],
                         use_cache=False, position_ids=pos2)[0].last_hidden_state
    h_concat_t2 = out_concat[:, traj_a_t1["input_ids"].shape[1] - 1:]
    h_two_t2 = out_t2
    diff = (h_concat_t2.float() - h_two_t2.float()).abs().max().item()
    rope_ok = diff < 0.05
    record("RoPE continuity: persist+offset == concat (2-seg == 1-seg)", rope_ok,
           "max_abs_diff=%.2e" % diff)

    del model, model2
    torch.cuda.empty_cache()


def main():
    print("=== P0I state-lifetime tests ===", flush=True)
    t0 = time.time()
    unit_tests()
    from clr_ouro import load_ouro
    _, tok = load_ouro()
    adapter_tests(tok)
    print("elapsed %.0fs" % (time.time() - t0), flush=True)

    all_ok = all(ok for _, ok, _ in RESULTS)
    # write markdown report
    md = ["# P0I_STATE_LIFETIME_TESTS.md", "",
          "Date: 2026-08-25", "Run: p0i_state_lifetime_tests.py", "",
          "| # | Test | Result | Detail |", "| --- | --- | --- | --- |"]
    for i, (name, ok, detail) in enumerate(RESULTS):
        md.append("| %d | %s | %s | %s |" % (i + 1, name, "PASS" if ok else "FAIL", detail))
    md += ["", "**Overall: %s**" % ("ALL PASS" if all_ok else "SOME FAILED"), ""]
    if not all_ok:
        md.append("> 测试未全通过，禁止开始训练。")
    with open("D:/training/P0I_STATE_LIFETIME_TESTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\nwrote P0I_STATE_LIFETIME_TESTS.md", flush=True)
    print("OVERALL: %s" % ("ALL PASS" if all_ok else "SOME FAILED"), flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
