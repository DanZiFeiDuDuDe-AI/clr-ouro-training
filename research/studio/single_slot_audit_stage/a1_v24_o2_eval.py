import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from a1_v24_o2_common import (CAPACITIES, DEFAULT_OUT, O2Cache, direct_selected_logits,
                               freeze_storage, json_dump, load_storage, rows_for,
                               slot_logits_from_heads, weighted_answer_logits)
from a1_v23_r2_common import CITY_CLASSES
from clr_v24_o2 import AllSlotsVisibleBaseline, RetrievalController


CHANCE = 1.0 / CITY_CLASSES
ROOT = Path(r"D:\training")


def load_controller(path, device):
    ck = torch.load(path, map_location=device)
    model = RetrievalController().to(device=device, dtype=torch.float32)
    model.load_state_dict(ck["controller"])
    model.eval()
    return model, ck


def load_baseline(path, device):
    ck = torch.load(path, map_location=device)
    model = AllSlotsVisibleBaseline().to(device=device, dtype=torch.float32)
    model.load_state_dict(ck["model"]); model.eval()
    return model


@torch.no_grad()
def storage_batch(core, cache, split, rows, cap, device, order=None):
    facts, labels, query, target_slot, target_y = cache.batch(split, rows, cap, device)
    if order is not None:
        if len(order) == 1:
            facts = facts[:, order[0], :]
        else:
            facts = torch.stack([facts[i, order[i], :] for i in range(len(rows))], dim=0)
        order_t = torch.as_tensor(order, dtype=torch.long, device=device)
        if order_t.ndim == 1:
            target_slot = (order_t == target_slot[:, None]).nonzero(as_tuple=False)[:, 1]
        else:
            target_slot = (order_t == target_slot[:, None]).nonzero(as_tuple=False)[:, 1]
    state, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
    return state, labels, query, target_slot, target_y


def pct(x):
    return 100.0 * float(x)


def metric_from_logits(pred, y):
    return float((pred == y).float().mean().item()) if len(y) else 0.0


@torch.no_grad()
def controller_metrics(controller, heads, state, query, target_slot, target_y, cap,
                       random_seed=0, forced_selected=None):
    scores = controller(query, state, cap)
    probs = scores.softmax(1)
    selected = scores.argmax(1) if forced_selected is None else forced_selected
    hard_pred = direct_selected_logits(heads, state, selected).argmax(1)
    soft_pred = weighted_answer_logits(heads, state, probs).argmax(1)
    oracle_pred = direct_selected_logits(heads, state, target_slot).argmax(1)
    g = torch.Generator(device=state.device); g.manual_seed(random_seed)
    random_selected = torch.randint(cap, (len(y:=target_y),), generator=g, device=state.device)
    random_pred = direct_selected_logits(heads, state, random_selected).argmax(1)
    ret_ok = selected == target_slot
    hard_ok = hard_pred == target_y
    soft_ok = soft_pred == target_y
    return {
        "retrieval_top1_acc": float(ret_ok.float().mean().item()),
        "hard_answer_acc": float(hard_ok.float().mean().item()),
        "soft_answer_acc": float(soft_ok.float().mean().item()),
        "oracle_answer_acc": float((oracle_pred == target_y).float().mean().item()),
        "random_answer_acc": float((random_pred == target_y).float().mean().item()),
        "conditional_answer_acc": float(hard_ok[ret_ok].float().mean().item()) if ret_ok.any() else 0.0,
        "answer_acc_when_retrieval_wrong": float(hard_ok[~ret_ok].float().mean().item()) if (~ret_ok).any() else 0.0,
        "routing_entropy": float((-(probs * probs.clamp_min(1e-8).log()).sum(1)).mean().item()),
        "selected": selected.detach().cpu().numpy(),
        "scores": scores.detach().cpu().numpy(),
    }


def strip_arrays(d):
    return {k: v for k, v in d.items() if k not in ("selected", "scores")}


def position_metrics(controller, heads, state, query, target_slot, target_y, cap):
    m = controller_metrics(controller, heads, state, query, target_slot, target_y, cap)
    out = {}
    sel = torch.as_tensor(m["selected"], device=state.device)
    for p in range(cap):
        mask = target_slot == p
        if mask.any():
            out[str(p)] = {"total": int(mask.sum()),
                           "retrieval_acc": float((sel[mask] == target_slot[mask]).float().mean()),
                           "hard_answer_acc": float((direct_selected_logits(heads, state[mask], sel[mask]).argmax(1) == target_y[mask]).float().mean()),
                           "oracle_answer_acc": float((direct_selected_logits(heads, state[mask], target_slot[mask]).argmax(1) == target_y[mask]).float().mean())}
    return out


@torch.no_grad()
def evaluate_primary(core, heads, hard, soft, baseline, cache, split, cap, device):
    rows = rows_for(cache, split, cap)
    state, labels, query, target_slot, target_y = storage_batch(core, cache, split, rows, cap, device)
    hm = controller_metrics(hard, heads, state, query, target_slot, target_y, cap)
    sm = controller_metrics(soft, heads, state, query, target_slot, target_y, cap)
    bpred = baseline(query, state).argmax(1)
    first = labels[:, 0] == target_y
    last = labels[:, cap - 1] == target_y
    return {
        "n": len(rows), "hard": strip_arrays(hm), "soft": strip_arrays(sm),
        "position": position_metrics(hard, heads, state, query, target_slot, target_y, cap),
        "all_slots_visible_answer_acc": float((bpred == target_y).float().mean()),
        "first_city_baseline_acc": float(first.float().mean()),
        "last_city_baseline_acc": float(last.float().mean()),
        "labels": labels.detach().cpu().numpy(), "state": state, "query": query,
        "target_slot": target_slot, "target_y": target_y,
    }


@torch.no_grad()
def intervention_suite(core, heads, controller, cache, split, cap, device):
    rows = rows_for(cache, split, cap)
    state, labels, query, target_slot, target_y = storage_batch(core, cache, split, rows, cap, device)
    oracle = controller_metrics(controller, heads, state, query, target_slot, target_y, cap)
    wrong_sel = (target_slot + 1) % cap
    wrong = controller_metrics(controller, heads, state, query, target_slot, target_y, cap, forced_selected=wrong_sel)

    s_correct_zero = state.clone(); s_correct_zero[torch.arange(len(rows), device=device), target_slot] = 0
    s_unrelated_zero = state.clone(); s_unrelated_zero[torch.arange(len(rows), device=device), (target_slot + 1) % cap] = 0
    s_all_zero = torch.zeros_like(state)
    corr_zero = controller_metrics(controller, heads, s_correct_zero, query, target_slot, target_y, cap)
    unrelated_zero = controller_metrics(controller, heads, s_unrelated_zero, query, target_slot, target_y, cap)
    all_zero = controller_metrics(controller, heads, s_all_zero, query, target_slot, target_y, cap)

    shuffled_state = state.roll(1, dims=0)
    memory_shuffle = controller_metrics(controller, heads, shuffled_state, query, target_slot, target_y, cap)
    external_query = query.roll(1, dims=0)
    external = controller_metrics(controller, heads, state, external_query, target_slot, target_y, cap)

    wrong_pos = (target_slot + 1) % cap
    wrong_query = cache.person_query_batch(split, rows, wrong_pos.detach().cpu().numpy(), device)
    wrong_target_y = labels[torch.arange(len(rows), device=device), wrong_pos]
    wrong_query_m = controller_metrics(controller, heads, state, wrong_query, wrong_pos, wrong_target_y, cap)

    # Query-only and memory-only controls.
    memory_only_q = torch.zeros_like(query)
    memory_only = controller_metrics(controller, heads, state, memory_only_q, target_slot, target_y, cap)
    query_only = controller_metrics(controller, heads, s_all_zero, query, target_slot, target_y, cap)

    return {
        "oracle_reference": strip_arrays(oracle),
        "forced_wrong_slot": strip_arrays(wrong),
        "wrong_slot_drop_vs_oracle_pp": pct(oracle["oracle_answer_acc"] - wrong["hard_answer_acc"]),
        "correct_slot_zeroed": strip_arrays(corr_zero),
        "unrelated_slot_zeroed": strip_arrays(unrelated_zero),
        "all_slots_zeroed": strip_arrays(all_zero),
        "memory_state_shuffle": strip_arrays(memory_shuffle),
        "external_query_shuffle": strip_arrays(external),
        "within_episode_wrong_query": strip_arrays(wrong_query_m),
        "wrong_query_switched_target_acc": wrong_query_m["hard_answer_acc"],
        "memory_only_query_zero": strip_arrays(memory_only),
        "query_only_memory_zero": strip_arrays(query_only),
        "state": state, "labels": labels, "query": query, "target_slot": target_slot, "target_y": target_y,
    }


@torch.no_grad()
def permutation_test(core, heads, controller, cache, split, cap, device, seed=123):
    rows = rows_for(cache, split, cap)
    state, labels, query, target_slot, target_y = storage_batch(core, cache, split, rows, cap, device)
    rng = np.random.default_rng(seed); perm = rng.permutation(cap)
    perm_state = state[:, perm, :]
    new_target = torch.as_tensor(np.asarray([np.flatnonzero(perm == int(x))[0] for x in target_slot.cpu().numpy()]), device=device)
    m = controller_metrics(controller, heads, perm_state, query, new_target, target_y, cap)
    return {"permutation": perm.tolist(), "original": strip_arrays(controller_metrics(controller, heads, state, query, target_slot, target_y, cap)),
            "permuted": strip_arrays(m)}


@torch.no_grad()
def fact_order_test(core, heads, controller, cache, split, cap, device, seed=321):
    rows = rows_for(cache, split, cap); facts, labels, query, target_slot, target_y = cache.batch(split, rows, cap, device)
    rng = np.random.default_rng(seed); perms = np.stack([rng.permutation(cap) for _ in range(len(rows))])
    pfacts = torch.stack([facts[i, perms[i]] for i in range(len(rows))], dim=0)
    state, _ = core.forward_facts(pfacts, slot_indices=list(range(cap)))
    new_target = torch.as_tensor(np.asarray([np.flatnonzero(perms[i] == int(x))[0] for i, x in enumerate(target_slot.cpu().numpy())]), device=device)
    original_state, _ = core.forward_facts(facts, slot_indices=list(range(cap)))
    orig = controller_metrics(controller, heads, original_state, query, target_slot, target_y, cap)
    perm = controller_metrics(controller, heads, state, query, new_target, target_y, cap)
    return {"original": strip_arrays(orig), "permuted": strip_arrays(perm), "seed": seed}


@torch.no_grad()
def geometry(controller, core, cache, split, cap, device):
    rows = rows_for(cache, split, cap); state, _, query, target_slot, _ = storage_batch(core, cache, split, rows, cap, device)
    scores = controller(query, state, cap)[:, :cap]
    pos = scores[torch.arange(len(rows), device=device), target_slot]
    mask = torch.ones_like(scores, dtype=torch.bool); mask[torch.arange(len(rows), device=device), target_slot] = False
    neg_vals = scores[mask].view(len(rows), cap - 1) if cap > 1 else scores.new_zeros((len(rows), 0))
    margin = pos - neg_vals.max(1).values if cap > 1 else pos
    conf = np.zeros((cap, cap), dtype=np.int64)
    pred = scores.argmax(1).cpu().numpy(); true = target_slot.cpu().numpy()
    for t, p in zip(true, pred): conf[int(t), int(p)] += 1
    return {"positive_score_mean": float(pos.mean()), "negative_max_mean": float(neg_vals.max(1).values.mean()) if cap > 1 else None,
            "margin_mean": float(margin.mean()), "score_std": float(scores.std()),
            "entropy_mean": float((-(scores.softmax(1) * scores.softmax(1).clamp_min(1e-8).log()).sum(1)).mean()),
            "confusion": conf.tolist()}


@torch.no_grad()
def state_geometry(core, cache, split, cap, device):
    rows = rows_for(cache, split, cap); state, _, query, target_slot, _ = storage_batch(core, cache, split, rows, cap, device)
    norms = state.norm(dim=2).mean(0)
    slot_cos = []
    for i in range(cap):
        for j in range(i + 1, cap):
            slot_cos.append(float(torch.nn.functional.cosine_similarity(state[:, i], state[:, j], dim=1).mean()))
    inter = float(torch.nn.functional.cosine_similarity(state[:-1].flatten(1), state[1:].flatten(1), dim=1).mean()) if len(rows) > 1 else 0.0
    qstate = state
    qmove = {"storage_query_update": "not_applicable_in_O2_controller_only_design",
             "memory_norm": float(state.flatten(1).norm(dim=1).mean()),
             "slot_norm_mean": norms.cpu().tolist(), "slot_pair_cosine_mean": float(np.mean(slot_cos)) if slot_cos else 1.0,
             "inter_episode_cosine_adjacent": inter}
    return qmove


def collect_probe_data(core, cache, split, cap, device, max_rows=None):
    rows = rows_for(cache, split, cap); rows = rows[:max_rows] if max_rows else rows
    states=[]; labels=[]
    for i in range(0, len(rows), 256):
        s, y, _, _, _ = storage_batch(core, cache, split, rows[i:i+256], cap, device)
        states.append(s.cpu()); labels.append(y.cpu())
    return torch.cat(states), torch.cat(labels)


def train_probe(x, y, steps=400, seed=7):
    torch.manual_seed(seed); device = x.device
    model = nn.Linear(x.shape[1], CITY_CLASSES).to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True); loss = nn.functional.cross_entropy(model(x), y); loss.backward(); opt.step()
    return model


@torch.no_grad()
def probe_acc(model, x, y): return float((model(x).argmax(1) == y).float().mean())


def fresh_probes(core, cache, device):
    tr_s, tr_y = collect_probe_data(core, cache, "train", 4, device, 6000)
    a_s, a_y = collect_probe_data(core, cache, "testA", 4, device)
    b_s, b_y = collect_probe_data(core, cache, "testB", 4, device)
    tr_s, a_s, b_s = tr_s.to(device), a_s.to(device), b_s.to(device)
    # Pre-query labels are the random query target, not the first fact.
    tr_rows = rows_for(cache, "train", 4)[:len(tr_s)]; a_rows = rows_for(cache, "testA", 4)[:len(a_s)]; b_rows = rows_for(cache, "testB", 4)[:len(b_s)]
    def targets(split, rows): return torch.from_numpy(cache.data[split]["Y"][rows].astype(np.int64)).to(device)
    tr_q, a_q, b_q = targets("train", tr_rows), targets("testA", a_rows), targets("testB", b_rows)
    pre = train_probe(tr_s.flatten(1), tr_q, steps=400)
    # Post-query representation is approximated by the selected local head input: the controller
    # itself is not a storage update, so use the query-conditioned selected slot feature.
    post_x_tr = tr_s[torch.arange(len(tr_s), device=device), torch.from_numpy(cache.data["train"]["query_idx"][tr_rows]).to(device)]
    post_x_a = a_s[torch.arange(len(a_s), device=device), torch.from_numpy(cache.data["testA"]["query_idx"][a_rows]).to(device)]
    post_x_b = b_s[torch.arange(len(b_s), device=device), torch.from_numpy(cache.data["testB"]["query_idx"][b_rows]).to(device)]
    post = train_probe(post_x_tr, tr_q, steps=400)
    return {"pre_query_unconditional_probe": {"testA": probe_acc(pre, a_s.flatten(1), a_q), "testB": probe_acc(pre, b_s.flatten(1), b_q)},
            "post_query_selected_slot_probe": {"testA": probe_acc(post, post_x_a, a_q), "testB": probe_acc(post, post_x_b, b_q)},
            "note": "O2 controller-only design has no S_query storage state; post-query probe uses the selected slot feature."}


def slot_probes(core, cache, device):
    out = {}
    for cap in CAPACITIES:
        tr_s, tr_y = collect_probe_data(core, cache, "train", cap, device, 6000)
        a_s, a_y = collect_probe_data(core, cache, "testA", cap, device)
        b_s, b_y = collect_probe_data(core, cache, "testB", cap, device)
        tr_s, a_s, b_s = tr_s.to(device), a_s.to(device), b_s.to(device)
        tr_y, a_y, b_y = tr_y.to(device), a_y.to(device), b_y.to(device)
        out[str(cap)] = {}
        for p in range(4):
            if p >= cap: out[str(cap)][f"slot{p}"] = None; continue
            m = train_probe(tr_s[:, p, :], tr_y[:, p], steps=250, seed=10+p)
            out[str(cap)][f"slot{p}"] = {"testA": probe_acc(m, a_s[:, p, :], a_y[:, p]), "testB": probe_acc(m, b_s[:, p, :], b_y[:, p]),
                                         "train": probe_acc(m, tr_s[:, p, :], tr_y[:, p])}
    return out


@torch.no_grad()
def pairwise_query(core, heads, controller, cache, split, cap, device, n=1000):
    rows = rows_for(cache, split, cap)[:n]; state, labels, _, _, _ = storage_batch(core, cache, split, rows, cap, device)
    pos_a = torch.zeros(len(rows), dtype=torch.long, device=device); pos_b = torch.ones(len(rows), dtype=torch.long, device=device)
    qa = cache.person_query_batch(split, rows, pos_a.cpu().numpy(), device); qb = cache.person_query_batch(split, rows, pos_b.cpu().numpy(), device)
    ma = controller_metrics(controller, heads, state, qa, pos_a, labels[:, 0], cap)
    mb = controller_metrics(controller, heads, state, qb, pos_b, labels[:, 1], cap)
    sel_a = torch.as_tensor(ma["selected"], device=device); sel_b = torch.as_tensor(mb["selected"], device=device)
    both = (sel_a == pos_a) & (sel_b == pos_b)
    hard_both = (direct_selected_logits(heads, state, sel_a).argmax(1) == labels[:, 0]) & (direct_selected_logits(heads, state, sel_b).argmax(1) == labels[:, 1])
    hard_any = (direct_selected_logits(heads, state, sel_a).argmax(1) == labels[:, 0]) | (direct_selected_logits(heads, state, sel_b).argmax(1) == labels[:, 1])
    return {"n_pairs": len(rows), "both_retrieval_correct": float(both.float().mean()),
            "both_correct_rate": float(hard_both.float().mean()),
            "at_least_one_correct_rate": float(hard_any.float().mean()),
            "at_least_one_retrieval_correct": float(((sel_a == pos_a) | (sel_b == pos_b)).float().mean())}


def main():
    t0 = time.perf_counter(); out = Path(os.environ.get("V24_O2_OUT_DIR", DEFAULT_OUT)); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = O2Cache(os.environ.get("V22_CACHE_DIR", r"D:\training\checkpoints\clr_v2_a1_ls2\cache"))
    core, heads, _ = load_storage(device); hard, _ = load_controller(out / "best.pt", device)
    log = json.loads((out / "train_log.json").read_text(encoding="utf-8")); soft_path = Path(log["soft_variant"]["best_path"])
    if not soft_path.exists(): soft_path = out / "soft_lambda1p0_best.pt"
    soft, _ = load_controller(soft_path, device); baseline = load_baseline(out / "all_slots_baseline.pt", device)

    primary = {split: {} for split in ("val", "testA", "testB")}; interventions={}; geometry_out={}; state_geo={}; position={}
    for split in primary:
        for cap in CAPACITIES:
            p = evaluate_primary(core, heads, hard, soft, baseline, cache, split, cap, device)
            primary[split][str(cap)] = {k:v for k,v in p.items() if k not in ("labels","state","query","target_slot","target_y")}
            position.setdefault(split, {})[str(cap)] = p["position"]
            geometry_out.setdefault(split,{})[str(cap)] = geometry(hard, core, cache, split, cap, device)
            state_geo.setdefault(split,{})[str(cap)] = state_geometry(core, cache, split, cap, device)
            if split in ("testA","testB"):
                interventions.setdefault(split,{})[str(cap)] = {k:v for k,v in intervention_suite(core, heads, hard, cache, split, cap, device).items() if k not in ("state","labels","query","target_slot","target_y")}

    pair = {s: pairwise_query(core, heads, hard, cache, s, 4, device) for s in ("testA","testB")}
    order = {s: fact_order_test(core, heads, hard, cache, s, 4, device) for s in ("testA","testB")}
    slot_perm = {s: permutation_test(core, heads, hard, cache, s, 4, device) for s in ("testA","testB")}
    probes = fresh_probes(core, cache, device); slots = slot_probes(core, cache, device)

    # Existing cache supports N=8/16; O1's four oracle slots are reused modulo 4 as an overload diagnostic only.
    overload = {}
    for split in ("testA","testB"):
        overload[split]={}
        for cap in (8,16):
            rows=rows_for(cache,split,cap); facts, labels, query, ts, y=cache.batch(split,rows,cap,device)
            state,_=core.forward_facts(facts,slot_indices=[i%4 for i in range(cap)])
            tsm=torch.remainder(ts,4); hm=controller_metrics(hard,heads,state,query,tsm,y,4)
            overload[split][str(cap)]={"hard":strip_arrays(hm),"n":len(rows),"note":"zero-shot O1 modulo-4 overload diagnostic; not an O2 training capacity."}

    def macro(split, key, field): return float(np.mean([primary[split][str(c)][key][field] for c in CAPACITIES]))
    chosen_summary={}
    for split in ("testA","testB"):
        chosen_summary[split]={"hard_macro_answer":macro(split,"hard","hard_answer_acc"),"hard_macro_retrieval":macro(split,"hard","retrieval_top1_acc"),
                              "oracle_n4":primary[split]["4"]["hard"]["oracle_answer_acc"],"hard_n4":primary[split]["4"]["hard"]["hard_answer_acc"],
                              "retrieval_n4":primary[split]["4"]["hard"]["retrieval_top1_acc"]}
    metrics={"chance":CHANCE,"primary":primary,"summary":chosen_summary,"overload_zero_shot":overload,"device":str(device),"wall_clock_s":time.perf_counter()-t0}
    interventions_out={"interventions":interventions,"pairwise_query":pair,"fact_order_permutation":order,"slot_permutation":slot_perm}
    json_dump(out/"metrics.json",metrics); json_dump(out/"intervention_results.json",interventions_out); json_dump(out/"retrieval_confusion_matrix.json",geometry_out); json_dump(out/"state_geometry.json",state_geo); json_dump(out/"slot_capacity.json",{"fresh_slot_probes":slots,"fresh_query_probes":probes})
    json_dump(out/"capacity_curve.json",chosen_summary); json_dump(out/"position_matrix.json",position); json_dump(out/"query_interventions.json",interventions); json_dump(out/"memory_interventions.json",interventions_out)
    json_dump(out/"pairwise_query.json", pair)
    json_dump(out/"shortcut_checks.json", {s: {str(c): {"first_city": primary[s][str(c)]["first_city_baseline_acc"],
                                                        "last_city": primary[s][str(c)]["last_city_baseline_acc"],
                                                        "query_only_memory_zero": interventions[s][str(c)]["query_only_memory_zero"],
                                                        "memory_shuffle": interventions[s][str(c)]["memory_state_shuffle"],
                                                        "all_slots_visible": primary[s][str(c)]["all_slots_visible_answer_acc"]}
                                      for c in CAPACITIES} for s in ("testA", "testB")})

    a=chosen_summary["testA"]; b=chosen_summary["testB"]; i4a=interventions["testA"]["4"]; i4b=interventions["testB"]["4"]
    pass_gate=(a["retrieval_n4"]>=.90 and b["retrieval_n4"]>=.85 and (a["oracle_n4"]-a["hard_n4"]<=.05) and (b["oracle_n4"]-b["hard_n4"]<=.05)
                and i4a["forced_wrong_slot"]["hard_answer_acc"] <= i4a["oracle_reference"]["oracle_answer_acc"]-.40
                and i4b["forced_wrong_slot"]["hard_answer_acc"] <= i4b["oracle_reference"]["oracle_answer_acc"]-.40
                and i4a["unrelated_slot_zeroed"]["hard_answer_acc"] >= a["hard_n4"]-.05
                and i4b["unrelated_slot_zeroed"]["hard_answer_acc"] >= b["hard_n4"]-.05)
    retrieval_bottleneck = (a["retrieval_n4"] < .90 or b["retrieval_n4"] < .85)
    answer_bypass = (primary["testA"]["4"]["all_slots_visible_answer_acc"] > a["hard_n4"] + .20 and
                     primary["testB"]["4"]["all_slots_visible_answer_acc"] > b["hard_n4"] + .20)
    verdict=("O2 SELECTIVE RETRIEVAL PASS" if pass_gate else
             "O2 FAIL: ANSWER BYPASS" if answer_bypass else
             "O2 FAIL: RETRIEVAL BOTTLENECK" if retrieval_bottleneck else
             "O2 PARTIAL: ROUTING LEARNED BUT NOT CAUSALLY USED")
    zero_path = out / "zero_shot_multiwrite.json"
    zero = json.loads(zero_path.read_text(encoding="utf-8")) if zero_path.exists() else None
    zero_rows = ""
    if zero is not None:
        zero_rows = ("\n## LS1 zero-shot multi-write baseline\n\n"
                     "| split | N=1 | N=2 | N=4 | N=8 | N=16 |\n|---|---:|---:|---:|---:|---:|\n" +
                     "\n".join("| %s | %s |" % (s, " | ".join(
                         f"{pct(zero['curves'][s][str(c)]['answer_acc']):.2f}%" for c in (1, 2, 4, 8, 16)))
                               for s in ("testA", "testB")) +
                     "\n\nThis diagnostic uses the completed LS1 checkpoint with query ignored; it does not alter O2 checkpoint selection.\n")
    report=f'''# CLR-v2.4-O2 Learned Selective Retrieval\n\n## Verdict\n\n**{verdict}**\n\nThis O2 run keeps the CLR-v2.3-OA-r2 oracle-position storage frozen and trains only a query-conditioned retrieval controller. No learned write allocation was added.\n\n## Main results\n\n| split | N=1 hard | N=2 hard | N=4 hard | N=4 retrieval | N=4 oracle |\n|---|---:|---:|---:|---:|---:|\n| TestA | {pct(primary["testA"]["1"]["hard"]["hard_answer_acc"]):.2f}% | {pct(primary["testA"]["2"]["hard"]["hard_answer_acc"]):.2f}% | {pct(a["hard_n4"]):.2f}% | {pct(a["retrieval_n4"]):.2f}% | {pct(a["oracle_n4"]):.2f}% |\n| TestB | {pct(primary["testB"]["1"]["hard"]["hard_answer_acc"]):.2f}% | {pct(primary["testB"]["2"]["hard"]["hard_answer_acc"]):.2f}% | {pct(b["hard_n4"]):.2f}% | {pct(b["retrieval_n4"]):.2f}% | {pct(b["oracle_n4"]):.2f}% |\n\nChance is {pct(CHANCE):.2f}%. Full matrices and interventions are in the JSON artifacts.\n{zero_rows}\n## Causal checks\n\n- Forced wrong-slot, correct-slot zeroing, unrelated-slot zeroing, memory shuffle, external query, and within-episode wrong-query are in `intervention_results.json`.\n- Same-memory/different-query pair results are in `pairwise_query.json`.\n- Fact-order and physical slot permutation results are included.\n- This O2 controller-only design does not create an `S_query` storage state; query-induced storage movement is therefore explicitly N/A, while query-conditioned controller movement is measured by pair/intervention outcomes.\n\n## Diagnostics and scope\n\nFresh probes, slot capacity, score confusion, state geometry, and N=8/N=16 modulo-4 overload diagnostics are saved separately. Existing cache metadata and `query_person_H.npz` were reused; Ouro was not rerun.\n\nThe capability boundary is: WRITE (prior LS0), RETENTION (prior LS1), MULTI-WRITE (this O2 storage substrate), and SELECTIVE RETRIEVAL (this O2 controller). A PASS would not imply latent reasoning, general memory, or O3.\n\n## Q1-Q15\n\n1. Storage was reused from the completed O1/r2 checkpoint.\n2. Only the retrieval controller was trained.\n3. N=1,2,4 were sampled uniformly during training.\n4. TestA is seen-person/new-binding and TestB is unseen-person.\n5. Query hidden is person-only from the existing cache; it contains no target city.\n6. Oracle, hard, soft, random, and all-slots-visible baselines are reported.\n7. Correct/wrong/external query and memory interventions are reported.\n8. Wrong-slot and zero-slot counterfactuals test causal use of routing.\n9. Pairwise query tests the same memory with two different keys.\n10. Fact-order and slot permutation tests are reported.\n11. Fresh probes and score confusion quantify geometry.\n12. N=8/N=16 are diagnostic overload only, not trained capacity.\n13. No architecture, storage primitive, Ouro weights, or learned writer was changed.\n14. PASS/partial/fail is determined by the preregistered retrieval and causal gates above.\n15. The experiment stops here; no O3 or A2 was entered.\n'''
    (out/"v24_o2_report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"verdict":verdict,"summary":chosen_summary,"wall_clock_s":metrics["wall_clock_s"]},indent=2))


if __name__ == "__main__": main()
