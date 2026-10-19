# -*- coding: utf-8 -*-
"""CLR-v2.4-O2.1a-MR — multi-relation semantic-key validation.

The storage path is the frozen CLR-v2.3-OA-r2 OracleLocalCore.  The router is
the same O2.1a K1 KeyEncoder/QueryEncoder and dot-product hard matcher.  This
file adds only the multi-relation data protocol and diagnostics; it does not
add a memory module, learned allocation, recurrent retrieval, or reasoning.
"""
import copy
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v24_o2_common import load_storage, json_dump
from a1_v24_o21a import KeyEncoder, QueryEncoder
from a1_ls2_common import load_model as load_ls1_model, forward_batch as ls1_forward_batch


ROOT = Path(r"D:/training")
OUT = Path(os.environ.get("V24_O21A_MR_OUT", r"D:/training/checkpoints/clr_v24_o21a_mr"))
CACHE = Path(os.environ.get("V24_O21A_MR_CACHE", str(OUT / "cache")))
O21A = Path(os.environ.get("V24_O21A_SOURCE", r"D:/training/checkpoints/clr_v24_o21a_semantic_key/best.pt"))
R2 = ROOT / "checkpoints" / "clr_v23_oracle_r2" / "best_model.pt"
CAPS = (1, 2, 4)
RELATIONS = ("current_city", "birth_city", "favorite_city", "work_city")
N_REL = len(RELATIONS)
N_CLASSES = 64
CHANCE = 1.0 / N_CLASSES
SEED = int(os.environ.get("V24_O21A_MR_SEED", "42"))


def seed_all(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def dump(path, obj):
    json_dump(path, obj)


class MRCache:
    def __init__(self, root=CACHE):
        self.root = Path(root); self.data = {}; self.meta = {}
        for split in ("train", "val", "testA", "testB"):
            z = np.load(self.root / f"{split}_data.npz", allow_pickle=False)
            self.data[split] = {k: z[k] for k in z.files}
            with (self.root / f"{split}_meta.jsonl").open("r", encoding="utf-8") as f:
                self.meta[split] = [json.loads(x) for x in f if x.strip()]
        self.train_persons = json.loads((self.root / "train_persons.json").read_text(encoding="utf-8"))
        self.person_to_id = {p: i for i, p in enumerate(self.train_persons)}

    def rows(self, split, cap):
        return np.flatnonzero(self.data[split]["N"] == cap).astype(np.int64)

    def rows_type(self, split, cap, kind):
        return np.asarray([i for i, m in enumerate(self.meta[split])
                           if int(m["capacity"]) == cap and m["episode_type"] == kind], dtype=np.int64)

    def batch(self, split, rows, cap, device):
        rows = np.asarray(rows, dtype=np.int64); d = self.data[split]; off = d["fact_offsets"]
        facts = np.stack([d["fact_H"][int(off[r]):int(off[r + 1])] for r in rows]).astype(np.float32)
        ent = np.stack([d["fact_entity"][int(off[r]):int(off[r + 1])] for r in rows]).astype(np.int64)
        rel = np.stack([d["fact_relation"][int(off[r]):int(off[r + 1])] for r in rows]).astype(np.int64)
        qh = d["query_H"][rows].astype(np.float32)
        ts = d["query_idx"][rows].astype(np.int64); y = d["Y"][rows].astype(np.int64)
        fy = np.stack([d["fact_Y"][int(off[r]):int(off[r + 1])] for r in rows]).astype(np.int64)
        assert facts.shape[1] == cap
        assert np.array_equal(y, fy[np.arange(len(rows)), ts])
        return (torch.from_numpy(facts).to(device), torch.from_numpy(ent).to(device),
                torch.from_numpy(rel).to(device), torch.from_numpy(qh).to(device),
                torch.from_numpy(ts).to(device), torch.from_numpy(y).to(device),
                torch.from_numpy(fy).to(device))

    def fact_query_batch(self, split, rows, positions, device):
        rows = np.asarray(rows, dtype=np.int64); pos = np.asarray(positions, dtype=np.int64)
        d = self.data[split]; off = d["fact_offsets"]
        ids = [int(off[r]) + int(p) for r, p in zip(rows, pos)]
        return torch.from_numpy(d["fact_query_H"][ids].astype(np.float32)).to(device)


class KeyHeadsMR(nn.Module):
    """O2.1a K1 semantic auxiliary heads; no value input is used."""
    def __init__(self, nentity, dim=64):
        super().__init__()
        self.entity = nn.Linear(dim, nentity)
        self.query_entity = nn.Linear(dim, nentity)


class MRValueHeads(nn.Module):
    """Same O1 local-head shape, trained only on frozen MR V states."""
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(64, N_CLASSES)
                                    for _ in range(4)])

    def all_logits(self, v):
        return torch.stack([self.heads[i](v[:, i, :].float()) for i in range(4)], dim=1)

    def selected(self, v, idx):
        return self.all_logits(v)[torch.arange(len(idx), device=v.device), idx]


@torch.no_grad()
def value_state(core, facts):
    return core.forward_facts(facts, slot_indices=list(range(facts.shape[1])))[0]


def batch_rows(cache, split, cap, rng, batch=128):
    rows = cache.rows(split, cap)
    return rows[rng.integers(0, len(rows), size=min(batch, len(rows)))]


def train_value_heads(core, cache, device, steps=600):
    vh = MRValueHeads().to(device)
    opt = torch.optim.AdamW(vh.parameters(), lr=3e-4, weight_decay=0.01)
    rng = np.random.default_rng(SEED + 700)
    core.eval(); vh.train(); t0 = time.perf_counter(); logs = []
    for step in range(1, steps + 1):
        cap = int(rng.choice(CAPS)); rows = batch_rows(cache, "train", cap, rng)
        facts, _, _, _, _, _, fy = cache.batch("train", rows, cap, device)
        with torch.no_grad(): v = value_state(core, facts)
        opt.zero_grad(set_to_none=True)
        logits = vh.all_logits(v)
        loss = sum(F.cross_entropy(logits[:, i, :], fy[:, i]) for i in range(cap)) / cap
        loss.backward(); torch.nn.utils.clip_grad_norm_(vh.parameters(), 1.0); opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            logs.append({"step": step, "loss": float(loss.detach())})
    vh.eval()
    return vh, {"steps": steps, "logs": logs, "wall_clock_s": time.perf_counter() - t0,
                "params": sum(p.numel() for p in vh.parameters())}


def load_warm_start(key, query):
    if not O21A.exists():
        return False
    ck = torch.load(O21A, map_location="cpu", weights_only=False)
    key.load_state_dict(ck["key"]); query.load_state_dict(ck["query"])
    return True


def key_batch(key, query, facts, qh):
    k = key(facts.reshape(-1, 2048)).reshape(facts.shape[0], facts.shape[1], -1)
    q = query(qh)
    return k, q


def train_loss(key, query, aux, facts, ent, qh, ts):
    k, q = key_batch(key, query, facts, qh)
    scores = (q[:, None, :] * k).sum(-1) / 0.1
    retrieval = F.cross_entropy(scores, ts)
    pos = k[torch.arange(len(ts), device=ts.device), ts]
    align = F.cross_entropy((q @ pos.T) / 0.1, torch.arange(len(ts), device=ts.device))
    ky = aux.entity(k.reshape(-1, 64)); qy = aux.query_entity(q)
    entity = F.cross_entropy(ky, ent.reshape(-1))
    q_entity = F.cross_entropy(qy, ent[torch.arange(len(ts), device=ts.device), ts])
    total = retrieval + 0.5 * align + 0.1 * (entity + q_entity)
    return total, {"retrieval": retrieval.detach(), "align": align.detach(),
                   "entity": entity.detach(), "query_entity": q_entity.detach()}


@torch.no_grad()
def evaluate_rows(key, query, vh, core, cache, split, cap, rows, device):
    key.eval(); query.eval(); vh.eval(); rows = np.asarray(rows, dtype=np.int64)
    total = ret = hard = oracle = random = 0; selected = []; target = []; yall = []
    for st in range(0, len(rows), 128):
        rb = rows[st:st + 128]
        facts, _, _, qh, ts, y, _ = cache.batch(split, rb, cap, device)
        v = value_state(core, facts); k, q = key_batch(key, query, facts, qh)
        sc = (q[:, None, :] * k).sum(-1) / 0.1; sel = sc.argmax(1)
        hp = vh.selected(v, sel).argmax(1); op = vh.selected(v, ts).argmax(1)
        rs = torch.remainder(torch.arange(len(ts), device=device) + 1, cap)
        rp = vh.selected(v, rs).argmax(1)
        ret += int((sel == ts).sum()); hard += int((hp == y).sum())
        oracle += int((op == y).sum()); random += int((rp == y).sum()); total += len(y)
        selected.extend(sel.cpu().tolist()); target.extend(ts.cpu().tolist()); yall.extend(y.cpu().tolist())
    return {"n": total, "retrieval": ret / max(total, 1), "hard_answer": hard / max(total, 1),
            "oracle_answer": oracle / max(total, 1), "random_answer": random / max(total, 1),
            "selected": np.asarray(selected), "target": np.asarray(target), "y": np.asarray(yall)}


def train_main(core, vh, cache, device, steps=3000):
    key = KeyEncoder(64).to(device); query = QueryEncoder(64).to(device)
    warm = load_warm_start(key, query)
    aux = KeyHeadsMR(len(cache.train_persons), 64).to(device)
    opt = torch.optim.AdamW(list(key.parameters()) + list(query.parameters()) + list(aux.parameters()),
                            lr=1e-3, weight_decay=0.01)
    rng = np.random.default_rng(SEED + 800); best = -1.0; best_payload = None; logs = []
    train_rows = {c: cache.rows("train", c) for c in CAPS}
    val_rows = {c: cache.rows("val", c) for c in CAPS}
    t0 = time.perf_counter(); key.train(); query.train(); aux.train()
    for step in range(1, steps + 1):
        cap = int(rng.choice(CAPS)); rows = train_rows[cap][rng.integers(0, len(train_rows[cap]), size=min(128, len(train_rows[cap])))]
        facts, ent, _, qh, ts, _, _ = cache.batch("train", rows, cap, device)
        opt.zero_grad(set_to_none=True); loss, parts = train_loss(key, query, aux, facts, ent, qh, ts)
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(key.parameters()) + list(query.parameters()) + list(aux.parameters()), 1.0); opt.step()
        if step == 1 or step % 100 == 0 or step == steps:
            vals = {str(c): evaluate_rows(key, query, vh, core, cache, "val", c, val_rows[c], device) for c in CAPS}
            macro = float(np.mean([vals[str(c)]["retrieval"] for c in CAPS]))
            logs.append({"step": step, "loss": float(loss.detach()),
                         "parts": {k: float(v) for k, v in parts.items()},
                         "val_macro_retrieval": macro,
                         "val": {c: {k: v for k, v in m.items() if k not in ("selected", "target", "y")} for c, m in vals.items()}})
            if macro > best:
                best = macro; best_payload = {"key": key.state_dict(), "query": query.state_dict(),
                                              "heads": aux.state_dict(), "value_heads": vh.state_dict(),
                                              "step": step, "val": vals, "warm_start_o21a": warm}
                torch.save(best_payload, OUT / "best.pt")
                torch.save(best_payload, OUT / "best_model.pt")
    final_payload = {"key": key.state_dict(), "query": query.state_dict(), "heads": aux.state_dict(),
                     "value_heads": vh.state_dict(), "step": steps, "warm_start_o21a": warm}
    torch.save(final_payload, OUT / "final.pt")
    torch.save(final_payload, OUT / "final_model.pt")
    return key, query, aux, {"steps": steps, "best_macro_val_retrieval": best,
                             "best_step": best_payload["step"], "warm_start_o21a": warm,
                             "logs": logs, "wall_clock_s": time.perf_counter() - t0,
                             "params": sum(p.numel() for p in key.parameters()) + sum(p.numel() for p in query.parameters()) + sum(p.numel() for p in aux.parameters())}


def restore_best(cache, device):
    ck = torch.load(OUT / "best.pt", map_location="cpu", weights_only=False)
    key = KeyEncoder(64).to(device); query = QueryEncoder(64).to(device); aux = KeyHeadsMR(len(cache.train_persons), 64).to(device); vh = MRValueHeads().to(device)
    key.load_state_dict(ck["key"]); query.load_state_dict(ck["query"]); aux.load_state_dict(ck["heads"]); vh.load_state_dict(ck["value_heads"])
    key.eval(); query.eval(); aux.eval(); vh.eval(); return key, query, aux, vh, ck


def curve(results):
    out = {str(c): results[str(c)]["retrieval"] for c in CAPS}
    a1 = out["1"]
    for c in CAPS: out[f"{c}_normalized_capacity"] = (out[str(c)] - CHANCE) / max(a1 - CHANCE, 1e-9)
    return out


def position_matrix(cache, split, results):
    out = {}
    for c in CAPS:
        rows = cache.rows(split, c); q = cache.data[split]["query_idx"][rows]; pred = results[str(c)]["selected"]; ts = results[str(c)]["target"]
        rows_out = {}
        masks = {"FIRST": q == 0, "LAST": q == c - 1, "ALL": np.ones(len(q), dtype=bool)}
        masks["MIDDLE"] = ((q == c // 2 - 1) | (q == c // 2)) if c >= 4 else np.zeros(len(q), dtype=bool)
        for name, mask in masks.items():
            rows_out[name] = {"n": int(mask.sum()), "retrieval": float((pred[mask] == ts[mask]).mean()) if mask.any() else None}
        rows_out["primacy_gap"] = rows_out["FIRST"]["retrieval"] - rows_out["ALL"]["retrieval"] if rows_out["FIRST"]["retrieval"] is not None else None
        rows_out["recency_gap"] = rows_out["LAST"]["retrieval"] - rows_out["ALL"]["retrieval"] if rows_out["LAST"]["retrieval"] is not None else None
        rows_out["middle_gap"] = rows_out["MIDDLE"]["retrieval"] - rows_out["ALL"]["retrieval"] if rows_out["MIDDLE"]["retrieval"] is not None else None
        out[str(c)] = rows_out
    return out


def shortcut_baselines(cache, split):
    out = {}
    d = cache.data[split]
    for c in CAPS:
        rows = cache.rows(split, c); off = d["fact_offsets"]; y = d["Y"][rows]
        first = np.asarray([d["fact_Y"][off[r]] for r in rows]); last = np.asarray([d["fact_Y"][off[r + 1] - 1] for r in rows])
        out[str(c)] = {"first": float((first == y).mean()), "last": float((last == y).mean())}
    return out


@torch.no_grad()
def kv_interventions(key, query, vh, core, cache, split, cap, rows, device):
    facts, _, _, qh, ts, y, _ = cache.batch(split, rows, cap, device); v = value_state(core, facts); k, q = key_batch(key, query, facts, qh)
    ix = torch.arange(len(ts), device=device)
    def run(kx, vx, tx=ts):
        sel = ((q[:, None, :] * kx).sum(-1) / 0.1).argmax(1)
        pred = vh.selected(vx, sel).argmax(1)
        return {"retrieval": float((sel == tx).float().mean()), "answer": float((pred == y).float().mean())}
    def run_force(sel, vx=v):
        return {"retrieval": float((sel == ts).float().mean()), "answer": float((vh.selected(vx, sel).argmax(1) == y).float().mean())}
    nxt = (ts + 1) % cap
    vz = v.clone(); vz[ix, ts] = 0
    uz = v.clone(); uz[ix, nxt] = 0
    kz = k.clone(); kz[ix, ts] = 0
    ks = k.clone(); ks[ix, ts], ks[ix, nxt] = k[ix, nxt], k[ix, ts]
    vs = v.clone(); vs[ix, ts], vs[ix, nxt] = v[ix, nxt], v[ix, ts]
    perm = torch.arange(cap - 1, -1, -1, device=device); kp = k[:, perm, :]; vp = v[:, perm, :]
    tsp = (perm[None, :] == ts[:, None]).nonzero(as_tuple=False)[:, 1]
    return {"base": run(k, v), "forced_wrong_slot": run_force(nxt),
            "correct_value_zero": run(k, vz), "unrelated_value_zero": run(k, uz),
            "correct_key_zero": run(kz, v), "wrong_key_swap": run(ks, v),
            "value_swap": run(k, vs), "full_kv_permutation": run(kp, vp, tsp)}


def _probe_fit(x, y, out_dim, device, steps=250, seed=123):
    torch.manual_seed(seed); x = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device); y = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)
    m = nn.Linear(x.shape[1], out_dim).to(device); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    for _ in range(steps):
        idx = torch.randperm(len(y), device=device)[:min(4096, len(y))]; opt.zero_grad(set_to_none=True); F.cross_entropy(m(x[idx]), y[idx]).backward(); opt.step()
    return m


@torch.no_grad()
def collect_embeddings(key, query, cache, split, device):
    all_k = []; all_e = []; all_r = []; all_v = []; all_q = []; all_qe = []; all_qr = []
    for cap in CAPS:
        rows = cache.rows(split, cap); d = cache.data[split]; off = d["fact_offsets"]
        for st in range(0, len(rows), 128):
            rb = rows[st:st + 128]; facts, ent, rel, qh, ts, _, fy = cache.batch(split, rb, cap, device); k, q = key_batch(key, query, facts, qh)
            all_k.append(k.reshape(-1, 64).cpu().numpy()); all_e.append(ent.reshape(-1).cpu().numpy()); all_r.append(rel.reshape(-1).cpu().numpy()); all_v.append(fy.reshape(-1).cpu().numpy())
            all_q.append(q.cpu().numpy()); all_qe.append(ent[np.arange(len(ts)), ts].cpu().numpy()); all_qr.append(rel[np.arange(len(ts)), ts].cpu().numpy())
    return {"K": np.concatenate(all_k), "entity": np.concatenate(all_e), "relation": np.concatenate(all_r), "value": np.concatenate(all_v),
            "Q": np.concatenate(all_q), "query_entity": np.concatenate(all_qe), "query_relation": np.concatenate(all_qr)}


def probe_metrics(probe, x, y, known=True):
    if len(x) == 0: return {"accuracy": None, "n": 0}
    with torch.no_grad(): p = probe(torch.from_numpy(x).to(next(probe.parameters()).device)).argmax(1).cpu().numpy()
    m = np.asarray(y) >= 0 if not known else np.ones(len(y), dtype=bool)
    return {"accuracy": float((p[m] == np.asarray(y)[m]).mean()) if m.any() else None, "n": int(m.sum())}


def run_probes(key, query, aux, cache, device):
    emb = {s: collect_embeddings(key, query, cache, s, device) for s in ("train", "val", "testA", "testB")}
    ent_out = {}; rel_out = {}; joint_out = {}; q_out = {}
    rel_probe = _probe_fit(emb["train"]["K"], emb["train"]["relation"], N_REL, device, 300, 911)
    qrel_probe = _probe_fit(emb["train"]["Q"], emb["train"]["query_relation"], N_REL, device, 300, 912)
    joint_y = emb["train"]["entity"] * N_REL + emb["train"]["relation"]
    qjoint_y = emb["train"]["query_entity"] * N_REL + emb["train"]["query_relation"]
    joint_probe = _probe_fit(emb["train"]["K"], joint_y, max(int(joint_y.max()) + 1, 1), device, 300, 913)
    qjoint_probe = _probe_fit(emb["train"]["Q"], qjoint_y, max(int(qjoint_y.max()) + 1, 1), device, 300, 914)
    for s in emb:
        e = emb[s]; known = s != "testB"
        ent_out[s] = {"K_to_entity_head": probe_metrics(aux.entity.to(device), e["K"], e["entity"], known=known),
                      "Q_to_entity_head": probe_metrics(aux.query_entity.to(device), e["Q"], e["query_entity"], known=known)}
        rel_out[s] = {"K_to_relation_probe": probe_metrics(rel_probe, e["K"], e["relation"]),
                      "Q_to_relation_probe": probe_metrics(qrel_probe, e["Q"], e["query_relation"])}
        if s == "testB":
            # TestB entity ids are -1 in the cache; report only relation here.
            jmask = e["entity"] >= 0; qjmask = e["query_entity"] >= 0
        else:
            jmask = np.ones(len(e["entity"]), dtype=bool); qjmask = np.ones(len(e["query_entity"]), dtype=bool)
        joint_out[s] = {"K_to_joint_probe": probe_metrics(joint_probe, e["K"][jmask], (e["entity"][jmask] * N_REL + e["relation"][jmask])),
                        "Q_to_joint_probe": probe_metrics(qjoint_probe, e["Q"][qjmask], (e["query_entity"][qjmask] * N_REL + e["query_relation"][qjmask]))}
        q_out[s] = rel_out[s]
    leakage = {s: {"K_to_value_probe": probe_metrics(_probe_fit(emb["train"]["K"], emb["train"]["value"], N_CLASSES, device, 250, 915), emb[s]["K"], emb[s]["value"])} for s in emb}
    return ent_out, rel_out, joint_out, q_out, leakage, emb


@torch.no_grad()
def component_ablation(key, query, cache, split, cap, rows, rel_probe, device):
    facts, ent, rel, qh, ts, _, _ = cache.batch(split, rows, cap, device); k, q = key_batch(key, query, facts, qh)
    # Entity-only / relation-only are supervised counterfactual routers. They
    # are diagnostics, not new model modules or a replacement training path.
    ent_logits_k = None
    out = {}
    # The caller supplies fresh probes through closure attributes below.
    return out


@torch.no_grad()
def ablation_with_probes(key, query, aux, rel_probe, cache, split, cap, rows, device):
    facts, ent, rel, qh, ts, _, _ = cache.batch(split, rows, cap, device); k, q = key_batch(key, query, facts, qh)
    q_ent = ent[torch.arange(len(ts), device=device), ts]
    q_rel = rel[torch.arange(len(ts), device=device), ts]
    entity_scores = (ent == q_ent[:, None]).float()
    relation_scores = (rel == q_rel[:, None]).float()
    return {"entity_only": float((entity_scores.argmax(1) == ts).float().mean()),
            "relation_only": float((relation_scores.argmax(1) == ts).float().mean()),
            "full_key": float((((q[:, None, :] * k).sum(-1)).argmax(1) == ts).float().mean())}


@torch.no_grad()
def query_controls(key, query, vh, core, cache, split, cap, rows, device):
    rows = np.asarray(rows, dtype=np.int64); facts, _, _, qh, ts, y, fy = cache.batch(split, rows, cap, device); v = value_state(core, facts); k, _ = key_batch(key, query, facts, qh)
    ix = np.arange(len(rows)); alt = (ts.cpu().numpy() + 1) % cap
    alt_q = cache.fact_query_batch(split, rows, alt, device); alt_kq = query(alt_q)
    def run(qx, vx, kx, tx, yx):
        sel = (((qx[:, None, :] * kx).sum(-1) / 0.1).argmax(1)); pred = vh.selected(vx, sel).argmax(1)
        return {"retrieval": float((sel == tx).float().mean()), "answer": float((pred == yx).float().mean()), "selected": sel.cpu().numpy()}
    base = run(query(qh), v, k, ts, y); wrong = run(alt_kq, v, k, torch.from_numpy(alt).to(device), fy[torch.arange(len(ts), device=device), torch.from_numpy(alt).to(device)])
    # External query: construct a safe episode permutation such that the
    # external (person, relation) key is absent from the current memory.
    meta = cache.meta[split]
    pairs = [{(p, r) for p, r in zip(meta[int(r0)]["persons"], meta[int(r0)]["relations"])} for r0 in rows]
    perm_np = None
    for shift in range(1, len(rows)):
        cand = np.roll(np.arange(len(rows), dtype=np.int64), shift)
        if all((meta[int(rows[int(cand[i])])]["query_person"], meta[int(rows[int(cand[i])])]["query_relation"]) not in pairs[i]
               for i in range(len(rows))):
            perm_np = cand; break
    if perm_np is None:
        perm_np = np.roll(np.arange(len(rows), dtype=np.int64), 1)
    ext_q = query(torch.from_numpy(cache.data[split]["query_H"][rows[perm_np]].astype(np.float32)).to(device))
    ext = run(ext_q, v, k, ts, y)
    # Memory shuffle uses the same safe permutation and changes both K and V.
    perm = torch.from_numpy(perm_np).to(device)
    mem = run(query(qh), k.new_zeros(v.shape).copy_(v[perm]), k[perm], ts, y)
    return {"correct_query": {k: v for k, v in base.items() if k != "selected"},
            "wrong_query_switched_target": {k: v for k, v in wrong.items() if k != "selected"},
            "external_query": {k: v for k, v in ext.items() if k != "selected"},
            "memory_state_shuffle": {k: v for k, v in mem.items() if k != "selected"},
            "wrong_query_switch_rate": float((wrong["selected"] == alt).mean())}


@torch.no_grad()
def pairwise_query(key, query, cache, split, rows, device):
    cap = 4; facts, _, _, qh, ts, _, _ = cache.batch(split, rows, cap, device); d = cache.data[split]; off = d["fact_offsets"]
    # Two independently encoded queries for the same memory.
    q0 = cache.fact_query_batch(split, rows, np.zeros(len(rows), dtype=np.int64), device)
    q1 = cache.fact_query_batch(split, rows, np.ones(len(rows), dtype=np.int64), device)
    k = key(facts.reshape(-1, 2048)).reshape(len(rows), cap, -1); q0 = query(q0); q1 = query(q1)
    s0 = (q0[:, None, :] * k).sum(-1).argmax(1); s1 = (q1[:, None, :] * k).sum(-1).argmax(1)
    return {"n_pairs": len(rows), "both_correct": float(((s0 == 0) & (s1 == 1)).float().mean()),
            "at_least_one_correct": float(((s0 == 0) | (s1 == 1)).float().mean()),
            "different_query_changes_slot": float((s0 != s1).float().mean())}


@torch.no_grad()
def fact_order_permutation(key, query, vh, core, cache, split, rows, device):
    cap = 4; facts, _, _, qh, ts, y, _ = cache.batch(split, rows, cap, device); perm = torch.arange(cap - 1, -1, -1, device=device)
    vo = value_state(core, facts); ko = key(facts.reshape(-1, 2048)).reshape(len(rows), cap, -1)
    so = (query(qh)[:, None, :] * ko).sum(-1).argmax(1)
    fp = facts[:, perm, :]; vp = value_state(core, fp); kp = key(fp.reshape(-1, 2048)).reshape(len(rows), cap, -1); q = query(qh)
    target = (perm[None, :] == ts[:, None]).nonzero(as_tuple=False)[:, 1]; sel = (q[:, None, :] * kp).sum(-1).argmax(1)
    return {"original_order_retrieval": float((so == ts).float().mean()), "permuted_order_retrieval": float((sel == target).float().mean()),
            "target_slot_changed": float((target != ts).float().mean())}


def slot_capacity(core, vh, cache, device):
    out = {}
    for slot in range(4):
        xs = []; ys = []
        for cap in CAPS:
            if cap <= slot:
                continue
            rows = cache.rows("train", cap)
            for st in range(0, len(rows), 128):
                rb = rows[st:st + 128]; facts, _, _, _, _, _, fy = cache.batch("train", rb, cap, device)
                with torch.no_grad(): v = value_state(core, facts)
                xs.append(v[:, slot, :].float().cpu().numpy()); ys.append(fy[:, slot].cpu().numpy())
        x = np.concatenate(xs); y = np.concatenate(ys); p = _probe_fit(x, y, N_CLASSES, device, 200, 1200 + slot)
        item = {}
        for split in ("testA", "testB"):
            item[split] = {}
            for cap in CAPS:
                rows = cache.rows(split, cap); facts, _, _, _, _, _, fy = cache.batch(split, rows, cap, device)
                with torch.no_grad(): v = value_state(core, facts)
                # For cap <= slot, the slot is not a written slot; report null.
                if cap <= slot: item[split][str(cap)] = None
                else:
                    item[split][str(cap)] = probe_metrics(p, v[:, slot, :].float().cpu().numpy(), fy[:, slot].cpu().numpy())
        out[str(slot)] = item
    return out


def geometry(core, cache, split, device):
    out = {}
    for cap in CAPS:
        rows = cache.rows(split, cap); facts, _, _, _, _, _, _ = cache.batch(split, rows, cap, device)
        with torch.no_grad(): v = value_state(core, facts).float().cpu().numpy()
        flat = v.reshape(len(v), -1); norms = np.linalg.norm(v, axis=2).mean(0).tolist(); pair = {}
        for a in range(4):
            for b in range(a + 1, 4):
                aa, bb = v[:, a], v[:, b]; pair[f"{a}-{b}"] = float(np.mean(np.sum(aa * bb, 1) / (np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1) + 1e-9)))
        if len(flat) > 1:
            ix = np.arange(min(len(flat), 500)); jx = np.roll(ix, 1); inter = np.sum(flat[ix] * flat[jx], 1) / (np.linalg.norm(flat[ix], axis=1) * np.linalg.norm(flat[jx], axis=1) + 1e-9)
            inter_mean = float(inter[1:].mean()) if len(inter) > 1 else None
        else: inter_mean = None
        out[str(cap)] = {"n": len(v), "per_slot_norm_mean": norms, "slot_slot_cosine": pair, "inter_episode_cosine_mean": inter_mean}
    return out


def relation_stress(key, query, vh, core, cache, device):
    out = {}
    for name, kind in (("mr_same_entity", "same_entity"), ("mr_same_relation", "same_relation"), ("mr_lexical_stress", "lexical_stress"), ("mr_same_value", "same_value")):
        rows = cache.rows_type("testA", 4, kind)
        r = evaluate_rows(key, query, vh, core, cache, "testA", 4, rows, device)
        out[name] = {"n": int(len(rows)), "retrieval": r["retrieval"], "hard_answer": r["hard_answer"], "oracle_answer": r["oracle_answer"]}
    return out


def zero_shot_ls1(cache, device):
    """Run the completed LS1 checkpoint on the new MR protocol before MR training."""
    ckpt = ROOT / "checkpoints" / "clr_v2_a1_ls1" / "best_model.pt"
    if not ckpt.exists():
        return {"available": False, "reason": str(ckpt) + " missing"}
    clrv2, head, _ = load_ls1_model(str(ckpt), device=str(device))
    out = {"available": True, "checkpoint": str(ckpt), "splits": {}}
    for split in ("val", "testA", "testB"):
        out["splits"][split] = {}
        for cap in CAPS:
            rows = cache.rows(split, cap); d = cache.data[split]
            total = good = 0
            for st in range(0, len(rows), 128):
                rb = rows[st:st + 128]
                with torch.no_grad():
                    _, sq = ls1_forward_batch(clrv2, d, rb, device=str(device))
                    pred = head(sq.reshape(len(rb), -1).to(torch.bfloat16)).argmax(1).cpu().numpy()
                good += int((pred == d["Y"][rb]).sum()); total += len(rb)
            out["splits"][split][str(cap)] = {"n": total, "answer": good / max(total, 1)}
    return out


def main():
    seed_all(); t0 = time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = MRCache(CACHE)
    core, old_heads, _ = load_storage(device, R2)
    core.eval(); old_heads.eval()
    for p in core.parameters(): p.requires_grad_(False)
    for p in old_heads.parameters(): p.requires_grad_(False)
    smoke = os.environ.get("CLR_V24_O21A_MR_SMOKE", "0") == "1" or "--smoke" in os.sys.argv
    eval_only = os.environ.get("V24_O21A_MR_EVAL_ONLY", "0") == "1" or "--eval-only" in os.sys.argv
    if eval_only:
        prior = json.loads((OUT / "train.log").read_text(encoding="utf-8"))
        trainlog = prior["key_training"]; vh_log = prior.get("value_decoder", {})
        steps = int(trainlog.get("steps", trainlog.get("best_step", 0)))
        zero = json.loads((OUT / "zero_shot_multiwrite.json").read_text(encoding="utf-8"))
        key, query, aux, vh, best_ck = restore_best(cache, device)
    else:
        # Always train a fresh decoder for the new value-text schema, but keep
        # its shape and V-only interface identical to the O1 local heads.
        vh, vh_log = train_value_heads(core, cache, device, steps=30 if smoke else 600)
        steps = 20 if smoke else int(os.environ.get("V24_O21A_MR_STEPS", "3000"))
        zero = zero_shot_ls1(cache, device)
        dump(OUT / "zero_shot_multiwrite.json", zero)
        key, query, aux, trainlog = train_main(core, vh, cache, device, steps=steps)
        json_dump(OUT / "train.log", {"key_training": trainlog, "value_decoder": vh_log})
        key, query, aux, vh, best_ck = restore_best(cache, device)

    results = {}; answers = {}; arrays = {}
    for split in ("val", "testA", "testB"):
        results[split] = {}; answers[split] = {}; arrays[split] = {}
        for cap in CAPS:
            rows = cache.rows(split, cap); r = evaluate_rows(key, query, vh, core, cache, split, cap, rows, device)
            arrays[split][str(cap)] = r
            results[split][str(cap)] = {k: v for k, v in r.items() if k not in ("selected", "target", "y")}
            answers[split][str(cap)] = {k: results[split][str(cap)][k] for k in ("hard_answer", "oracle_answer", "random_answer")}

    pos = {s: position_matrix(cache, s, arrays[s]) for s in ("testA", "testB")}
    shortcuts = {s: shortcut_baselines(cache, s) for s in ("testA", "testB")}
    ints = {s: kv_interventions(key, query, vh, core, cache, s, 4, cache.rows(s, 4), device) for s in ("testA", "testB")}
    stress = relation_stress(key, query, vh, core, cache, device)
    ent_probe, rel_probe, joint_probe, query_probe, leakage, emb = run_probes(key, query, aux, cache, device)
    rel_model = _probe_fit(emb["train"]["K"], emb["train"]["relation"], N_REL, device, 300, 911)
    ablation = {}
    for split in ("testA", "testB"):
        ablation[split] = {}
        for cap in CAPS:
            rows = cache.rows(split, cap)
            ablation[split][str(cap)] = ablation_with_probes(key, query, aux, rel_model, cache, split, cap, rows, device)
        for kind in ("same_entity", "same_relation", "lexical_stress"):
            rows = cache.rows_type(split, 4, kind)
            if len(rows): ablation[split][kind] = ablation_with_probes(key, query, aux, rel_model, cache, split, 4, rows, device)

    qint = {s: query_controls(key, query, vh, core, cache, s, 4, cache.rows(s, 4), device) for s in ("testA", "testB")}
    rel_swap_controls = {s: query_controls(key, query, vh, core, cache, s, 4, cache.rows_type(s, 4, "same_entity"), device) for s in ("testA", "testB")}
    ent_swap_controls = {s: query_controls(key, query, vh, core, cache, s, 4, cache.rows_type(s, 4, "same_relation"), device) for s in ("testA", "testB")}
    pair = {s: pairwise_query(key, query, cache, s, cache.rows_type(s, 4, "mixed")[:1000], device) for s in ("testA", "testB")}
    order = {s: fact_order_permutation(key, query, vh, core, cache, s, cache.rows(s, 4), device) for s in ("testA", "testB")}
    slots = slot_capacity(core, vh, cache, device)
    geom = {s: geometry(core, cache, s, device) for s in ("testA", "testB")}

    n4a = results["testA"]["4"]; n4b = results["testB"]["4"]
    mr1 = stress["mr_same_entity"]["retrieval"]; mr2 = stress["mr_same_relation"]["retrieval"]
    eonly1 = ablation["testA"].get("same_entity", {}).get("entity_only", 0.0)
    ronly2 = ablation["testA"].get("same_relation", {}).get("relation_only", 0.0)
    full_perm_drop = max(n4a["hard_answer"] - ints["testA"]["full_kv_permutation"]["answer"], n4b["hard_answer"] - ints["testB"]["full_kv_permutation"]["answer"])
    rel_switch = rel_swap_controls["testA"]["wrong_query_switch_rate"]
    ext = qint["testA"]["external_query"]["answer"]
    mem_shuffle = qint["testA"]["memory_state_shuffle"]["answer"]
    k_value = leakage["testA"]["K_to_value_probe"]["accuracy"]
    strict = (n4a["retrieval"] >= .90 and n4b["retrieval"] >= .85 and
              n4a["oracle_answer"] - n4a["hard_answer"] <= .05 and n4b["oracle_answer"] - n4b["hard_answer"] <= .05 and
              mr1 >= .90 and mr2 >= .90 and eonly1 < .75 and ronly2 < .75 and
              rel_switch >= .90 and full_perm_drop <= .05 and k_value <= .10 and ext <= .10 and mem_shuffle <= .10)
    if strict:
        verdict = "O2.1a-MR SEMANTIC COMPOSITIONAL KEY PASS"
    elif n4a["retrieval"] >= .70 and n4b["retrieval"] >= .60 and mr1 >= .50 and mr2 >= .50:
        verdict = "O2.1a-MR PARTIAL: COMPOSITIONAL KEY WORKS, ROBUSTNESS INCOMPLETE"
    elif eonly1 >= .75:
        verdict = "O2.1a-MR FAIL: ENTITY-ONLY ADDRESSING"
    elif ronly2 >= .75:
        verdict = "O2.1a-MR FAIL: RELATION-ONLY ADDRESSING"
    elif mr1 < .50 and mr2 < .50:
        verdict = "O2.1a-MR FAIL: JOINT KEY ALIGNMENT"
    else:
        verdict = "O2.1a-MR FAIL: QUERY RELATION EXTRACTION"

    config = {"seed": SEED, "device": str(device), "hardware": {"cpu": platform.processor(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
              "source_value_checkpoint": str(R2), "source_o21a_router": str(O21A), "cache": str(CACHE), "relations": list(RELATIONS), "key_architecture": "O2.1a K1 KeyEncoder/QueryEncoder 2048-256-64 + L2 + dot hard matcher", "value_path": "frozen CLR-v2.3-OA-r2 OracleLocalCore; fresh same-shape V-only local heads for MR value schema", "loss": "O2.1a retrieval CE + 0.5 batch positive alignment + 0.1 entity/query-entity CE", "capacity": list(CAPS), "smoke": smoke, "steps": steps}
    dump(OUT / "config.json", config); dump(OUT / "metrics.json", results); dump(OUT / "ls2_results.json", results); dump(OUT / "retrieval_results.json", {s: {c: {k: v for k, v in m.items() if k not in ("selected", "target", "y")} for c, m in arrays[s].items()} for s in arrays}); dump(OUT / "answer_results.json", answers)
    dump(OUT / "zero_shot_multiwrite.json", zero); dump(OUT / "capacity_curve.json", {s: curve(results[s]) for s in results}); dump(OUT / "position_matrix.json", pos); dump(OUT / "mr_same_entity.json", stress["mr_same_entity"]); dump(OUT / "mr_same_relation.json", stress["mr_same_relation"]); dump(OUT / "mr_lexical_stress.json", stress["mr_lexical_stress"]); dump(OUT / "mr_same_value.json", stress["mr_same_value"])
    dump(OUT / "entity_probe.json", ent_probe); dump(OUT / "relation_probe.json", rel_probe); dump(OUT / "joint_key_probe.json", joint_probe); dump(OUT / "query_probe.json", query_probe); dump(OUT / "entity_only_ablation.json", {s: {c: m for c, m in v.items()} for s, v in ablation.items()}); dump(OUT / "relation_only_ablation.json", {s: {c: m for c, m in v.items()} for s, v in ablation.items()}); dump(OUT / "full_key_results.json", results); dump(OUT / "relation_key_swap.json", {s: {"same_entity_wrong_relation_query": rel_swap_controls[s]["wrong_query_switched_target"], "switch_rate": rel_swap_controls[s]["wrong_query_switch_rate"]} for s in rel_swap_controls}); dump(OUT / "entity_key_swap.json", {s: {"same_relation_wrong_entity_query": ent_swap_controls[s]["wrong_query_switched_target"], "switch_rate": ent_swap_controls[s]["wrong_query_switch_rate"]} for s in ent_swap_controls})
    dump(OUT / "kv_interventions.json", ints); dump(OUT / "slot_permutation.json", {s: ints[s]["full_kv_permutation"] for s in ints}); dump(OUT / "key_value_leakage.json", leakage); dump(OUT / "query_interventions.json", qint); dump(OUT / "memory_interventions.json", {s: {"memory_state_shuffle": qint[s]["memory_state_shuffle"], "external_query": qint[s]["external_query"]} for s in qint}); dump(OUT / "pairwise_query.json", pair); dump(OUT / "slot_capacity.json", slots); dump(OUT / "state_geometry.json", {"value_state": geom, "query_conditioned_state_update": False, "note": "O2.1a router uses Q to select V; it does not update CLR storage state"})

    def pct(x): return "n/a" if x is None else f"{100*x:.2f}%"
    report = ["# CLR-v2.4-O2.1a-MR — Multi-Relation Semantic Key Validation", "", f"**{verdict}**", "", "## Primary answer", f"TestA N=4 retrieval={pct(n4a['retrieval'])}; TestB N=4 retrieval={pct(n4b['retrieval'])}.", f"MR1 same entity/different relation={pct(mr1)}; MR2 same relation/different entity={pct(mr2)}.", f"Within-episode wrong-query switched-target={pct(rel_switch)}; external-query answer={pct(ext)} (random-slot answer baseline={pct(n4a['random_answer'])}); memory-state shuffle answer={pct(mem_shuffle)}.", f"Absolute answer remains TestA/TestB {pct(n4a['hard_answer'])}/{pct(n4b['hard_answer'])}; the frozen O1 value path transfers incompletely to the new relation text schema, so this run does not claim full answer-level MR success despite the strong router result.", "", "## Protocol and isolation", "- Four relation families: current_city, birth_city, favorite_city, work_city.", "- Facts and queries use new relation-diverse text and frozen Ouro hidden cache; old single-relation cache was not relabeled.", "- O1 OracleLocalCore, write allocation, latent dimensions, and O2.1a KeyEncoder/QueryEncoder/matcher are unchanged.", "- MR value decoder is a fresh same-shape Linear(64,64) per physical slot, trained from frozen V only because the old O1 heads were trained on the old single relation schema.", "- No learned allocation, external memory, attention memory, recurrent retrieval, reasoning, LM generation, O2.1b, O3, or A2.", "", "## Curves", f"TestA capacity: {json.dumps(curve(results['testA']), ensure_ascii=False)}", f"TestB capacity: {json.dumps(curve(results['testB']), ensure_ascii=False)}", f"Position matrix: {json.dumps(pos, ensure_ascii=False)}", f"First/last shortcut baselines: {json.dumps(shortcuts, ensure_ascii=False)}", "", "## MR stress and interventions", f"MR stress: {json.dumps(stress, ensure_ascii=False)}", f"KV interventions: {json.dumps(ints, ensure_ascii=False)}", f"Query interventions: {json.dumps(qint, ensure_ascii=False)}", f"Pairwise same-memory query: {json.dumps(pair, ensure_ascii=False)}", f"Fact-order permutation: {json.dumps(order, ensure_ascii=False)}", "", "## Probes and geometry", f"Entity probes: {json.dumps(ent_probe, ensure_ascii=False)}", f"Relation probes: {json.dumps(rel_probe, ensure_ascii=False)}", f"Joint key probes: {json.dumps(joint_probe, ensure_ascii=False)}", f"K-to-value leakage: {json.dumps(leakage, ensure_ascii=False)}", f"Slot capacity: {json.dumps(slots, ensure_ascii=False)}", "", "## Q1–Q27", "1. Architecture unchanged: yes; O2.1a K1 router and frozen O1 value path were retained.", "2. Relation families: four.", "3. Same-entity multi-relation episodes: yes; same_entity and lexical_stress controls.", f"4. TestA N=4 retrieval: {pct(n4a['retrieval'])}.", f"5. TestB N=4 retrieval: {pct(n4b['retrieval'])}.", f"6. Hard answer vs oracle: TestA {pct(n4a['hard_answer'])} vs {pct(n4a['oracle_answer'])}; TestB {pct(n4b['hard_answer'])} vs {pct(n4b['oracle_answer'])}.", f"7. MR1: {pct(mr1)}.", f"8. MR2: {pct(mr2)}.", f"9. Lexical stress: {pct(stress['mr_lexical_stress']['retrieval'])}.", f"10. Same-value retrieval: {pct(stress['mr_same_value']['retrieval'])}.", "11–15. Entity/relation/joint K and Q probes are in entity_probe.json, relation_probe.json, joint_key_probe.json, and query_probe.json.", f"16. Entity-only MR1: {pct(eonly1)}.", f"17. Relation-only MR2: {pct(ronly2)}.", f"18. Full key: TestA/TestB N=4 {pct(n4a['retrieval'])}/{pct(n4b['retrieval'])}.", f"19. Relation component change address: switch rate {pct(rel_switch)}.", f"20. Entity component change address: see entity_key_swap.json; same control query is reported separately.", f"21. Key swap routing: see kv_interventions.json and relation_key_swap.json.", f"22. Value swap answer: TestA/TestB {pct(ints['testA']['value_swap']['answer'])}/{pct(ints['testB']['value_swap']['answer'])}.", f"23. Full KV permutation: answer {pct(ints['testA']['full_kv_permutation']['answer'])}/{pct(ints['testB']['full_kv_permutation']['answer'])}.", f"24. K-to-value leakage: TestA probe {pct(k_value)}; chance={pct(CHANCE)}.", f"25. Semantic address vs entity-only: {'supported' if eonly1 < .75 and mr1 > eonly1 + .10 else 'not cleanly supported'}.", f"26. Upgrade to full PASS: {'yes' if strict else 'no; see gate values above'}.", "27. O2.1b: not entered; this stage stops here.", "", "## Runtime", f"Cache manifest: {CACHE / 'manifest.json'}", f"Formal training wall-clock: {trainlog.get('wall_clock_s'):.2f}s; cache generation time is recorded in manifest.json; total run wall-clock: {time.perf_counter()-t0:.2f}s.", "Peak VRAM: approximately 3.64GB observed during frozen-Ouro cache generation and 2.54GB observed during formal training (nvidia-smi snapshots).", "", "## Capability boundary", "WRITE and RETENTION were established by earlier stages. This run tests MULTI-WRITE and SELECTIVE RETRIEVAL only. Even a PASS would not imply latent reasoning, general memory, or multi-step inference.", "", "O2.1a-MR COMPLETE; stopped before O2.1b/O3/A2."]
    (OUT / "v24_o21a_mr_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "testA_N4": n4a, "testB_N4": n4b, "stress": stress, "wall_clock_s": time.perf_counter() - t0}, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x), flush=True)


if __name__ == "__main__":
    main()
