# -*- coding: utf-8 -*-
"""CLR v2.4 O3b-G: learned binding-aware key, offline K-K only.

The original O2.1b KeyEncoder output is frozen.  This experiment learns a
small compositional binding head on top of that hidden-derived K.  Entity and
relation labels are used only by auxiliary training classifiers; inference
uses only the input K and never receives oracle labels.  No real memory
pipeline is reconnected.
"""
import copy
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import a1_v24_o3b_f as fdiag
from a1_v24_o3b import FactPool, MRCache, KeyEncoder, DIM
from a1_v24_o3b_diag import KeyBank, sample_pair
from a1_v24_o3b_e import pair_refs


OUT = Path(os.environ.get("V24_O3BG_OUT", r"D:/training/checkpoints/clr_v24_o3b_g_binding_key"))
CACHE_ROOT = Path(os.environ.get("V24_O3BG_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
O21B = Path(os.environ.get("V24_O3BG_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
O3BF_OUT = Path(os.environ.get("V24_O3BG_O3BF_OUT", r"D:/training/checkpoints/clr_v24_o3b_f_binding_diag"))
SEED = int(os.environ.get("V24_O3BG_SEED", "42"))
DEVICE_NAME = os.environ.get("V24_O3BG_DEVICE", "cuda")
SMOKE = os.environ.get("V24_O3BG_SMOKE", "0") == "1" or "--smoke" in os.sys.argv
STEPS = int(os.environ.get("V24_O3BG_STEPS", "3000"))
EVAL_EVERY = int(os.environ.get("V24_O3BG_EVAL_EVERY", "200"))
PROBE_STEPS = int(os.environ.get("V24_O3BG_PROBE_STEPS", "600"))
PAIR_TRAIN = 512 if SMOKE else 6000
PAIR_EVAL = 256 if SMOKE else 3000


def seed_all(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer, np.floating)): return x.item()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    if isinstance(x, Path): return str(x)
    return x


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def tensor(x, device):
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def g_pair_refs(bank, count, seed):
    """Training protocol: 25% P2 and 25% P3 of all negatives."""
    rng = np.random.default_rng(seed); npos = count // 2; nneg = count - npos
    pairs = []
    for _ in range(npos):
        a, b = sample_pair(bank, rng, "same_key"); pairs.append((a, b, 1, "P1_same_key"))
    neg_cats = (["same_entity_diff_relation"] * (nneg // 4) +
                ["same_relation_diff_entity"] * (nneg // 4))
    rest = nneg - len(neg_cats); other = ["same_value_diff_key", "fully_different", "lexical_hard_negative", "similar_entity_names"]
    neg_cats += [other[i % len(other)] for i in range(rest)]; rng.shuffle(neg_cats)
    names = {"same_entity_diff_relation": "P2_same_entity_diff_relation", "same_relation_diff_entity": "P3_same_relation_diff_entity", "same_value_diff_key": "P4_same_value_diff_key", "fully_different": "P5_fully_different", "lexical_hard_negative": "P6_lexical_hard_negative", "similar_entity_names": "P7_similar_entity_names"}
    for cat in neg_cats:
        a, b = sample_pair(bank, rng, cat); pairs.append((a, b, 0, names[cat]))
    rng.shuffle(pairs)
    ia = np.asarray([a["bank_idx"] for a, _, _, _ in pairs], dtype=np.int64); ib = np.asarray([b["bank_idx"] for _, b, _, _ in pairs], dtype=np.int64)
    return {"ka": bank.K[ia].astype(np.float32), "kb": bank.K[ib].astype(np.float32), "y": np.asarray([p[2] for p in pairs], dtype=np.int64), "categories": np.asarray([p[3] for p in pairs]), "ra": np.asarray([a["relation"] for a, _, _, _ in pairs], dtype=np.int64), "rb": np.asarray([b["relation"] for _, b, _, _ in pairs], dtype=np.int64), "idxa": ia, "idxb": ib, "pair_counts": Counter([p[3] for p in pairs]), "pair_meta": [(dict(a), dict(b)) for a, b, _, _ in pairs]}


class BindingKey(nn.Module):
    def __init__(self, variant, nentity):
        super().__init__(); self.variant = variant; self.in_dim = DIM
        self.branch_dim = 16 if variant == "G3_projected_tensor" else 32
        self.entity = nn.Linear(DIM, self.branch_dim); self.relation = nn.Linear(DIM, self.branch_dim)
        if variant == "G1_hadamard": self.compose = nn.Linear(self.branch_dim, DIM)
        elif variant.startswith("G2_residual"):
            self.lam = float(variant.rsplit("l", 1)[1].replace("p", "."))
            self.pe = nn.Linear(self.branch_dim, DIM); self.pr = nn.Linear(self.branch_dim, DIM); self.pb = nn.Linear(self.branch_dim, DIM)
        elif variant == "G3_projected_tensor": self.compose = nn.Linear(self.branch_dim * self.branch_dim, DIM)
        elif variant == "G4_learned_concat": self.compose = nn.Linear(self.branch_dim * 2, DIM)
        else: raise ValueError(variant)
        self.entity_classifier = nn.Linear(self.branch_dim, nentity)
        self.relation_classifier = nn.Linear(self.branch_dim, 4)

    def factors(self, x):
        return F.normalize(self.entity(x), dim=-1), F.normalize(self.relation(x), dim=-1)

    def forward(self, x):
        e, r = self.factors(x); b = e * r
        if self.variant == "G1_hadamard": z = self.compose(b)
        elif self.variant.startswith("G2_residual"): z = self.pe(e) + self.pr(r) + self.lam * self.pb(b)
        elif self.variant == "G3_projected_tensor": z = self.compose((e.unsqueeze(2) * r.unsqueeze(1)).flatten(1))
        else: z = self.compose(torch.cat([e, r], -1))
        return F.normalize(z, dim=-1), e, r


def score_pair(model, d, device):
    model.eval(); out = {"score": [], "k_a": [], "k_b": [], "e_a": [], "e_b": [], "r_a": [], "r_b": []}
    with torch.no_grad():
        for st in range(0, len(d["y"]), 1024):
            a = tensor(d["ka"][st:st + 1024], device); b = tensor(d["kb"][st:st + 1024], device)
            ka, ea, ra = model(a); kb, eb, rb = model(b)
            out["score"].append((ka * kb).sum(-1).cpu().numpy())
            for n, v in (("k_a", ka), ("k_b", kb), ("e_a", ea), ("e_b", eb), ("r_a", ra), ("r_b", rb)): out[n].append(v.cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def pair_record(d, s, name):
    th = fdiag.choose_threshold(d["y"], s)
    return fdiag.metric(d["y"], s, th, d["categories"]), th


def train_variant(train, val, variant, nentity, emap, device, steps):
    seed_all(SEED + sum(map(ord, variant))); model = BindingKey(variant, nentity).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01); rng = np.random.default_rng(SEED + 7300 + len(variant)); bs = 256 if not SMOKE else 128
    y = torch.from_numpy(train["y"].astype(np.float32)).to(device); rel_a = torch.from_numpy(train["ra"]).long().to(device); rel_b = torch.from_numpy(train["rb"]).long().to(device)
    ent_a = torch.tensor([emap[m[0]["person"]] for m in train["pair_meta"]], dtype=torch.long, device=device); ent_b = torch.tensor([emap[m[1]["person"]] for m in train["pair_meta"]], dtype=torch.long, device=device)
    hard = torch.from_numpy(np.isin(train["categories"], ["P2_same_entity_diff_relation", "P3_same_relation_diff_entity"])).to(device)
    p2 = torch.from_numpy(train["categories"] == "P2_same_entity_diff_relation").to(device); p3 = torch.from_numpy(train["categories"] == "P3_same_relation_diff_entity").to(device)
    logs = []; best = None; nsteps = 20 if SMOKE else steps; start = time.perf_counter()
    for step in range(1, nsteps + 1):
        ix = rng.integers(0, len(train["y"]), size=min(bs, len(train["y"])))
        ka, ea, ra = model(tensor(train["ka"][ix], device)); kb, eb, rb = model(tensor(train["kb"][ix], device)); sim = (ka * kb).sum(-1)
        loss_pair = F.binary_cross_entropy_with_logits(5.0 * sim, y[ix])
        loss_ent = .5 * (F.cross_entropy(model.entity_classifier(ea), ent_a[ix]) + F.cross_entropy(model.entity_classifier(eb), ent_b[ix]))
        loss_rel = .5 * (F.cross_entropy(model.relation_classifier(ra), rel_a[ix]) + F.cross_entropy(model.relation_classifier(rb), rel_b[ix]))
        hi = hard[ix]; loss_hard = F.softplus(5.0 * sim[hi]).mean() if hi.any() else sim.sum() * 0.0
        loss = loss_pair + .25 * loss_ent + .25 * loss_rel + .5 * loss_hard
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step % EVAL_EVERY == 0 or step == nsteps:
            vf = score_pair(model, val, device); vm, th = pair_record(val, vf["score"], variant); logs.append({"step": step, "loss": float(loss.detach()), "loss_pair": float(loss_pair.detach()), "loss_entity": float(loss_ent.detach()), "loss_relation": float(loss_rel.detach()), "loss_hard": float(loss_hard.detach()), "val": vm, "threshold": th, "batch_expected": {"P2": .25, "P3": .25}})
            if best is None or (vm["f1"], vm["balanced_accuracy"]) > (best["val"]["f1"], best["val"]["balanced_accuracy"]): best = {"state": copy.deepcopy(model.state_dict()), "step": step, "val": vm, "threshold": th}
    model.load_state_dict(best["state"]); ck = OUT / "checkpoints" / f"{variant}_best.pt"; ck.parent.mkdir(parents=True, exist_ok=True); torch.save({"model": model.state_dict(), "variant": variant, "best_step": best["step"], "threshold": best["threshold"], "params": sum(p.numel() for p in model.parameters()), "inference_uses_oracle_labels": False}, ck)
    return model, {"variant": variant, "best_step": best["step"], "best_val": best["val"], "threshold": best["threshold"], "steps": nsteps, "params": sum(p.numel() for p in model.parameters()), "wall_clock_s": time.perf_counter() - start, "logs": logs, "train_pair_counts": dict(train["pair_counts"])}


def probe_model(xtr, ytr, evals, nclass, device, shallow, name, steps):
    seed_all(SEED + 4100 + nclass + (17 if shallow else 0)); model = nn.Sequential(nn.Linear(xtr.shape[1], 64), nn.GELU(), nn.Linear(64, nclass)).to(device) if shallow else nn.Linear(xtr.shape[1], nclass).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01); rng = np.random.default_rng(SEED + 5100 + nclass); x = tensor(xtr, device); y = torch.from_numpy(ytr.astype(np.int64)).to(device); bs = min(512, len(y)); nsteps = 20 if SMOKE else steps
    for _ in range(nsteps):
        ix = rng.integers(0, len(ytr), size=bs); loss = F.cross_entropy(model(x[ix]), y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    (OUT / "probes").mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "nclass": nclass, "shallow": shallow}, OUT / "probes" / f"{name}.pt")
    out = {"params": sum(p.numel() for p in model.parameters()), "shallow": shallow}
    for s, (xx, yy) in evals.items():
        m = yy >= 0
        if not m.any(): out[s] = {"accuracy": None, "known_fraction": 0.0, "n_known": 0}; continue
        with torch.no_grad(): lg = model(tensor(xx[m], device)).cpu().numpy()
        out[s] = {**fdiag.multiclass_stats(yy[m], lg), "known_fraction": float(m.mean()), "n_known": int(m.sum())}
    return model, out


def variant_probes(model, banks, device, maps):
    feats = {}
    for s, bank in banks.items():
        with torch.no_grad(): k, e, r = model(tensor(bank.K, device)); feats[s] = {"K": k.cpu().numpy(), "E": e.cpu().numpy(), "R": r.cpu().numpy()}
    pmap, nentity = maps; yent = {}; yrel = {}; yjoint = {}
    for s, bank in banks.items():
        yent[s] = np.asarray([pmap.get(r["person"], -1) for r in bank.refs], dtype=np.int64); yrel[s] = np.asarray([int(r["relation"]) for r in bank.refs], dtype=np.int64); yjoint[s] = np.asarray([pmap.get(r["person"], -1) * 4 + int(r["relation"]) if r["person"] in pmap else -1 for r in bank.refs], dtype=np.int64)
    out = {"testB_note": "TestB unseen-person; entity/joint closed-world probes are N/A.", "representations": {}}
    for rep in ("K", "E", "R"):
        out["representations"][rep] = {}
        xtr = feats["train"][rep]
        for target, ys, nc in (("entity", yent, nentity), ("relation", yrel, 4), ("joint", yjoint, nentity * 4)):
            if rep == "E" and target == "relation": pass
            lin, lr = probe_model(xtr, ys["train"], {s: (feats[s][rep], ys[s]) for s in banks}, nc, device, False, f"{model.variant}_{rep}_{target}_linear", PROBE_STEPS)
            mlp, mr = probe_model(xtr, ys["train"], {s: (feats[s][rep], ys[s]) for s in banks}, nc, device, True, f"{model.variant}_{rep}_{target}_mlp", PROBE_STEPS)
            out["representations"][rep][target] = {"linear": lr, "mlp": mr}
    # E->entity plus R->relation exact factorized reconstruction.
    eprobe = probe_model(xtr=feats["train"]["E"], ytr=yent["train"], evals={s: (feats[s]["E"], yent[s]) for s in banks}, nclass=nentity, device=device, shallow=False, name=f"{model.variant}_factor_entity", steps=PROBE_STEPS)[0]
    rprobe = probe_model(xtr=feats["train"]["R"], ytr=yrel["train"], evals={s: (feats[s]["R"], yrel[s]) for s in banks}, nclass=4, device=device, shallow=False, name=f"{model.variant}_factor_relation", steps=PROBE_STEPS)[0]
    fac = {}
    for s in banks:
        known = yjoint[s] >= 0
        if not known.any(): fac[s] = {"accuracy": None, "known_fraction": 0.0}; continue
        with torch.no_grad(): pe = eprobe(tensor(feats[s]["E"][known], device)).argmax(1).cpu().numpy(); pr = rprobe(tensor(feats[s]["R"][known], device)).argmax(1).cpu().numpy()
        fac[s] = {"accuracy": float(np.mean((pe == yent[s][known]) & (pr == yrel[s][known]))), "entity_accuracy": float(np.mean(pe == yent[s][known])), "relation_accuracy": float(np.mean(pr == yrel[s][known])), "known_fraction": float(known.mean()), "n_known": int(known.sum())}
    out["factorized_reconstruction"] = fac
    return out, feats


def qstats(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x): return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()), "p10": float(np.quantile(x, .1)), "p25": float(np.quantile(x, .25)), "p50": float(np.quantile(x, .5)), "p75": float(np.quantile(x, .75)), "p90": float(np.quantile(x, .9))}


def overlap(a, b):
    a = np.asarray(a); b = np.asarray(b); lo = min(float(a.min()), float(b.min())); hi = max(float(a.max()), float(b.max()))
    h1, _ = np.histogram(a, bins=np.linspace(lo, hi, 101)); h2, _ = np.histogram(b, bins=np.linspace(lo, hi, 101)); h1 = h1 / max(h1.sum(), 1); h2 = h2 / max(h2.sum(), 1); return float(np.minimum(h1, h2).sum())


def geometry(d, scores):
    out = {}
    for cat in sorted(set(d["categories"].tolist())):
        m = d["categories"] == cat; out[cat] = qstats(scores[m])
    p1 = scores[d["categories"] == "P1_same_key"]; p2 = scores[d["categories"] == "P2_same_entity_diff_relation"]; p3 = scores[d["categories"] == "P3_same_relation_diff_entity"]
    out["overlap_P1_P2"] = overlap(p1, p2); out["overlap_P1_P3"] = overlap(p1, p3); return out


def retrieval(bank, karr):
    eps = [ep for ep in bank.pool.episodes if len(ep) == 4]; refs = {(r["episode"], r["pos"]): r for r in bank.refs}; errors = []; margins = []; p2m = []; p3m = []; groups = Counter(); total = correct = 0
    for ep in eps:
        for target0 in ep:
            target = refs[(target0["episode"], target0["pos"])]
            cand = [refs[(x["episode"], x["pos"])] for x in ep]; q = karr[target["bank_idx"]]; ks = karr[[x["bank_idx"] for x in cand]]; sc = ks @ q; j = int(np.argmax(sc)); ok = j == target["pos"]; total += 1; correct += int(ok)
            wrong = [i for i in range(len(cand)) if i != target["pos"]]; margins.append(float(sc[target["pos"]] - max(sc[wrong])))
            p2 = [i for i, x in enumerate(cand) if i != target["pos"] and x["person"] == target["person"] and x["relation"] != target["relation"]]; p3 = [i for i, x in enumerate(cand) if i != target["pos"] and x["person"] != target["person"] and x["relation"] == target["relation"]]
            if p2: p2m.append(float(sc[target["pos"]] - max(sc[p2])))
            if p3: p3m.append(float(sc[target["pos"]] - max(sc[p3])))
            if not ok:
                pred = cand[j]; same_e = pred["person"] == target["person"]; same_r = pred["relation"] == target["relation"]; kind = "E+_R-" if same_e and not same_r else ("E-_R+" if (not same_e and same_r) else "E-_R-"); groups[kind] += 1
                if len(errors) < 250: errors.append({"kind": kind, "true_pos": int(target["pos"]), "pred_pos": int(j), "true_person": target["person"], "true_relation": int(target["relation"]), "pred_person": pred["person"], "pred_relation": int(pred["relation"]), "margin": margins[-1]})
    return {"n": total, "correct": correct, "accuracy": float(correct / max(total, 1)), "error_counts": dict(groups), "error_fractions": {k: float(v / max(sum(groups.values()), 1)) for k, v in groups.items()}, "errors": errors, "margin": {"all_negatives": {**qstats(margins), "negative_rate": float(np.mean(np.asarray(margins) < 0))}, "same_entity_wrong_relation": {**qstats(p2m), "negative_rate": float(np.mean(np.asarray(p2m) < 0)) if p2m else None}, "wrong_entity_same_relation": {**qstats(p3m), "negative_rate": float(np.mean(np.asarray(p3m) < 0)) if p3m else None}}}


def same_entity_retrieval(bank, karr, unique=False):
    groups = defaultdict(list)
    for r in bank.refs: groups[r["person"]].append(r)
    pred = []; true = []
    for rs in groups.values():
        for r in rs:
            cand = [x for x in rs if x["bank_idx"] != r["bank_idx"]]
            if unique:
                one = {}
                for x in cand: one.setdefault(int(x["relation"]), x)
                cand = list(one.values())
            if len({int(x["relation"]) for x in cand}) < 2: continue
            sc = karr[[x["bank_idx"] for x in cand]] @ karr[r["bank_idx"]]; pred.append(int(cand[int(np.argmax(sc))]["relation"])); true.append(int(r["relation"]))
    return {"n": len(true), "accuracy": float(np.mean(np.asarray(pred) == np.asarray(true))) if true else None}


def collision(bank, karr):
    by = defaultdict(dict)
    for r in bank.refs: by[r["person"]][int(r["relation"])] = r
    chosen = next((v for v in by.values() if len(v) >= 4), None)
    if chosen is None: return {"available": False}
    rels = sorted(chosen)[:4]; idx = [chosen[x]["bank_idx"] for x in rels]; mat = karr[idx] @ karr[idx].T; off = [float(mat[i, j]) for i in range(4) for j in range(4) if i != j]
    return {"available": True, "relations": rels, "matrix": mat, "offdiag_mean": float(np.mean(off)), "offdiag_min": float(np.min(off)), "offdiag_max": float(np.max(off))}


def main():
    seed_all(); OUT.mkdir(parents=True, exist_ok=True); start = time.perf_counter(); device = torch.device("cuda" if DEVICE_NAME == "cuda" and torch.cuda.is_available() else "cpu")
    cache = MRCache(CACHE_ROOT); pools = {s: FactPool(cache, s) for s in ("train", "val", "testA", "testB")}
    kc = torch.load(O21B, map_location="cpu", weights_only=False); base_key = KeyEncoder(DIM).to(device); base_key.load_state_dict(kc["key"], strict=True); base_key.eval()
    for p in base_key.parameters(): p.requires_grad_(False)
    banks = {s: KeyBank(pools[s], base_key, device) for s in pools}; exact = {"train": pair_refs(banks["train"], PAIR_TRAIN, SEED + 1), "val": pair_refs(banks["val"], PAIR_EVAL, SEED + 2), "testA": pair_refs(banks["testA"], PAIR_EVAL, SEED + 3), "testB": pair_refs(banks["testB"], PAIR_EVAL, SEED + 4)}; train = g_pair_refs(banks["train"], PAIR_TRAIN, SEED + 101); val, ta, tb = exact["val"], exact["testA"], exact["testB"]
    raw = {s: (d["ka"] * d["kb"]).sum(1) for s, d in exact.items()}; raw_th = fdiag.choose_threshold(val["y"], raw["val"]); baseline = {"protocol": "exact O3b-D/E/F K-K; q=K_a, k=K_b", "threshold_val": raw_th, "val": fdiag.metric(val["y"], raw["val"], raw_th, val["categories"]), "testA": fdiag.metric(ta["y"], raw["testA"], raw_th, ta["categories"]), "testB": fdiag.metric(tb["y"], raw["testB"], raw_th, tb["categories"])}
    prior = None
    if (O3BF_OUT / "kk_metric_addendum.json").exists():
        prior = json.loads((O3BF_OUT / "kk_metric_addendum.json").read_text(encoding="utf-8")); baseline["prior_kk_addendum_testA_f1"] = prior["cosine"]["testA"]["f1"]; baseline["prior_kk_addendum_testB_f1"] = prior["cosine"]["testB"]["f1"]
    dump(OUT / "baseline_reproduction.json", baseline); expected = {"testA_f1": .8843853820598006, "testB_f1": .8655913978494623}; baseline_ok = abs(baseline["testA"]["f1"] - expected["testA_f1"]) <= .02 and abs(baseline["testB"]["f1"] - expected["testB_f1"]) <= .02
    dump(OUT / "pair_dataset_stats.json", {"train_exact": {"n": len(exact["train"]["y"]), "counts": dict(exact["train"]["pair_counts"])}, "train_hard": {"n": len(train["y"]), "counts": dict(train["pair_counts"])}, "val_test": {s: {"n": len(exact[s]["y"]), "counts": dict(exact[s]["pair_counts"])} for s in ("val", "testA", "testB")}, "sampling": {"P2_negative_fraction": .25, "P3_negative_fraction": .25, "same_episode_pairing": False}})
    if not baseline_ok:
        dump(OUT / "gate_evaluation.json", {"baseline_reproduced": False, "verdict": "STOP: G0 baseline reproduction failed", "reconnect": False, "o3c": False}); (OUT / "v24_o3b_g_binding_key_report.md").write_text("# CLR v2.4 O3b-G\n\n**STOP: G0 baseline reproduction failed.**\n", encoding="utf-8"); return
    pmap = {p: i for i, p in enumerate(sorted({r["person"] for r in banks["train"].refs}))}; nentity = len(pmap); variants = ["G1_hadamard", "G2_residual_l0p5", "G2_residual_l1", "G2_residual_l2", "G3_projected_tensor", "G4_learned_concat"]; results = {}; all_probe = {}; all_retr = {}; all_margin = {}; all_geom = {}; collisions = {}; best_models = {}
    for variant in variants:
        model, rec = train_variant(train, val, variant, nentity, pmap, device, STEPS); results[variant] = rec; probes, feats = variant_probes(model, banks, device, (pmap, nentity)); all_probe[variant] = probes; scores = {s: score_pair(model, exact[s], device)["score"] for s in ("val", "testA", "testB")}; th = fdiag.choose_threshold(val["y"], scores["val"]); results[variant]["testA"] = fdiag.metric(ta["y"], scores["testA"], th, ta["categories"]); results[variant]["testB"] = fdiag.metric(tb["y"], scores["testB"], th, tb["categories"]); results[variant]["threshold_val"] = th; results[variant]["geometry"] = {s: geometry(exact[s], scores[s]) for s in ("testA", "testB")}; results[variant]["same_entity_relation"] = {s: same_entity_retrieval(banks[s], feats[s]["K"], False) for s in ("testA", "testB")}; results[variant]["entity_fixed_relation"] = {s: same_entity_retrieval(banks[s], feats[s]["K"], True) for s in ("testA", "testB")}; all_retr[variant] = {s: retrieval(banks[s], feats[s]["K"]) for s in ("testA", "testB")}; all_margin[variant] = {s: all_retr[variant][s]["margin"] for s in ("testA", "testB")}; all_geom[variant] = results[variant]["geometry"]; collisions[variant] = {s: collision(banks[s], feats[s]["K"]) for s in ("testA", "testB")}; best_models[variant] = {"params": rec["params"], "checkpoint": str(OUT / "checkpoints" / f"{variant}_best.pt")}; torch.save({"model": model.state_dict(), "variant": variant, "params": rec["params"], "inference_uses_oracle_labels": False}, OUT / "checkpoints" / f"{variant}_final.pt"); del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    best_name = max(variants, key=lambda n: (results[n]["best_val"]["f1"], results[n]["best_val"]["balanced_accuracy"])); best_test_name = max(variants, key=lambda n: results[n]["testB"]["f1"]); best = results[best_name]; explicit = {}
    if (O3BF_OUT / "explicit_binding_results.json").exists(): explicit = json.loads((O3BF_OUT / "explicit_binding_results.json").read_text(encoding="utf-8"))
    dump(OUT / "variant_results.json", results); dump(OUT / "probe_results.json", all_probe); dump(OUT / "retrieval_error_decomposition.json", all_retr); dump(OUT / "margin_analysis.json", all_margin); dump(OUT / "binding_geometry.json", all_geom); dump(OUT / "collision_matrices.json", collisions); dump(OUT / "explicit_binding_results.json", explicit)
    p = all_probe[best_name]; best_k = p["representations"]["K"]; eplus = all_retr[best_name]["testB"]["error_fractions"].get("E+_R-", 0.0); gate = {"baseline_reproduced": baseline_ok, "selection": "best variant selected by validation F1; TestB not used for selection", "best_variant": best_name, "best_testB_variant_diagnostic": best_test_name, "testB_f1_ge_95": best["testB"]["f1"] >= .95, "testB_recall_ge_95": best["testB"]["recall"] >= .95, "testB_P2_ge_95": fdiag.per_category(best["testB"], "P2_same_entity_diff_relation") >= .95, "same_entity_relation_ge_90": best["same_entity_relation"]["testB"]["accuracy"] >= .90, "entity_fixed_relation_ge_90": best["entity_fixed_relation"]["testB"]["accuracy"] >= .90, "joint_linear_improved": (best_k["joint"]["linear"]["testA"]["accuracy"] or 0) > .09391080617495712, "joint_mlp_improved": (best_k["joint"]["mlp"]["testA"]["accuracy"] or 0) > .056174957118353346, "eplus_rminus_error_fraction": eplus, "verdict": None, "reconnect": False, "o3c": False, "repeated_v_update_contamination": "independent unresolved issue"}
    all_pass = all(gate[k] for k in ("testB_f1_ge_95", "testB_recall_ge_95", "testB_P2_ge_95", "same_entity_relation_ge_90", "entity_fixed_relation_ge_90", "joint_linear_improved", "joint_mlp_improved"))
    improved = any(results[n]["testB"]["f1"] > baseline["testB"]["f1"] + .02 and fdiag.per_category(results[n]["testB"], "P2_same_entity_diff_relation") > fdiag.per_category(baseline["testB"], "P2_same_entity_diff_relation") for n in variants)
    gate["partial_binding_recovery"] = bool(improved); gate["verdict"] = "O3b-G PASS: LEARNED COMPOSITIONAL ADDRESS BINDING RECOVERED" if all_pass else "O3b-G FAIL: LEARNED BINDING INDUCTIVE BIAS INSUFFICIENT"; gate["secondary_diagnosis"] = "PARTIAL BINDING RECOVERY" if (not all_pass and improved) else None; dump(OUT / "gate_evaluation.json", gate)
    sha = "NO_GIT_REPO"; diff = "NO_GIT_REPO"
    try: sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(Path.cwd()), text=True).strip(); diff = subprocess.check_output(["git", "diff", "--stat"], cwd=str(Path.cwd()), text=True).strip()
    except Exception: pass
    cfg = {"phase": "CLR-v2.4-O3b-G", "seed": SEED, "git_sha": sha, "device": str(device), "cache": str(CACHE_ROOT), "cache_reused": True, "frozen": ["backbone/Ouro", "V/ValueEncoder/GRU", "allocator", "memory write/update", "downstream answer head", "original O2.1b KeyEncoder"], "trainable": ["small learned entity/relation binding heads", "auxiliary train-only entity/relation classifiers"], "variants": variants, "loss": "L_pair + 0.25 L_entity + 0.25 L_relation + 0.5 L_hard; labels not used at inference", "hard_negative_sampling": {"P2": .25, "P3": .25}, "pair_sizes": {"train": len(train["y"]), "val": len(val["y"]), "testA": len(ta["y"]), "testB": len(tb["y"])}, "best_variant": best_name, "best_testB_variant_diagnostic": best_test_name, "peak_vram": {"allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None, "reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None}, "wall_clock_s": time.perf_counter() - start, "verdict": gate["verdict"], "reconnect": False, "o3c": False, "repeated_v_update_contamination": "independent unresolved issue"}; dump(OUT / "config.json", cfg); dump(OUT / "train_log.json", {"variants": {n: results[n]["logs"] for n in variants}, "best_variant": best_name, "best_testB_variant_diagnostic": best_test_name})
    def pct(x): return "n/a" if x is None else f"{100*x:.2f}%"
    rows = ["# CLR v2.4 — O3b-G: Learned Binding-Aware Key", "", f"**{gate['verdict']}**", "", "## Executive Summary", f"G0 K-K cosine reproduced: TestA F1={pct(baseline['testA']['f1'])}, TestB F1={pct(baseline['testB']['f1'])}. Validation-selected best variant={best_name}; diagnostic best TestB variant={best_test_name}. No oracle labels are used at inference; no real write/read reconnect was run.", "", "## Protocol Correction / Boundary", "This is the exact O3b-D/E/F K-K protocol: q=K_a and k=K_b. Original KeyEncoder, backbone, V/GRU, allocator, memory and downstream head remain frozen. Auxiliary entity/relation labels are training-only.", "", "## Baseline Reproduction", f"TestA/TestB F1={pct(baseline['testA']['f1'])}/{pct(baseline['testB']['f1'])}; P1={pct(fdiag.per_category(baseline['testA'], 'P1_same_key'))}/{pct(fdiag.per_category(baseline['testB'], 'P1_same_key'))}; P2={pct(fdiag.per_category(baseline['testA'], 'P2_same_entity_diff_relation'))}/{pct(fdiag.per_category(baseline['testB'], 'P2_same_entity_diff_relation'))}.", "", "## Variant Results", "| Variant | Params | TestA F1 | TestB F1 | TestA recall | TestB recall | TestA P2 | TestB P2 | same-E rel B | entity-fixed B |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n in variants:
        r = results[n]; rows.append(f"| {n} | {r['params']} | {pct(r['testA']['f1'])} | {pct(r['testB']['f1'])} | {pct(r['testA']['recall'])} | {pct(r['testB']['recall'])} | {pct(fdiag.per_category(r['testA'], 'P2_same_entity_diff_relation'))} | {pct(fdiag.per_category(r['testB'], 'P2_same_entity_diff_relation'))} | {pct(r['same_entity_relation']['testB']['accuracy'])} | {pct(r['entity_fixed_relation']['testB']['accuracy'])} |")
    rows += ["", "## Probes", f"Best {best_name}: K→joint linear={pct(best_k['joint']['linear']['testA']['accuracy'])}, K→joint MLP={pct(best_k['joint']['mlp']['testA']['accuracy'])}; E→entity={pct(p['representations']['E']['entity']['linear']['testA']['accuracy'])}, E→relation={pct(p['representations']['E']['relation']['linear']['testA']['accuracy'])}; R→entity={pct(p['representations']['R']['entity']['linear']['testA']['accuracy'])}, R→relation={pct(p['representations']['R']['relation']['linear']['testA']['accuracy'])}; full K→entity={pct(best_k['entity']['linear']['testA']['accuracy'])}, full K→relation={pct(best_k['relation']['linear']['testA']['accuracy'])}.", "", "## Explicit Controls", "Explicit concat/Hadamard/Kronecker results are reused from O3b-F as symbolic offline upper bounds; they are not learned-key or reconnect results.", "", "## Geometry, Margins and Error Audit", f"Best variant E+/R− error fraction on TestB episode retrieval={pct(eplus)}. Full collision, P1/P2/P3 distributions, margin quantiles and error cases are in JSON artifacts.", "", "## Gate", json.dumps(gate, ensure_ascii=False), "", "## Compute", f"Git SHA={sha}; cache reused; device={device}; wall-clock={cfg['wall_clock_s']:.2f}s; peak VRAM={cfg['peak_vram']}; single seed limitation.", "", "## Reconnect / O3c", "Reconnect: NO. O3c: NO. Do not design O3b-H.", "", "## Repeated V Update Contamination", "Repeated V update contamination remains an independent unresolved issue.", "", "## Final Interpretation", "O3b-G tests learned compositional key binding only. A PASS would show stable learned `(entity, relation)` addressing under the offline K-K protocol; it would not establish multi-fact memory or reasoning."]
    (OUT / "v24_o3b_g_binding_key_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8"); print(json.dumps({"verdict": gate["verdict"], "best_variant": best_name, "best_testB_variant": best_test_name, "best_testA_f1": best["testA"]["f1"], "best_testB_f1": best["testB"]["f1"], "best_testB_P2": fdiag.per_category(best["testB"], "P2_same_entity_diff_relation"), "reconnect": False, "o3c": False, "wall_clock_s": cfg["wall_clock_s"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
