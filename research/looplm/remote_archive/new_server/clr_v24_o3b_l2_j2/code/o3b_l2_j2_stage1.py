# -*- coding: utf-8 -*-
"""CLR-v2.4 O3b-L2 Stage 0/1 measurement calibration; frozen weights only."""
import csv
import json
import math
import os
import pickle
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# The frozen checkpoint contains NumPy pickle globals from a newer NumPy.
# Remap globals only during deserialization; checkpoint bytes are untouched.
class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


_pm = types.ModuleType("pickle_compat")
_pm.Unpickler = _NumpyCoreCompatUnpickler
_pm.Pickler = pickle.Pickler
_pm.dump = pickle.dump
_pm.dumps = pickle.dumps
_pm.load = pickle.load
_pm.loads = pickle.loads
_torch_load = torch.load


def _compat_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is False and "pickle_module" not in kwargs:
        kwargs["pickle_module"] = _pm
    return _torch_load(*args, **kwargs)


torch.load = _compat_torch_load

import a1_v24_o3b_j as j
from a1_v24_o3b import MRCache, KeyEncoder, DIM


ROOT = Path("/root/private_data/clr_v24_o3b_l2_j2")
OUT = ROOT / "sanity"
FEAT = ROOT / "features"
LOGDIR = ROOT / "logs"
OLD = Path("/root/private_data/clr_v24_o3b_l/output")
CACHE = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache")
HOUT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_h_entity_carrier_diag")
CKPT = Path("/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21b/b1_best.pt")
SPLITS = ("train", "val", "testA", "testB")
SEEDS = [11, 23, 37, 53, 71]
MILESTONES = [20, 50, 100, 180, 300, 500, 750, 1000]
TASK_OFFSET = {"entity": 1, "relation": 2, "joint": 3}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer, np.floating)): return x.item()
    if isinstance(x, Path): return str(x)
    return x


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def ci(values, seed=991, draws=20000):
    a = np.asarray(values, dtype=np.float64); a = a[np.isfinite(a)]
    if not len(a): return {"mean": None, "std": None, "t95_low": None, "t95_high": None, "boot95_low": None, "boot95_high": None, "n": 0}
    mean = float(a.mean()); sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(a), 1.96)
    if len(a) == 1: lo = hi = mean
    else: lo, hi = mean - tcrit * sd / math.sqrt(len(a)), mean + tcrit * sd / math.sqrt(len(a))
    if len(a) == 1: blo = bhi = mean
    else:
        rng = np.random.default_rng(seed)
        bs = a[rng.integers(0, len(a), size=(draws, len(a)))].mean(1)
        blo, bhi = np.quantile(bs, [.025, .975])
    return {"mean": mean, "std": sd, "t95_low": float(lo), "t95_high": float(hi), "boot95_low": float(blo), "boot95_high": float(bhi), "n": int(len(a))}


def accuracy(logits, y):
    mask = y >= 0
    if not bool(mask.any()): return None
    return float((logits[mask].argmax(-1) == y[mask]).float().mean().item())


def standardize(reps):
    out, stats = {}, {}
    for name, xs in reps.items():
        mu = np.asarray(xs["train"], dtype=np.float64).mean(0)
        sd = np.asarray(xs["train"], dtype=np.float64).std(0)
        low = sd < 1e-8
        sd[low] = 1.0
        out[name] = {s: ((np.asarray(xs[s], dtype=np.float64) - mu) / sd).astype(np.float32) for s in SPLITS}
        stats[name] = {"mu": mu.tolist(), "sd": sd.tolist(), "zero_or_tiny_std_dims": int(low.sum()), "train_only": True}
    return out, stats


def feature_conditioning(x):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(0); xc = x - mu
    cov = xc.T @ xc / max(1, len(x) - 1)
    eig = np.linalg.eigvalsh(cov)
    pos = eig[eig > max(float(eig.max()) * 1e-10, 1e-12)]
    p = pos / pos.sum()
    diag = np.diag(cov); sd = np.sqrt(np.maximum(diag, 0))
    return {"n": int(len(x)), "d": int(x.shape[1]), "mean_abs": float(np.abs(mu).mean()),
            "feature_norm_mean": float(np.linalg.norm(x, axis=1).mean()), "feature_norm_std": float(np.linalg.norm(x, axis=1).std()),
            "std_min": float(sd.min()), "std_max": float(sd.max()), "std_max_min_ratio": float(sd.max()/max(sd[sd>1e-12].min(), 1e-30)),
            "cov_condition": float(eig.max()/max(pos.min(), 1e-30)), "effective_rank": float(np.exp(-np.sum(p*np.log(np.maximum(p,1e-30))))),
            "participation_ratio": float((eig.sum()**2)/max((eig**2).sum(),1e-30)), "top_pc_variance_fraction": float(eig[-1]/max(eig.sum(),1e-30)),
            "spectrum_desc": eig[::-1].tolist()}


def probe_curves(reps, labels, nclass, seeds, task, norm_name):
    names = list(reps)
    y0 = np.asarray(labels["train"], dtype=np.int64); mask = y0 >= 0
    ytr_np = y0[mask]
    xnp = np.stack([np.asarray(reps[n]["train"], dtype=np.float32)[mask] for n in names])
    nr, n, d = xnp.shape; ns = len(seeds); nm = nr * ns
    X = torch.from_numpy(np.tile(xnp, (ns,1,1))).to(DEVICE)
    Y = torch.from_numpy(ytr_np).long().to(DEVICE)
    weight = nn.Parameter(torch.empty(nm, nclass, d, device=DEVICE))
    bias = nn.Parameter(torch.empty(nm, nclass, device=DEVICE))
    bound = 1.0/math.sqrt(d)
    with torch.no_grad():
        for si, sd in enumerate(seeds):
            torch.manual_seed(int(sd)+TASK_OFFSET[task]*1009)
            weight[si*nr:(si+1)*nr].uniform_(-bound,bound)
            bias[si*nr:(si+1)*nr].uniform_(-bound,bound)
    opt = torch.optim.AdamW([weight,bias],lr=3e-4,weight_decay=.01)
    rng = np.random.default_rng(42+913+TASK_OFFSET[task]); bs=min(1024,n)
    ev = {}
    for split in SPLITS:
        yy = torch.from_numpy(np.asarray(labels[split],dtype=np.int64)).long().to(DEVICE)
        xx = np.stack([np.asarray(reps[nm_][split],dtype=np.float32) for nm_ in names])
        ev[split] = (torch.from_numpy(np.tile(xx,(ns,1,1))).to(DEVICE), yy)
    rows=[]; snapshots={}
    for step in range(1,max(MILESTONES)+1):
        ix = torch.as_tensor(rng.integers(0,n,size=bs),dtype=torch.long,device=DEVICE)
        logits = torch.bmm(X[:,ix,:],weight.transpose(1,2))+bias[:,None,:]
        target=Y[ix][None,:].expand(nm,-1)
        loss=F.cross_entropy(logits.reshape(nm*bs,nclass),target.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        gnorm=float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in (weight,bias))).item())
        opt.step()
        if step not in MILESTONES: continue
        with torch.no_grad():
            tr_logits=torch.bmm(X,weight.transpose(1,2))+bias[:,None,:]
            tr_acc=(tr_logits.argmax(-1)==Y[None,:]).float().mean(1).cpu().numpy()
            wnorm=float(torch.sqrt((weight**2).sum(1).mean()).item())
            split_acc={}
            for split,(xe,ye) in ev.items():
                # Chunk eval to avoid an unnecessarily large [models,samples,classes] temporary.
                correct=np.zeros(nm,dtype=np.int64); total=np.zeros(nm,dtype=np.int64)
                for st in range(0,xe.shape[1],512):
                    en=min(st+512,xe.shape[1]); lg=torch.bmm(xe[:,st:en],weight.transpose(1,2))+bias[:,None,:]
                    valid=ye[st:en]>=0
                    if bool(valid.any()):
                        pred=lg[:,valid].argmax(-1)
                        correct+=(pred==ye[st:en][valid][None,:]).sum(1).cpu().numpy()
                        total+=int(valid.sum())
                split_acc[split]=[None if t==0 else float(c/t) for c,t in zip(correct,total)]
            for si,sd in enumerate(seeds):
                for ri,name in enumerate(names):
                    mi=si*nr+ri
                    row={"normalization":norm_name,"task":task,"representation":name,"seed":sd,"step":step,
                         "train_accuracy":float(tr_acc[mi]),"train_loss_minibatch":float(loss.item()),"gradient_norm":gnorm,"weight_norm":wnorm}
                    for split in SPLITS: row[split]=split_acc[split][mi]
                    rows.append(row)
        snapshots[step]={"weight":weight.detach().cpu().numpy().reshape(ns,nr,nclass,d).copy(),
                         "bias":bias.detach().cpu().numpy().reshape(ns,nr,nclass).copy()}
    del opt, X,Y,weight,bias,ev
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows,snapshots


def studentized_paired(a,b):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64)
    return {"delta_lhs_minus_rhs":ci(d),"paired_deltas":d.tolist()}


def audit_and_features():
    t0=time.perf_counter(); OLD.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True); FEAT.mkdir(parents=True,exist_ok=True); LOGDIR.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((OLD/"config.json").read_text(encoding="utf-8"))
    seeds=cfg["longer_seeds"]
    if seeds != SEEDS: raise RuntimeError(f"O3b-L exact longer seeds mismatch: {seeds}")
    cache=MRCache(CACHE); infos,pmap=j.build_infos(cache); token,_=j.load_token_arrays(); carriers=j.aligned_carriers(infos,token)
    H={s:np.asarray(carriers["J-I2b"][s],dtype=np.float32) for s in SPLITS}
    jlabels,jmap=j.joint_labels(infos,pmap)
    ck=torch.load(CKPT,map_location="cpu",weights_only=False)
    key=KeyEncoder(DIM); key.load_state_dict(ck["key"],strict=True); key.eval()
    layer=key.net[0]; W=layer.weight.detach().float().cpu().numpy(); b=layer.bias.detach().float().cpu().numpy() if layer.bias is not None else None
    U,S,Vh=np.linalg.svd(W,full_matrices=False)
    reps={k:{} for k in ("R0","R1","R2")}; eq=[]
    rng=np.random.default_rng(7781)
    for split in SPLITS:
        h=H[split]; r0=h@Vh.T; r1=r0*S[None,:]; r2=h@W.T; r2v=r1@U.T
        diff=(r2-r2v).astype(np.float64); den=max(float(np.linalg.norm(r2)),1e-30)
        n=min(len(h),1024); pairs=rng.integers(0,n,size=(min(30000,max(5000,n*8)),2))
        d1=np.linalg.norm(r1[pairs[:,0]]-r1[pairs[:,1]],axis=1); d2=np.linalg.norm(r2[pairs[:,0]]-r2[pairs[:,1]],axis=1)
        corr=float(np.corrcoef(d1,d2)[0,1])
        # Ensure exactly matched samples/labels by construction and verify row counts.
        full_lens={"entity":len(infos[split]["entity"]),"relation":len(infos[split]["relation"]),"joint":len(jlabels[split])}
        if len(set(full_lens.values()))!=1 or len(h)>next(iter(full_lens.values())): raise RuntimeError(f"sample alignment mismatch {split}: H={len(h)}, labels={full_lens}")
        lens={"carrier_rows":len(h),"full_cache_label_rows":next(iter(full_lens.values())),"aligned_prefix_rows":min(len(h),next(iter(full_lens.values())))}
        cos=float(np.mean(np.sum(r2*r2v,axis=1)/np.maximum(np.linalg.norm(r2,axis=1)*np.linalg.norm(r2v,axis=1),1e-30)))
        eq.append({"split":split,"n":len(h),"max_abs_R2_minus_R1U":float(np.abs(diff).max()),"mean_abs_R2_minus_R1U":float(np.abs(diff).mean()),
                   "relative_fro_error":float(np.linalg.norm(diff)/den),"cosine_similarity_mean":cos,
                   "pairwise_distance_correlation":corr,"max_abs_R2_direct_W":float(np.abs(r2-h@W.T).max()),"sample_lengths":lens})
        reps["R0"][split]=r0.astype(np.float32); reps["R1"][split]=r1.astype(np.float32); reps["R2"][split]=r2.astype(np.float32)
    # A compact re-usable feature cache, with sample order preserved by split.
    np.savez_compressed(FEAT/"l2_i2b_features.npz", **{f"H_{s}":H[s] for s in SPLITS}, **{f"{k}_{s}":reps[k][s] for k in reps for s in SPLITS})
    metadata={"carrier":"L-I2b / J-I2b","steps_h_reused_from_cache":True,"checkpoint_read_only":str(CKPT),"checkpoint_sha256":__import__("hashlib").sha256(CKPT.read_bytes()).hexdigest(),
              "W1_shape":list(W.shape),"W1_dtype":str(W.dtype),"bias_present":b is not None,"bias_shape":None if b is None else list(b.shape),
              "rank":int(np.linalg.matrix_rank(W)),"singular_values":S.tolist(),"seeds_from_o3b_l_config":seeds,"probe_steps_primary":500,"extension_steps":1000,
              "probe_optimizer":"AdamW(lr=3e-4, weight_decay=0.01)","batch_size":"min(1024,n_train_known)","milestones":MILESTONES,"train_only_standardization":True,
              "entity_classes":len(pmap),"relation_classes":4,"joint_train_classes":len(jmap),"label_coverage":{s:{"carrier_rows":len(H[s]),"entity_known":int(np.sum(infos[s]["entity"][:len(H[s])]>=0)),"relation_known":int(np.sum(infos[s]["relation"][:len(H[s])]>=0)),"joint_known":int(np.sum(jlabels[s][:len(H[s])]>=0))} for s in SPLITS},
              "l2_config_used":cfg,"equivalence":eq}
    dump(OUT/"stage0_metadata.json",metadata)
    lines=["# Stage 0 — reproduction audit","",f"O3b-L config reused: `{OLD/'config.json'}`.",f"Primary L2 carrier: `L-I2b` (J-I2b aligned rows); fixed seeds: `{seeds}`.","",f"Frozen checkpoint: `{CKPT}`; SHA-256: `{metadata['checkpoint_sha256']}`. It was loaded read-only; no model/checkpoint writes.",f"W1 actual shape/dtype: `{W.shape}` / `{W.dtype}`; bias present: `{b is not None}`; numerical rank: `{np.linalg.matrix_rank(W)}`.","", "Definitions match O3b-L: `R0=H@Vh.T`, `R1=R0*Sigma`, `R2=H@W1.T`; therefore `R2=R1@U.T` up to floating-point error. The cache is loaded once and all transformations preserve the existing split/sample order.","", "Exact inherited probe settings: 5 long-run seeds from config; linear `nn.Linear`; AdamW `lr=3e-4`, `weight_decay=0.01`; batch size `min(1024,n)`; seed+task initialization offsets and task-specific fixed NumPy minibatch schedule as in O3b-L; primary endpoint 500 steps, continuation to 1000 only for convergence diagnosis.","", "Sample/label maps are rebuilt using the O3b-J source functions from the same immutable MR cache; entity class ids come from train persons, relation ids are unchanged, joint ids are train-seen pairs only. Test-only unseen entity/joint metrics remain `n/a`.","", "## Algebraic equivalence checks","", "| split | max abs R2−R1Uᵀ | mean abs | relative Fro error | mean cosine | pairwise-distance corr |", "|---|---:|---:|---:|---:|---:|"]
    for r in eq: lines.append(f"| {r['split']} | {r['max_abs_R2_minus_R1U']:.3e} | {r['mean_abs_R2_minus_R1U']:.3e} | {r['relative_fro_error']:.3e} | {r['cosine_similarity_mean']:.9f} | {r['pairwise_distance_correlation']:.9f} |")
    lines += ["", "R2 is compared both to `R1 @ U.T` and the direct frozen `H @ W1.T`. Any large discrepancy would halt downstream mechanism interpretation. Pairwise checks use same-row random pairs and do not fit on evaluation data.","", "## Sample alignment and label coverage","", "O3b-H caches mention carriers for only the first 4096 train facts; MR cache contains 11665 train facts. As in O3b-L, the carrier and every label are aligned to the same deterministic prefix via `[:len(carrier)]` (validation/test are fully covered). Full label count and aligned carrier-prefix count are both recorded in `stage0_metadata.json`; entity/relation/joint label arrays have identical source ordering. No test split contributes to standardization, hyperparameter selection, or stopping.",""]
    (OUT/"01_stage0_audit.md").write_text("\n".join(lines),encoding="utf-8")
    return H,reps,infos,pmap,jlabels,jmap,U,S,Vh,W,b,metadata


def run_stage1(H,reps,infos,pmap,jlabels,jmap,U,S,Vh,metadata):
    started=time.perf_counter(); names=("R0","R1","R2"); raw={k:reps[k] for k in names}; std,std_stats=standardize(raw)
    dump(OUT/"standardization_train_stats.json",std_stats)
    # All feature condition numbers are computed from TRAIN only; test features are never consulted.
    cond={}
    for norm,rr in (("raw",raw),("standardized",std)):
        for name in names: cond[f"{norm}:{name}"]=feature_conditioning(rr[name]["train"])
    dump(OUT/"conditioning_raw_standardized.json",cond)
    lengths={s:len(H[s]) for s in SPLITS}
    tasks={"entity":({s:infos[s]["entity"][:lengths[s]] for s in SPLITS},len(pmap)),
           "relation":({s:infos[s]["relation"][:lengths[s]] for s in SPLITS},4),
           "joint":({s:jlabels[s][:lengths[s]] for s in SPLITS},len(jmap))}
    all_curve=[]; endpoint=[]; endpoints={}
    for norm,rr in (("raw",raw),("standardized",std)):
        for task,(lab,nc) in tasks.items():
            rows,snaps=probe_curves(rr,lab,nc,SEEDS,task,norm)
            all_curve += rows; endpoints[(norm,task)]=snaps
    np.savez_compressed(FEAT/"probe_endpoint_parameters.npz", **{
        f"{norm}_{task}_{step}_{kind}": endpoints[(norm,task)][step][kind]
        for norm in ("raw","standardized") for task in tasks for step in (500,1000) for kind in ("weight","bias")
    })
    write_csv(OUT/"03_probe_convergence.csv",all_curve)
    # Endpoint seed records at 500 and 1000; primary = 500.
    for r in all_curve:
        if r["step"] in (500,1000): endpoint.append(r)
    write_csv(OUT/"probe_seed_endpoints.csv",endpoint)
    # Aggregate by condition/task/representation/split/step and calculate paired contrasts.
    agg=[]
    keys=sorted({(r["normalization"],r["task"],r["representation"],r["step"],s) for r in all_curve for s in ("train","val","testA","testB")})
    for norm,task,rep,step,split in keys:
        vals=[r[split] for r in all_curve if r["normalization"]==norm and r["task"]==task and r["representation"]==rep and r["step"]==step and r[split] is not None]
        agg.append({"normalization":norm,"task":task,"representation":rep,"step":step,"split":split,**ci(vals,seed=1741+len(agg))})
    write_csv(OUT/"probe_seed_stats.csv",agg)
    paired=[]
    for norm in ("raw","standardized"):
        for task in tasks:
            for step in (180,300,500,750,1000):
                for split in ("val","testA","testB"):
                    rows={(r["representation"],r["seed"]):r[split] for r in all_curve if r["normalization"]==norm and r["task"]==task and r["step"]==step}
                    for a,b in (("R1","R0"),("R2","R1"),("R2","R0")):
                        av=[rows.get((a,s)) for s in SEEDS]; bv=[rows.get((b,s)) for s in SEEDS]
                        ds=[None if x is None or y is None else float(x-y) for x,y in zip(av,bv)]
                        d=[x for x in ds if x is not None]
                        paired.append({"normalization":norm,"task":task,"step":step,"split":split,"lhs":a,"rhs":b,"paired_delta_lhs_minus_rhs":ci(d,seed=1977+len(paired)),"paired_deltas":json.dumps(d)})
    write_csv(OUT/"paired_deltas.csv",paired)
    # Explicit R1 -> R2 transfer using R2 = R1 U^T. For row-vector activations, Wdst = Wsrc U^T.
    transfer=[]; raw_r1={k:raw[k] for k in ("R0","R1","R2")}; raw_labels=tasks
    for task,(lab,nc) in tasks.items():
        snap=endpoints[("raw",task)][1000]; ri=1
        src_w=snap["weight"][:,ri]; src_b=snap["bias"][:,ri]
        dst_w=np.einsum("scd,de->sce",src_w,U.T,optimize=True) # Wdst = Wsrc U.T
        # Verify equivalent logits from R2 features against source R1 classifier.
        for split in SPLITS:
            x1=torch.from_numpy(raw["R1"][split]).to(DEVICE); x2=torch.from_numpy(raw["R2"][split]).to(DEVICE)
            y=torch.from_numpy(np.asarray(lab[split],dtype=np.int64)).long().to(DEVICE)
            for si,seed in enumerate(SEEDS):
                w1=torch.from_numpy(src_w[si]).to(DEVICE); w2=torch.from_numpy(dst_w[si]).to(DEVICE); bb=torch.from_numpy(src_b[si]).to(DEVICE)
                with torch.no_grad():
                    lg1=x1@w1.T+bb; lg2=x2@w2.T+bb
                    err=float((lg1-lg2).abs().max().item())
                    a1=accuracy(lg1,y); a2=accuracy(lg2,y)
                transfer.append({"normalization":"raw","task":task,"seed":seed,"step":1000,"split":split,"max_abs_logit_error":err,"source_accuracy":a1,"transferred_accuracy":a2,"accuracy_delta":None if a1 is None or a2 is None else a2-a1})
        # The same classifier transfer for standardized views is affine (means/scales included).
        m1=np.asarray(std_stats["R1"]["mu"]); d1=np.asarray(std_stats["R1"]["sd"]); m2=np.asarray(std_stats["R2"]["mu"]); d2=np.asarray(std_stats["R2"]["sd"])
        M=(d2[:,None]*U)/d1[None,:] # x2_std @ M -> x1_std linear part
        c=(m2@U-m1)/d1
        sw=endpoints[("standardized",task)][1000]["weight"][:,1]; sb=endpoints[("standardized",task)][1000]["bias"][:,1]
        dw=np.einsum("scd,ed->sce",sw,M,optimize=True).astype(np.float32)
        db=(sb+np.einsum("scd,d->sc",sw,c,optimize=True)).astype(np.float32)
        for split in SPLITS:
            x1=torch.from_numpy(std["R1"][split]).to(DEVICE); x2=torch.from_numpy(std["R2"][split]).to(DEVICE); y=torch.from_numpy(np.asarray(lab[split],dtype=np.int64)).long().to(DEVICE)
            for si,seed in enumerate(SEEDS):
                w1=torch.from_numpy(sw[si]).to(DEVICE); w2=torch.from_numpy(dw[si]).to(DEVICE); b1=torch.from_numpy(sb[si]).to(DEVICE); b2=torch.from_numpy(db[si]).to(DEVICE)
                with torch.no_grad():
                    lg1=x1@w1.T+b1; lg2=x2@w2.T+b2; err=float((lg1-lg2).abs().max().item()); a1=accuracy(lg1,y); a2=accuracy(lg2,y)
                transfer.append({"normalization":"standardized_affine_transfer","task":task,"seed":seed,"step":1000,"split":split,"max_abs_logit_error":err,"source_accuracy":a1,"transferred_accuracy":a2,"accuracy_delta":None if a1 is None or a2 is None else a2-a1})
    write_csv(OUT/"09_weight_transfer_control.csv",transfer)
    # A direct feature-level invariance check for independently standardized R0 and R1.
    invariance={}
    for split in SPLITS:
        a=std["R0"][split].astype(np.float64); b=std["R1"][split].astype(np.float64)
        invariance[split]={"max_abs_R0std_R1std":float(np.abs(a-b).max()),"mean_abs_R0std_R1std":float(np.abs(a-b).mean()),
                           "pearson_flattened":float(np.corrcoef(a.ravel(),b.ravel())[0,1])}
    dump(OUT/"feature_standardized_equivalence.json",invariance)
    # Generate simple SVG plots without external plotting dependencies.
    plots=ROOT/"plots"; plots.mkdir(parents=True,exist_ok=True)
    svg_curves(all_curve,plots)
    run_s=time.perf_counter()-started
    dump(OUT/"stage1_runtime.json",{"wall_clock_s":run_s,"device":str(DEVICE),"device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu","stage1_complete":True})
    # Gate is intentionally provisional and evidence-based; the report carries the full curve and paired CI.
    gate={"status":"REVIEW_REQUIRED","primary_steps":500,"extension_steps":1000,"reason":"Requires joint review of paired 500/1000-step results, curve plateau, exact standardized feature equivalence, and weight-transfer logit invariance. No mechanism decision is encoded from accuracy alone.","j_revalidation_started":False}
    dump(OUT/"stage1_gate.json",gate)
    write_stage1_report(OUT,metadata,cond,agg,paired,transfer,invariance,run_s)
    print(json.dumps({"stage":"stage1_complete","elapsed_s":run_s,"seeds":SEEDS,"features":str(FEAT/"l2_i2b_features.npz"),"out":str(OUT)}),flush=True)


def svg_curves(rows,plots):
    # Each SVG is small, self-contained, and reports validation/train curves by step.
    colors={"R0":"#1f77b4","R1":"#d62728","R2":"#2ca02c"}
    for norm in ("raw","standardized"):
        for task in ("entity","relation","joint"):
            ss=[r for r in rows if r["normalization"]==norm and r["task"]==task and r.get("val") is not None]
            steps=MILESTONES; width=850;height=500; left=70;top=35; pw=730;ph=390
            parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/>',
                   f'<text x="{left}" y="24" font-size="16">{norm} {task}: validation accuracy (mean)</text>']
            for t in range(0,101,20):
                y=top+ph*(1-t/100); parts.append(f'<line x1="{left}" x2="{left+pw}" y1="{y}" y2="{y}" stroke="#ddd"/><text x="20" y="{y+4}" font-size="11">{t}%</text>')
            for rep in ("R0","R1","R2"):
                xs=[]; ys=[]
                for st in steps:
                    vals=[r["val"] for r in ss if r["representation"]==rep and r["step"]==st]
                    if not vals: continue
                    x=left+pw*(st/max(steps)); y=top+ph*(1-np.mean(vals)); xs.append(f"{x:.1f},{y:.1f}")
                parts.append(f'<polyline fill="none" stroke="{colors[rep]}" stroke-width="2.5" points="{" ".join(xs)}"/>')
                parts.append(f'<text x="{left+pw-45}" y="{top+18+(int(rep[-1]))*20}" fill="{colors[rep]}" font-size="13">{rep}</text>')
            parts.append(f'<text x="{left+pw/2}" y="{height-15}" text-anchor="middle" font-size="12">optimization step</text></svg>')
            (plots/f"convergence_{norm}_{task}.svg").write_text("".join(parts),encoding="utf-8")


def write_stage1_report(out,metadata,cond,agg,paired,transfer,invariance,run_s):
    def find(norm,task,rep,split,step):
        r=next((r for r in agg if r["normalization"]==norm and r["task"]==task and r["representation"]==rep and r["split"]==split and r["step"]==step),None)
        return "n/a" if r is None or r["mean"] is None else f"{100*r['mean']:.2f} ± {100*r['std']:.2f}% (paired-seed t95 {100*r['t95_low']:.2f}–{100*r['t95_high']:.2f})"
    lines=["# O3b-L2-500STD — Stage 1 measurement calibration","","## Scope","","Frozen L-I2b carrier and frozen W1 only. No Ouro/backbone retraining, no checkpoint/model modification, no reconnect/O3c, and no repeated-V update work. All newly generated data are under this new experiment directory.","","Exact five seeds were read from the O3b-L `config.json`: `11, 23, 37, 53, 71`. All conditions use the inherited linear probe, AdamW 3e-4 / 0.01, batch ≤1024, same seed/task initialization rule, same task-specific minibatch schedule, and fixed 500-step endpoint; 750/1000 are convergence-only extensions. Standardization uses train-only per-dimension mean/std.","","## Paired results (mean ± SD, percent; paired t-CI also shown)",""]
    for norm in ("raw","standardized"):
        lines += [f"### {norm}","","| task | split | step | R0 | R1 | R2 |","|---|---|---:|---:|---:|---:|"]
        for task in ("entity","relation","joint"):
            for split in ("val","testA","testB"):
                lines.append(f"| {task} | {split} | 500 | {find(norm,task,'R0',split,500)} | {find(norm,task,'R1',split,500)} | {find(norm,task,'R2',split,500)} |")
        lines += [""]
    runtime_text="not captured" if run_s is None else f"{run_s:.2f}s"
    lines += ["## Conditioning","","Train-only raw and standardized covariance spectra, condition numbers, effective rank, participation ratio, feature norms and per-dimension standard deviations are in `conditioning_raw_standardized.json`. Learning curves at 20/50/100/180/300/500/750/1000 are in `03_probe_convergence.csv`; only validation metrics are used to inspect plateau, never to stop training.","","## Algebraic / transfer controls","",f"Train-standardized R0/R1 feature-level differences: `{json.dumps(invariance,ensure_ascii=False)}`.","","R2 is algebraically `R1 @ U.T`; transferred classifiers use the derived inverse-coordinate transform and include an affine transform for independently standardized coordinates. Per-seed max logit errors and accuracy parity are in `09_weight_transfer_control.csv`.","","## Gate status","","No result is interpreted as information content. Stage 1 gate remains `REVIEW_REQUIRED` pending evidence-based assessment of 500/1000-step convergence, paired seed variation, standardized feature equivalence and weight-transfer invariance. Full raw metrics are in the accompanying CSV/JSON files.","",f"Runtime: {runtime_text} on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}. Output root: `{ROOT}`.",""]
    (out/"02_l2_500std_results.md").write_text("\n".join(lines),encoding="utf-8")


def read_rows(path):
    with path.open("r",newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    number_fields={"seed","step","train_accuracy","train_loss_minibatch","gradient_norm","weight_norm","train","val","testA","testB","mean","std","t95_low","t95_high","boot95_low","boot95_high","n","paired_delta_lhs_minus_rhs","source_accuracy","transferred_accuracy","accuracy_delta","max_abs_logit_error"}
    for row in rows:
        for k in number_fields:
            if k in row and row[k] != "":
                try:
                    v=float(row[k]); row[k]=int(v) if k in ("seed","step","n") else v
                except ValueError: pass
            elif k in row and row[k]=="": row[k]=None
    return rows


def finalize_existing():
    all_curve=read_rows(OUT/"03_probe_convergence.csv")
    agg=read_rows(OUT/"probe_seed_stats.csv")
    paired=read_rows(OUT/"paired_deltas.csv")
    transfer=read_rows(OUT/"09_weight_transfer_control.csv")
    metadata=json.loads((OUT/"stage0_metadata.json").read_text(encoding="utf-8"))
    cond=json.loads((OUT/"conditioning_raw_standardized.json").read_text(encoding="utf-8"))
    invariance=json.loads((OUT/"feature_standardized_equivalence.json").read_text(encoding="utf-8"))
    plots=ROOT/"plots"; plots.mkdir(parents=True,exist_ok=True); svg_curves(all_curve,plots)
    gate={"status":"REVIEW_REQUIRED","primary_steps":500,"extension_steps":1000,"reason":"Assess paired 500/1000-step results, learning curves, exact feature equivalence and weight-transfer invariance before selecting the automatic branch.","j_revalidation_started":False}
    dump(OUT/"stage1_gate.json",gate)
    dump(OUT/"stage1_runtime.json",{"wall_clock_s":None,"runtime_note":"Training and weight-transfer computation completed; wall-clock summary was not emitted because the first plotting attempt hit a harness error after all result CSVs were written.","device":"cuda","device_name":"BW","stage1_complete":True})
    write_stage1_report(OUT,metadata,cond,agg,paired,transfer,invariance,None)
    print(json.dumps({"stage":"stage1_finalize_existing","curve_rows":len(all_curve),"aggregate_rows":len(agg),"transfer_rows":len(transfer),"out":str(OUT)}),flush=True)


def main():
    if "--finalize-existing" in sys.argv:
        finalize_existing(); return
    print(json.dumps({"stage":"start","out":str(ROOT),"device":str(DEVICE),"device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}),flush=True)
    H,reps,infos,pmap,jlabels,jmap,U,S,Vh,W,b,metadata=audit_and_features()
    run_stage1(H,reps,infos,pmap,jlabels,jmap,U,S,Vh,metadata)


if __name__=="__main__": main()
