# -*- coding: utf-8 -*-
"""CLR-v2.4 O3b-K: first-linear subspace attribution.

Diagnostic-only analysis of the frozen O2.1b KeyEncoder first Linear layer.
No production model, memory path, matcher, or addressing architecture is
modified.  Temporary probes are used only as measurement instruments.
"""
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import a1_v24_o3b_j as j
from a1_v24_o3b import MRCache, KeyEncoder, DIM


OUT = Path(os.environ.get("V24_O3BK_OUT", r"D:/training/checkpoints/clr_v24_o3b_k_first_linear_subspace"))
CACHE_ROOT = Path(os.environ.get("V24_O3BK_CACHE", r"D:/training/checkpoints/clr_v24_o21a_mr/cache"))
HOUT = Path(os.environ.get("V24_O3BK_HOUT", r"D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag"))
O21B = Path(os.environ.get("V24_O3BK_O21B", r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt"))
SEED = int(os.environ.get("V24_O3BK_SEED", "42"))
PROBE_STEPS = int(os.environ.get("V24_O3BK_PROBE_STEPS", "180"))
CONTROL_STEPS = int(os.environ.get("V24_O3BK_CONTROL_STEPS", "30"))
SWEEP_STEPS = int(os.environ.get("V24_O3BK_SWEEP_STEPS", "30"))
CLASSIFIER_STEPS = int(os.environ.get("V24_O3BK_CLASSIFIER_STEPS", "60"))
PAIR_N = int(os.environ.get("V24_O3BK_PAIR_N", "1800"))
N_RANDOM = int(os.environ.get("V24_O3BK_N_RANDOM", "10"))
DEVICE_NAME = os.environ.get("V24_O3BK_DEVICE", "cuda")
SPLITS = ("train", "val", "testA", "testB")
KSWEEP = (8, 16, 32, 64, 128, 256)


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer, np.floating)): return x.item()
    if isinstance(x, Path): return str(x)
    return x


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("", encoding="utf-8"); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows({k:r.get(k,"") for k in fields} for r in rows)


def stage(label, started):
    (OUT/"progress.txt").write_text(json.dumps({"stage":label,"elapsed_s":time.perf_counter()-started},ensure_ascii=False),encoding="utf-8")


def pct(x): return "n/a" if x is None else f"{100*float(x):.2f}%"


def seed_all(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def pair_summary(reps, pairsets):
    """Cosine AUROC/F1, overlap, and median margin for P0/P1/P2."""
    rows=[]
    for name, arrs in reps.items():
        for split in ("testA","testB"):
            for neg in ("P1_diff_entity_same_relation","P2_same_entity_diff_relation"):
                def score(typ):
                    ab=pairsets[split][typ]; return j.cosine(arrs[split][ab[:,0]],arrs[split][ab[:,1]])
                p0=score("P0_same_entity_same_relation"); pn=score(neg)
                vp0=j.cosine(arrs["val"][pairsets["val"]["P0_same_entity_same_relation"][:,0]],arrs["val"][pairsets["val"]["P0_same_entity_same_relation"][:,1]])
                vpn=j.cosine(arrs["val"][pairsets["val"][neg][:,0]],arrs["val"][pairsets["val"][neg][:,1]])
                y=np.r_[np.ones(len(p0),dtype=np.int64),np.zeros(len(pn),dtype=np.int64)]; sc=np.r_[p0,pn]
                th=j.best_threshold(np.r_[np.ones(len(vp0),dtype=np.int64),np.zeros(len(vpn),dtype=np.int64)],np.r_[vp0,vpn])
                m=j.binary_metrics(y,sc,th)
                rows.append({"representation":name,"split":split,"contrast":f"P0_vs_{neg}","cosine_auroc":m["auroc"],"best_f1":m["f1"],"balanced_accuracy":m["balanced_accuracy"],"histogram_overlap":j.overlap(p0,pn),"median_margin":float(np.median(p0)-np.median(pn)),"p0_median":float(np.median(p0)),"negative_median":float(np.median(pn))})
    return rows


def fit_classifier(x, y, nclass, device, steps=PROBE_STEPS, seed=SEED):
    x=np.asarray(x,dtype=np.float32); y=np.asarray(y,dtype=np.int64); m=y>=0; x=x[m]; y=y[m]
    torch.manual_seed(seed); model=nn.Linear(x.shape[1],nclass).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01); X=torch.from_numpy(x).to(device); Y=torch.from_numpy(y).long().to(device); rng=np.random.default_rng(seed+177); bs=min(1024,len(Y))
    model.train()
    for _ in range(max(1,steps)):
        ix=rng.integers(0,len(Y),size=bs); loss=F.cross_entropy(model(X[ix]),Y[ix]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); w=model.weight.detach().float().cpu().numpy(); b=model.bias.detach().float().cpu().numpy(); del model,X,Y
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return w,b


def batched_eval_probes(names, reps, label_by_split, nclass, device, steps=PROBE_STEPS, seed=SEED, eval_splits=SPLITS):
    """Train batched probes and evaluate them on all four frozen splits."""
    ytr=np.asarray(label_by_split["train"],dtype=np.int64); mask=ytr>=0; ytr=ytr[mask]
    Xnp=np.stack([np.asarray(reps[n]["train"],dtype=np.float32)[mask] for n in names],axis=0)
    R,N,D=Xnp.shape; X=torch.from_numpy(Xnp).to(device); Y=torch.from_numpy(ytr).long().to(device)
    weight=nn.Parameter(torch.empty(R,nclass,D,device=device)); bias=nn.Parameter(torch.empty(R,nclass,device=device)); bound=1.0/math.sqrt(max(D,1))
    torch.manual_seed(seed)
    with torch.no_grad(): weight.uniform_(-bound,bound); bias.uniform_(-bound,bound)
    opt=torch.optim.AdamW([weight,bias],lr=3e-4,weight_decay=.01); rng=np.random.default_rng(seed+913); bs=min(1024,N); target_batch=Y.unsqueeze(0).expand(R,-1)
    for _ in range(max(1,steps)):
        ix=rng.integers(0,N,size=bs); logits=torch.bmm(X[:,ix,:],weight.transpose(1,2))+bias[:,None,:]
        loss=F.cross_entropy(logits.reshape(R*bs,nclass),target_batch[:,ix].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    out={n:{"train_n":int(N),"nclass":int(nclass),"mlp":False} for n in names}
    with torch.no_grad():
        for split in eval_splits:
            yy=np.asarray(label_by_split[split],dtype=np.int64); m=yy>=0; recs=[]
            if m.any():
                Xev=torch.from_numpy(np.stack([np.asarray(reps[n][split],dtype=np.float32)[m] for n in names],axis=0)).to(device)
                pred=torch.bmm(Xev,weight.transpose(1,2))+bias[:,None,:]; pred=pred.argmax(2).cpu().numpy(); del Xev
            for ri,n in enumerate(names):
                rec={"known_fraction":float(m.mean()) if len(yy) else 0.0,"n_known":int(m.sum()),"accuracy":float(np.mean(pred[ri]==yy[m])) if m.any() else None}; out[n][split]=rec
    del opt,weight,bias,X,Y,target_batch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def probe_reps(reps, infos, pmap, jlabels, device, name, eval_splits=SPLITS, steps=PROBE_STEPS):
    out={r:{"entity":{},"relation":{},"joint":{}} for r in reps}
    groups={}
    for r,arrs in reps.items(): groups.setdefault(arrs["train"].shape[1],[]).append(r)
    tasks=(("entity",{s:infos[s]["entity"][:len(reps[next(iter(reps))][s])] for s in SPLITS},len(pmap),SEED+1),
           ("relation",{s:infos[s]["relation"][:len(reps[next(iter(reps))][s])] for s in SPLITS},4,SEED+2),
           ("joint",{s:jlabels[s][:len(reps[next(iter(reps))][s])] for s in SPLITS},len(set(jlabels["train"][jlabels["train"]>=0])),SEED+3))
    for _,names in groups.items():
        for task,label_by_split,nclass,seed in tasks:
            rec=batched_eval_probes(names,{n:reps[n] for n in names},label_by_split,nclass,device,steps=steps,seed=seed,eval_splits=eval_splits)
            for n in names: out[n][task]=rec[n]
    return out


def orth_basis(indim, outdim, seed):
    rng=np.random.default_rng(seed); A=rng.normal(size=(indim,outdim)).astype(np.float64); Q,_=np.linalg.qr(A,mode="reduced"); return Q.T.astype(np.float32)


def pca_basis(X, d):
    n=min(len(X),1024); idx=np.linspace(0,len(X)-1,n,dtype=np.int64); sample=np.asarray(X[idx],dtype=np.float32); mean=sample.mean(0).astype(np.float32)
    return mean,top_right_dirs(sample-mean[None,:],d,seed=SEED+701)


def stats(v):
    v=np.asarray(v,dtype=np.float64); return {"mean":float(v.mean()),"std":float(v.std()),"median":float(np.median(v)),"p10":float(np.quantile(v,.10)),"p90":float(np.quantile(v,.90))}


def alignment(U, B):
    """U and B are row-orthonormal direction matrices."""
    U=np.asarray(U,dtype=np.float64); B=np.asarray(B,dtype=np.float64); vals=np.linalg.svd(U@B.T,compute_uv=False); vals=np.clip(vals,-1,1); ang=np.degrees(np.arccos(vals)); energy=np.sum((U@B.T)**2,axis=1)
    return {"mean_principal_angle_deg":float(ang.mean()),"median_principal_angle_deg":float(np.median(ang)),"smallest_principal_angle_deg":float(ang.min()),"largest_principal_angle_deg":float(ang.max()),"mean_squared_cosine":float(np.mean(vals*vals)),"alignment":float(np.mean(energy)),"angles_deg":ang.tolist()}


def fisher_dirs(X, y, kmax=64):
    X=np.asarray(X,dtype=np.float64); y=np.asarray(y,dtype=np.int64); m=y>=0; X=X[m]; y=y[m]; mu=X.mean(0); rows=[]; weights=[]
    for c in sorted(set(y.tolist())):
        xc=X[y==c]
        if len(xc): rows.append(xc.mean(0)-mu); weights.append(math.sqrt(len(xc)))
    A=np.asarray(rows,dtype=np.float32)*np.asarray(weights,dtype=np.float32)[:,None]
    return top_right_dirs(A, kmax)


def classifier_dirs(W, kmax=64):
    W=np.asarray(W,dtype=np.float32); W=W-W.mean(0,keepdims=True); return top_right_dirs(W, kmax)


def top_right_dirs(A, kmax=64, seed=SEED+703):
    """Top right-singular directions without materializing a huge exact SVD.

    The exact W1 SVD above remains unchanged.  This helper is only for the
    temporary classifier/Fisher diagnostic matrices, where an exact SVD of
    thousands of classes by 2048 dimensions is needlessly expensive.  The
    resulting orthonormal directions are the same diagnostic object (top
    right singular subspace) to randomized-PCA numerical precision.
    """
    A=np.asarray(A,dtype=np.float32)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] == 0:
        return np.zeros((0, A.shape[1] if A.ndim == 2 else 0), dtype=np.float32)
    q=min(int(kmax), A.shape[0], A.shape[1])
    if q <= 0: return np.zeros((0,A.shape[1]),dtype=np.float32)
    # Halko-style randomized range finder.  This is only for temporary
    # classifier/Fisher/PCA diagnostic subspaces; the W1 SVD remains exact.
    rng=np.random.default_rng(seed); oversample=min(16,max(4,A.shape[1]-q)); l=min(q+oversample,min(A.shape))
    omega=rng.normal(size=(A.shape[1],l)).astype(np.float32); Y=A@omega; Q,_=np.linalg.qr(Y,mode="reduced"); B=Q.T@A
    # Diagonalize the small l-by-l Gram matrix instead of invoking a large
    # rectangular SVD in the remote NumPy build.
    gram=B@B.T; eigval,eigvec=np.linalg.eigh(gram); order=np.argsort(eigval)[::-1][:q]; dirs=eigvec[:,order].T@B
    dirs/=np.maximum(np.linalg.norm(dirs,axis=1,keepdims=True),1e-12)
    return dirs.astype(np.float32)


def energy_rows(carriers, V, infos):
    rows=[]
    for carrier,arrs in carriers.items():
        for split in SPLITS:
            H=arrs[split]; HR=(H@V.T)@V; HN=H-HR; re=np.sum(HR*HR,axis=1)/np.maximum(np.sum(H*H,axis=1),1e-12); rn=np.sum(HN*HN,axis=1)/np.maximum(np.sum(H*H,axis=1),1e-12)
            rows.append({"carrier":carrier,"split":split,"component":"row_energy_fraction",**stats(re)}); rows.append({"carrier":carrier,"split":split,"component":"null_energy_fraction",**stats(rn)})
    return rows


def difference_rows(carriers, V, pairsets):
    rows=[]
    for carrier,arrs in carriers.items():
        for split in ("testA","testB"):
            H=arrs[split]
            for typ,label in (("P1_diff_entity_same_relation","entity_difference"),("P2_same_entity_diff_relation","relation_difference")):
                ab=pairsets[split][typ]; d=H[ab[:,0]]-H[ab[:,1]]; denom=np.maximum(np.linalg.norm(d,axis=1),1e-12); dr=d@V.T; rf=np.linalg.norm(dr,axis=1)/denom; nf=np.linalg.norm(d-dr@V,axis=1)/denom
                rows.append({"carrier":carrier,"split":split,"difference_type":label,"row_fraction_mean":float(rf.mean()),"row_fraction_std":float(rf.std()),"row_fraction_median":float(np.median(rf)),"null_fraction_mean":float(nf.mean()),"null_fraction_std":float(nf.std()),"null_fraction_median":float(np.median(nf)),"n":len(d)})
    return rows


def contraction_rows(carriers, pairsets, transforms):
    rows=[]
    for carrier,base in carriers.items():
        for repname,arrs in transforms[carrier].items():
            for split in ("testA","testB"):
                H=base[split]
                for typ,label in (("P1_diff_entity_same_relation","entity_contraction"),("P2_same_entity_diff_relation","relation_contraction")):
                    ab=pairsets[split][typ]; d0=np.linalg.norm(H[ab[:,0]]-H[ab[:,1]],axis=1); d1=np.linalg.norm(arrs[split][ab[:,0]]-arrs[split][ab[:,1]],axis=1); ratio=d1/np.maximum(d0,1e-12); rows.append({"carrier":carrier,"representation":repname,"split":split,"difference_type":label,**stats(ratio),"n":len(ratio)})
    return rows


def simple_svg(path,title,series,ymin=0,ymax=1,labels=None):
    W,H,L,R,T,B=900,540,80,30,55,70; pw,ph=W-L-R,H-T-B; labels=labels or [str(i) for i in range(len(series[0][1]))]
    def yy(v): return None if v is None else T+ph*(1-(float(v)-ymin)/max(ymax-ymin,1e-12))
    cols=["#2563eb","#dc2626","#059669","#9333ea","#ea580c","#0891b2","#111827"]
    p=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="{W/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text><line x1="{L}" y1="{T}" x2="{L}" y2="{T+ph}" stroke="#333"/><line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="#333"/>']
    for si,(name,vals) in enumerate(series):
        pts=[]
        for i,v in enumerate(vals):
            y=yy(v)
            if y is not None: pts.append(f"{L+pw*i/max(len(vals)-1,1):.1f},{y:.1f}")
        if pts: p.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{cols[si%len(cols)]}" stroke-width="2"/>')
        p.append(f'<text x="{L+pw-5}" y="{T+15+18*si}" text-anchor="end" font-family="Arial" font-size="12" fill="{cols[si%len(cols)]}">{name}</text>')
    for i,l in enumerate(labels): p.append(f'<text x="{L+pw*i/max(len(labels)-1,1):.1f}" y="{T+ph+22}" text-anchor="middle" font-family="Arial" font-size="12">{l}</text>')
    p.append('</svg>'); path.write_text("".join(p),encoding="utf-8")


def make_figures(probe_rows, align_rows, random_rows, sweep_rows, contraction, difference):
    d=OUT/"figures"; d.mkdir(parents=True,exist_ok=True)
    # SVGs are intentionally dependency-free on the remote training host.
    def series(metric, reps): return [(r,[next((x.get(metric) for x in probe_rows if x["representation"]==r and x["split"]=="testA" and x["carrier"]=="K-I2b" and x["representation"]==r),None)]) for r in reps]
    # 1: row/null probe bars represented as a depth-like line.
    names=[r["representation"] for r in probe_rows if r["carrier"]=="K-I2b" and r["split"]=="testA" and r["representation"] in ("H","H_R","H_N","W1H","W1H_no_bias","GELU_W1H","white_row")]
    names=list(dict.fromkeys(names));
    for metric,file,title in (("entity","01_row_null_entity.svg","Entity probes on H/row/null"),("relation","02_row_null_relation.svg","Relation probes on H/row/null"),("joint","03_row_null_joint.svg","Joint probes on H/row/null")):
        vals=[next((r.get(metric) for r in probe_rows if r["carrier"]=="K-I2b" and r["representation"]==n and r["split"]=="testA"),None) for n in names]; simple_svg(d/file,title,[("K-I2b",vals)],0,1,names)
    ar=[r for r in align_rows if r["carrier"]=="K-I2b" and r["subspace_kind"]=="classifier" and r["k"]==64]; simple_svg(d/"04_entity_relation_alignment.svg","Entity vs relation alignment to W1 row-space",[("entity",[next((x["alignment"] for x in ar if x["label_kind"]=="entity"),0)]),("relation",[next((x["alignment"] for x in ar if x["label_kind"]=="relation"),0)])],0,1,["alignment"])
    rr=[r for r in random_rows if r["carrier"]=="K-I2b"]; vals=[next((x["entity_testA"] for x in rr if x["subspace"]==s),None) for s in ["learned_row","pca256"]]+[x["entity_testA"] for x in rr if x["subspace"].startswith("random_")]; names=["learned_row","pca256"]+[f"random_{i+1}" for i in range(len(vals)-2)]; simple_svg(d/"05_learned_pca_random.svg","Learned vs PCA vs random-256 entity",[("entity",vals)],0,1,names)
    rv=[x["entity_testA"] for x in rr if x["subspace"].startswith("random_")]; simple_svg(d/"06_random_subspace_distribution.svg","Random subspace entity distribution",[("random",rv),("learned",[next((x["entity_testA"] for x in rr if x["subspace"]=="learned_row"),None)]*len(rv))],0,1,[str(i+1) for i in range(len(rv))])
    ks=sorted(set(x["k"] for x in sweep_rows if x["carrier"]=="K-I2b")); simple_svg(d/"07_singular_direction_sweep.svg","Probe vs singular-direction count",[("top_entity",[next((x["entity_testA"] for x in sweep_rows if x["carrier"]=="K-I2b" and x["sweep"]=="top" and x["k"]==k and x["metric"]=="entity"),None) for k in ks]),("top_relation",[next((x["entity_testA"] for x in sweep_rows if x["carrier"]=="K-I2b" and x["sweep"]=="top" and x["k"]==k and x["metric"]=="relation"),None) for k in ks])],0,1,[str(k) for k in ks])
    cr=[x for x in contraction if x["carrier"]=="K-I2b" and x["split"]=="testA" and x["representation"] in ("W1H","PCA256","random256_0")]; simple_svg(d/"08_contraction_ratio.svg","Entity vs relation contraction ratio",[(x["representation"],[next((float(y["contraction_mean"]) for y in cr if y["representation"]==x["representation"] and y["difference_type"]==t),None) for t in ("entity_contraction","relation_contraction")]) for x in cr],0,max([float(x["contraction_mean"]) for x in cr]+[1])*1.1,["entity","relation"])


def main():
    seed_all(); t0=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if DEVICE_NAME=="cuda" and torch.cuda.is_available() else "cpu")
    dump(OUT/"runtime_device.json",{"requested":DEVICE_NAME,"selected":str(device),"cuda_available":bool(torch.cuda.is_available()),"cuda_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    cache=MRCache(CACHE_ROOT); infos,pmap=j.build_infos(cache); token,_=j.load_token_arrays(); all_carriers=j.aligned_carriers(infos,token)
    carriers={"K-I0":all_carriers["J-I0"],"K-I2b":all_carriers["J-I2b"],"K-I2a":all_carriers["J-I2a"]}
    jlabels,jmap=j.joint_labels(infos,pmap)
    pair_infos={};
    for s in SPLITS:
        n=min(len(carriers[c][s]) for c in carriers); pair_infos[s]={"persons":infos[s]["persons"][:n],"relation":infos[s]["relation"][:n]}
    pairsets={s:j.make_address_pairs(pair_infos[s],max(100,PAIR_N//4),SEED+100+i) for i,s in enumerate(SPLITS)}
    ck=torch.load(O21B,map_location="cpu",weights_only=False); key=KeyEncoder(DIM).to(device); key.load_state_dict(ck["key"],strict=True); key.eval()
    for p in key.parameters(): p.requires_grad_(False)
    W=key.net[0].weight.detach().float().cpu().numpy(); b=key.net[0].bias.detach().float().cpu().numpy() if key.net[0].bias is not None else None
    U,S,Vh=np.linalg.svd(W,full_matrices=False); tol=max(W.shape)*np.finfo(W.dtype).eps*float(S.max()); rank=int(np.sum(S>tol)); VR=Vh[:rank].astype(np.float32); VN=np.zeros((0,W.shape[1]),dtype=np.float32)
    P_R=VR.T@VR; numerical_rank=rank
    sv={"shape":list(W.shape),"dtype":str(W.dtype),"rank":int(np.linalg.matrix_rank(W)),"numerical_rank":numerical_rank,"singular_values":S.tolist(),"condition_number":float(S.max()/max(S[S>tol].min(),1e-30)),"frobenius_norm":float(np.linalg.norm(W,"fro")),"spectral_norm":float(S.max()),"bias_present":b is not None,"bias_shape":None if b is None else list(b.shape),"row_dim":int(rank),"input_dim":int(W.shape[1]),"svd_tolerance":float(tol)}; dump(OUT/"w1_svd_metrics.json",sv)
    transforms={}
    for carrier,arrs in carriers.items():
        transforms[carrier]={};
        for split in SPLITS:
            H=arrs[split]; HR=(H@VR.T)@VR; HN=H-HR; Y=H@W.T; Y0=Y-(b if b is not None else 0); G=F.gelu(torch.from_numpy(Y)).numpy(); white=H@VR.T
            transforms[carrier]["H"]={s:arrs[s] for s in SPLITS} if split==SPLITS[0] else transforms[carrier]["H"]
            if split==SPLITS[0]:
                transforms[carrier].update({"H_R":{},"H_N":{},"W1H":{},"W1H_no_bias":{},"GELU_W1H":{},"white_row":{}})
            transforms[carrier]["H_R"][split]=HR; transforms[carrier]["H_N"][split]=HN; transforms[carrier]["W1H"][split]=Y; transforms[carrier]["W1H_no_bias"][split]=Y0; transforms[carrier]["GELU_W1H"][split]=G; transforms[carrier]["white_row"][split]=white
    energy=energy_rows(carriers, VR, infos)
    dump(OUT/"energy_attribution.json",energy); write_csv(OUT/"energy_attribution.csv",energy)
    stage("energy",t0)
    # Probe representations required by the protocol.
    probe_rows=[]; probe_json={}
    for carrier in carriers:
        probe_json[carrier]={}; selected={k:transforms[carrier][k] for k in ("H","H_R","H_N","W1H","W1H_no_bias","GELU_W1H","white_row")}; ps=probe_reps(selected,infos,pmap,jlabels,device,carrier); probe_json[carrier]=ps
        for rep,rec in ps.items():
            for split in SPLITS: probe_rows.append({"carrier":carrier,"representation":rep,"split":split,"entity":rec["entity"][split]["accuracy"],"relation":rec["relation"][split]["accuracy"],"joint":rec["joint"][split]["accuracy"],"entity_known_fraction":rec["entity"][split]["known_fraction"],"relation_known_fraction":rec["relation"][split]["known_fraction"],"joint_known_fraction":rec["joint"][split]["known_fraction"]})
    write_csv(OUT/"subspace_probe.csv",probe_rows); dump(OUT/"subspace_probe.json",probe_json)
    stage("subspace_probe",t0)
    # PCA and ten random 256-D controls.
    control_rows=[]; control_subspaces={}; align_rows=[]
    classifier_dirs_by={}; fisher_dirs_by={}
    for carrier in carriers:
        H=carriers[carrier]["train"]; we,be=fit_classifier(H,infos["train"]["entity"][:len(H)],len(pmap),device,steps=CLASSIFIER_STEPS,seed=SEED+300); wr,br=fit_classifier(H,infos["train"]["relation"][:len(H)],4,device,steps=CLASSIFIER_STEPS,seed=SEED+301); classifier_dirs_by[carrier]={"entity":classifier_dirs(we),"relation":classifier_dirs(wr)}; fisher_dirs_by[carrier]={"entity":fisher_dirs(H,infos["train"]["entity"][:len(H)]),"relation":fisher_dirs(H,infos["train"]["relation"][:len(H)])}
        for kind,dirs in (("classifier",classifier_dirs_by[carrier]),("fisher",fisher_dirs_by[carrier])):
            for label_kind in ("entity","relation"):
                for k in (8,16,32,64):
                    A=dirs[label_kind][:k]; ar=alignment(A,VR); ar.update({"carrier":carrier,"subspace_kind":kind,"label_kind":label_kind,"k":k}); align_rows.append(ar)
        mean,pca=pca_basis(H,256); subspaces={"learned_row":VR,"pca256":pca}
        for ri in range(N_RANDOM): subspaces[f"random_{ri}"]=orth_basis(H.shape[1],256,SEED+1000+ri)
        control_subspaces[carrier]=subspaces
        control_arrs={name:{s:carriers[carrier][s]@B.T for s in SPLITS} for name,B in subspaces.items()}
        control_probe=probe_reps(control_arrs,infos,pmap,jlabels,device,carrier,eval_splits=("testA","testB"),steps=CONTROL_STEPS)
        for name,B in subspaces.items():
            rec=control_probe[name]; row={"carrier":carrier,"subspace":name,"entity_testA":rec["entity"]["testA"]["accuracy"],"relation_testA":rec["relation"]["testA"]["accuracy"],"joint_testA":rec["joint"]["testA"]["accuracy"],"entity_testB_known_fraction":rec["entity"]["testB"]["known_fraction"]}
            for label_kind in ("entity","relation"):
                row[f"{label_kind}_alignment_classifier64"]=alignment(classifier_dirs_by[carrier][label_kind][:64],B)["alignment"]; row[f"{label_kind}_alignment_fisher64"]=alignment(fisher_dirs_by[carrier][label_kind][:64],B)["alignment"]
            control_rows.append(row)
        stage(f"controls_{carrier}",t0)
    write_csv(OUT/"principal_angles.csv",align_rows); write_csv(OUT/"random_subspace_controls.csv",control_rows)
    stage("controls",t0)
    # Singular-direction top/bottom sweep, primary I2b plus controls.
    sweep_rows=[]; all_sweep_arrs={}; sweep_meta={}
    for carrier in carriers:
        Hs=carriers[carrier];
        for sweep,dirs in (("top",Vh[:rank]),("bottom",Vh[:rank][::-1])):
            for k in KSWEEP:
                name=f"{carrier}__{sweep}_{k}"; B=dirs[:min(k,len(dirs))].astype(np.float32); all_sweep_arrs[name]={s:Hs[s]@B.T for s in SPLITS}; sweep_meta[name]=(carrier,sweep,int(k))
    all_sweep_probe=probe_reps(all_sweep_arrs,infos,pmap,jlabels,device,"all-carriers",steps=SWEEP_STEPS)
    for name,(carrier,sweep,k) in sweep_meta.items():
            rec=all_sweep_probe[name]
            for metric in ("entity","relation","joint"): sweep_rows.append({"carrier":carrier,"sweep":sweep,"k":k,"metric":metric,"train":rec[metric]["train"]["accuracy"],"val":rec[metric]["val"]["accuracy"],"entity_testA":rec[metric]["testA"]["accuracy"],"testA":rec[metric]["testA"]["accuracy"],"testB":rec[metric]["testB"]["accuracy"]})
    write_csv(OUT/"singular_direction_sweep.csv",sweep_rows)
    stage("singular_sweep",t0)
    # Geometry and direct contraction/difference tests.
    geom_reps={}
    for carrier in carriers:
        geom_reps[carrier]={"H":transforms[carrier]["H"],"H_R":transforms[carrier]["H_R"],"H_N":transforms[carrier]["H_N"],"W1H":transforms[carrier]["W1H"],"white_row":transforms[carrier]["white_row"],"PCA256":{s:carriers[carrier][s]@control_subspaces[carrier]["pca256"].T for s in SPLITS},"random256_0":{s:carriers[carrier][s]@control_subspaces[carrier]["random_0"].T for s in SPLITS}}
    geom_rows=[]
    for carrier, reps in geom_reps.items():
        geom_rows += [{"carrier":carrier,**r} for r in pair_summary(reps,pairsets)]
    write_csv(OUT/"geometry_metrics.csv",geom_rows)
    contraction=[]; diff=[]
    for carrier,reps in geom_reps.items():
        base=carriers[carrier]
        for repname,arrs in reps.items():
            for split in ("testA","testB"):
                for typ,label in (("P1_diff_entity_same_relation","entity"),("P2_same_entity_diff_relation","relation")):
                    ab=pairsets[split][typ]; d0=np.linalg.norm(base[split][ab[:,0]]-base[split][ab[:,1]],axis=1); d1=np.linalg.norm(arrs[split][ab[:,0]]-arrs[split][ab[:,1]],axis=1); ratio=d1/np.maximum(d0,1e-12); contraction.append({"carrier":carrier,"representation":repname,"split":split,"difference_type":label,"contraction_mean":float(ratio.mean()),"contraction_std":float(ratio.std()),"contraction_median":float(np.median(ratio)),"contraction_p10":float(np.quantile(ratio,.10)),"contraction_p90":float(np.quantile(ratio,.90))})
        H=base["testA"]
        for typ,label in (("P1_diff_entity_same_relation","entity_difference"),("P2_same_entity_diff_relation","relation_difference")):
            ab=pairsets["testA"][typ]; d=H[ab[:,0]]-H[ab[:,1]]; den=np.maximum(np.linalg.norm(d,axis=1),1e-12); dr=d@VR.T; rp=np.linalg.norm(dr,axis=1)/den; npart=np.linalg.norm(d-dr@VR,axis=1)/den; diff.append({"carrier":carrier,"split":"testA","difference_type":label,"row_fraction_mean":float(rp.mean()),"row_fraction_std":float(rp.std()),"row_fraction_median":float(np.median(rp)),"null_fraction_mean":float(npart.mean()),"null_fraction_std":float(npart.std()),"null_fraction_median":float(np.median(npart))})
    write_csv(OUT/"contraction_metrics.csv",contraction); write_csv(OUT/"difference_projection.csv",diff)
    stage("geometry",t0)
    dump(OUT/"pair_sets.json",{s:{k:int(len(v)) for k,v in pairsets[s].items()} for s in SPLITS})
    arch=["W1 SVD / subspace protocol","",f"W1 shape = {tuple(W.shape)}; PyTorch forward is Y = H @ W1.T + b1.",f"rank(W1) = {rank}; row-space dimension = {rank}; nullspace dimension = {W.shape[1]-rank}.","Retained row-space projector: P_R = V_R.T @ V_R, where V_R = Vh[:rank].","Nullspace projector: P_N = I - P_R; H_R = H @ P_R; H_N = H - H_R.","Whitened row-space coordinates: H @ V_R.T (no singular-value weighting).","Real first-layer output: H @ W1.T + b1.","All KeyEncoder weights, Ouro/backbone, memory path, matcher, and production architecture remained frozen."]
    (OUT/"w1_projection_definition.txt").write_text("\n".join(arch)+"\n",encoding="utf-8")
    # Verdict: evidence-based and conservative.
    pr=[r for r in probe_rows if r["carrier"]=="K-I2b" and r["split"]=="testA"]; get=lambda rep,metric:next((r[metric] for r in pr if r["representation"]==rep),None)
    ent_hr=get("H_R","entity"); ent_hn=get("H_N","entity"); rel_hr=get("H_R","relation"); rel_hn=get("H_N","relation"); ent_white=get("white_row","entity"); ent_y=get("W1H","entity")
    rand=[r["entity_testA"] for r in control_rows if r["carrier"]=="K-I2b" and r["subspace"].startswith("random_")]; rand_mean=float(np.mean(rand)); rand_std=float(np.std(rand)); percentile=100*float(np.mean(np.asarray(rand)<=ent_hr))
    ce=next(r["contraction_mean"] for r in contraction if r["carrier"]=="K-I2b" and r["representation"]=="W1H" and r["split"]=="testA" and r["difference_type"]=="entity"); cr=next(r["contraction_mean"] for r in contraction if r["carrier"]=="K-I2b" and r["representation"]=="W1H" and r["split"]=="testA" and r["difference_type"]=="relation")
    ae=next(r["alignment"] for r in align_rows if r["carrier"]=="K-I2b" and r["subspace_kind"]=="classifier" and r["label_kind"]=="entity" and r["k"]==64); ar=next(r["alignment"] for r in align_rows if r["carrier"]=="K-I2b" and r["subspace_kind"]=="classifier" and r["label_kind"]=="relation" and r["k"]==64)
    if ent_hr is not None and ent_hn is not None and ent_hn-ent_hr>=.15 and ar-ae>=.10: primary="K-A ROW-SPACE SELECTIVE ENTITY SUPPRESSION"
    elif ent_white is not None and ent_y is not None and ent_white-ent_y>=.15: primary="K-C SINGULAR-WEIGHTING DOMINATED"
    elif (ent_hr is not None and ent_white is not None and ent_hr<.50 and ent_white<.50) and rand_mean-ent_hr<.08: primary="K-B ROW-SPACE GENERALLY COMPRESSIVE, NOT ENTITY-SPECIFIC"
    elif ent_hr is not None and ent_y is not None and ent_hr<.50 and ent_y<.50: primary="K-D MIXED ROW-SPACE + SINGULAR-WEIGHTING FAILURE"
    else: primary="K-E NO DIRECT EVIDENCE FOR LEARNED ENTITY INVARIANCE"
    secondary="singular-value weighting and downstream geometry remain contributory" if abs(ce-cr)>.10 else "no independent secondary mechanism established"
    def val(rep,metric): return pct(get(rep,metric))
    lines=["# CLR v2.4 — O3b-K First-Linear Subspace Attribution","","O3b-K Verdict","",f'"O3b-K: {primary}"',"",f"Secondary:\n\"{secondary}\"","","Row-space attribution","",f'Entity in row-space:\n"{val("H_R","entity")}"',f'Entity in nullspace:\n"{val("H_N","entity")}"',f'Relation in row-space:\n"{val("H_R","relation")}"',f'Relation in nullspace:\n"{val("H_N","relation")}"',f'Joint:\n"row={val("H_R","joint")}; null={val("H_N","joint")}"',"","Discriminative-subspace alignment","",f'Entity alignment:\n"classifier top-64 alignment={ae:.4f}; Fisher top-64 alignment={next(r["alignment"] for r in align_rows if r["carrier"]=="K-I2b" and r["subspace_kind"]=="fisher" and r["label_kind"]=="entity" and r["k"]==64):.4f}"',f'Relation alignment:\n"classifier top-64 alignment={ar:.4f}; Fisher top-64 alignment={next(r["alignment"] for r in align_rows if r["carrier"]=="K-I2b" and r["subspace_kind"]=="fisher" and r["label_kind"]=="relation" and r["k"]==64):.4f}"',"","Learned vs controls","",f'Learned-256:\n"row-space entity={val("H_R","entity")}; relation={val("H_R","relation")}; joint={val("H_R","joint")}"',f'PCA-256:\n"entity={pct(next(r["entity_testA"] for r in control_rows if r["carrier"]=="K-I2b" and r["subspace"]=="pca256") )}; relation={pct(next(r["relation_testA"] for r in control_rows if r["carrier"]=="K-I2b" and r["subspace"]=="pca256"))}; joint={pct(next(r["joint_testA"] for r in control_rows if r["carrier"]=="K-I2b" and r["subspace"]=="pca256"))}"',f'Random-256 mean ± std:\n"entity={pct(rand_mean)} ± {pct(rand_std)}"',f'Learned entity percentile among random:\n"{percentile:.1f}th percentile"',"","Singular weighting","",f'Whitened row-space:\n"entity={val("white_row","entity")}; relation={val("white_row","relation")}; joint={val("white_row","joint")}"',f'Real W1 output:\n"entity={val("W1H","entity")}; relation={val("W1H","relation")}; joint={val("W1H","joint")}"',f'Interpretation:\n"Whitening changes entity from {pct(ent_white)} to {pct(ent_y)}; the difference is used to separate row-space selection from singular-value weighting."',"","Direct invariance test","",f'Entity contraction:\n"C_e={ce:.4f}"',f'Relation contraction:\n"C_r={cr:.4f}"',f'Interpretation:\n"C_e/C_r={ce/max(cr,1e-12):.4f}; PCA/random controls are in contraction_metrics.csv."',"","Core answer","",f"The I2b carrier is analyzed before any production addressing path: W1 row-space and nullspace probes show whether entity and relation directions are geometrically retained or discarded. Learned row-space is compared with label-free PCA-256 and ten random orthogonal 256-D subspaces, while whitened coordinates isolate singular-value weighting from row-space selection. The direct same-relation/different-entity contraction test and counterfactual difference projection provide a geometric test of entity invariance rather than relying only on classifier accuracy. No architecture, matcher, memory policy, or Ouro computation was changed.","","Reconnect allowed","","No","","O3c allowed","","No","","Repeated V update contamination","","Unresolved; not touched.","","Recommended next experiment","","Only propose one narrowly scoped follow-up targeting the primary diagnosed mechanism; do not execute it in O3b-K.",""]
    (OUT/"v24_o3b_k_first_linear_subspace_report.md").write_text("\n".join(lines),encoding="utf-8")
    dump(OUT/"gate_evaluation.json",{"diagnostic_only":True,"reconnect":False,"o3c":False,"production_architecture_changed":False,"new_projection_trained":False,"repeated_v_update_contamination":"unresolved; not touched","stop_rule":"O3b-K complete; stop after report"})
    dump(OUT/"config.json",{"phase":"CLR-v2.4-O3b-K","cache":str(CACHE_ROOT),"source_key_checkpoint":str(O21B),"carriers":list(carriers),"n_random":N_RANDOM,"probe_steps_core":PROBE_STEPS,"probe_steps_controls":CONTROL_STEPS,"probe_steps_sweep":SWEEP_STEPS,"classifier_steps":CLASSIFIER_STEPS,"w1_shape":list(W.shape),"rank":rank,"reconnect":False,"o3c":False,"architecture_changed":False,"wall_clock_s":time.perf_counter()-t0})
    make_figures(probe_rows,align_rows,control_rows,sweep_rows,contraction,diff)
    stage("complete",t0)
    print(json.dumps({"out":str(OUT),"report":str(OUT/"v24_o3b_k_first_linear_subspace_report.md"),"primary":primary,"entity_row":ent_hr,"entity_null":ent_hn,"relation_row":rel_hr,"relation_null":rel_hn,"Ce":ce,"Cr":cr,"wall_clock_s":time.perf_counter()-t0},ensure_ascii=False))


if __name__ == "__main__": main()
