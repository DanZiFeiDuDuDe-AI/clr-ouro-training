# -*- coding: utf-8 -*-
import os, json, random
import numpy as np
import torch
import torch.nn as nn
from clr_v2 import CLRv2, LATENT_SLOTS, LATENT_DIM

N_CLASSES=64
CHANCE=1.0/N_CLASSES
CAPACITIES=(1,2,4,8,16)
MAX_N=16
LS0_DIR=os.environ.get("LS0_DIR","D:/training/checkpoints/clr_v2_a1_ls0")
LS1_DIR=os.environ.get("LS1_DIR","D:/training/checkpoints/clr_v2_a1_ls1")
LS2_DIR=os.environ.get("LS2_OUT_DIR","D:/training/checkpoints/clr_v2_a1_ls2")
CACHE_DIR=os.path.join(LS2_DIR,"cache")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_cache(cache_dir=CACHE_DIR):
    mf=json.load(open(os.path.join(cache_dir,"manifest.json"),encoding="utf-8"))
    dm=json.load(open(os.path.join(cache_dir,"data_manifest.json"),encoding="utf-8"))
    out={}
    qp=json.load(open(os.path.join(cache_dir,"query_persons.json"),encoding="utf-8"))
    qh=np.load(os.path.join(cache_dir,"query_person_H.npz"))["H"].astype(np.float32)
    out["_query_map"]={p:qh[k] for k,p in enumerate(qp)}
    for split in ["train","val","testA","testB"]:
        z=np.load(os.path.join(cache_dir,split+"_data.npz"))
        meta=[json.loads(x) for x in open(os.path.join(cache_dir,split+"_meta.jsonl"),encoding="utf-8")]
        out[split]={"fact_H":z["fact_H"].astype(np.float32),
                    "fact_offsets":z["fact_offsets"].astype(np.int64),
                    "query_H":z["query_H"].astype(np.float32),
                    "Y":z["Y"].astype(np.int64),
                    "N":z["N"].astype(np.int64),
                    "query_idx":z["query_idx"].astype(np.int64),
                    "fact_Y":z["fact_Y"].astype(np.int64),
                    "meta":meta}
    return mf,dm,out

def load_model(ckpt_path,device="cuda"):
    clrv2=CLRv2()
    head=nn.Linear(LATENT_SLOTS*LATENT_DIM,N_CLASSES,dtype=torch.bfloat16,device=device)
    ck=torch.load(ckpt_path,map_location="cpu",weights_only=False)
    clrv2.load_state_dict(ck["clrv2"],strict=True)
    head.load_state_dict(ck["head"],strict=True)
    for p in clrv2.write.parameters(): p.requires_grad=False
    clrv2.eval(); head.eval()
    return clrv2,head,ck

def trainable_params(clrv2,head):
    p=[x for x in clrv2.parameters() if x.requires_grad]+list(head.parameters())
    n=[x for x,xp in clrv2.named_parameters() if xp.requires_grad]+["state_head.weight","state_head.bias"]
    return p,n

def batch_fact_H(data,idx,t,orders=None):
    idx=np.asarray(idx,dtype=np.int64)
    ns=data["N"][idx]; b=len(idx)
    h=np.zeros((b,2048),dtype=np.float32)
    active=ns>t
    if np.any(active):
        rows=idx[active]
        if orders is None:
            pos=data["fact_offsets"][rows]+t
        else:
            pos=data["fact_offsets"][rows]+orders[idx[active],t]
        h[active]=data["fact_H"][pos]
    return h,active

def forward_batch(clrv2,data,idx,device="cuda",orders=None):
    idx=np.asarray(idx,dtype=np.int64); b=len(idx)
    ns=data["N"][idx]
    S=clrv2.init_state(b,device)
    for t in range(int(ns.max())):
        h,active=batch_fact_H(data,idx,t,orders)
        S_new=clrv2.update(S,torch.from_numpy(h).to(device).to(torch.bfloat16),latent_steps=1)
        mask=torch.from_numpy(active[:,None,None]).to(device)
        S=torch.where(mask,S_new,S)
    S_memory=S
    qh=torch.from_numpy(data["query_H"][idx]).to(device).to(torch.bfloat16)
    S_query=clrv2.update(S_memory,qh,latent_steps=1)
    return S_memory,S_query

def collect_states(clrv2,data,device="cuda",batch=512,seed=20260907,orders=None):
    n=len(data["Y"])
    mem=np.zeros((n,LATENT_SLOTS,LATENT_DIM),dtype=np.float32)
    qry=np.zeros_like(mem)
    torch.manual_seed(seed); clrv2.eval()
    with torch.no_grad():
        for st in range(0,n,batch):
            en=min(n,st+batch); idx=np.arange(st,en,dtype=np.int64)
            m,q=forward_batch(clrv2,data,idx,device,orders)
            mem[st:en]=m.float().cpu().numpy(); qry[st:en]=q.float().cpu().numpy()
    return mem,qry

def predict_head(head,states,device="cuda",batch=1024):
    out=[]; head.eval()
    with torch.no_grad():
        for st in range(0,len(states),batch):
            x=torch.from_numpy(states[st:st+batch]).to(device).to(torch.bfloat16)
            out.append(head(x.reshape(x.shape[0],-1)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def capacity_curve(pred,y,ns):
    out={}
    for n in CAPACITIES:
        mask=ns==n; a=float((pred[mask]==y[mask]).mean()) if np.any(mask) else None
        out[str(n)]=a
    a1=out["1"]
    for n in CAPACITIES:
        out[str(n)+"_normalized_capacity"]=float((out[str(n)]-CHANCE)/(a1-CHANCE)) if a1 is not None and abs(a1-CHANCE)>1e-9 else None
    return out

def position_matrix(pred,y,ns,qidx):
    out={}
    for n in CAPACITIES:
        base=ns==n
        rows={}
        specs={"first":qidx==0,"last":qidx==n-1}
        if n>=4: specs["middle"]=(qidx==(n//2-1))|(qidx==(n//2))
        else: specs["middle"]=np.zeros(len(y),dtype=bool)
        specs["all"]=np.ones(len(y),dtype=bool)
        for label,mask0 in specs.items():
            mask=base&mask0
            rows[label]={"n":int(mask.sum()),"acc":float((pred[mask]==y[mask]).mean()) if mask.any() else None}
        if rows["all"]["acc"] is not None:
            rows["primacy_gap"]=None if rows["first"]["acc"] is None else rows["first"]["acc"]-rows["all"]["acc"]
            rows["recency_gap"]=None if rows["last"]["acc"] is None else rows["last"]["acc"]-rows["all"]["acc"]
            rows["middle_gap"]=None if rows["middle"]["acc"] is None else rows["middle"]["acc"]-rows["all"]["acc"]
        out[str(n)]=rows
    return out

def fact_shortcuts(data):
    y=data["Y"]; ns=data["N"]; offs=data["fact_offsets"]; fy=data["fact_Y"]
    first=np.zeros(len(y),dtype=np.int64); last=np.zeros(len(y),dtype=np.int64)
    for i in range(len(y)):
        first[i]=fy[offs[i]]; last[i]=fy[offs[i]+ns[i]-1]
    return {"first_city_baseline":capacity_curve(first,y,ns),
            "last_city_baseline":capacity_curve(last,y,ns)}

def safe_perm(n,forbidden,seed=1234):
    rng=np.random.RandomState(seed); p=np.roll(rng.permutation(n),1)
    for _ in range(max(10,n*5)):
        bad=[i for i in range(n) if p[i]==i or forbidden(i,int(p[i]))]
        if not bad: return p
        i=bad[0]; found=None
        for j0 in rng.permutation(n):
            j=int(j0)
            if j==i or p[j]==i or p[i]==j: continue
            if forbidden(i,int(p[j])) or forbidden(j,int(p[i])): continue
            found=j; break
        if found is None: p=np.roll(p,1)
        else: p[i],p[found]=p[found],p[i]
    raise RuntimeError("could not construct safe permutation")

def mean_cos(a,b):
    af=a.reshape(len(a),-1); bf=b.reshape(len(b),-1)
    return np.sum(af*bf,axis=1)/(np.linalg.norm(af,axis=1)*np.linalg.norm(bf,axis=1)+1e-9)

def movement(mem,q):
    c=mean_cos(mem,q); d=np.linalg.norm(q.reshape(len(q),-1)-mem.reshape(len(mem),-1),axis=1)
    return {"mean_cosine_memory_query":float(c.mean()),"std_cosine_memory_query":float(c.std()),
            "mean_l2_memory_query":float(d.mean()),"std_l2_memory_query":float(d.std())}

def geometry(states,ns,seed=777):
    out={}
    rng=np.random.RandomState(seed)
    for n in CAPACITIES:
        mask=ns==n; x=states[mask]
        if len(x)==0: continue
        per=np.linalg.norm(x,axis=2).mean(axis=0)
        pair={}
        for a in range(LATENT_SLOTS):
            for b in range(a+1,LATENT_SLOTS):
                aa=x[:,a,:]; bb=x[:,b,:]
                pair[str(a)+"-"+str(b)]=float(np.mean(np.sum(aa*bb,axis=1)/(np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1)+1e-9)))
        flat=x.reshape(len(x),-1); vals=[]
        for _ in range(min(5000,len(x)*5)):
            i=int(rng.randint(0,len(x))); j=int(rng.randint(0,len(x)))
            if i!=j: vals.append(float(np.dot(flat[i],flat[j])/(np.linalg.norm(flat[i])*np.linalg.norm(flat[j])+1e-9)))
        out[str(n)]={"n":int(len(x)),"per_slot_norm_mean":[float(v) for v in per],
                     "slot_slot_cosine":pair,"inter_episode_cosine_mean":float(np.mean(vals)) if vals else None}
    return out

def slot_probe(train_states,train_y,eval_states,eval_y,ns_train,ns_eval,device="cuda",steps=200,seed=20260907):
    out={}
    for n in CAPACITIES:
        out[str(n)]={}
        mt=ns_train==n
        for slot in range(LATENT_SLOTS):
            X=torch.from_numpy(train_states[mt,n*0:][:,slot,:]).to(device).to(torch.bfloat16)
            Y=torch.from_numpy(train_y[mt]).to(device)
            lin=nn.Linear(LATENT_DIM,N_CLASSES,dtype=torch.bfloat16,device=device)
            nn.init.zeros_(lin.bias); opt=torch.optim.AdamW(lin.parameters(),lr=3e-4,weight_decay=.01)
            for step in range(steps):
                idx=torch.arange(len(Y),device=device) if len(Y)<=4096 else torch.randperm(len(Y),device=device)[:4096]
                opt.zero_grad(); nn.CrossEntropyLoss()(lin(X[idx]),Y[idx]).backward(); opt.step()
            item={}
            with torch.no_grad():
                item["train_acc"]=float((lin(X).argmax(1)==Y).float().mean())
                for name,(es,ey,ens) in eval_states.items():
                    me=ens==n; Xe=torch.from_numpy(es[me][:,slot,:]).to(device).to(torch.bfloat16)
                    Ye=torch.from_numpy(ey[me]).to(device)
                    item[name+"_acc"]=float((lin(Xe).argmax(1)==Ye).float().mean())
            out[str(n)][str(slot)]=item
            del lin,opt,X,Y
            torch.cuda.empty_cache()
    return out

def peak_vram_gb():
    return round(torch.cuda.max_memory_allocated()/(1024**3),3) if torch.cuda.is_available() else None
