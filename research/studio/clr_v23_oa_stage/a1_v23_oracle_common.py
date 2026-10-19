import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v22_storage_common import DEFAULT_CACHE, FactCache, get_device, json_dump, seed_all
from clr_v23_oracle import OracleAllocationCore


DEFAULT_OUT = r"D:\training\checkpoints\clr_v23_oracle"
CITY_CLASSES = 64
SLOTS = 4
DIM = 64


def rows_for(cache, split, cap, limit=None):
    rows = cache.rows_for_capacity(split, cap)
    return rows if limit is None else rows[:min(len(rows), int(limit))]


def make_heads(device, dtype):
    return nn.ModuleList([nn.Linear(256, 64) for _ in range(4)]).to(device=device, dtype=dtype)


def select_logits(heads, state, pos):
    flat = state.flatten(1); chunks=[]; ys=[]
    for p in range(4):
        m = pos == p
        if m.any(): chunks.append(heads[p](flat[m])); ys.append(m)
    return chunks, ys


@torch.no_grad()
def eval_capacity(model, heads, cache, split, cap, rows=None, batch_size=128, fixed_position=None):
    model.eval(); device=next(model.parameters()).device
    if rows is None: rows=rows_for(cache, split, cap)
    correct=total=0; by=[0]*cap; bt=[0]*cap
    for i in range(0,len(rows),batch_size):
        rb=rows[i:i+batch_size]; facts,labels,pos,y=cache.batch(split,rb,cap,device)
        slots=list(range(cap))
        state,_=model.forward_facts(facts,slot_indices=slots)
        if fixed_position is not None:
            pos=torch.full_like(pos,int(fixed_position)); y=labels[:,int(fixed_position)]
        for p in range(4):
            m=pos==p
            if m.any():
                pred=heads[p](state[m].flatten(1)).argmax(1); yy=y[m]
                correct += int((pred==yy).sum()); total += int(m.sum())
                for q in range(cap):
                    qm=m & (pos==q); bt[q]+=int(qm.sum())
                    if qm.any(): by[q]+=int((heads[p](state[qm].flatten(1)).argmax(1)==y[qm]).sum())
    return {"acc":correct/max(total,1),"correct":correct,"total":total,
            "by_position":[by[i]/max(bt[i],1) for i in range(cap)],"position_total":bt}


def fixed_matrix(model, heads, cache, split, cap, rows=None):
    return [eval_capacity(model,heads,cache,split,cap,rows=rows,fixed_position=p) for p in range(cap)]


def save_ckpt(path, model, heads, optimizer=None, step=0, stage="", metrics=None, extra=None):
    d={"model":model.state_dict(),"heads":heads.state_dict(),"step":step,"stage":stage,"metrics":metrics or {},
       "config":{"slots":4,"dim":64,"allocation":"oracle_position","architecture":"v2.3-OA"}}
    if optimizer is not None:d["optimizer"]=optimizer.state_dict()
    if extra:d.update(extra)
    torch.save(d,path)


def load_ckpt(path, device):
    d=torch.load(path,map_location=device); dtype=torch.float32
    m=OracleAllocationCore(device=device,dtype=dtype); h=make_heads(device,dtype); m.load_state_dict(d["model"]); h.load_state_dict(d["heads"]); return m,h,d


class Probe(nn.Module):
    def __init__(self, dim): super().__init__(); self.fc=nn.Linear(dim,64)
    def forward(self,x): return self.fc(x)


def probe_train(xtr,ytr,xv,yv,xte,ye,dim,device,steps=100,save_path=None):
    torch.manual_seed(42); model=Probe(dim).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.0)
    xtr=torch.from_numpy(xtr).float().to(device); ytr=torch.from_numpy(ytr).long().to(device)
    xv=torch.from_numpy(xv).float().to(device); yv=torch.from_numpy(yv).long().to(device)
    best=-1.; best_state=None; stale=0
    for ep in range(1,steps+1):
        model.train(); perm=torch.randperm(len(xtr),device=device)
        for i in range(0,len(perm),128):
            idx=perm[i:i+128]; opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(xtr[idx]),ytr[idx]); loss.backward(); opt.step()
        with torch.no_grad(): va=(model(xv).argmax(1)==yv).float().mean().item()
        if va>best: best=va; stale=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=10: break
    if best_state:model.load_state_dict(best_state)
    with torch.no_grad(): tr=(model(xtr).argmax(1)==ytr).float().mean().item(); ta=(model(torch.from_numpy(xte).float().to(device)).argmax(1)==torch.from_numpy(ye).long().to(device)).float().mean().item()
    if save_path:torch.save({"model":model.state_dict(),"val_acc":best},save_path)
    return {"train_acc":tr,"val_acc":best,"test_acc":ta,"epochs":ep}
