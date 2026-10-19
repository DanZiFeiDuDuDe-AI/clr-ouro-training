import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v23_oracle_common import DEFAULT_CACHE, DEFAULT_OUT, FactCache, eval_capacity, fixed_matrix, get_device, json_dump, load_ckpt, rows_for


@torch.no_grad()
def collect(model, cache, split, cap, rows):
    dev=next(model.parameters()).device; finals=[]; stages=[]; labels=[]; pos=[]
    slots=[i%4 for i in range(cap)]
    for i in range(0,len(rows),128):
        rb=rows[i:i+128]; facts,y,p,_=cache.batch(split,rb,cap,dev); s,ss=model.forward_facts(facts,slot_indices=slots,keep_states=True)
        finals.append(s.float().cpu().numpy()); stages.append(torch.stack(ss,1).float().cpu().numpy()); labels.append(y.cpu().numpy()); pos.append(p.cpu().numpy())
    return {"final":np.concatenate(finals),"stages":np.concatenate(stages),"labels":np.concatenate(labels),"positions":np.concatenate(pos),"rows":np.asarray(rows)}


def fresh_probe(xtr,ytr,xv,yv,xA,yA,xB,yB,dim,device):
    model=nn.Linear(dim,64).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.0); xtr=torch.from_numpy(xtr).float().to(device); ytr=torch.from_numpy(ytr).long().to(device); xv=torch.from_numpy(xv).float().to(device); yv=torch.from_numpy(yv).long().to(device); best=-1.; bs=None; stale=0
    for ep in range(1,101):
        model.train(); perm=torch.randperm(len(xtr),device=device)
        for i in range(0,len(perm),128):
            idx=perm[i:i+128]; opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(xtr[idx]),ytr[idx]); loss.backward(); opt.step()
        with torch.no_grad(): va=(model(xv).argmax(1)==yv).float().mean().item()
        if va>best: best=va; stale=0; bs={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=10:break
    if bs:model.load_state_dict(bs)
    with torch.no_grad():
        tr=(model(xtr).argmax(1)==ytr).float().mean().item(); a=(model(torch.from_numpy(xA).float().to(device)).argmax(1)==torch.from_numpy(yA).long().to(device)).float().mean().item(); b=(model(torch.from_numpy(xB).float().to(device)).argmax(1)==torch.from_numpy(yB).long().to(device)).float().mean().item()
    return {"train_acc":tr,"val_acc":best,"testA_acc":a,"testB_acc":b,"epochs":ep}


def full_probe(train, val, ta, tb, position):
    return fresh_probe(train["final"].reshape(len(train["final"]),-1),train["labels"][:,position],val["final"].reshape(len(val["final"]),-1),val["labels"][:,position],ta["final"].reshape(len(ta["final"]),-1),ta["labels"][:,position],tb["final"].reshape(len(tb["final"]),-1),tb["labels"][:,position],256,next(iter([]),None))


def probe_matrix(train,val,ta,tb,device,slot_local=False):
    cap=train["final"].shape[1]; out=[]
    for slot in range(cap):
        row=[]
        for p in range(cap):
            tr=train["final"][:,slot,:] if slot_local else train["final"].reshape(len(train["final"]),-1)
            va=val["final"][:,slot,:] if slot_local else val["final"].reshape(len(val["final"]),-1)
            aa=ta["final"][:,slot,:] if slot_local else ta["final"].reshape(len(ta["final"]),-1)
            bb=tb["final"][:,slot,:] if slot_local else tb["final"].reshape(len(tb["final"]),-1)
            row.append(fresh_probe(tr,train["labels"][:,p],va,val["labels"][:,p],aa,ta["labels"][:,p],bb,tb["labels"][:,p],tr.shape[1],device))
        out.append(row)
    return out


def write_survival(train,val,ta,tb,device):
    cap=4; mat=[]; exact=[]
    for p in range(cap):
        row=[]
        for stage in range(cap+1):
            if stage < p+1: row.append(None); continue
            tr=train["stages"][:,stage,p,:]; va=val["stages"][:,stage,p,:]; aa=ta["stages"][:,stage,p,:]; bb=tb["stages"][:,stage,p,:]
            row.append(fresh_probe(tr,train["labels"][:,p],va,val["labels"][:,p],aa,ta["labels"][:,p],bb,tb["labels"][:,p],64,device))
        mat.append(row)
    # Differences on non-target slots must be exactly zero by construction.
    for split,rec in [("train",train),("testA",ta),("testB",tb)]:
        vals=[]
        for t in range(cap):
            before=rec["stages"][:,t]; after=rec["stages"][:,t+1]; other=[j for j in range(cap) if j!=t]
            vals.append(float(np.max(np.abs(after[:,other,:]-before[:,other,:]))))
        exact.append({"split":split,"max_abs_non_target_diff_by_write":vals,"assertion_pass":max(vals or [0.0])==0.0})
    return {"rows_fact_position":list(range(cap)),"columns_after_write_stage":list(range(cap+1)),"matrix":mat,"numerical_preservation":exact}


@torch.no_grad()
def overload(model,heads,cache,split,cap,rows):
    dev=next(model.parameters()).device; good=[0]*cap; total=[0]*cap
    for i in range(0,len(rows),128):
        rb=rows[i:i+128]; facts,labels,_,_=cache.batch(split,rb,cap,dev); s,_=model.forward_facts(facts,slot_indices=[p%4 for p in range(cap)])
        for p in range(cap):
            pred=heads[p%4](s.flatten(1)).argmax(1); good[p]+=int((pred==labels[:,p]).sum()); total[p]+=len(rb)
    return {"capacity":cap,"acc":sum(good)/max(sum(total),1),"by_position":[good[p]/max(total[p],1) for p in range(cap)],"note":"zero-shot position modulo 4; no N>4 training"}


def main():
    out=Path(os.environ.get("V23_OUT_DIR",DEFAULT_OUT)); dev=get_device(); cache=FactCache(os.environ.get("V22_CACHE_DIR",DEFAULT_CACHE)); log=json.loads((out/"train_log.json").read_text(encoding="utf-8")); s1=bool(log["result"].get("o1_ran",False)); model,heads,ck=load_ckpt(out/"best_model.pt",dev); t0=time.perf_counter()
    caps=[1] if not s1 else [1,2,4]; result={"stage":ck.get("stage",""),"device":str(dev),"capacity":{},"position_matrix":{},"fresh_position_probes":{},"slot_local_probes":{},"slot_purity":{},"write_survival":{},"overload_capacity":{},"params":model.parameter_report()}
    recs={}
    for split in ("train","val","testA","testB"):
        result["capacity"][split]={}
        for cap in caps:
            rows=rows_for(cache,split,cap); result["capacity"][split][str(cap)]=eval_capacity(model,heads,cache,split,cap,rows=rows); result["position_matrix"].setdefault(split,{})[str(cap)]=fixed_matrix(model,heads,cache,split,cap,rows=rows); recs[(split,cap)]=collect(model,cache,split,cap,rows)
    ta=result["capacity"].get("testA",{}); tb=result["capacity"].get("testB",{}); posA=result["position_matrix"].get("testA",{}).get("4",[]); posB=result["position_matrix"].get("testB",{}).get("4",[])
    o1_pass=bool(s1 and ta.get("1",{}).get("acc",0)>=.80 and ta.get("2",{}).get("acc",0)>=.75 and ta.get("4",{}).get("acc",0)>=.70 and tb.get("1",{}).get("acc",0)>=.75 and tb.get("2",{}).get("acc",0)>=.70 and tb.get("4",{}).get("acc",0)>=.65 and min([x["acc"] for x in posA] or [0])>=.60 and min([x["acc"] for x in posB] or [0])>=.60)
    for cap in caps:
        tr=recs[("train",cap)]; va=recs[("val",cap)]; a=recs[("testA",cap)]; b=recs[("testB",cap)]; result["fresh_position_probes"][str(cap)]=[fresh_probe(tr["final"].reshape(len(tr["final"]),-1),tr["labels"][:,p],va["final"].reshape(len(va["final"]),-1),va["labels"][:,p],a["final"].reshape(len(a["final"]),-1),a["labels"][:,p],b["final"].reshape(len(b["final"]),-1),b["labels"][:,p],256,dev) for p in range(cap)]
    if s1:
        tr=recs[("train",4)]; va=recs[("val",4)]; a=recs[("testA",4)]; b=recs[("testB",4)]; result["slot_local_probes"]["testA"]=probe_matrix(tr,va,a,b,dev,True); result["slot_local_probes"]["testB"]=probe_matrix(tr,va,b,a,dev,True); result["slot_purity"]={"testA":result["slot_local_probes"]["testA"],"testB":result["slot_local_probes"]["testB"]}; result["write_survival"]=write_survival(tr,va,a,b,dev)
    else:
        result["slot_local_probes"]={"skipped":"O0 gate failed; O1 not entered"}; result["slot_purity"]={"skipped":"O0 gate failed"}; result["write_survival"]={"skipped":"O0 gate failed"}
    if o1_pass:
        for split in ("testA","testB"): result["overload_capacity"][split]={str(c):overload(model,heads,cache,split,c,rows_for(cache,split,c)) for c in (8,16)}
    else: result["overload_capacity"]={"skipped":"O1 storage gate not passed"}
    result["o0_gate"]=log["result"].get("o0_gate",{}); result["o1_gate"]={"pass":o1_pass}; result["baseline_comparison"]={"original_LS2_N4":{"testA":.165,"testB":.1925},"v21a_N4":{"testA":.185,"testB":.19},"v21b_N4":{"testA":.1275,"testB":.155},"v22_S0_N1":{"testA":.4175,"testB":.375}}
    if not result["o0_gate"].get("pass",False): verdict="O0 FAIL"
    elif o1_pass: verdict="O1 STORAGE PASS"
    else: verdict="PARTIAL" if ta.get("4",{}).get("acc",0)>1/64+.05 else "FAIL"
    result["verdict"]=verdict; result["wall_clock_eval_s"]=time.perf_counter()-t0; json_dump(out/"v23_results.json",result); json_dump(out/"position_matrix.json",result["position_matrix"]); json_dump(out/"fresh_position_probes.json",result["fresh_position_probes"]); json_dump(out/"slot_purity.json",result["slot_purity"]); json_dump(out/"slot_local_probes.json",result["slot_local_probes"]); json_dump(out/"write_survival_matrix.json",result["write_survival"]); json_dump(out/"overload_capacity.json",result["overload_capacity"]); report(out,result)
    print(json.dumps({"verdict":verdict,"capacity":result["capacity"],"o0_gate":result["o0_gate"],"o1_gate":result["o1_gate"]},indent=2))


def report(out,r):
    lines=["# CLR-v2.3-OA — Oracle Allocation Sanity","","Oracle allocation only: fact position k always writes slot k. No learned router, query, or A2.","","## Verdict","",r["verdict"],"","## Capacity"]
    for s in ("testA","testB"):
        lines += ["",s]
        for c,m in r["capacity"].get(s,{}).items(): lines.append("- N=%s: %.3f"%(c,m["acc"]))
    lines += ["","## Position matrix","```json",json.dumps(r["position_matrix"],indent=2),"```","","## Fresh full-state probes","```json",json.dumps(r["fresh_position_probes"],indent=2),"```","","## Slot purity/local probes","```json",json.dumps(r["slot_purity"],indent=2),"```","","## Write survival and exact preservation","```json",json.dumps(r["write_survival"],indent=2),"```","","## Baselines","```json",json.dumps(r["baseline_comparison"],indent=2),"```","","## Interpretation","", "- Oracle allocation isolates storage from learned allocation.","- N>4 is only evaluated zero-shot after an O1 pass.","- This run stops after CLR-v2.3-OA."]
    (out/"v23_report.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":main()
