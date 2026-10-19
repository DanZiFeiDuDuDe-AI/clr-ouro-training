import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from a1_v23_oracle_common import (
    DEFAULT_CACHE, DEFAULT_OUT, FactCache, OracleAllocationCore, eval_capacity,
    get_device, json_dump, load_ckpt, make_heads, rows_for, save_ckpt, seed_all,
)


def sanity(model, loss):
    loss.backward(); out={}
    for name,p in [("content_read",model.content_read.weight),("shared_gru",model.shared_gru.weight_ih),("init_s",model.init_s)]:
        out[name]=p.grad is not None and bool(torch.isfinite(p.grad).all())
    out["all_required"]=all(out.values())
    if not out["all_required"]: raise RuntimeError("oracle gradient sanity failed: "+repr(out))
    return out


def train_stage(model,heads,cache,out_dir,stage,steps,caps,eval_caps,train_limit=None,val_limit=None,smoke=False,optimizer=None):
    dev=get_device(); model.train(); heads.train()
    if optimizer is None: optimizer=torch.optim.AdamW(list(model.parameters())+list(heads.parameters()),lr=3e-4,weight_decay=0.01)
    rng=np.random.default_rng(2300+(0 if stage=="O0" else 1)); batch=32 if smoke else 64
    tr={c:rows_for(cache,"train",c,train_limit) for c in caps}; va={c:rows_for(cache,"val",c,val_limit) for c in eval_caps}
    logs=[]; best_score=-1.; best_worst=-1.; best_metrics={}; first_grad=None; t0=time.perf_counter()
    best_path=Path(out_dir)/("best_o0.pt" if stage=="O0" else "best_model.pt")
    for step in range(1,steps+1):
        cap=int(caps[rng.integers(len(caps))]); rows=tr[cap]
        rb=rows if len(rows)<=batch else rng.choice(rows,size=batch,replace=False)
        facts,labels,_,_=cache.batch("train",rb,cap,dev)
        target_pos=torch.from_numpy(rng.integers(0,cap,size=len(rb),dtype=np.int64)).to(dev)
        y=labels[torch.arange(len(rb),device=dev),target_pos]
        optimizer.zero_grad(set_to_none=True); state,_=model.forward_facts(facts)
        parts=[]
        for p in range(4):
            m=target_pos==p
            if m.any(): parts.append(F.cross_entropy(heads[p](state[m].flatten(1)).float(),y[m]))
        loss=sum(parts)/len(parts)
        if first_grad is None:first_grad=sanity(model,loss)
        else:loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(heads.parameters()),1.0); optimizer.step()
        if step==1 or step% (5 if smoke else 100)==0 or step==steps:
            val={str(c):eval_capacity(model,heads,cache,"val",c,rows=va[c]) for c in eval_caps}
            score=float(np.mean([val[str(c)]["acc"] for c in eval_caps])); worst=min(val.get("4",{}).get("by_position",[score]))
            rec={"step":step,"stage":stage,"loss":float(loss.detach().float()),"grad_norm":float(gn),"val":val,"elapsed_s":time.perf_counter()-t0}; logs.append(rec)
            if score>best_score or (abs(score-best_score)<1e-12 and worst>best_worst):
                best_score=score; best_worst=worst; best_metrics=val
                save_ckpt(best_path,model,heads,optimizer,step,stage,val,{"gradient_sanity":first_grad})
    return {"stage":stage,"steps":steps,"wall_clock_s":time.perf_counter()-t0,"best_score":best_score,"best_worst_position":best_worst,"best_metrics":best_metrics,"gradient_sanity":first_grad,"logs":logs,"optimizer":optimizer,"best_path":str(best_path)}


def main():
    seed_all(2300); smoke=os.environ.get("CLR_V23_OA_SMOKE","0")=="1"; dev=get_device(); cache=FactCache(os.environ.get("V22_CACHE_DIR",DEFAULT_CACHE)); out=Path(os.environ.get("V23_OUT_DIR",DEFAULT_OUT)); out.mkdir(parents=True,exist_ok=True)
    o0_steps,o1_steps=(20,20) if smoke else (2000,3000); limit=256 if smoke else None; vlimit=128 if smoke else None
    # Match the proven SS-64 diagnostic path: fresh float32 ContentRead/GRU/head.
    model=OracleAllocationCore(device=dev,dtype=torch.float32); heads=make_heads(dev,torch.float32); all_logs=[]; t0=time.perf_counter()
    o0=train_stage(model,heads,cache,out,"O0",o0_steps,[1],[1],limit,vlimit,smoke); all_logs.append({k:v for k,v in o0.items() if k!="optimizer"})
    ck=torch.load(out/"best_o0.pt",map_location=dev); model.load_state_dict(ck["model"]); heads.load_state_dict(ck["heads"])
    ta=eval_capacity(model,heads,cache,"testA",1,rows=rows_for(cache,"testA",1,128 if smoke else None)); tb=eval_capacity(model,heads,cache,"testB",1,rows=rows_for(cache,"testB",1,128 if smoke else None))
    gate={"testA_n1":ta,"testB_n1":tb,"pass":bool(ta["acc"]>=.80 and tb["acc"]>=.80),"stop_below_75":bool(ta["acc"]<.75 or tb["acc"]<.75)}; json_dump(out/"o0_gate.json",gate)
    run_o1=gate["pass"] or smoke
    if not run_o1:
        save_ckpt(out/"best_model.pt",model,heads,o0["optimizer"],o0_steps,"O0-only",{"o0_gate":gate}); save_ckpt(out/"final_model.pt",model,heads,o0["optimizer"],o0_steps,"O0-only",{"o0_gate":gate})
        result={"smoke":smoke,"device":str(dev),"cache":str(cache.root),"o0":{k:v for k,v in o0.items() if k!="optimizer"},"o0_gate":gate,"o1_ran":False,"wall_clock_s":time.perf_counter()-t0,"parameter_report":model.parameter_report()}
    else:
        o0["optimizer"].load_state_dict(ck.get("optimizer",o0["optimizer"].state_dict()))
        o1=train_stage(model,heads,cache,out,"O1",o1_steps,[1,2,4],[1,2,4],limit,vlimit,smoke,o0["optimizer"]); all_logs.append({k:v for k,v in o1.items() if k!="optimizer"})
        save_ckpt(out/"final_model.pt",model,heads,o1["optimizer"],o1_steps,"O1-final",o1["best_metrics"])
        result={"smoke":smoke,"device":str(dev),"cache":str(cache.root),"o0":{k:v for k,v in o0.items() if k!="optimizer"},"o0_gate":gate,"o1":{k:v for k,v in o1.items() if k!="optimizer"},"o1_ran":True,"wall_clock_s":time.perf_counter()-t0,"parameter_report":model.parameter_report()}
    result["peak_vram_gb"]=(torch.cuda.max_memory_allocated(dev)/(1024**3)) if dev.type=="cuda" else 0.0; result["protocol"]={"lr":3e-4,"weight_decay":.01,"grad_clip":1.0,"batch_size":32 if smoke else 64,"full_bptt":True,"oracle_allocation":True,"non_target_exact_unchanged":True}
    json_dump(out/"train_log.json",{"stages":all_logs,"result":result}); print(json.dumps(result,indent=2,default=str))


if __name__=="__main__":main()
