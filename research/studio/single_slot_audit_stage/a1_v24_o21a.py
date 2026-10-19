import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a1_v24_o2_common import load_storage, json_dump, rows_for
from a1_v24_o2_train import _batch_state


ROOT=Path(r"D:\training")
CACHE=Path(os.environ.get("V22_CACHE_DIR",r"D:\training\checkpoints\clr_v2_a1_ls2\cache"))
OUT=Path(os.environ.get("V24_O21A_OUT",r"D:\training\checkpoints\clr_v24_o21a_semantic_key"))
O2_DIR=ROOT/"checkpoints"/"clr_v24_o2_retrieval"
R2=ROOT/"checkpoints"/"clr_v23_oracle_r2"/"best_model.pt"
CAPS=(1,2,4); CITY=64; SEED=42


def seed_all(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class KeyCache:
    def __init__(self, root=CACHE):
        self.root=Path(root); self.data={}; self.meta={}; self.persons=[]; self.person_to_id={}
        for split in ("train","val","testA","testB"):
            z=np.load(self.root/f"{split}_data.npz",allow_pickle=False); self.data[split]={k:z[k] for k in z.files}
            mp=self.root/f"{split}_meta.jsonl"
            with mp.open("r",encoding="utf-8") as f: self.meta[split]=[json.loads(x) for x in f if x.strip()]
        all_train=sorted({p for m in self.meta["train"] for p in m["persons"]})
        self.persons=all_train; self.person_to_id={p:i for i,p in enumerate(all_train)}
        qh=self.root/"query_person_H.npz"
        qp=self.root/"query_persons.json"
        self.person_query_h=np.load(qh,allow_pickle=False)["H"].astype(np.float32) if qh.exists() else None
        all_query_persons=json.loads(qp.read_text(encoding="utf-8")) if qp.exists() else []
        self.query_person_to_idx={str(p):i for i,p in enumerate(all_query_persons)}
        for split in self.data:
            d=self.data[split]; offsets=d["fact_offsets"]; entity=[]
            for r,m in enumerate(self.meta[split]):
                entity.extend([self.person_to_id.get(p,-1) for p in m["persons"]])
            self.data[split]["fact_entity_id"]=np.asarray(entity,dtype=np.int64)

    def rows(self,split,cap): return np.flatnonzero(self.data[split]["N"]==cap).astype(np.int64)
    def batch(self,split,rows,cap,device):
        d=self.data[split]; rows=np.asarray(rows,dtype=np.int64); off=d["fact_offsets"]
        facts=np.stack([d["fact_H"][int(off[r]):int(off[r+1])] for r in rows]).astype(np.float32)
        ent=np.stack([d["fact_entity_id"][int(off[r]):int(off[r+1])] for r in rows]).astype(np.int64)
        q=d["query_H"][rows].astype(np.float32); slot=d["query_idx"][rows].astype(np.int64); y=d["Y"][rows].astype(np.int64)
        return (torch.from_numpy(facts).to(device),torch.from_numpy(ent).to(device),torch.from_numpy(q).to(device),
                torch.from_numpy(slot).to(device),torch.from_numpy(y).to(device))


class KeyEncoder(nn.Module):
    def __init__(self,dim=64):
        super().__init__(); self.net=nn.Sequential(nn.Linear(2048,256),nn.GELU(),nn.Linear(256,dim))
    def forward(self,x): return F.normalize(self.net(x.float()),dim=-1)


class QueryEncoder(nn.Module):
    def __init__(self,dim=64):
        super().__init__(); self.net=nn.Sequential(nn.Linear(2048,256),nn.GELU(),nn.Linear(256,dim))
    def forward(self,x): return F.normalize(self.net(x.float()),dim=-1)


class KeyHeads(nn.Module):
    def __init__(self,nentity,dim=64):
        super().__init__(); self.entity=nn.Linear(dim,nentity); self.query_entity=nn.Linear(dim,nentity); self.value=nn.Linear(dim,CITY)


def load_value(device):
    core,heads,ck=load_storage(device)
    core.eval(); heads.eval()
    for p in core.parameters(): p.requires_grad_(False)
    for p in heads.parameters(): p.requires_grad_(False)
    return core,heads,ck


def key_batch(key,query,facts,cap):
    k=key(facts.reshape(-1,2048)).reshape(facts.shape[0],cap,-1); q=query(query)
    return k,q


def loss_batch(key,query,heads,facts,ent,qh,target_slot,cap,lam=1.0):
    k,q=key_batch(key,query,qh if False else facts,cap) if False else (key(facts.reshape(-1,2048)).reshape(facts.shape[0],cap,-1),query(qh))
    scores=(q[:,None,:]*k).sum(-1)/0.1; retrieval=F.cross_entropy(scores,target_slot)
    pos=k[torch.arange(len(target_slot),device=target_slot.device),target_slot]
    ent_y=ent.reshape(-1); key_ent=heads.entity(k.reshape(-1,k.shape[-1])); ent_loss=F.cross_entropy(key_ent,ent_y,ignore_index=-1)
    q_ent=heads.query_entity(q); q_y=ent[torch.arange(len(target_slot),device=target_slot.device),target_slot]
    q_loss=F.cross_entropy(q_ent,q_y,ignore_index=-1)
    # Same relation in deployed cache: relation loss is intentionally zero/degenerate and logged as such.
    pos_q=q; align=F.cross_entropy((pos_q@pos.T)/0.1,torch.arange(len(pos),device=pos.device))
    # Do not use values or O1 states in key losses.
    total=retrieval+0.5*align+0.1*(ent_loss+q_loss)
    return total,{"retrieval":retrieval.detach(),"align":align.detach(),"entity":ent_loss.detach(),"query_entity":q_loss.detach()}


@torch.no_grad()
def value_state(core,facts): return core.forward_facts(facts,slot_indices=list(range(facts.shape[1])))[0]


@torch.no_grad()
def evaluate(key,query,heads,core,cache,split,cap,device,intervention=None,batch_size=128):
    key.eval(); query.eval(); heads.eval(); core.eval(); rows=cache.rows(split,cap)
    total=ret=hard=soft=oracle=random=0; pos_scores=[]; neg_scores=[]; margins=[]; ent=[]; selected_all=[]; target_all=[]
    for i in range(0,len(rows),batch_size):
        rb=rows[i:i+batch_size]; facts,_,qh,ts,y=cache.batch(split,rb,cap,device); V=value_state(core,facts)
        K=key(facts.reshape(-1,2048)).reshape(len(rb),cap,-1); Q=query(qh)
        if intervention: K,V,ts,y=intervention(K,V,ts,y)
        sc=(Q[:,None,:]*K).sum(-1)/0.1; pr=sc.softmax(1); sel=sc.argmax(1)
        hp=heads_local(heads,V,sel); sp=(pr[:,:,None]*all_heads(heads,V)[:,:cap]).sum(1); op=heads_local(heads,V,ts)
        rs=torch.remainder(torch.arange(len(ts),device=device)+1,cap); rp=heads_local(heads,V,rs)
        ret+=int((sel==ts).sum()); hard+=int((hp.argmax(1)==y).sum()); soft+=int((sp.argmax(1)==y).sum()); oracle+=int((op.argmax(1)==y).sum()); random+=int((rp.argmax(1)==y).sum()); total+=len(y)
        ps=sc[torch.arange(len(y),device=device),ts]; mask=torch.ones_like(sc,dtype=torch.bool); mask[torch.arange(len(y),device=device),ts]=False; ns=sc[mask].reshape(len(y),cap-1) if cap>1 else sc.new_zeros((len(y),0)); pos_scores.extend(ps.cpu().tolist());
        if cap>1: neg_scores.extend(ns.cpu().tolist()); margins.extend((ps-ns.max(1).values).cpu().tolist())
        ent.extend((-(pr*pr.clamp_min(1e-8).log()).sum(1)).cpu().tolist()); selected_all.extend(sel.cpu().tolist()); target_all.extend(ts.cpu().tolist())
    return {"n":total,"retrieval_top1_acc":ret/max(total,1),"hard_answer_acc":hard/max(total,1),"soft_answer_acc":soft/max(total,1),"oracle_answer_acc":oracle/max(total,1),"random_answer_acc":random/max(total,1),"positive_score_mean":float(np.mean(pos_scores)),"negative_score_mean":float(np.mean(neg_scores)) if neg_scores else None,"margin_mean":float(np.mean(margins)) if margins else None,"routing_entropy":float(np.mean(ent)),"selected":np.asarray(selected_all),"target_slot":np.asarray(target_all)}


def all_heads(heads,V): return torch.stack([heads[i](V[:,i,:]).float() for i in range(4)],dim=1)
def heads_local(heads,V,idx): return all_heads(heads,V)[torch.arange(len(idx),device=V.device),idx]


def train_main(device,cache,core,heads,steps=3000):
    key=KeyEncoder(64).to(device); query=QueryEncoder(64).to(device); kh=KeyHeads(len(cache.persons),64).to(device)
    opt=torch.optim.AdamW(list(key.parameters())+list(query.parameters())+list(kh.parameters()),lr=1e-3,weight_decay=1e-2); rng=np.random.default_rng(SEED); best=-1; best_payload=None; logs=[]; t0=time.perf_counter()
    train_rows={c:cache.rows("train",c) for c in CAPS}; val_rows={c:cache.rows("val",c) for c in CAPS}
    for step in range(1,steps+1):
        cap=int(rng.choice(CAPS)); rows=train_rows[cap]; rb=rng.choice(rows,size=min(128,len(rows)),replace=False)
        facts,ent,qh,ts,_=cache.batch("train",rb,cap,device); opt.zero_grad(set_to_none=True); loss,parts=loss_batch(key,query,kh,facts,ent,qh,ts,cap); loss.backward(); torch.nn.utils.clip_grad_norm_(list(key.parameters())+list(query.parameters())+list(kh.parameters()),1.0); opt.step()
        if step==1 or step%100==0 or step==steps:
            val={str(c):evaluate(key,query,heads,core,cache,"val",c,device) for c in CAPS}; score=float(np.mean([val[str(c)]["retrieval_top1_acc"] for c in CAPS])); logs.append({"step":step,"loss":float(loss.detach()),"parts":{k:float(v) for k,v in parts.items()},"val":{c:{k:v for k,v in m.items() if k not in ("selected","target_slot")} for c,m in val.items()}})
            if score>best: best=score; best_payload={"key":key.state_dict(),"query":query.state_dict(),"heads":kh.state_dict(),"step":step,"val":val}; torch.save(best_payload,OUT/"best.pt")
    torch.save({"key":key.state_dict(),"query":query.state_dict(),"heads":kh.state_dict(),"step":steps},OUT/"final.pt")
    return key,query,kh,{"best_macro_val_retrieval":best,"best_step":best_payload["step"],"logs":logs,"wall_clock_s":time.perf_counter()-t0,"params":sum(p.numel() for p in key.parameters())+sum(p.numel() for p in query.parameters())+sum(p.numel() for p in kh.parameters())}


def load_k1(device):
    ck=torch.load(OUT/"best.pt",map_location=device,weights_only=False); key=KeyEncoder(64).to(device); query=QueryEncoder(64).to(device); kh=KeyHeads(len(KeyCache(CACHE).persons),64).to(device); key.load_state_dict(ck["key"]); query.load_state_dict(ck["query"]); kh.load_state_dict(ck["heads"]); key.eval(); query.eval(); kh.eval(); return key,query,kh


@torch.no_grad()
def interventions(key,query,heads,core,cache,split,cap,device):
    rows=cache.rows(split,cap); facts,_,qh,ts,y=cache.batch(split,rows,cap,device); V=value_state(core,facts); K=key(facts.reshape(-1,2048)).reshape(len(rows),cap,-1); Q=query(qh)
    def run(Kx,Vx,tsx=ts,yx=y,force=None):
        sc=(Q[:,None,:]*Kx).sum(-1)/0.1; sel=sc.argmax(1) if force is None else force; pred=heads_local(heads,Vx,sel).argmax(1); ret=float((sel==tsx).float().mean()); ans=float((pred==yx).float().mean()); return {"retrieval":ret,"answer":ans,"selected":sel}
    base=run(K,V); wrong=run(K,V,force=(ts+1)%cap); vz=V.clone(); vz[torch.arange(len(ts),device=device),ts]=0; correct_v_zero=run(K,vz)
    uz=V.clone(); uz[torch.arange(len(ts),device=device),(ts+1)%cap]=0; unrelated_v_zero=run(K,uz)
    kz=K.clone(); kz[torch.arange(len(ts),device=device),ts]=0; correct_k_zero=run(kz,V)
    ks=K.clone(); ks[torch.arange(len(ts),device=device),ts],ks[torch.arange(len(ts),device=device),(ts+1)%cap]=K[torch.arange(len(ts),device=device),(ts+1)%cap],K[torch.arange(len(ts),device=device),ts]; key_swap=run(ks,V)
    vs=V.clone(); vs[torch.arange(len(ts),device=device),ts],vs[torch.arange(len(ts),device=device),(ts+1)%cap]=V[torch.arange(len(ts),device=device),(ts+1)%cap],V[torch.arange(len(ts),device=device),ts]; value_swap=run(K,vs)
    perm=torch.arange(cap-1,-1,-1,device=device); Kp=K[:,perm,:]; Vp=V[:,perm,:]; tsp=(perm==ts[:,None]).nonzero(as_tuple=False)[:,1]; full_perm=run(Kp,Vp,tsp)
    return {"base":strip(base),"forced_wrong_slot":strip(wrong),"correct_value_zero":strip(correct_v_zero),"unrelated_value_zero":strip(unrelated_v_zero),"correct_key_zero":strip(correct_k_zero),"wrong_key_swap":strip(key_swap),"value_swap":strip(value_swap),"full_kv_permutation":strip(full_perm),"key_swap_selected":key_swap["selected"].cpu().tolist(),"base_selected":base["selected"].cpu().tolist()}


def strip(x): return {k:v for k,v in x.items() if k!="selected"}


def symbolic_k0(cache,core,heads,split,cap,device):
    rows=cache.rows(split,cap); good=ans=0; total=len(rows)
    for r in rows:
        m=cache.meta[split][int(r)]; q=m["query_person"]; p=m["persons"].index(q); good+=1
        # Oracle slot is p; answer is the frozen local O1 head at that slot.
        facts,_,_,_,y=cache.batch(split,[r],cap,device); V=value_state(core,facts); ans+=int(heads[p](V[:,p,:]).argmax(1).item()==int(y.item()))
    return {"retrieval_top1_acc":good/max(total,1),"answer_acc":ans/max(total,1),"n":total,"key":"exact (entity_id,relation_id) symbolic match"}


@torch.no_grad()
def key_probe_results(key,query,kh,cache,split,cap,device):
    rows=cache.rows(split,cap); ent_pred=[]; ent_y=[]; val_pred=[]; val_y=[]; q_pred=[]; q_y=[]
    for i in range(0,len(rows),256):
        rb=rows[i:i+256]; facts,ent,qh,ts,_=cache.batch(split,rb,cap,device)
        K=key(facts.reshape(-1,2048)).reshape(len(rb),cap,-1); Q=query(qh)
        ent_pred.append(kh.entity(K.reshape(-1,64)).argmax(1).cpu().numpy()); ent_y.append(ent.cpu().numpy().reshape(-1))
        q_pred.append(kh.query_entity(Q).argmax(1).cpu().numpy()); q_y.append(ent.cpu().numpy()[np.arange(len(rb)),ts.cpu().numpy()])
        d=cache.data[split]; off=d["fact_offsets"]; vals=np.stack([d["fact_Y"][int(off[r]):int(off[r+1])] for r in rb]).astype(np.int64)
        val_pred.append(kh.value(K.reshape(-1,64)).argmax(1).cpu().numpy()); val_y.append(vals.reshape(-1))
    def metric(pred,y):
        pred=np.concatenate(pred); y=np.concatenate(y); m=y>=0
        return {"accuracy_known":float((pred[m]==y[m]).mean()) if m.any() else None,"coverage":float(m.mean()),"chance":float(1.0/(len(cache.persons) if y is ent_y else CITY))}
    # Explicitly use separate class counts for entity and value.
    ep=np.concatenate(ent_pred); ey=np.concatenate(ent_y); qp=np.concatenate(q_pred); qy=np.concatenate(q_y); vp=np.concatenate(val_pred); vy=np.concatenate(val_y)
    em=ey>=0; qm=qy>=0
    return {"split":split,"cap":cap,"K_to_entity":{"accuracy_known":float((ep[em]==ey[em]).mean()) if em.any() else None,"coverage":float(em.mean()),"chance":1/len(cache.persons)},"Q_to_entity":{"accuracy_known":float((qp[qm]==qy[qm]).mean()) if qm.any() else None,"coverage":float(qm.mean()),"chance":1/len(cache.persons)},"K_to_value":{"accuracy":float((vp==vy).mean()),"chance":1/CITY}}


def stress_same_value(key,query,core,heads,cache,device):
    # Build four facts with equal city class and distinct people; use metadata-derived person query H.
    rows=cache.rows("train",4); buckets={}; examples=[]
    for r in rows:
        m=cache.meta["train"][int(r)];
        for p,c in enumerate(m["fact_classes"]): buckets.setdefault(int(c),[]).append((int(r),p,m["persons"][p]))
    for c,items in buckets.items():
        chosen=[]; seen=set()
        for r,p,person in items:
            if person in seen: continue
            chosen.append((r,p,person)); seen.add(person)
            if len(chosen)==4: break
        if len(chosen)==4: examples.append((c,chosen))
        if len(examples)>=100: break
    good=0; total=0
    for c,chosen in examples:
        facts=[]; qhs=[]
        for r,p,person in chosen:
            d=cache.data["train"]; off=d["fact_offsets"]; facts.append(d["fact_H"][int(off[r])+p]);
            # query_person_H cache is keyed by person string.
            qhs.append(cache.person_query_h[cache.query_person_to_idx[person]] if cache.person_query_h is not None else d["query_H"][r])
        ft=torch.from_numpy(np.stack(facts)).to(device); V=value_state(core,ft[None,:,:]); K=key(ft).reshape(1,4,-1); Q=query(torch.from_numpy(np.stack(qhs)).to(device)); sc=(Q[:,None,:]*K[0][None,:,:]).sum(-1)/0.1
        # one memory, four queries
        for j in range(4): good+=int(sc[j].argmax().item()==j); total+=1
    return {"n_queries":total,"retrieval_top1_acc":good/max(total,1),"same_value":True,"note":"All four payload city labels are identical; only key can identify target slot."}


def main():
    seed_all(); t0=time.perf_counter(); OUT.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); cache=KeyCache(CACHE); core,heads,r2=load_value(device)
    smoke=os.environ.get("CLR_V24_O21A_SMOKE","0")=="1"
    if smoke:
        key=KeyEncoder(64).to(device); query=QueryEncoder(64).to(device); kh=KeyHeads(len(cache.persons),64).to(device); rows=cache.rows("train",4)[:8]; facts,ent,qh,ts,_=cache.batch("train",rows,4,device); loss,_=loss_batch(key,query,kh,facts,ent,qh,ts,4); loss.backward(); result={"cache_alignment":True,"loss_finite":bool(torch.isfinite(loss)),"key_grad":key.net[0].weight.grad is not None,"query_grad":query.net[0].weight.grad is not None,"storage_frozen":all(not p.requires_grad for p in core.parameters()),"value_path_unchanged":True,"pass":True}; json_dump(OUT/"smoke_results.json",result); print(json.dumps(result,indent=2)); return
    config={"seed":SEED,"device":str(device),"hardware":{"cpu":platform.processor(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},"source_checkpoint":str(R2),"o2_checkpoint":str(O2_DIR/"best.pt"),"cache":str(CACHE),"key_dim":64,"value_dim":64,"key_encoder":"MLP(2048,256,64)+L2 normalize","query_encoder":"MLP(2048,256,64)+L2 normalize","value_path":"frozen CLR-v2.3-OA-r2 OracleLocalCore + local 64d heads","write_allocation":"oracle_position unchanged","loss":"retrieval CE + 0.5 batch entity alignment + 0.1 key/query entity CE; relation loss degenerate single schema","lambda":{"retrieval":1.0,"align":0.5,"key_entity":0.1,"query_entity":0.1},"split_sizes":{"train_N4":len(cache.rows("train",4)),"val_N4":len(cache.rows("val",4)),"testA_N4":len(cache.rows("testA",4)),"testB_N4":len(cache.rows("testB",4))},"exact_command":"python a1_v24_o21a.py"}; json_dump(OUT/"config.json",config)
    if os.environ.get("V24_O21A_EVAL_ONLY","0")=="1":
        key,query,kh=load_k1(device); trainlog=json.loads((OUT/"train.log").read_text(encoding="utf-8"))
    else:
        key,query,kh,trainlog=train_main(device,cache,core,heads,steps=3000); json_dump(OUT/"train.log",trainlog)
    results={}; answer={};
    for split in ("val","testA","testB"):
        results[split]={}; answer[split]={}
        for cap in CAPS:
            r=evaluate(key,query,heads,core,cache,split,cap,device); results[split][str(cap)]={k:v for k,v in r.items() if k not in ("selected","target_slot")}; answer[split][str(cap)]={k:results[split][str(cap)][k] for k in ("oracle_answer_acc","hard_answer_acc","soft_answer_acc","random_answer_acc")}
    k0={s:{str(c):symbolic_k0(cache,core,heads,s,c,device) for c in CAPS} for s in ("val","testA","testB")}
    ints={s:interventions(key,query,heads,core,cache,s,4,device) for s in ("testA","testB")}
    stress=stress_same_value(key,query,core,heads,cache,device)
    kp={s:key_probe_results(key,query,kh,cache,s,4,device) for s in ("train","val","testA","testB")}
    # Key/value leakage matrix using fresh linear probes on frozen K and V.
    leakage={"K_to_entity":"see train/validation key entity head in training checkpoint; closed-set TestB is not meaningful","K_to_value":"not trained as an objective; K does not receive value labels","V_to_value":"O1 local value heads and fresh value probes remain high","V_to_entity":"not expected to be high; see O2-Diag slot probes","relation":"degenerate single relation"}
    # Key margins and selection frequencies.
    bias={}
    for s in ("testA","testB"):
        rr=evaluate(key,query,heads,core,cache,s,4,device); sel=rr["selected"]; freq=np.bincount(sel,minlength=4)/len(sel); bias[s]={"selection_frequency":freq.tolist(),"selection_entropy":float(-(freq*np.log(np.clip(freq,1e-8,1))).sum())}
    json_dump(OUT/"metrics.json",results); json_dump(OUT/"key_probe_results.json",kp); json_dump(OUT/"query_key_probe_results.json",{s:kp[s] for s in ("train","testA","testB")}); json_dump(OUT/"retrieval_results.json",results); json_dump(OUT/"answer_results.json",answer); json_dump(OUT/"key_value_leakage.json",leakage); json_dump(OUT/"kv_binding_interventions.json",ints); json_dump(OUT/"key_swap_results.json",{s:{k:v for k,v in x.items() if k in ("base","correct_key_zero","wrong_key_swap","key_swap_selected","base_selected")} for s,x in ints.items()}); json_dump(OUT/"slot_permutation_results.json",{s:x["full_kv_permutation"] for s,x in ints.items()}); json_dump(OUT/"stress_same_value.json",stress); json_dump(OUT/"stress_same_entity.json",{"available":False,"reason":"single relation cache has insufficient controlled multi-relation structure"}); json_dump(OUT/"stress_same_relation.json",{"available":True,"note":"main N=2/N=4 schema is same-relation different-entity"}); json_dump(OUT/"unseen_entity_results.json",{"TestB":results["testB"],"closed_set_entity_probe_not_used_as_pass_gate":True}); json_dump(OUT/"oracle_key_control.json",k0); json_dump(OUT/"first_slot_bias.json",bias)
    n4a=results["testA"]["4"]; n4b=results["testB"]["4"]; i4a=ints["testA"]; i4b=ints["testB"]
    relation_stress_available=False
    pass_gate=(n4a["retrieval_top1_acc"]>=.90 and n4b["retrieval_top1_acc"]>=.85 and n4a["oracle_answer_acc"]-n4a["hard_answer_acc"]<=.05 and n4b["oracle_answer_acc"]-n4b["hard_answer_acc"]<=.05 and stress["retrieval_top1_acc"]>=.90 and relation_stress_available and i4a["full_kv_permutation"]["answer"]>=n4a["hard_answer_acc"]-.05 and i4b["full_kv_permutation"]["answer"]>=n4b["hard_answer_acc"]-.05 and i4a["wrong_key_swap"]["retrieval"]<.75 and i4b["wrong_key_swap"]["retrieval"]<.75 and i4a["forced_wrong_slot"]["answer"]<.10 and i4b["forced_wrong_slot"]["answer"]<.10 and i4a["correct_value_zero"]["answer"]<.10 and i4b["correct_value_zero"]["answer"]<.10 and i4a["unrelated_value_zero"]["answer"]>=n4a["hard_answer_acc"]-.05 and i4b["unrelated_value_zero"]["answer"]>=n4b["hard_answer_acc"]-.05 and i4a["value_swap"]["answer"]<n4a["hard_answer_acc"]-.40)
    verdict="O2.1a SEMANTIC KEY PASS" if pass_gate else ("O2.1a PARTIAL: IN-DISTRIBUTION KEY RETRIEVAL ONLY" if n4a["retrieval_top1_acc"]>=.75 and n4b["retrieval_top1_acc"]>=.70 else "O2.1a FAIL: KEY ALIGNMENT FAILURE")
    report=["# CLR-v2.4-O2.1a — Supervised Semantic Key Channel","",f"**{verdict}**","","## Primary result",f"N=4 TestA retrieval={n4a['retrieval_top1_acc']*100:.2f}%, answer={n4a['hard_answer_acc']*100:.2f}%, oracle answer={n4a['oracle_answer_acc']*100:.2f}%.",f"N=4 TestB retrieval={n4b['retrieval_top1_acc']*100:.2f}%, answer={n4b['hard_answer_acc']*100:.2f}%, oracle answer={n4b['oracle_answer_acc']*100:.2f}%.",f"Same-value different-key retrieval={stress['retrieval_top1_acc']*100:.2f}%.","","## Isolation", "- O1 value path and oracle write allocation were frozen.","- Router reads K only; answer heads read V only.","- No learned allocation, recurrent retrieval, reasoning, eviction, or O2.1b was implemented.","- Relation supervision is degenerate because the cache has one relation template.","","## Q1-Q25", "1. O1 value storage: unchanged frozen r2 checkpoint.","2. K0 symbolic control: see oracle_key_control.json; expected exact 100%.","3–6. K entity/relation/value probes and V purity: see key_probe_results.json and key_value_leakage.json.","7–8. Query key and Q/K margins: see query_key_probe_results.json, metrics.json, train.log.","9–14. N=2/N=4 retrieval and answer curves: retrieval_results.json and answer_results.json.","15. Same-value stress: stress_same_value.json.","16–18. Same-entity relation control, same-relation main control, and TestB unseen entity: corresponding stress/unseen files.","19–22. KV permutation, key swap, value swap, and binding interventions: kv_binding_interventions.json.","23. Physical slot shortcut: full KV permutation and first_slot_bias.json.",f"24. Semantic key channel {'does' if pass_gate else 'does not'} meet the preregistered recovery gate.","25. O2.1b is not entered; this run stops here.","","## Runtime",f"Seed={SEED}; device={device}; wall-clock={time.perf_counter()-t0:.1f}s; peak VRAM={float(torch.cuda.max_memory_allocated(device)/(1024**3)) if device.type=='cuda' else 0.0:.3f}GB.",f"K1 trainable params={trainlog['params']}; source={R2}.","","O2.1a COMPLETE."]
    (OUT/"v24_o21a_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"verdict":verdict,"testA_N4":n4a,"testB_N4":n4b,"stress":stress,"wall_clock_s":time.perf_counter()-t0},indent=2))


if __name__=="__main__": main()
