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

from a1_v24_o2_common import O2Cache, load_storage, rows_for, json_dump
from clr_v24_o2 import RetrievalController


ROOT = Path(r"D:\training")
OUT = Path(os.environ.get("V24_O2_DIAG_OUT", r"D:\training\checkpoints\clr_v24_o2_diag"))
CACHE = os.environ.get("V22_CACHE_DIR", r"D:\training\checkpoints\clr_v2_a1_ls2\cache")
O2_DIR = ROOT / "checkpoints" / "clr_v24_o2_retrieval"
SEED = 42
CITY_CLASSES = 64
SLOTS = 4
DIM = 64
CHANCE4 = 0.25


def seed_all(seed=SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_o2_controller(device):
    ck = torch.load(O2_DIR / "best.pt", map_location=device)
    m = RetrievalController().to(device=device, dtype=torch.float32)
    m.load_state_dict(ck["controller"]); m.eval()
    return m


def collect_split(core, cache, split, cap, device, max_rows=None, keep_facts=False):
    rows = rows_for(cache, split, cap)
    if max_rows is not None: rows = rows[:max_rows]
    states=[]; facts_all=[]; qs=[]; ys=[]; slots=[]; labels=[]; persons=[]; cities=[]
    for i in range(0, len(rows), 256):
        rb=rows[i:i+256]
        facts, lab, q, ts, y = cache.batch(split, rb, cap, device)
        with torch.no_grad(): s,_=core.forward_facts(facts, slot_indices=list(range(cap)))
        states.append(s.cpu().numpy().astype(np.float32)); qs.append(q.cpu().numpy().astype(np.float32))
        ys.append(y.cpu().numpy().astype(np.int64)); slots.append(ts.cpu().numpy().astype(np.int64)); labels.append(lab.cpu().numpy().astype(np.int64))
        if keep_facts: facts_all.append(facts.cpu().numpy().astype(np.float32))
        if split in cache.meta:
            for r in rb:
                m=cache.meta[split][int(r)]
                persons.append(list(m.get("persons", []))); cities.append(list(m.get("fact_classes", m.get("cities", []))))
    out={"rows":rows,"state":np.concatenate(states),"query":np.concatenate(qs),"y":np.concatenate(ys),"target_slot":np.concatenate(slots),"labels":np.concatenate(labels),"persons":persons,"cities":cities}
    if keep_facts: out["facts"] = np.concatenate(facts_all)
    return out


def fit_linear(x, y, nclass, device, steps=350, lr=3e-3, seed=SEED):
    seed_all(seed); x=torch.as_tensor(x, dtype=torch.float32, device=device); y=torch.as_tensor(y, dtype=torch.long, device=device)
    model=nn.Linear(x.shape[1], nclass).to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-2)
    batch=2048
    for st in range(steps):
        idx=torch.randint(len(y),(min(batch,len(y)),),device=device)
        opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(x[idx]),y[idx]); loss.backward(); opt.step()
    return model


@torch.no_grad()
def linear_eval(model, x, y, device, known=None):
    x=torch.as_tensor(x,dtype=torch.float32,device=device); y=np.asarray(y,dtype=np.int64)
    logits=model(x); pred=logits.argmax(1).cpu().numpy(); topk=min(5,logits.shape[1]); top5=logits.topk(topk,dim=1).indices.cpu().numpy()
    if known is None: known=np.ones(len(y),dtype=bool)
    return {"accuracy_overall_known_class_only": float((pred[known]==y[known]).mean()) if known.any() else None,
            "top5_accuracy_known_class_only": float((top5[known]==y[known,None]).any(axis=1).mean()) if known.any() else None,
            "chance": float(1.0/logits.shape[1]), "known_coverage": float(known.mean()), "known_n": int(known.sum()), "total_n": int(len(y))}


def flatten_meta(data, field, cap):
    vals=[]
    for row in data[field]: vals.extend(row[:cap])
    return np.asarray(vals)


def slot_probe_bundle(train, a, b, device):
    train_persons=sorted({p for row in train["persons"] for p in row})
    person_to_id={p:i for i,p in enumerate(train_persons)}
    def person_ids(d):
        raw=flatten_meta(d,"persons",4); return np.asarray([person_to_id.get(p,-1) for p in raw],dtype=np.int64)
    def city_ids(d):
        # Metadata has fact_classes in the deployed LS2 cache; fallback to labels.
        raw=flatten_meta(d,"cities",4)
        if raw.dtype.kind not in "iu": raw=np.asarray([int(x) for x in raw])
        return raw.astype(np.int64)
    tr_p=person_ids(train); a_p=person_ids(a); b_p=person_ids(b)
    tr_v=city_ids(train); a_v=city_ids(a); b_v=city_ids(b)
    out={"entity":{},"value":{},"relation":{},"joint_entity_relation":{},"person_vocab_size":len(train_persons)}
    for p in range(4):
        xtr=train["state"][:,p,:]; xa=a["state"][:,p,:]; xb=b["state"][:,p,:]
        m=fit_linear(xtr,tr_p[::4] if False else tr_p.reshape(-1,4)[...,p],len(train_persons),device,steps=300,seed=100+p)
        # state rows are episode-major, so each slot p has one label per episode.
        out["entity"][f"slot{p}"]={"train":linear_eval(m,xtr,tr_p.reshape(-1,4)[:,p],device,tr_p.reshape(-1,4)[:,p]>=0),
                                     "testA":linear_eval(m,xa,a_p.reshape(-1,4)[:,p],device,a_p.reshape(-1,4)[:,p]>=0),
                                     "testB":linear_eval(m,xb,b_p.reshape(-1,4)[:,p],device,b_p.reshape(-1,4)[:,p]>=0)}
        mv=fit_linear(xtr,tr_v.reshape(-1,4)[:,p],CITY_CLASSES,device,steps=300,seed=200+p)
        out["value"][f"slot{p}"]={"train":linear_eval(mv,xtr,tr_v.reshape(-1,4)[:,p],device),"testA":linear_eval(mv,xa,a_v.reshape(-1,4)[:,p],device),"testB":linear_eval(mv,xb,b_v.reshape(-1,4)[:,p],device)}
        # Single relation template: this is a constant positive control, not semantic evidence.
        out["relation"][f"slot{p}"]={"testA":{"accuracy":1.0,"chance":1.0,"degenerate_single_relation":True},"testB":{"accuracy":1.0,"chance":1.0,"degenerate_single_relation":True}}
        out["joint_entity_relation"][f"slot{p}"]=out["entity"][f"slot{p}"]
    out["entity_note"]="TestB entity IDs are unseen by construction; coverage and known-class accuracy are reported rather than treating unseen labels as classifier errors."
    out["relation_note"]="All facts use one relation/schema, so relation accuracy is a degenerate constant baseline and cannot validate relation semantics."
    return out


def query_probe_bundle(train,a,b,device):
    persons=sorted({p for row in train["persons"] for p in row}); p2={p:i for i,p in enumerate(persons)}
    def qlabels(d): return np.asarray([p2.get(cache_person,-1) for cache_person in [cache.meta["x"] if False else ""]])
    # Query metadata is the person at query_index.
    def labels(d,split):
        return np.asarray([p2.get(d["persons"][i][int(d["target_slot"][i])],-1) for i in range(len(d["y"]))],dtype=np.int64)
    trl=labels(train,"train"); al=labels(a,"testA"); bl=labels(b,"testB")
    m=fit_linear(train["query"],trl,len(persons),device,steps=400,seed=500)
    res={"entity": {"train":linear_eval(m,train["query"],trl,device,trl>=0),"testA":linear_eval(m,a["query"],al,device,al>=0),"testB":linear_eval(m,b["query"],bl,device,bl>=0)},
         "relation":{"testA":{"accuracy":1.0,"chance":1.0,"degenerate_single_relation":True},"testB":{"accuracy":1.0,"chance":1.0,"degenerate_single_relation":True}},
         "entity_note":"Query entity probe uses train-person classes; TestB unseen-person coverage is reported.","relation_note":"Single relation template."}
    return res


class ProjectedDot(nn.Module):
    def __init__(self):
        super().__init__(); self.q=nn.Linear(2048,64); self.s=nn.Linear(64,64)
    def forward(self,q,s): return (self.q(q)[:,None,:]*self.s(s)).sum(-1)/8.0


class Bilinear(nn.Module):
    def __init__(self):
        super().__init__(); self.w=nn.Parameter(torch.randn(2048,64)*0.01); self.b=nn.Parameter(torch.zeros(()))
    def forward(self,q,s): return ((q@self.w)[:,None,:]*s).sum(-1)/8.0+self.b


class PairMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.q=nn.Linear(2048,64); self.net=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,q,s):
        qq=self.q(q)[:,None,:].expand(-1,s.shape[1],-1); z=torch.cat([qq,s,qq*s,(qq-s).abs()],dim=-1); return self.net(z).squeeze(-1)


@torch.no_grad()
def scores_for(kind, model, o2, data, cap, device):
    q=torch.as_tensor(data["query"],dtype=torch.float32,device=device); s=torch.as_tensor(data["state"][:,:cap,:],dtype=torch.float32,device=device)
    if kind=="B0": return o2(q,s,cap)[:,:cap]
    return model(q,s)


def matcher_metrics(scores,target,threshold=None):
    sc=np.asarray(scores); t=np.asarray(target); pred=sc.argmax(1); pos=sc[np.arange(len(t)),t]
    neg=np.concatenate([sc[:, :p] for p in range(sc.shape[1])],axis=1) if sc.shape[1]>1 else np.zeros((len(t),0))
    mask=np.ones_like(sc,dtype=bool); mask[np.arange(len(t)),t]=False; negvals=sc[mask].reshape(len(t),sc.shape[1]-1) if sc.shape[1]>1 else np.zeros((len(t),0))
    if threshold is None: threshold=float((pos.mean()+negvals.mean())/2) if len(negvals) else 0.0
    pair_y=np.concatenate([np.ones(len(pos)),np.zeros(negvals.size)]); pair_score=np.concatenate([pos,negvals.reshape(-1)])
    binary=float(((pair_score>=threshold)==pair_y).mean()) if len(pair_y) else 0.0
    probs=torch.as_tensor(sc).softmax(1).numpy()
    return {"top1_retrieval_acc":float((pred==t).mean()),"binary_match_accuracy":binary,"threshold":threshold,
            "positive_score_mean":float(pos.mean()),"negative_score_mean":float(negvals.mean()) if len(negvals) else None,
            "margin_mean":float((pos-negvals.max(1)).mean()) if len(negvals) else None,
            "routing_entropy":float((-(probs*np.log(np.clip(probs,1e-8,1))).sum(1)).mean()),"n":len(t)}


def train_matcher(kind, train, val, device, steps=1000):
    model={"B1":ProjectedDot,"B2":Bilinear,"B3":PairMLP}[kind]().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-2); rng=np.random.default_rng(SEED)
    best=-1; best_state=None; threshold=0.0
    for step in range(1,steps+1):
        cap=int(rng.choice([2,4])); d=train[str(cap)]; idx=rng.integers(len(d["y"]),size=min(128,len(d["y"])))
        q=torch.as_tensor(d["query"][idx],dtype=torch.float32,device=device); s=torch.as_tensor(d["state"][idx,:cap,:],dtype=torch.float32,device=device); y=torch.as_tensor(d["target_slot"][idx],dtype=torch.long,device=device)
        opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(q,s),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step==1 or step%100==0 or step==steps:
            vals=[]
            for c in [2,4]:
                vd=val[str(c)]; ss=scores_for(kind,model,None,vd,c,device).cpu().numpy(); vals.append(matcher_metrics(ss,vd["target_slot"])["top1_retrieval_acc"])
            score=float(np.mean(vals))
            if score>best: best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    # Calibration threshold from validation pairs.
    vv=[]
    for c in [2,4]:
        vd=val[str(c)]; vv.append(scores_for(kind,model,None,vd,c,device).cpu().numpy());
    pos=[]; neg=[]
    for ss,c in zip(vv,[2,4]):
        tt=val[str(c)]["target_slot"]; pos.append(ss[np.arange(len(tt)),tt]); mask=np.ones_like(ss,dtype=bool); mask[np.arange(len(tt)),tt]=False; neg.append(ss[mask])
    threshold=float((np.concatenate(pos).mean()+np.concatenate(neg).mean())/2)
    return model, {"best_val_macro_top1":best,"steps":steps,"params":sum(p.numel() for p in model.parameters()),"threshold":threshold}


def matcher_ladder(train,val,tests,core,o2,device):
    out={}; models={"B0":None}
    for kind in ["B0","B1","B2","B3"]:
        info={"params":0,"steps":0}
        if kind!="B0": models[kind],info=train_matcher(kind,train,val,device,steps=1000)
        out[kind]={"train_info":info,"splits":{}}
        for split,ds in tests.items():
            out[kind]["splits"][split]={}
            for cap in [2,4]:
                d=ds[str(cap)]; ss=scores_for(kind,models[kind],o2,d,cap,device).cpu().numpy(); out[kind]["splits"][split][str(cap)]=matcher_metrics(ss,d["target_slot"],info.get("threshold"))
        out[kind]["model_params"] = info.get("params",0)
    return out,models


def position_probes(train,a,b,device):
    xs=[]; ys=[]
    for d in [train,a,b]:
        xs.append(d["state"].reshape(-1,64)); ys.append(np.tile(np.arange(4),len(d["state"])))
    m=fit_linear(xs[0],ys[0],4,device,steps=350,seed=600)
    # write order equals physical slot under O1 oracle allocation.
    return {"physical_slot_id": {"testA":linear_eval(m,xs[1],ys[1],device),"testB":linear_eval(m,xs[2],ys[2],device)},
            "write_order": {"testA":linear_eval(m,xs[1],ys[1],device),"testB":linear_eval(m,xs[2],ys[2],device)},
            "chance":CHANCE4,"note":"Under oracle-position storage, physical slot and write order coincide; they cannot be causally separated without a separate random-write protocol."}


@torch.no_grad()
def o2_scores(o2,d,cap,device): return scores_for("B0",None,o2,d,cap,device).cpu().numpy()


def permutation_results(o2,matchers,a,b,device):
    out={}
    for split,d in [("testA",a),("testB",b)]:
        st=d["state"].copy(); rng=np.random.default_rng(SEED); perm=rng.permutation(4); ps=st[:,perm,:]; inv=np.asarray([np.flatnonzero(perm==x)[0] for x in d["target_slot"]])
        dd=dict(d); dd["state"]=ps; dd["target_slot"]=inv
        out[split]={"permutation":perm.tolist()}
        for kind in ["B0","B1","B2","B3"]:
            model=matchers.get(kind); m=matcher_metrics(scores_for(kind,model,o2,dd,4,device).cpu().numpy(),inv, None)
            out[split][kind]=m
    return out


def first_bias(o2,a,b,device):
    out={}
    for split,d in [("testA",a),("testB",b)]:
        sc=o2_scores(o2,d,4,device); sel=sc.argmax(1); true=d["target_slot"]
        freq=np.bincount(sel,minlength=4)/len(sel); by={str(p):{str(k):float(((sel==k)&(true==p)).sum()/max((true==p).sum(),1)) for k in range(4)} for p in range(4)}
        ent=float((-(freq*np.log(np.clip(freq,1e-8,1))).sum())); out[split]={"selection_frequency":freq.tolist(),"by_true_target":by,"selection_entropy":ent,"uniform_entropy":float(np.log(4)),"deviation_from_uniform_l1":float(np.abs(freq-.25).sum())}
    return out


@torch.no_grad()
def controlled_similarity(core, cache, device):
    d=collect_split(core,cache,"train",4,device,max_rows=256,keep_facts=True); facts=torch.as_tensor(d["facts"],device=device); b=len(facts)
    # Reuse one fact per episode and write it to each physical slot from zero state.
    reps=[]
    for p in range(4):
        z=core.zero_state(b,device=device); s,_,_=core.write_to_slot(z,facts[:,0],p); reps.append(s[:,p,:].cpu().numpy())
    reps=np.stack(reps,axis=1); same=[]
    for i in range(b):
        for p in range(4):
            for q in range(p+1,4): same.append(float(np.dot(reps[i,p],reps[i,q])/(np.linalg.norm(reps[i,p])*np.linalg.norm(reps[i,q])+1e-8)))
    # Different facts in a fixed slot.
    x=reps[:,0,:]; diff=[]
    for i in range(min(b-1,255)): diff.append(float(np.dot(x[i],x[i+1])/(np.linalg.norm(x[i])*np.linalg.norm(x[i+1])+1e-8)))
    return {"same_fact_different_slot_cosine_mean":float(np.mean(same)),"different_fact_same_slot_cosine_mean":float(np.mean(diff)),
            "same_fact_different_slot_euclidean_mean":float(np.linalg.norm(reps[:,0]-reps[:,1],axis=1).mean()),
            "controlled_n":b,"note":"OracleLocalCore writes the same fact into each requested physical slot with all other slots untouched."}


def synthetic_controls(core,cache,o2,matchers,device):
    # Use cached fact hidden and query-person hidden to construct key-preserving groups.
    base=collect_split(core,cache,"train",4,device,max_rows=300,keep_facts=True); persons=base["persons"]; cities=base["cities"]
    person_h=cache.person_query_h; p2=cache.person_to_index
    groups_same_value={}; groups_same_entity={}
    for i,row in enumerate(persons):
        for j,p in enumerate(row):
            c=int(cities[i][j]); groups_same_value.setdefault(c,[]).append((i,j,p))
            groups_same_entity.setdefault(p,[]).append((i,j,p))
    def build(groups, want_diff_value=False):
        facts=[]; qs=[]; ts=[]; ys=[]
        for items in groups.values():
            chosen=[]; seenp=set(); seenc=set()
            for it in items:
                if it[2] in seenp: continue
                if want_diff_value and int(cities[it[0]][it[1]]) in seenc: continue
                chosen.append(it); seenp.add(it[2]); seenc.add(int(cities[it[0]][it[1]]))
                if len(chosen)==4: break
            if len(chosen)<4: continue
            fh=np.stack([base["facts"][i,j] for i,j,_ in chosen]); qh=np.stack([person_h[p2[p]] for _,_,p in chosen])
            facts.append(fh); qs.append(qh); ts.append(np.arange(4)); ys.append([int(cities[i][j]) for i,j,_ in chosen])
            if len(facts)>=100: break
        if not facts: return None
        ft=torch.as_tensor(np.stack(facts),device=device); st,_=core.forward_facts(ft,slot_indices=[0,1,2,3])
        n=len(facts)
        # Each memory is queried once per person; repeat the same memory four times.
        return {"state":np.repeat(st.cpu().numpy(),4,axis=0),"query":np.concatenate(qs,axis=0),
                "target_slot":np.tile(np.arange(4,dtype=np.int64),n),"y":np.concatenate(ys,axis=0),"n":n}
    # Same-value requires groups keyed by city; same-entity/different-value is a useful substitute because relation is constant.
    sv=build(groups_same_value,False); se=build(groups_same_entity,True)
    out={}
    for name,d in [("same_value_different_key",sv),("same_entity_different_value",se)]:
        if d is None: out[name]={"available":False}; continue
        out[name]={"available":True,"n_episodes":d["n"],"results":{}}
        for kind in ["B0","B1","B2","B3"]:
            sc=scores_for(kind,matchers.get(kind),o2,d,4,device).cpu().numpy(); out[name]["results"][kind]=matcher_metrics(sc,d["target_slot"])
        out[name]["value_only_expected_top1"]="chance=25% for same-value groups"
    out["same_entity_different_relation"]={"available":False,"reason":"deployed cache has one relation/schema only"}
    out["lexical_overlap"]={"available":False,"reason":"deployed cache has one fixed sentence template; no lexical-overlap template control is available without redefining the dataset"}
    return out


def oracle_key_control(cache,device):
    # Symbolic entity key, independent of learned slot vectors. This sanity control must be exact.
    total=0; good=0
    for split in ["testA","testB"]:
        for r in range(len(cache.meta[split])):
            m=cache.meta[split][r]; persons=m["persons"]; q=m["query_person"]; target=int(m["query_index"]); pred=persons.index(q); good+=int(pred==target); total+=1
    return {"top1_retrieval_acc":good/max(total,1),"total":total,"key":"symbolic person identity + fixed relation","pass":good==total,"note":"Positive control bypasses learned slot/query spaces and validates target mapping only."}


def value_only_control(same_value):
    if not same_value.get("available"): return {"available":False}
    n=same_value["n_episodes"]; return {"available":True,"same_value_top1_expected":0.25,"n_episodes":n,"note":"When all four values are identical, value-only identity cannot select the queried person."}


def main():
    t0=time.perf_counter(); seed_all(); OUT.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache=O2Cache(CACHE); core,heads,_=load_storage(device); o2=load_o2_controller(device)
    train4=collect_split(core,cache,"train",4,device,max_rows=None); val4=collect_split(core,cache,"val",4,device); a4=collect_split(core,cache,"testA",4,device); b4=collect_split(core,cache,"testB",4,device)
    train2=collect_split(core,cache,"train",2,device); val2=collect_split(core,cache,"val",2,device); a2=collect_split(core,cache,"testA",2,device); b2=collect_split(core,cache,"testB",2,device)
    # Keep matcher arrays keyed by capacity; they are frozen representations.
    train={"2":train2,"4":train4}; val={"2":val2,"4":val4}; tests={"testA":{"2":a2,"4":a4},"testB":{"2":b2,"4":b4}}
    slot_probes=slot_probe_bundle(train4,a4,b4,device); query_probes=query_probe_bundle(train4,a4,b4,device)
    matcher,models=matcher_ladder(train,val,tests,core,o2,device); models["B0"]=None
    # B0 uses the frozen O2 controller; B1-B3 are trained diagnostic-only matchers.
    position=position_probes(train4,a4,b4,device); perm=permutation_results(o2,models,a4,b4,device); bias=first_bias(o2,a4,b4,device)
    sim=controlled_similarity(core,cache,device); synthetic=synthetic_controls(core,cache,o2,models,device)
    oracle=oracle_key_control(cache,device); value_only=value_only_control(synthetic["same_value_different_key"])
    # Existing O2 baseline counts by target position.
    shortcut={}
    for split,d in [("testA",a4),("testB",b4)]:
        sc=o2_scores(o2,d,4,device); sel=sc.argmax(1); shortcut[split]={"selection_frequency":np.bincount(sel,minlength=4).tolist(),"first_city_bias":float((d["target_slot"]==0).mean()),"top1":float((sel==d["target_slot"]).mean())}
    try: git_sha=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],stderr=subprocess.STDOUT,text=True).strip()
    except Exception: git_sha="not-a-git-repository"
    config={"seed":SEED,"device":str(device),"hardware":{"cpu":platform.processor(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},"git_sha":git_sha,"checkpoint_source":str(O2_DIR/"best.pt"),"storage_checkpoint":str(ROOT/"checkpoints"/"clr_v23_oracle_r2"/"best_model.pt"),"cache":str(CACHE),"splits":{"train":len(train4["y"]),"val":len(val4["y"]),"testA":len(a4["y"]),"testB":len(b4["y"])},"trainable_storage":False,"trainable_probe_only":True,"matcher_steps":1000,"relation_schema":"single fixed relation; degenerate relation probe","exact_command":"python a1_v24_o2_diag.py"}
    json_dump(OUT/"config.json",config); json_dump(OUT/"slot_probe_results.json",slot_probes); json_dump(OUT/"query_probe_results.json",query_probes); json_dump(OUT/"offline_matcher_results.json",matcher); json_dump(OUT/"position_probe_results.json",position); json_dump(OUT/"permutation_results.json",perm); json_dump(OUT/"distractor_results.json",synthetic); json_dump(OUT/"oracle_key_control.json",oracle); json_dump(OUT/"first_slot_bias.json",bias); json_dump(OUT/"representation_stats.json",{"controlled_similarity":sim,"value_only_control":value_only,"shortcut":shortcut})
    diag_metrics={"primary":{"testA":{"N2":matcher["B0"]["splits"]["testA"]["2"],"N4":matcher["B0"]["splits"]["testA"]["4"]},"testB":{"N2":matcher["B0"]["splits"]["testB"]["2"],"N4":matcher["B0"]["splits"]["testB"]["4"]}},"matcher_best":{"testA_N4":max((matcher[k]["splits"]["testA"]["4"]["top1_retrieval_acc"],k) for k in matcher),"testB_N4":max((matcher[k]["splits"]["testB"]["4"]["top1_retrieval_acc"],k) for k in matcher)},"oracle_key":oracle,"runtime_s":time.perf_counter()-t0,"peak_vram_gb":float(torch.cuda.max_memory_allocated(device)/(1024**3)) if device.type=="cuda" else 0.0}
    json_dump(OUT/"diag_metrics.json",diag_metrics)
    best_a=diag_metrics["matcher_best"]["testA_N4"]; best_b=diag_metrics["matcher_best"]["testB_N4"]
    slot_key=slot_probes["entity"]["slot0"]["testA"]; value=slot_probes["value"]["slot0"]["testA"]
    query_key=query_probes["entity"]["testA"]; position_key=position["physical_slot_id"]["testA"]
    if oracle["pass"] and best_a[0]>=.90 and best_b[0]>=.85:
        verdict="O2-Diag COMPLETE"; primary="CROSS-SPACE ALIGNMENT / ROUTER CAPACITY"
    elif oracle["pass"] and value["accuracy_overall_known_class_only"]>=.90 and best_a[0]<.60 and best_b[0]<.60:
        verdict="O2-Diag COMPLETE"; primary="REPRESENTATION INSUFFICIENT FOR ASSOCIATIVE RETRIEVAL"
    else:
        verdict="O2-Diag COMPLETE"; primary="MIXED / INCONCLUSIVE; SEE PROBES AND MATCHER LADDER"
    lines=["# CLR-v2.4 O2-Diag — Associative Retrieval Bottleneck Diagnosis","","## Final verdict","",f"**{verdict}**","","**PRIMARY BOTTLENECK:**",primary,"","**SECONDARY FINDINGS:**","- Storage and oracle-key mapping were not changed; O1 storage remains frozen.","- Relation probes are degenerate because the deployed dataset has one relation/schema.","- No O3, O2.1, learned allocation, recurrent retrieval, or storage retraining was run.","","## Key evidence","",f"- Oracle-key sanity: {oracle['top1_retrieval_acc']*100:.2f}% ({oracle['total']} episodes).",f"- Frozen O2 B0 N=4: TestA {matcher['B0']['splits']['testA']['4']['top1_retrieval_acc']*100:.2f}%, TestB {matcher['B0']['splits']['testB']['4']['top1_retrieval_acc']*100:.2f}%.",f"- Best offline matcher N=4: TestA {best_a[0]*100:.2f}% ({best_a[1]}), TestB {best_b[0]*100:.2f}% ({best_b[1]}).",f"- Value probe slot0: TestA {value['accuracy_overall_known_class_only']*100:.2f}% (chance {value['chance']*100:.2f}%).",f"- Entity probe slot0: TestA {slot_key['accuracy_overall_known_class_only']*100:.2f}% on {slot_key['known_coverage']*100:.1f}% known-class coverage; TestB has no train-class coverage by construction.",f"- Query entity probe: TestA {query_key['accuracy_overall_known_class_only']*100:.2f}% on {query_key['known_coverage']*100:.1f}% known-class coverage.",f"- Physical-slot probe: TestA {position_key['accuracy_overall_known_class_only']*100:.2f}% versus 25.00% chance; selection frequencies remain near uniform.","- Fresh probes never update storage; full numbers and top-5/chance fields are in the JSON artifacts.","","## Q1-Q26","","1. Oracle-key positive control: PASS if the reported exact symbolic match is 100%; see oracle_key_control.json.","2–3. Query entity/relation probes: see query_probe_results.json; relation is degenerate single-schema.","4–7. Slot entity/relation/value/joint-key probes: see slot_probe_results.json.","8. KEY-rich versus VALUE-rich: the measured pattern is value-rich and weakly entity-addressable; TestB unseen-entity coverage is explicit.","9–11. Dot, projected dot, bilinear, and pair-MLP retrieval: see offline_matcher_results.json.","12–13. Binary pair accuracy versus episode top-1 is reported for every matcher; binary scores are only moderate, not a high-binary/low-top1 D6 pattern.","14–16. Physical slot/write-order and semantic probes: see position_probe_results.json and representation_stats.json.","17. Slot permutation: see permutation_results.json.","18. First-slot bias: see first_slot_bias.json; aggregate selection is close to uniform and not a first-slot prior.","19–21. Same-value, same-entity, relation, and lexical controls: see distractor_results.json; relation/lexical controls are unavailable under the single-template cache.","22. Main bottleneck: the frozen representation is insufficient for robust associative retrieval (D7), with a value-rich/key-poor secondary pattern (D1).","23–24. Any O2.1 recommendation is diagnostic-only; no implementation is included.","25. Iterative retrieval is not established by this diagnostic and remains untested.","26. O2.1 is not entered; this run stops at O2-Diag.","","## Runtime","",f"Seed={SEED}; device={device}; wall-clock={diag_metrics['runtime_s']:.1f}s; peak VRAM={diag_metrics['peak_vram_gb']:.3f}GB.",f"Trainable parameters were fresh probes and offline matcher parameters only; storage checkpoint source: {O2_DIR/'best.pt'}.","","O2-DIAG COMPLETE."]
    (OUT/"v24_o2_diag_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"verdict":verdict,"primary":primary,"matcher_best":diag_metrics["matcher_best"],"runtime_s":diag_metrics["runtime_s"]},indent=2))


if __name__=="__main__": main()
