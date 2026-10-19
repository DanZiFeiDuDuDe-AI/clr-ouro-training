import json
import numpy as np
import torch
from a1_v24_o3b import MRCache, FactPool, make_examples, KeyEncoder, O21B, CACHE_ROOT, SEED

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache = MRCache(CACHE_ROOT)
kc = torch.load(O21B, map_location="cpu", weights_only=False)
key = KeyEncoder(64).to(device); key.load_state_dict(kc["key"], strict=True); key.eval()

def scores(examples, pool):
    vals=[]; ys=[]; cats=[]
    for ex in examples:
        bh=np.stack([pool.hidden(x) for x in ex['base']]+[np.zeros(2048,dtype=np.float32) for _ in range(4-len(ex['base']))])
        kn=pool.hidden(ex['incoming'])
        with torch.no_grad():
            kb=key(torch.from_numpy(bh).to(device)); q=key(torch.from_numpy(kn).to(device))
        vals.append(float((kb[:len(ex['base'])]@q).max().item())); ys.append(ex['action']); cats.append(ex['category'])
    return np.asarray(vals),np.asarray(ys),cats

def metrics(y,s,t):
    p=(s>=t).astype(int); tp=((p==1)&(y==1)).sum(); tn=((p==0)&(y==0)).sum(); fp=((p==1)&(y==0)).sum(); fn=((p==0)&(y==1)).sum(); pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); f=2*pr*rc/max(pr+rc,1e-9); return {'acc':float((p==y).mean()),'f1':float(f),'recall':float(rc),'new_recall':float(tn/max(tn+fp,1)),'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn)}

def main():
    pv=FactPool(cache,'val'); pa=FactPool(cache,'testA'); pb=FactPool(cache,'testB')
    ev=make_examples(pv,1000,SEED+2,True); ea=make_examples(pa,1000,SEED+3,True); eb=make_examples(pb,1000,SEED+4,True)
    sv,yv,cv=scores(ev,pv); sa,ya,ca=scores(ea,pa); sb,yb,cb=scores(eb,pb)
    best=(0,0)
    for t in np.linspace(-1,1,4001):
        m=metrics(yv,sv,t)
        if m['f1']>best[0]: best=(m['f1'],t)
    t=best[1]
    out={'threshold':t,'val':metrics(yv,sv,t),'testA':metrics(ya,sa,t),'testB':metrics(yb,sb,t),'quantiles':{'val_new':np.quantile(sv[yv==0],[0,.1,.5,.9,1]).tolist(),'val_update':np.quantile(sv[yv==1],[0,.1,.5,.9,1]).tolist()},'hard_negative':{}}
    for name,s,y,c in [('testA',sa,ya,ca),('testB',sb,yb,cb)]:
        out['hard_negative'][name]={}
        for cat in sorted(set(c)):
            m=np.asarray([x==cat for x in c]); out['hard_negative'][name][cat]=metrics(y[m],s[m],t)
    print(json.dumps(out,ensure_ascii=False,indent=2))
main()
