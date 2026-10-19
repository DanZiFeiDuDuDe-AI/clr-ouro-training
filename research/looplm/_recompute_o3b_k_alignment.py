import csv, json, os, time
from pathlib import Path
import numpy as np
import torch
import a1_v24_o3b_k as k
import a1_v24_o3b_j as j
from a1_v24_o3b import MRCache, KeyEncoder, DIM

OUT=Path(os.environ.get('V24_O3BK_OUT',r'D:/training/checkpoints/clr_v24_o3b_k_first_linear_subspace'))
cache=MRCache(k.CACHE_ROOT); infos,pmap=j.build_infos(cache); token,_=j.load_token_arrays(); all_carriers=j.aligned_carriers(infos,token)
ck=torch.load(k.O21B,map_location='cpu',weights_only=False); key=KeyEncoder(DIM); key.load_state_dict(ck['key'],strict=True); W=key.net[0].weight.detach().float().cpu().numpy(); _,_,Vh=np.linalg.svd(W,full_matrices=False); VR=Vh.astype(np.float32)

def exact_classifier_dirs(A,kmax=64):
    A=np.asarray(A,dtype=np.float64); A=A-A.mean(0,keepdims=True); _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:min(kmax,len(vt))].astype(np.float32)

def exact_fisher_dirs(X,y,kmax=64):
    X=np.asarray(X,dtype=np.float64); y=np.asarray(y,dtype=np.int64); m=y>=0; X=X[m]; y=y[m]; mu=X.mean(0); rows=[]; weights=[]
    for c in sorted(set(y.tolist())):
        xc=X[y==c]
        if len(xc): rows.append(xc.mean(0)-mu); weights.append(np.sqrt(len(xc)))
    A=np.asarray(rows,dtype=np.float64)*np.asarray(weights,dtype=np.float64)[:,None]; _,_,vt=np.linalg.svd(A,full_matrices=False); return vt[:min(kmax,len(vt))].astype(np.float32)

t0=time.perf_counter(); rows=[]
for carrier,source in (('K-I0','J-I0'),('K-I2b','J-I2b'),('K-I2a','J-I2a')):
    H=all_carriers[source]['train']; y=infos['train']['entity'][:len(H)]; r=infos['train']['relation'][:len(H)]
    we,_=k.fit_classifier(H,y,len(pmap),torch.device('cuda'),steps=60,seed=k.SEED+300); wr,_=k.fit_classifier(H,r,4,torch.device('cuda'),steps=60,seed=k.SEED+301)
    dirs={'classifier':{'entity':exact_classifier_dirs(we),'relation':exact_classifier_dirs(wr)},'fisher':{'entity':exact_fisher_dirs(H,y),'relation':exact_fisher_dirs(H,r)}}
    for kind,bylabel in dirs.items():
        for label_kind in ('entity','relation'):
            for kk in (8,16,32,64):
                z=k.alignment(bylabel[label_kind][:kk],VR); z.update({'carrier':carrier,'subspace_kind':kind,'label_kind':label_kind,'k':kk,'direction_svd':'exact'}); rows.append(z)
with (OUT/'principal_angles.csv').open(encoding='utf-8',newline='') as f: old=list(csv.DictReader(f))
old=[r for r in old if r.get('carrier') not in {'K-I0','K-I2b','K-I2a'}]
k.write_csv(OUT/'principal_angles.csv',old+rows)
(OUT/'alignment_protocol.json').write_text(json.dumps({'carriers':['K-I0','K-I2b','K-I2a'],'classifier_steps':60,'direction_svd':'exact','rows':len(rows),'elapsed_s':time.perf_counter()-t0},indent=2),encoding='utf-8')
print(json.dumps({'rows':len(rows),'elapsed_s':time.perf_counter()-t0}))
