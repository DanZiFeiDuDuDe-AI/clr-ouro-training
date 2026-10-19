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
t0=time.perf_counter(); jlabels,_=j.joint_labels(infos,pmap); rows=[]
for carrier,source in (('K-I0','J-I0'),('K-I2b','J-I2b'),('K-I2a','J-I2a')):
    H=all_carriers[source]['train']; _,pca=k.pca_basis(H,256); subspaces={'learned_row':VR,'pca256':pca}
    for ri in range(k.N_RANDOM): subspaces[f'random_{ri}']=k.orth_basis(H.shape[1],256,k.SEED+1000+ri)
    arrs={name:{s:all_carriers[source][s]@B.T for s in k.SPLITS} for name,B in subspaces.items()}
    recs=k.probe_reps(arrs,infos,pmap,jlabels,torch.device('cuda'), f'{carrier}-recalibrated',eval_splits=('testA','testB'),steps=180)
    for name,B in subspaces.items():
        rec=recs[name]; rows.append({'carrier':carrier,'subspace':name,'entity_testA':rec['entity']['testA']['accuracy'],'relation_testA':rec['relation']['testA']['accuracy'],'joint_testA':rec['joint']['testA']['accuracy'],'entity_testB_known_fraction':rec['entity']['testB']['known_fraction'],'probe_steps':180})
with (OUT/'random_subspace_controls.csv').open(encoding='utf-8',newline='') as f: old=list(csv.DictReader(f))
old=[r for r in old if r.get('carrier') not in {'K-I0','K-I2b','K-I2a'}]
k.write_csv(OUT/'random_subspace_controls.csv',old+rows)
k.write_csv(OUT/'random_subspace_controls_180.csv',rows)
all_sweep_arrs={}; sweep_meta={}
for carrier,source in (('K-I0','J-I0'),('K-I2b','J-I2b'),('K-I2a','J-I2a')):
    Hs=all_carriers[source]
    for sweep,dirs in (('top',Vh),('bottom',Vh[::-1])):
        for kk in k.KSWEEP:
            name=f'{carrier}__{sweep}_{kk}'; B=dirs[:kk].astype(np.float32); all_sweep_arrs[name]={s:Hs[s]@B.T for s in k.SPLITS}; sweep_meta[name]=(carrier,sweep,int(kk))
sweep_recs=k.probe_reps(all_sweep_arrs,infos,pmap,jlabels,torch.device('cuda'),'all-carriers-recalibrated',steps=180)
sweep_rows=[]
for name,(carrier,sweep,kk) in sweep_meta.items():
    rec=sweep_recs[name]
    for metric in ('entity','relation','joint'):
        sweep_rows.append({'carrier':carrier,'sweep':sweep,'k':kk,'metric':metric,'train':rec[metric]['train']['accuracy'],'val':rec[metric]['val']['accuracy'],'entity_testA':rec[metric]['testA']['accuracy'],'testA':rec[metric]['testA']['accuracy'],'testB':rec[metric]['testB']['accuracy'],'probe_steps':180})
k.write_csv(OUT/'singular_direction_sweep.csv',sweep_rows)
(OUT/'controls_protocol.json').write_text(json.dumps({'carriers':['K-I0','K-I2b','K-I2a'],'probe_steps':180,'n_random':k.N_RANDOM,'eval_splits':['testA','testB'],'sweep_probe_steps':180,'elapsed_s':time.perf_counter()-t0},indent=2),encoding='utf-8')
cfg=json.loads((OUT/'config.json').read_text(encoding='utf-8')); cfg.update({'probe_steps_controls':180,'probe_steps_sweep':180}); (OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
print(json.dumps({'rows':len(rows),'elapsed_s':time.perf_counter()-t0,'out':str(OUT)}))
