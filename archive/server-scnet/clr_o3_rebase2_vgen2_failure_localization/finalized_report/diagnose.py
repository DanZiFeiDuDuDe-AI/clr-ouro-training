"""Train/VAL-only VGen2 diagnosis. Does not import or open test splits."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path('/root/private_data/clr_o3_rebase2_vgen2_failure_localization')
PARENT=Path('/root/private_data/clr_o3_rebase1_fresh_operational_baseline')
GOOD=Path('/root/private_data/clr_o3_rebase1_vgen2_10k_diag')
MR=Path('/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache')
J3=Path('/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o3b_j3c_fresh_certified_feature_cache')
spec=importlib.util.spec_from_file_location('vgen_rebase',PARENT/'code/train_vgen2_rebase.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
torch.set_num_threads(4)
torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def finish(out):
 files=[{'path':str(p.relative_to(out)), 'sha256':base.sha256(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file() and p.name not in {'manifest.json','COMPLETE.json'}]
 dump(out/'manifest.json',{'files':files}); dump(out/'COMPLETE.json',{'success':True,'time':time.time()})

def load_model(p):
 ck=torch.load(p,map_location='cpu',weights_only=False)
 core=base.OracleLocalCore().to(DEVICE); heads=base.MRValueHeads().to(DEVICE)
 core.load_state_dict(ck['core'],strict=True); heads.load_state_dict(ck['value_heads'],strict=True)
 core.eval(); heads.eval(); return core,heads,ck

def csvrows(p): return base.read_csv(p)

@torch.no_grad()
def representations(data,core,heads):
 d=data.d; x=d['fact_H'].astype(np.float32); y=d['fact_Y'].astype(np.int64)
 ns=np.repeat(d['N'],np.diff(d['fact_offsets'])).astype(np.int64)
 slots=np.zeros(len(y),dtype=np.int64); rng=np.random.default_rng(73042)
 for i,n in enumerate(d['N']): slots[int(d['fact_offsets'][i]):int(d['fact_offsets'][i+1])]=rng.choice(4,size=int(n),replace=False)
 pairs=data.pairs; indices=np.concatenate([np.arange(len(y)),pairs[:,1]])
 target=np.concatenate([y,pairs[:,3]])
 cap=np.concatenate([ns,np.zeros(len(pairs),dtype=np.int64)])
 slot=np.concatenate([slots,(np.arange(len(pairs))//64)%4])
 source=np.asarray(x[indices]); proj=[]; latent=[]; logits=[]
 for start in range(0,len(indices),512):
  end=min(start+512,len(indices)); h=torch.as_tensor(source[start:end],device=DEVICE)
  z=core.content_read(h); prior=torch.zeros(len(h),64,device=DEVICE)
  mask=np.arange(start,end)>=len(y)
  if mask.any():
   pi=np.arange(start,end)[mask]-len(y)
   old=torch.as_tensor(x[pairs[pi,0]],device=DEVICE)
   prior[torch.as_tensor(mask,device=DEVICE)]=core.shared_gru(core.content_read(old),torch.zeros(len(old),64,device=DEVICE))
  value=core.shared_gru(z,prior)
  all_logits=torch.stack([head(value) for head in heads.heads],dim=1)
  selected=all_logits[torch.arange(len(h),device=DEVICE),torch.as_tensor(slot[start:end],device=DEVICE)]
  proj.append(z.cpu().numpy()); latent.append(value.cpu().numpy()); logits.append(selected.cpu().numpy())
 raw=np.load(J3/f'H_base_I0_{data.split}.npy',allow_pickle=False)[indices].astype(np.float32)
 return {'R0':raw,'R1':source,'R2':np.concatenate(proj),'R3':np.concatenate(latent),'R4':np.concatenate(logits),
         'y':target,'N':cap,'index':indices,'slot':slot,'single_count':np.asarray(len(y))}

def metrics(pred,d):
 y=d['y']; n=d['N']; out={}
 for name,m in [('value_identity',np.ones(len(y),bool)),('single_write',n>0),('N1',n==1),('N2',n==2),('N4',n==4),('UPDATE_latest',n==0)]:
  out[name]={'accuracy':float((pred[m]==y[m]).mean()),'correct':int((pred[m]==y[m]).sum()),'count':int(m.sum())}
 return out

def geometry(x,y):
 xx=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12); rng=np.random.default_rng(42)
 a=rng.integers(len(y),size=100000); b=rng.integers(len(y),size=100000)
 good=a!=b; a=a[good]; b=b[good]; cos=(xx[a]*xx[b]).sum(1); same=y[a]==y[b]
 def stats(z): return {'mean':float(z.mean()),'std':float(z.std()),'count':len(z)}
 sc,dc=cos[same],cos[~same]
 # Pairwise AUC: probability that a random same-value cosine exceeds a different-value cosine.
 sample=rng.choice(dc,size=len(sc),replace=True)
 return {'within_class_cosine':stats(sc),'between_class_cosine':stats(dc),'same_vs_different_cosine_auc':float(((sc>sample)+0.5*(sc==sample)).mean())}

def prepare():
 out=ROOT/'prepare'; out.mkdir(parents=True,exist_ok=True)
 train=base.SplitData('train',MR,J3); val=base.SplitData('val',MR,J3)
 overlap=set(x['canonical_fact_id'] for x in train.labels)&set(x['canonical_fact_id'] for x in val.labels)
 if overlap: raise ValueError('Cross-split canonical source contamination')
 audit={'train':train.audit,'val':val.audit,'canonical_fact_overlap':len(overlap),'git_sha':None,'git_sha_reason':'workspace and source snapshot are not Git repositories','test_files_opened':False,'candidate_b_modified':False,'rule_a_threshold':0.8504285216331482}
 audit['source_hashes']={}
 for split in ('train','val'):
  for path in [MR/f'{split}_data.npz',J3/f'labels_{split}.csv',J3/f'H_fresh_{split}_sidecar.csv',J3/f'H_base_I0_{split}.npy']:
   audit['source_hashes'][str(path)]=base.sha256(path)
 audit['artifacts']={}
 paths={'3k':PARENT/'components/vgen2/vgen2_rebase.pt','10k':GOOD/'vgen2_rebase_10k_best.pt'}
 valpacks={}
 for name,path in paths.items():
  core,heads,ck=load_model(path); audit['artifacts'][name]={'sha256':base.sha256(path),'step':ck['step'],'config':ck.get('config')}
  for split,data in [('train',train),('val',val)]:
   pack=representations(data,core,heads); np.savez_compressed(out/f'{name}_{split}.npz',**pack)
   if split=='val': valpacks[name]=pack
  audit[name]={'single':base.eval_single(core,heads,val,DEVICE,73042),'update':base.eval_updates(core,heads,val,DEVICE)}
  audit[name]['geometry']={stage:geometry(valpacks[name][stage],valpacks[name]['y']) for stage in ('R0','R1','R2','R3','R4')}
  audit[name]['R0_R1_exact']=bool(np.array_equal(valpacks[name]['R0'],valpacks[name]['R1']))
 a,b=valpacks['3k'],valpacks['10k']; pa=a['R4'].argmax(1); pb=b['R4'].argmax(1); yy=a['y']; n=a['N']; idx=a['index']
 mask=(pb==yy)&(pa!=yy)
 replay={'comparison':'paired 3K failing versus 10K passing rebase, NOT an authenticated historical VGen2 checkpoint',
         'examples':len(yy),'labels_equal':bool(np.array_equal(yy,b['y'])),'raw_input_equal':bool(np.array_equal(a['R1'],b['R1'])),
         'projection_max_abs_diff':float(np.abs(a['R2']-b['R2']).max()),'latent_max_abs_diff':float(np.abs(a['R3']-b['R3']).max()),
         'passing_correct_failing_wrong':int(mask.sum()),'both_wrong':int(((pa!=yy)&(pb!=yy)).sum()),
         'failing_correct_passing_wrong':int(((pa==yy)&(pb!=yy)).sum()),'breakdown':{},'historical_checkpoint_replay':'UNAVAILABLE; prior bounded recovery found no historical VGen2/R2 weights'}
 for field in ('relation_semantic_value','entity_semantic_value','object_value'):
  source=val.sidecar if field=='object_value' else val.labels
  values=np.array([source[int(i)].get(field,'') for i in idx]); groups={}
  for key in sorted(set(values)):
   m=values==key; groups[str(key)]={'count':int(m.sum()),'3k_wrong':int(((pa!=yy)&m).sum()),'10k_wrong':int(((pb!=yy)&m).sum()),'rescued':int((mask&m).sum())}
  replay['breakdown'][field]=groups
 replay['by_context']={k:{'rescued':int((mask&m).sum()),'3k_wrong':int(((pa!=yy)&m).sum()),'count':int(m.sum())} for k,m in [('single_write',n>0),('UPDATE_latest',n==0)]}
 rows=[]
 for i in range(len(yy)):
  rows.append({'canonical_fact_id':val.labels[int(idx[i])]['canonical_fact_id'],'context':'UPDATE' if n[i]==0 else f'N{n[i]}','slot':int(a['slot'][i]),'label':int(yy[i]),'pred_3k':int(pa[i]),'pred_10k':int(pb[i]),'passing_correct_failing_wrong':bool(mask[i])})
 base.write_csv(out/'paired_replay.csv',rows)
 # Check vocabulary consistency from source sidecars; an object must map to one class across both splits.
 mapping={}; conflicts=[]
 for data in (train,val):
  for row,y in zip(data.sidecar,data.d['fact_Y']):
   value=row.get('object_value',''); previous=mapping.setdefault(value,int(y))
   if previous!=int(y): conflicts.append({'object':value,'old':previous,'new':int(y)})
 audit['value_class_mapping']=mapping; audit['value_mapping_conflicts']=conflicts
 audit['class_histogram_train']=np.bincount(train.d['fact_Y'],minlength=64).tolist()
 audit['class_histogram_val']=np.bincount(val.d['fact_Y'],minlength=64).tolist()
 for name in ('carrier_construction.json','extraction_policy_frozen.json','joint_label_map.json','environment_manifest.json'):
  path=J3/name
  if path.exists(): (out/name).write_bytes(path.read_bytes()); audit['source_hashes'][str(path)]=base.sha256(path)
 dump(out/'input_audit.json',audit); dump(out/'paired_replay.json',replay); finish(out)

def probe(snapshot,stage,kind):
 out=ROOT/f'probe_{snapshot}_{stage}_{kind}'; out.mkdir(parents=True,exist_ok=True)
 tr=dict(np.load(ROOT/'prepare'/f'{snapshot}_train.npz')); va=dict(np.load(ROOT/'prepare'/f'{snapshot}_val.npz'))
 x=tr[stage].astype(np.float32); v=va[stage].astype(np.float32); y=tr['y']; yy=va['y']
 torch.manual_seed(42); rng=np.random.default_rng(42)
 mu=x.mean(0); sd=x.std(0); sd=np.maximum(sd,1e-5)
 if kind=='centroid':
  z=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
  cent=np.stack([z[y==i].mean(0) if (y==i).any() else np.zeros(x.shape[1]) for i in range(64)])
  cent/=np.linalg.norm(cent,axis=1,keepdims=True)+1e-12
  vv=v/(np.linalg.norm(v,axis=1,keepdims=True)+1e-12); pred=(vv@cent.T).argmax(1)
  torch.save({'centroids':torch.from_numpy(cent),'kind':kind,'fit_split':'train'},out/'probe.pt'); curves=[]
 else:
  tx=torch.from_numpy((x-mu)/sd).to(DEVICE); vx=torch.from_numpy((v-mu)/sd).to(DEVICE); ty=torch.from_numpy(y).to(DEVICE)
  model=nn.Linear(x.shape[1],64) if kind=='linear' else nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,64))
  model=model.to(DEVICE); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=.01); curves=[]
  # Fixed budget, no validation-driven selection or early stopping.
  single=np.flatnonzero(tr['N']>0); update=np.flatnonzero(tr['N']==0)
  for step in range(1,2001):
   ids=np.concatenate([rng.choice(single,128),rng.choice(update,128)])
   opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(tx[ids]),ty[ids]); loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.); opt.step()
   if step%200==0:
    with torch.no_grad(): acc=float((model(tx).argmax(1)==ty).float().mean())
    curves.append({'step':step,'train_loss':float(loss.detach()),'train_accuracy':acc,'gradient_norm':float(norm)})
  with torch.no_grad(): logits=model(vx); pred=logits.argmax(1).cpu().numpy(); trainacc=float((model(tx).argmax(1)==ty).float().mean())
  torch.save({'state_dict':model.cpu().state_dict(),'mean':torch.from_numpy(mu),'std':torch.from_numpy(sd),'kind':kind,'stage':stage,'hidden_width':64,'fit_split':'train','steps':2000,'seed':42},out/'probe.pt')
  dump(out/'train_metrics.json',{'train_accuracy':trainacc,'parameters':sum(p.numel() for p in model.parameters()),'curves':curves})
 result={'snapshot':snapshot,'stage':stage,'kind':kind,'fit':'train only','evaluation':'frozen validation only','selection':'final fixed-step model; no validation selection','steps':0 if kind=='centroid' else 2000,'preprocessing':'cosine' if kind=='centroid' else 'train-only per-feature standardization','metrics':metrics(pred,va)}
 dump(out/'metrics.json',result); finish(out); print(json.dumps(result),flush=True)

def sanity(size,scale):
 # Only authorized after raw MLP clears the information diagnostic gate.
 raw=[json.loads((ROOT/f'probe_3k_R1_{kind}/metrics.json').read_text()) for kind in ('mlp','linear')]
 if not any(r['metrics']['single_write']['accuracy']>=.95 and r['metrics']['UPDATE_latest']['accuracy']>=.95 for r in raw): raise RuntimeError('Raw supervised probe evidence insufficient: stop optimization sanity')
 out=ROOT/f'sanity_{size}_{scale:g}'; out.mkdir(parents=True,exist_ok=True)
 data=base.SplitData('train',MR,J3); val=base.SplitData('val',MR,J3)
 rng=np.random.default_rng(42); ids=rng.choice(len(data.d['fact_Y']),size=min(size,len(data.d['fact_Y'])),replace=False)
 x=torch.tensor(data.d['fact_H'][ids].astype(np.float32),device=DEVICE); y=torch.tensor(data.d['fact_Y'][ids],device=DEVICE)
 core,heads,ck=load_model(PARENT/'components/vgen2/vgen2_rebase.pt'); params=list(core.parameters())+list(heads.parameters())
 opt=torch.optim.AdamW(params,lr=3e-4*scale,weight_decay=.01); curves=[]
 core.train(); heads.train()
 for step in range(1,2001):
  chosen=rng.choice(len(ids),64); slot=int(rng.integers(4)); state=core.write_to_slot(core.zero_state(len(chosen),device=DEVICE),x[chosen],slot)
  loss=F.cross_entropy(heads.logits(state,slot),y[chosen]); opt.zero_grad(set_to_none=True); loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.); opt.step()
  if step%200==0:
   core.eval(); heads.eval()
   with torch.no_grad():
    allpred=torch.stack([heads.logits(core.write_to_slot(core.zero_state(len(x),device=DEVICE),x,s),s) for s in range(4)])
    acc=float((allpred.argmax(-1)==y[None,:]).float().mean()); entropy=float((-(allpred.softmax(-1)*allpred.log_softmax(-1)).sum(-1)).mean())
   single=base.eval_single(core,heads,val,DEVICE,73042); update=base.eval_updates(core,heads,val,DEVICE)
   curves.append({'step':step,'loss':float(loss.detach()),'gradient_norm_before_clip':float(norm),'train_accuracy_all_slots':acc,'prediction_entropy':entropy,'val_single':single,'val_update':update})
   core.train(); heads.train()
 torch.save({'core':core.cpu().state_dict(),'value_heads':heads.cpu().state_dict(),'train_ids':ids,'steps':2000,'lr_scale':scale,'initialization':'archived failing 3K checkpoint','scope':'single-write overfit sanity; not a production repair'},out/'sanity.pt')
 matrix=np.zeros((64,64),dtype=int)
 for truth,pred in zip(np.tile(y.cpu().numpy(),4),allpred.argmax(-1).cpu().numpy().reshape(-1)): matrix[truth,pred]+=1
 dump(out/'metrics.json',{'size':size,'lr_scale':scale,'steps':2000,'curves':curves,'confusion_matrix':matrix.tolist(),'class_histogram':np.bincount(y.cpu().numpy(),minlength=64).tolist(),'selection':'final fixed-step; no test access or model selection'})
 finish(out); print(json.dumps({'sanity':size,'lr_scale':scale,'train_accuracy':acc}),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('task'); ap.add_argument('--snapshot',default='3k'); ap.add_argument('--stage',default='R1'); ap.add_argument('--kind',default='mlp'); ap.add_argument('--size',type=int,default=256); ap.add_argument('--scale',type=float,default=1.)
 args=ap.parse_args()
 if args.task=='prepare': prepare()
 elif args.task=='probe': probe(args.snapshot,args.stage,args.kind)
 elif args.task=='sanity': sanity(args.size,args.scale)
 else: raise ValueError(args.task)
