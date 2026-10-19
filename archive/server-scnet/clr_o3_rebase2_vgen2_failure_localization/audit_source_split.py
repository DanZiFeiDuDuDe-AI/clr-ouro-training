"""Supplemental read-only source-level audit, train/VAL only."""
import hashlib,json
from pathlib import Path
import numpy as np
MR=Path('/root/private_data/clr_v24_o3b_l/checkpoints/clr_v24_o21a_mr/cache')
OUT=Path('/root/private_data/clr_o3_rebase2_vgen2_failure_localization/source_audit')
OUT.mkdir(exist_ok=True)
datasets={};meta={};texts={};semantic={};result={'test_files_opened':False,'splits':{}}
for split in ('train','val'):
 with np.load(MR/f'{split}_data.npz',allow_pickle=False) as z: d={k:z[k] for k in z.files}
 datasets[split]=d
 meta[split]=[json.loads(s) for s in (MR/f'{split}_meta.jsonl').read_text().splitlines() if s.strip()]
 texts[split]=[t for ep in meta[split] for t in ep['fact_texts']]
 semantic[split]=[(p,r,c) for ep in meta[split] for p,r,c in zip(ep['persons'],ep['relations'],ep['cities'])]
 if len(texts[split])!=len(d['fact_Y']): raise ValueError('Text row offset mismatch')
 offsets=d['fact_offsets']; target=d['fact_Y'][offsets[:-1]+d['query_idx']]
 mismatches=int((target!=d['Y']).sum())
 semantic_targets=np.array([y for ep in meta[split] for y in ep['fact_classes']])
 wrong=int((semantic_targets!=d['fact_Y']).sum())
 result['splits'][split]={'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in d.items()},'query_target_offset_mismatches':mismatches,'metadata_fact_class_mismatches':wrong,'duplicate_fact_text_rows':len(texts[split])-len(set(texts[split])),'raw_norm_mean':float(np.linalg.norm(d['fact_H'],axis=1).mean())}
shared=set(texts['train'])&set(texts['val']);shared_sem=set(semantic['train'])&set(semantic['val'])
val_overlap=np.array([t in shared for t in texts['val']],dtype=bool)
result['cross_split']={'unique_shared_exact_fact_texts':len(shared),'validation_fact_rows_seen_as_exact_text_in_train':int(val_overlap.sum()),'validation_fact_rows':len(val_overlap),'shared_semantic_entity_relation_value_triples':len(shared_sem),'episode_ids_overlap':len(set(x['episode_id'] for x in meta['train'])&set(x['episode_id'] for x in meta['val'])),'qualification':'Split-prefixed canonical IDs alone cannot exclude duplicate source text or semantic triples.'}
# Pair/representation row indexing allows measurement on VAL facts unseen as exact train text.
for task in ['probe_3k_R1_linear','probe_3k_R1_mlp']:
 import torch
 from torch import nn
 root=OUT.parent
 pack=dict(np.load(root/'prepare/3k_val.npz'))
 ck=torch.load(root/task/'probe.pt',map_location='cpu',weights_only=False)
 model=nn.Linear(2048,64) if ck['kind']=='linear' else nn.Sequential(nn.Linear(2048,64),nn.ReLU(),nn.Linear(64,64))
 model.load_state_dict(ck['state_dict'],strict=True);model.eval()
 with torch.no_grad(): pred=model((torch.from_numpy(pack['R1'])-ck['mean'])/ck['std']).argmax(1).numpy()
 unseen=~val_overlap[pack['index']]
 result[task]={}
 for name,m in [('single_unseen_exact_text',unseen&(pack['N']>0)),('update_unseen_latest_exact_text',unseen&(pack['N']==0))]:
  result[task][name]={'count':int(m.sum()),'accuracy':float((pred[m]==pack['y'][m]).mean())}
np.save(OUT/'validation_seen_exact_train_text.npy',val_overlap)
(OUT/'source_split_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['cross_split']))
