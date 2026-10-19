# -*- coding: utf-8 -*-
import os, sys, json, math, random, statistics
from pathlib import Path
os.environ['OURO_MODEL_PATH']='D:/training/Ouro-1.4B'
sys.path.insert(0,'D:/training')
import numpy as np, torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention, _RetentionAttentionAdapter
from clutrr_data import generate_dataset, RELATIONS

REPLACED=[3,7,11,15,19,23]; MAX_TOKENS=128
CK='D:/training/checkpoints/ouro_retention_m2_h2v2_final_rerun/best_model.pt'
BASE='D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt'
SEED=42
class LoRALinear(torch.nn.Module):
    def __init__(self, orig, r, alpha):
        super().__init__(); self.orig=orig; self.r=r; self.alpha=alpha
        self.A=torch.nn.Parameter(torch.empty(orig.in_features,r,dtype=torch.bfloat16))
        self.B=torch.nn.Parameter(torch.zeros(r,orig.out_features,dtype=torch.bfloat16))
        self.orig.requires_grad_(False)
    def forward(self,x): return self.orig(x)+(x@self.A@self.B)*(self.alpha/self.r)

def build_teacher():
    m,t=load_ouro(); return m,t

def build_student():
    m,t=load_ouro(); conv=convert_ouro_to_retention(m,REPLACED,gamma_mode='sigmoid_logit')
    for ret,idx in conv:
        layer=m.model.layers[idx]; layer.self_attn=_RetentionAttentionAdapter(ret,layer.self_attn.src_attn)
    for p in m.parameters(): p.requires_grad_(False)
    ck=torch.load(BASE,map_location='cuda',weights_only=False)
    for ret,idx in conv: ret.gamma_logit.data.copy_(ck['retention_gamma'][str(idx)].to('cuda'))
    for li in REPLACED:
        for nm,w in ck['post_norm_weights'][str(li)].items(): getattr(m.model.layers[li],nm).weight.data.copy_(w.to('cuda'))
    h2=torch.load(CK,map_location='cuda',weights_only=False)
    for ret,idx in conv:
        for proj in ['q_proj','k_proj','v_proj','o_proj']:
            orig=getattr(ret,proj); info=h2['lora'][str(idx)][proj]
            l=LoRALinear(orig,int(info['r']),float(info['alpha'])).to(orig.weight.device)
            l.A.data.copy_(info['A'].to('cuda')); l.B.data.copy_(info['B'].to('cuda'))
            setattr(ret,proj,l)
    m.eval(); return m,t

def prompt_ids(tok,s): return tok(s,truncation=True,max_length=MAX_TOKENS)['input_ids']
@torch.no_grad()
def score(m,tok,pids,cands,pad):
    n=len(cands); P=len(pids); L=P+max(map(len,cands)); ids=torch.full((n,L),pad,dtype=torch.long); mask=torch.zeros((n,L),dtype=torch.long)
    for i,c in enumerate(cands):
        seq=pids+c; ids[i,:len(seq)]=torch.tensor(seq); mask[i,:len(seq)]=1
    out=m.model(input_ids=ids.cuda(),attention_mask=mask.cuda(),use_cache=False)
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, 'last_hidden_state'):
        out = out.last_hidden_state
    elif isinstance(out, tuple):
        out = out[0]
    lp=torch.log_softmax(m.lm_head(out).float(),dim=-1); scores=[]
    for i,c in enumerate(cands): scores.append(sum(lp[i,P-1+j,t].item() for j,t in enumerate(c))/len(c))
    return np.array(scores), lp[0,P-1].detach().cpu().numpy()
def rankdata(a):
    order=np.argsort(a); r=np.empty(len(a),float); r[order]=np.arange(1,len(a)+1); return r
def spearman(a,b):
    x=rankdata(a); y=rankdata(b); return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else 0.0

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    _=generate_dataset(600); data=generate_dataset(150)[:150]
    print('P0H5 data=%d'%len(data),flush=True)
    teacher,tok=build_teacher(); student,_=build_student(); cands=[tok(r,add_special_tokens=False)['input_ids'] for r in RELATIONS]; pad=tok.pad_token_id
    rows=[]; teacher_vecs=[]; student_vecs=[]; sranks=[]; smargins=[]; tranks=[]; tmargins=[]; s_ent=[]; t_ent=[]; s_cos=[]; s_dnll=[]
    for i,tr in enumerate(data):
        turn=tr['turns'][0]; p=prompt_ids(tok,turn['prompt']); ci=RELATIONS.index(turn['answer'])
        ts,tl=score(teacher,tok,p,cands,pad); ss,sl=score(student,tok,p,cands,pad)
        for vec,out in [(ts,tl),(ss,sl)]:
            pass
        def met(x):
            order=np.argsort(-x); rank=int(np.where(order==ci)[0][0])+1; margin=float(x[ci]-np.max(np.delete(x,ci))); return rank,margin
        r,m=met(ts); tranks.append(r); tmargins.append(m); r,m=met(ss); sranks.append(r); smargins.append(m)
        teacher_vecs.extend(ts.tolist()); student_vecs.extend(ss.tolist())
        s_ent.append(float(-(F.softmax(torch.tensor(sl),dim=0)*F.log_softmax(torch.tensor(sl),dim=0)).sum()))
        t_ent.append(float(-(F.softmax(torch.tensor(tl),dim=0)*F.log_softmax(torch.tensor(tl),dim=0)).sum()))
        s_cos.append(float(np.dot(sl,tl)/(np.linalg.norm(sl)*np.linalg.norm(tl))))
        ans=tok(turn['answer'],add_special_tokens=False)['input_ids'][0]
        s_dnll.append(float(-F.log_softmax(torch.tensor(sl),dim=0)[ans]+F.log_softmax(torch.tensor(tl),dim=0)[ans]))
        if (i+1)%25==0: print('eval %d/150'% (i+1),flush=True)
    res={'n':150,'checkpoint':CK,'seed':SEED,'p0h5_protocol':'seed42 -> generate_dataset(600) -> generate_dataset(150)',
      'M0_teacher':{'candidate_accuracy':sum(r==1 for r in tranks),'mean_rank':statistics.mean(tranks),'mean_margin':statistics.mean(tmargins),'entropy':statistics.mean(t_ent)},
      'H2_v2':{'candidate_accuracy':sum(r==1 for r in sranks),'mean_rank':statistics.mean(sranks),'mean_margin':statistics.mean(smargins),'entropy':statistics.mean(s_ent),'spearman_vs_M0':spearman(np.array(teacher_vecs),np.array(student_vecs)),'logit_cosine':statistics.mean(s_cos),'delta_nll':statistics.mean(s_dnll)},
      'diagnostic':{'teacher_ranks':tranks,'student_ranks':sranks}}
    Path('P0H2_V2_P0H5_EVAL.json').write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__': main()
