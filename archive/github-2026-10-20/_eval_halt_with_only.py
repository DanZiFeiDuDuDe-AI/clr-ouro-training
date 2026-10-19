#!/usr/bin/env python3
import os,sys,json,math,torch
sys.path.insert(0,r'D:/LoopLM')
os.environ.setdefault('PYTHONUNBUFFERED','1')
from clutrr_data import generate_dataset, bayes_ceiling
from clr_ouro import load_ouro, inject_film, turn_forward_v1, PlaaCore, HaltGate

CKPT=os.environ.get('CLR_EVAL_CKPT', r'D:/LoopLM/checkpoints/clr_ouro_v1_halt_s42_cw0p3_rep_ho/best_model.pt')
N=int(os.environ.get('CLR_EVAL_N','80'))
SEED=int(os.environ.get('CLR_SEED','42'))
REPLACE=os.environ.get('CLR_HALT_REPLACE','1')=='1'
MAX_TOKENS=128
DO_WITHOUT=os.environ.get('CLR_EVAL_WITHOUT','0')=='1'

def ans_id(tok,a):
    ids=tok(a, add_special_tokens=False)['input_ids']
    if isinstance(ids[0], list): ids=ids[0]
    return ids[0]

@torch.no_grad()
def run(model,tok,trajs,film,core,halt,ablate=False):
    c=t=0; es=0.0; en=0; gates=[]
    for i,traj in enumerate(trajs):
        S=core.init_state(1,'cuda')
        for turn in traj['turns']:
            inp=tok(turn['prompt'], return_tensors='pt', truncation=True, max_length=MAX_TOKENS).to('cuda')
            logits,S,_,_, = None,None,None,None
            out=turn_forward_v1(model,inp,film,core,S,ablate_s=ablate,
                halt_gate=None if ablate else halt, use_weighted_exit=True)
            logits,S_new,_,aux=out
            ok=int(logits.argmax().item()==ans_id(tok,turn['answer']))
            # only t2p: turns after first
            # count all non-first as t2p below
            S=S_new
            es+=float(aux['expected_exit_step']); en+=1
            gates.append(aux['gate_means'])
        # recompute t2p properly
        if (i+1)%10==0:
            print(f'  traj {i+1}/{len(trajs)}', flush=True)
        del inp
    # second pass for accuracy with proper t2p - actually redo cleanly
    return None

# clean implementation
@torch.no_grad()
def eval_once(model,tok,trajs,film,core,halt,ablate=False):
    t2p_c=t2p_t=all_c=all_t=0
    es=0.0; en=0; gsum=None; gn=0
    for i,traj in enumerate(trajs):
        S=core.init_state(1,'cuda')
        for ti,turn in enumerate(traj['turns']):
            inp=tok(turn['prompt'], return_tensors='pt', truncation=True, max_length=MAX_TOKENS).to('cuda')
            logits,S_new,_,aux=turn_forward_v1(model,inp,film,core,S,ablate_s=ablate,
                halt_gate=None if ablate else halt, use_weighted_exit=True)
            ok=int(logits.argmax().item()==ans_id(tok,turn['answer']))
            S=S_new
            all_t+=1; all_c+=ok
            if ti>0:
                t2p_t+=1; t2p_c+=ok
            es+=float(aux['expected_exit_step']); en+=1
            gm=aux['gate_means']
            if gsum is None: gsum=[0.0]*len(gm)
            for j,v in enumerate(gm): gsum[j]+=float(v)
            gn+=1
        if (i+1)%10==0:
            print(f'  {i+1}/{len(trajs)} t2p_run={t2p_c/max(t2p_t,1):.3f} Eexit={es/max(en,1):.3f}', flush=True)
    return {
        't2p':{'correct':t2p_c,'total':t2p_t,'acc':t2p_c/max(t2p_t,1)},
        'all':{'correct':all_c,'total':all_t,'acc':all_c/max(all_t,1)},
        'mean_expected_exit':es/max(en,1),
        'mean_gate_lambda':[x/max(gn,1) for x in gsum],
    }

def main():
    print('CKPT',CKPT,'N',N,'replace',REPLACE, flush=True)
    torch.manual_seed(SEED)
    data=generate_dataset(N)
    print('ceiling', bayes_ceiling(data), flush=True)
    model,tok=load_ouro()
    film=inject_film(model)
    core=PlaaCore(); halt=HaltGate(gain_init=5.0, replace=REPLACE)
    ck=torch.load(CKPT, map_location='cpu', weights_only=False)
    core.load_state_dict(ck['plaa_core'])
    for l,sd in zip(film, ck['scale_proj']): l.scale_proj.load_state_dict(sd)
    if 'halt_gate' in ck:
        print('halt', halt.load_state_dict(ck['halt_gate'], strict=False), flush=True)
    print('gain', float(halt.gain.float()), flush=True)
    print('WITH_S...', flush=True)
    w=eval_once(model,tok,data,film,core,halt,False)
    print('with', w, flush=True)
    o=None
    if DO_WITHOUT:
        print('WITHOUT_S...', flush=True)
        o=eval_once(model,tok,data,film,core,halt,True)
        print('without', o, flush=True)
    out=str(Path(CKPT).parent / f'final_eval_n{N}_with{"_wo" if DO_WITHOUT else ""}.json')
    json.dump({'ckpt':CKPT,'n':N,'with_S':w,'without_S':o,'step':ck.get('step'),'score':ck.get('best_score')}, open(out,'w'), indent=2)
    print('wrote', out, flush=True)

if __name__=='__main__':
    from pathlib import Path
    main()
