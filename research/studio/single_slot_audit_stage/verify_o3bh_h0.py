import json
from pathlib import Path
import numpy as np
import torch
from a1_v24_o3b import FactPool, MRCache, KeyEncoder, DIM
from a1_v24_o3b_diag import KeyBank
from a1_v24_o3b_e import pair_refs
import a1_v24_o3b_f as fdiag
from a1_v24_o3b_g import BindingKey

CACHE=Path(r"D:/training/checkpoints/clr_v24_o21a_mr/cache"); O21B=Path(r"D:/training/checkpoints/clr_v24_o21b/b1_best.pt")
GOUT=Path(r"D:/training/checkpoints/clr_v24_o3b_g_binding_key"); HOUT=Path(r"D:/training/checkpoints/clr_v24_o3b_h_entity_carrier_diag")
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache=MRCache(CACHE); pools={s:FactPool(cache,s) for s in ("train","val","testA","testB")}
key=KeyEncoder(DIM).to(dev); key.load_state_dict(torch.load(O21B,map_location="cpu",weights_only=False)["key"]); key.eval()
for p in key.parameters(): p.requires_grad_(False)
banks={s:KeyBank(pools[s],key,dev) for s in pools}
pair_seeds={"train":43,"val":44,"testA":45,"testB":46}
sets={s:pair_refs(banks[s], 6000 if s=="train" else 3000, pair_seeds[s]) for s in ("train","val","testA","testB")}
raw={s:(d["ka"]*d["kb"]).sum(1) for s,d in sets.items()}; th=fdiag.choose_threshold(sets["val"]["y"],raw["val"])
out={"device":str(dev),"training":False,"protocol":"exact O3b-G K-K pair_refs; train/val/testA/testB seeds 43/44/45/46","baseline":{"threshold_val":th}}
for s in ("val","testA","testB"): out["baseline"][s]=fdiag.metric(sets[s]["y"],raw[s],th,sets[s]["categories"])
nentity=len({r["person"] for r in banks["train"].refs}); g3=BindingKey("G3_projected_tensor",nentity).to(dev); g3.load_state_dict(torch.load(GOUT/"checkpoints"/"G3_projected_tensor_best.pt",map_location=dev,weights_only=False)["model"]); g3.eval()
def score(d):
  vals=[]
  with torch.no_grad():
    for st in range(0,len(d["y"]),1024):
      a=torch.from_numpy(d["ka"][st:st+1024]).float().to(dev); b=torch.from_numpy(d["kb"][st:st+1024]).float().to(dev); ka,_,_=g3(a); kb,_,_=g3(b); vals.append((ka*kb).sum(-1).cpu().numpy())
  return np.concatenate(vals)
gs={s:score(sets[s]) for s in sets}; gth=fdiag.choose_threshold(sets["val"]["y"],gs["val"]); out["G3_projected_tensor"]={"threshold_val":gth}
for s in ("val","testA","testB"): out["G3_projected_tensor"][s]=fdiag.metric(sets[s]["y"],gs[s],gth,sets[s]["categories"])
(HOUT/"h0_frozen_recheck.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(out,ensure_ascii=False,indent=2))
