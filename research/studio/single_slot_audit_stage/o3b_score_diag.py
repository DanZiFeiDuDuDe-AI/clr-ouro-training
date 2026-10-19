import json
import numpy as np
import torch
from pathlib import Path
from a1_v24_o3b import MRCache, FactPool, make_examples, KeyEncoder, UpdateRouter, O21B, OUT, CACHE_ROOT, SEED, router_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache = MRCache(CACHE_ROOT)
kc = torch.load(O21B, map_location="cpu", weights_only=False)
key = KeyEncoder(64).to(device); key.load_state_dict(kc["key"], strict=True); key.eval()
pool = FactPool(cache, "val")
examples = make_examples(pool, 1000, SEED + 2, True)
router = UpdateRouter().to(device)
ck = torch.load(OUT / "router_best.pt", map_location=device, weights_only=False)
router.load_state_dict(ck["router"], strict=True); router.eval()
raw = router_eval(router, key, examples, pool, device, 0.5)
y = raw["y"]; s = raw["score"]
print(json.dumps({
    "threshold_saved": ck["threshold"],
    "score_quantiles_all": np.quantile(s, [0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]).tolist(),
    "score_quantiles_new": np.quantile(s[y==0], [0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]).tolist(),
    "score_quantiles_update": np.quantile(s[y==1], [0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]).tolist(),
    "threshold_metrics": {str(t): router_eval(router, key, examples, pool, device, float(t))["binary"] for t in [0.000001,0.00001,0.00003,0.0001,0.001,0.01,0.05,0.1,0.2,0.5]},
}, ensure_ascii=False, indent=2))
