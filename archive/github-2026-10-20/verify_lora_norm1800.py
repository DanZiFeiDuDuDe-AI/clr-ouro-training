# -*- coding: utf-8 -*-
"""Verify training-script load path (incl. attach_lora) on P0H5 batch."""
import os, sys, random, json, math
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"
sys.path.insert(0, "D:/training")
import torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention
from clutrr_data import generate_dataset, RELATIONS

OUT = "D:/training/checkpoints/ouro_retention_m2_h2v2/verify_lora_result.json"
REPLACED = [3, 7, 11, 15, 19, 23]
POST_NORM_NAMES = ["input_layernorm_2", "post_attention_layernorm_2"]
MAX_TOKENS = 128
BASE = "D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt"
RANK, ALPHA = 4, 8.0
SEED = 42


class LoRALinear(torch.nn.Module):
    def __init__(self, orig, r, alpha):
        super().__init__()
        self.orig = orig
        self.r, self.alpha = r, alpha
        in_f, out_f = orig.in_features, orig.out_features
        _a = torch.empty(in_f, r)
        torch.nn.init.kaiming_uniform_(_a, a=math.sqrt(5))
        self.A = torch.nn.Parameter(_a.to(torch.bfloat16))
        self.B = torch.nn.Parameter(torch.zeros(r, out_f, dtype=torch.bfloat16))
        self.orig.requires_grad_(False)

    def forward(self, x):
        return self.orig(x) + (x @ self.A @ self.B) * (self.alpha / self.r)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def attach_lora(converted, r, alpha):
    for ret, idx in converted:
        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            orig = getattr(ret, proj_name)
            lora = LoRALinear(orig, r, alpha)
            lora = lora.to(orig.weight.device)
            setattr(ret, proj_name, lora)


def load_base(model, converted):
    ck = torch.load(BASE, map_location="cuda", weights_only=False)
    for ret, idx in converted:
        ret.gamma_logit.data.copy_(ck["retention_gamma"][str(idx)])
    pnw = ck.get("post_norm_weights")
    if pnw:
        for li in REPLACED:
            layer = model.model.layers[li]
            for nm, w in pnw[str(li)].items():
                getattr(layer, nm).weight.data.copy_(w.to("cuda"))
    return ck


random.seed(SEED)
torch.manual_seed(SEED)
model, tok = load_ouro()
converted = convert_ouro_to_retention(model, REPLACED, gamma_mode="sigmoid_logit")
for ret, idx in converted:
    layer = model.model.layers[idx]
    src_attn = layer.self_attn.src_attn
    from retention_ouro import _RetentionAttentionAdapter
    layer.self_attn = _RetentionAttentionAdapter(ret, src_attn)
freeze_all(model)
load_base(model, converted)
attach_lora(converted, RANK, ALPHA)
model.eval()

# verify B all zero
bs = [getattr(ret, p).B.abs().sum().item() for ret, idx in converted for p in ["q_proj", "k_proj", "v_proj", "o_proj"]]
print("lora B sums (all should be 0):", max(bs), flush=True)

random.seed(SEED)
torch.manual_seed(SEED)
_ = generate_dataset(600)
eval_data = generate_dataset(150)[:150]
cand_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
pad_id = tok.pad_token_id


@torch.no_grad()
def cand_scores(model, tok, prompt_ids, cand_ids, pad_id):
    n = len(cand_ids); P = len(prompt_ids); L = P + max(len(c) for c in cand_ids)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for i, c in enumerate(cand_ids):
        seq = prompt_ids + c
        input_ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
    bo, _, _ = model.model(input_ids=input_ids.cuda(), attention_mask=attn.cuda(), use_cache=False)
    logprobs = torch.log_softmax(model.lm_head(bo.last_hidden_state).float(), dim=-1)
    out = []
    for i, c in enumerate(cand_ids):
        lp = sum(logprobs[i, P - 1 + j, t].item() for j, t in enumerate(c)) / len(c)
        out.append(lp)
    return out


ranks, margins = [], []
for ti, traj in enumerate(eval_data):
    turn = traj["turns"][0]
    prompt_ids = tok(turn["prompt"], truncation=True, max_length=MAX_TOKENS)["input_ids"]
    s = cand_scores(model, tok, prompt_ids, cand_ids, pad_id)
    ci = RELATIONS.index(turn["answer"])
    order = sorted(range(len(s)), key=lambda i: s[i], reverse=True)
    ranks.append(order.index(ci) + 1)
    margins.append(s[ci] - max(x for i, x in enumerate(s) if i != ci))
    if (ti + 1) % 50 == 0:
        with open(OUT, "w") as f:
            json.dump({"progress": ti + 1}, f)

res = {
    "acc": sum(1 for r in ranks if r == 1), "n": len(ranks),
    "mean_rank": sum(ranks) / len(ranks),
    "mean_margin": sum(margins) / len(margins),
    "rank_gt3": sum(1 for r in ranks if r > 3),
    "p0h5_ref": {"acc": 0, "mean_rank": 9.85, "mean_margin": -4.9246, "rank_gt3": 146},
}
with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print("RESULT: %s" % json.dumps(res), flush=True)
