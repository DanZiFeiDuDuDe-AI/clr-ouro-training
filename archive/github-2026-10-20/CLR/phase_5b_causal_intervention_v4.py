#!/usr/bin/env python3
"""
Phase 5B v4.1 — Self-contained causal intervention.
Embeds evaluate() and score_candidates() directly instead of importing from v2.
"""
import os, json, random, time
import torch
import numpy as np

from clutrr_data import RELATIONS, generate_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

WORKSPACE = "H:/CLR"
MODEL_PATH = os.environ.get("CLR_MODEL_PATH", f"{WORKSPACE}/models/Qwen3.5-0.8B")
DS = int(os.environ.get("CLR_DS", "64"))
FILM_LAYERS_LIST = json.loads(os.environ.get("CLR_FILM_LAYERS", "[3, 7, 11, 15, 19, 23]"))
MAX_TOKENS = 128
CKPT = "local_clutrr_v2_nolora"
CKPT_SEED = 42
device = "cuda"

# ── FiLMWrapper ──
class FiLMWrapper(torch.nn.Module):
    def __init__(self, layer, d_model):
        super().__init__()
        self.layer = layer
        self._s = None
        self.scale_proj = torch.nn.Linear(DS, d_model, dtype=torch.bfloat16, device='cuda')
        torch.nn.init.normal_(self.scale_proj.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                  position_ids=None, past_key_values=None, **kwargs):
        o = self.layer(hidden_states, position_embeddings,
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, **kwargs)
        h = o[0] if isinstance(o, (tuple, list)) else o
        if self._s is not None:
            scale = torch.tanh(self.scale_proj(self._s).unsqueeze(1))
            h = h * (1 + 0.1 * scale)
        return (h,) + o[1:] if isinstance(o, (tuple, list)) else h

# ── PlaaCore ──
class PlaaCore(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.read_proj = torch.nn.Linear(d_model, DS, dtype=torch.bfloat16, device='cuda')
        self.gru = torch.nn.GRUCell(DS, DS, dtype=torch.bfloat16, device='cuda')
        self.init_s = torch.nn.Parameter(torch.zeros(1, DS, dtype=torch.bfloat16))

    def forward(self, S, h_last):
        return self.gru(self.read_proj(h_last.to(torch.bfloat16)), S)

    def init_state(self, B, device):
        base = self.init_s.expand(B, -1)
        return (base + torch.randn_like(base) * 0.01).contiguous().to(device)

# ── score_candidates (with surgery_fn support) ──
def score_candidates(model, film_layers, prompt_ids, s_vec, cand_ids, pad_id, surgery_fn=None):
    n = len(cand_ids)
    P = len(prompt_ids)
    L = P + max(len(c) for c in cand_ids)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for i, c in enumerate(cand_ids):
        seq = prompt_ids + c
        input_ids[i, :len(seq)] = torch.tensor(seq)
        attn[i, :len(seq)] = 1
    input_ids, attn = input_ids.cuda(), attn.cuda()
    if surgery_fn is not None:
        s_vec = surgery_fn(s_vec)
    for l in film_layers:
        l._s = s_vec
    logprobs = torch.log_softmax(
        model(input_ids=input_ids, attention_mask=attn).logits.float(), dim=-1)
    best, best_i = -1e30, 0
    for i, c in enumerate(cand_ids):
        lp = sum(logprobs[i, P - 1 + j, t].item() for j, t in enumerate(c)) / len(c)
        if lp > best:
            best, best_i = lp, i
    return best_i

# ── evaluate (with permanent surgery_fn that affects BOTH forward and score) ──
@torch.no_grad()
def evaluate(model, tok, trajs, film_layers, plaa_core, cand_ids, ablate_s=False,
             static_s=False, surgery_fn=None):
    """
    surgery_fn(S, ti, traj_idx, film_layers) -> S_modified
    Applied EVERY time FiLM _s is set, including in score_candidates.
    """
    model.eval()
    counts = {"t1": [0, 0], "t2p": [0, 0], "last": [0, 0], "all": [0, 0]}
    pad_id = tok.pad_token_id
    for traj_idx, traj in enumerate(trajs):
        S = plaa_core.init_state(1, "cuda")
        turns = traj["turns"]
        for ti, turn in enumerate(turns):
            prompt_ids = tok(turn["prompt"], truncation=True,
                             max_length=MAX_TOKENS)["input_ids"]
            pin = torch.tensor([prompt_ids], device="cuda")
            pmask = torch.ones_like(pin)
            s_val = torch.zeros_like(S) if ablate_s else S

            # Create a closure that captures ti/traj_idx and applies surgery
            def mk_sc_fn(_ti, _tj):
                if surgery_fn is None:
                    return None
                def fn(s_vec):
                    return surgery_fn(s_vec, _ti, _tj, film_layers)
                return fn

            sc_fn_ti = mk_sc_fn(ti, traj_idx)

            # Forward: apply surgery BEFORE model
            for l in film_layers:
                l._s = sc_fn_ti(s_val) if sc_fn_ti else s_val
            hs = model(input_ids=pin, attention_mask=pmask,
                       output_hidden_states=True).hidden_states[-1]
            if static_s:
                S_next = S
            else:
                S_next = plaa_core(S, hs[0, -1].unsqueeze(0))
            s_pred = torch.zeros_like(S) if ablate_s else S_next

            # Predict: surgery also applied inside score_candidates
            pred = score_candidates(model, film_layers, prompt_ids, s_pred,
                                    cand_ids, pad_id, surgery_fn=sc_fn_ti)
            ok = int(RELATIONS[pred] == turn["answer"])
            counts["all"][0] += ok; counts["all"][1] += 1
            key = "t1" if ti == 0 else "t2p"
            counts[key][0] += ok; counts[key][1] += 1
            if ti == len(turns) - 1:
                counts["last"][0] += ok; counts["last"][1] += 1
            S = S_next
        torch.cuda.empty_cache()
    model.train()
    return {k: (c / t if t else 0.0) for k, (c, t) in counts.items()}

# ── Load model ──
print("Loading model...", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
    quantization_config=bnb, device_map="auto", trust_remote_code=True,
    torch_dtype=torch.bfloat16, attn_implementation="sdpa")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token if tok.eos_token else "<|endoftext|>"
for p in model.parameters():
    p.requires_grad = False
DM = model.config.hidden_size

layers = model.model.layers
film_layers = []
for idx in FILM_LAYERS_LIST:
    layers[idx] = FiLMWrapper(layers[idx], DM)
    film_layers.append(layers[idx])
plaa_core = PlaaCore(DM).cuda()
model.eval()

ck = torch.load(f"{WORKSPACE}/checkpoints/{CKPT}/best_model.pt",
    map_location=device, weights_only=False)
for idx, fl in zip(FILM_LAYERS_LIST, film_layers):
    fl.scale_proj.load_state_dict(ck["scale_proj"][idx])
plaa_core.load_state_dict(ck["plaa_core"])
cand_ids = [tok(" " + r, add_special_tokens=False)["input_ids"] for r in RELATIONS]

# ── Data ──
rng_eval = random.Random(CKPT_SEED + 1000)
eval_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_eval)
rng_other = random.Random(9999)
other_data = generate_dataset(100, (2, 3), (0.6, 0.4), rng_other)

# ── Collect states ──
print("Collecting states...", flush=True)
def collect_full_states(data):
    all_states = []
    for traj in data:
        S = plaa_core.init_state(1, device)
        traj_S = []
        for ti, turn in enumerate(traj["turns"]):
            prompt_ids = tok(turn["prompt"], truncation=True, max_length=MAX_TOKENS)["input_ids"]
            for l in film_layers:
                l._s = S
            pin = torch.tensor([prompt_ids], device=device)
            hs = model(input_ids=pin, attention_mask=torch.ones_like(pin),
                       output_hidden_states=True).hidden_states[-1]
            S_next = plaa_core(S, hs[0, -1].unsqueeze(0))
            traj_S.append(S_next[0].detach().cpu().float().numpy().copy())
            S = S_next
        all_states.append(traj_S)
        torch.cuda.empty_cache()
    return all_states

print("Collecting eval states...", flush=True)
eval_states = collect_full_states(eval_data)
print("Using eval states for diff control (avoid 2nd heavy collect)", flush=True)
other_states = eval_states

# ── Pre-allocate intervention tensors ──
backward_s = [torch.from_numpy(eval_states[t][0]).unsqueeze(0).to(device, dtype=torch.bfloat16)
              for t in range(len(eval_states))]
forward_s = [torch.from_numpy(eval_states[t][1]).unsqueeze(0).to(device, dtype=torch.bfloat16)
             if len(eval_states[t]) > 1 else None for t in range(len(eval_states))]
diff_s = [None] * len(eval_states)
for t in range(len(eval_states)):
    oi = (t + 37) % len(other_states)
    diff_s[t] = torch.from_numpy(other_states[oi][1]).unsqueeze(0).to(device, dtype=torch.bfloat16)

# ── Surgery functions ──
def no_surgery(S, ti, traj_idx, film_layers):
    return S

def hop_backward(S, ti, traj_idx, film_layers):
    return backward_s[traj_idx] if ti == 1 else S

def hop_forward(S, ti, traj_idx, film_layers):
    return forward_s[traj_idx] if (ti == 0 and forward_s[traj_idx] is not None) else S

def diff_control(S, ti, traj_idx, film_layers):
    return diff_s[traj_idx] if ti == 1 else S

def add_noise(S, ti, traj_idx, film_layers):
    return S + torch.randn_like(S) * 0.1 if ti == 1 else S

def zero_ablation(S, ti, traj_idx, film_layers):
    return torch.zeros_like(S) if ti == 1 else S

# ── Run ──
results = {}
names = ["A: Normal (baseline)", "B: Hop backward (S0->ti=1)",
         "C: Hop forward (S1->ti=0)", "D: Diff traj S (control)",
         "E: Noise (control)", "F: Zero S (ablation)"]
fns = [no_surgery, hop_backward, hop_forward, diff_control, add_noise, zero_ablation]

print("\n=== Causal Intervention v4.1 (self-contained, dual hook) ===", flush=True)
for name, fn in zip(names, fns):
    t0 = time.time()
    r = evaluate(model, tok, eval_data, film_layers, plaa_core, cand_ids, surgery_fn=fn)
    elapsed = time.time() - t0
    results[name] = r
    print("  %-35s: t2p=%.3f in %ds" % (name, r["t2p"], elapsed), flush=True)

baseline = results["A: Normal (baseline)"]["t2p"]
print("\n" + "=" * 70, flush=True)
print("CAUSAL INTERVENTION — FINAL SUMMARY (v4.1)", flush=True)
print("=" * 70, flush=True)
print("%-35s %7s %7s %8s %7s" % ("Condition", "t2p", "t1", "Delta", "Dpp"), flush=True)
print("-" * 70, flush=True)
for name in names:
    r = results[name]
    t2p = r["t2p"]
    t1 = r.get("t1", 0)
    delta = t2p - baseline
    print("%-35s %7.3f %7.3f %+8.3f %+6.1fpp" % (name, t2p, t1, delta, delta * 100), flush=True)

output = {"checkpoint": CKPT, "conditions": {k: {"t2p": v["t2p"], "t1": v.get("t1", 0)} for k, v in results.items()}, "baseline_t2p": baseline}
with open(f"{WORKSPACE}/02_results/phase_5b_causal_intervention_v4.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved: v4 JSON", flush=True)
print("Done.", flush=True)
