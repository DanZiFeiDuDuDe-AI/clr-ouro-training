#!/usr/bin/env python3
"""
CLR Phase 2.2 — CLUTRR Local Edition (v2, fixed after U+FFFD damage)
Minimal module providing classes/functions needed by Phase 5B intervention experiments.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, json, random
from torch.utils.data import Dataset

WORKSPACE = "H:/CLR"
MODEL_PATH = os.environ.get("CLR_MODEL_PATH", f"{WORKSPACE}/models/Qwen3.5-0.8B")
DS = int(os.environ.get("CLR_DS", "64"))
DM = int(os.environ.get("CLR_DM", "1024"))
FULL_ATTN_LAYERS = json.loads(os.environ.get("CLR_FILM_LAYERS", "[3, 7, 11, 15, 19, 23]"))
MAX_TOKENS = 128
SMOKE = os.environ.get("CLR_SMOKE") == "1"
OUT_SUFFIX = os.environ.get("CLR_OUT_SUFFIX", "seed_{seed}")
TOTAL_STEPS = 4 if SMOKE else 400
EVAL_EVERY = 2 if SMOKE else 100
TRAIN_SAMPLES = 24 if SMOKE else 600
EVAL_SAMPLES = 6 if SMOKE else 100
LR = 3e-4
USE_LORA = False
STATIC_S = os.environ.get("CLR_STATIC_S", "0") == "1"
STATIC_SEED = os.environ.get("CLR_STATIC_SEED", "0")

from clutrr_data import RELATIONS, REL2ID, generate_dataset, bayes_ceiling
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_scheduler

# ── FiLMWrapper ──
class FiLMWrapper(nn.Module):
    """Multiplicative FiLM: hs = hs * (1 + scale(S_t))."""
    def __init__(self, layer, d_model):
        super().__init__()
        self.layer = layer
        self._s = None
        self.scale_proj = nn.Linear(DS, d_model, dtype=torch.bfloat16, device='cuda')
        nn.init.normal_(self.scale_proj.weight, mean=0.0, std=0.01)

    def forward(self, h, *args, **kwargs):
        if self._s is not None:
            s = self._s
            if s.shape[0] != h.shape[0]:
                s = s.expand(h.shape[0], -1)
            scale = 0.1 * torch.tanh(self.scale_proj(s.to(torch.bfloat16)))
            if h.dim() == 3:
                h = h * (1 + scale.unsqueeze(1))
            else:
                h = h * (1 + scale)
        o = self.layer(h, *args, **kwargs)
        if isinstance(o, tuple):
            return (h,) + o[1:] if isinstance(o, (tuple, list)) else h
        return o

# ── PlaaCore ──
class PlaaCore(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.read_proj = nn.Linear(d_model, DS, dtype=torch.bfloat16, device='cuda')
        self.gru = nn.GRUCell(DS, DS, dtype=torch.bfloat16, device='cuda')
        self.init_s = nn.Parameter(torch.zeros(1, DS, dtype=torch.bfloat16))

    def forward(self, S, h_last):
        return self.gru(self.read_proj(h_last.to(torch.bfloat16)), S)

    def init_state(self, B, device):
        base = self.init_s.expand(B, -1)
        return (base + torch.randn_like(base) * 0.01).contiguous().to(device)

# ── score_candidates ──
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

# ── evaluate ──
@torch.no_grad()
def evaluate(model, tok, trajs, film_layers, plaa_core, cand_ids, ablate_s=False,
             static_s=False, surgery_fn=None):
    """
    Evaluate per-trajectory on CLUTRR.
    surgery_fn(S, ti, traj_idx, film_layers) -> S_modified or None
    If not None, applied AFTER l._s is set and BEFORE model forward.
    The modified S from surgery_fn is also used for score_candidates.
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
            for l in film_layers:
                l._s = s_val
            if surgery_fn is not None:
                s_surg = surgery_fn(s_val, ti, traj_idx, film_layers)
                if s_surg is not None:
                    for l in film_layers:
                        l._s = s_surg
            hs = model(input_ids=pin, attention_mask=pmask,
                       output_hidden_states=True).hidden_states[-1]
            if static_s:
                S_next = S
            else:
                S_next = plaa_core(S, hs[0, -1].unsqueeze(0))
            s_pred = torch.zeros_like(S) if ablate_s else S_next
            # Pass surgery to score_candidates too
            sc_fn = None
            if surgery_fn is not None:
                def sc_fn(s_vec, _ti=ti, _tj=traj_idx):
                    return surgery_fn(s_vec, _ti, _tj, film_layers)
            pred = score_candidates(model, film_layers, prompt_ids, s_pred,
                                    cand_ids, pad_id, surgery_fn=sc_fn)
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

# ── Forward-only FiLM apply (for score_candidates consistency) ──
def apply_film(film_layers, S):
    """Set _s across all film layers. Used by external scripts."""
    for l in film_layers:
        l._s = S

# ── Dataset (legacy, not used by Phase 5B) ──
class CLUTRRTrajectoryDataset(Dataset):
    def __init__(self, data):
        self.trajs = data
        print(f"Dataset: {len(self.trajs)} trajectories", flush=True)
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, idx):
        turns = self.trajs[idx]["turns"]
        adapted = []
        for t in turns:
            adapted.append({"question": t["prompt"], "answer": t["answer"]})
        return adapted

def collate_fn(batch):
    max_t = max(len(t) for t in batch)
    result = []
    for ti in range(max_t):
        prompts, answers = [], []
        for t in batch:
            if ti < len(t):
                prompts.append(t[ti]["question"])
                answers.append(t[ti]["answer"])
        result.append({"prompts": prompts, "answers": answers})
    return result
