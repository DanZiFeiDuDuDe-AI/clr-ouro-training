# -*- coding: utf-8 -*-
"""
P0I pilot training: direct task-objective training (candidate CE),
NOT teacher imitation. 4 conditions (2 inits x 2 state lifetimes).

Inits:
  M1      = checkpoints/ouro_retention_m1/best_model.pt (migration-only)
  N1800   = checkpoints/ouro_retention_m2_d3/best_model.pt (norm1800)

State lifetime:
  reset   = retention state zeroed at each turn boundary
  persist = retention state carried from prev turn end (loop-0 only)

Objective: candidate cross-entropy on real CLUTRR ground truth.
  L = CE(candidate_scores, correct_index)
  No teacher cache, no KL, no block loss.

Ouro loop boundary: always reset (current_ut>0 -> _state=None).
Episode boundary: always reset.
Position: C-persist uses continuous offset; C-reset uses 0 each turn.

Seed 42. Output: checkpoints/p0i_pilot_{init}_{mode}_s42/
"""
import os, sys, json, time, math, glob, random
# --- stdout safety: redirect ALL output to a file, never touch the pipe ---
_OUT_DIR_EARLY = os.environ.get("P0I_OUT_DIR", "D:/training/checkpoints/p0i_pilot_tmp")
os.makedirs(_OUT_DIR_EARLY, exist_ok=True)
_LOG_FH = open(os.path.join(_OUT_DIR_EARLY, "train_stdout.log"), "a", encoding="utf-8", buffering=1)
def _safe_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args) + "\n"
    _LOG_FH.write(msg)
    _LOG_FH.flush()
# Replace builtins.print so all print() calls go to file only
import builtins
builtins.print = _safe_print
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"

import numpy as np
import torch
import torch.nn.functional as F
from clr_ouro import load_ouro
from retention_ouro import convert_ouro_to_retention, _RetentionAttentionAdapter
from clutrr_data import generate_dataset, RELATIONS

REPLACED = [3, 7, 11, 15, 19, 23]
POST_NORM_NAMES = ["input_layernorm_2", "post_attention_layernorm_2"]
MAX_TOKENS = 128
SEED = int(os.environ.get("P0I_SEED", "42"))
N_STEPS = int(os.environ.get("P0I_STEPS", "600"))
LR = float(os.environ.get("P0I_LR", "3e-4"))
INIT = os.environ.get("P0I_INIT", "m1")        # m1 | n1800
MODE = os.environ.get("P0I_MODE", "reset")      # reset | persist
OUT_DIR = os.environ.get("P0I_OUT_DIR",
                         "D:/training/checkpoints/p0i_pilot_%s_%s_s%d" % (INIT, MODE, SEED))
SMOKE = os.environ.get("P0I_SMOKE", "") == "1"

CKPT_MAP = {
    "m1":    "D:/training/checkpoints/ouro_retention_m1/best_model.pt",
    "n1800": "D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt",
}


def _detach_state(state):
    """Recursively detach retention state so cross-turn injection never
    carries a grad_fn graph (tensor / tuple / list / dict / None)."""
    if state is None:
        return None
    if isinstance(state, torch.Tensor):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(_detach_state(s) for s in state)
    if isinstance(state, list):
        return [_detach_state(s) for s in state]
    if isinstance(state, dict):
        return {k: _detach_state(v) for k, v in state.items()}
    return state


class P0IAdapter(_RetentionAttentionAdapter):
    """Adapter with Ouro-loop-boundary reset and turn-state management."""
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False, cache_position=None,
                position_embeddings=None, **kwargs):
        ut = kwargs.get("current_ut", 0)
        if ut > 0:
            self._state = None  # loop boundary reset
        out, state = self.ret.forward(
            hidden_states,
            q_proj=self.ret.q_proj, k_proj=self.ret.k_proj,
            v_proj=self.ret.v_proj, o_proj=self.ret.o_proj,
            position_embeddings=position_embeddings,
            state=getattr(self, "_state", None),
            mode=getattr(self, "_mode", "recurrent"))
        self._last_state = state
        if ut == 0:
            # record DETACHED turn-end state: injection into next turn must
            # not carry this turn's autograd graph (turn-boundary detach).
            self._turn_state = _detach_state(state)
        return out, None


def build_model(init_tag):
    model, tok = load_ouro()
    converted = convert_ouro_to_retention(model, REPLACED, gamma_mode="sigmoid_logit")
    adapters = []
    for ret, idx in converted:
        layer = model.model.layers[idx]
        adapter = P0IAdapter(ret, layer.self_attn.src_attn)
        layer.self_attn = adapter
        adapters.append(adapter)
    for p in model.parameters():
        p.requires_grad_(False)
    ck_path = CKPT_MAP[init_tag]
    ck = torch.load(ck_path, map_location="cuda", weights_only=False)
    for ret, idx in converted:
        if "retention_gamma" in ck and str(idx) in ck["retention_gamma"]:
            ret.gamma_logit.data.copy_(ck["retention_gamma"][str(idx)].to("cuda"))
    pnw = ck.get("post_norm_weights")
    train_params = []
    if pnw:
        for li in REPLACED:
            layer = model.model.layers[li]
            for nm, w in pnw[str(li)].items():
                getattr(layer, nm).weight.data.copy_(w.to("cuda"))
    # Unfreeze post-norm weights for training
    for li in REPLACED:
        layer = model.model.layers[li]
        for nm in POST_NORM_NAMES:
            w = getattr(layer, nm).weight
            w.requires_grad_(True)
            train_params.append(w)
    print("init=%s ckpt=%s trainable=%d (post-norm)" % (
        init_tag, ck_path, sum(p.numel() for p in train_params)), flush=True)
    return model, tok, adapters, train_params


def reset_turn_state(adapters):
    for a in adapters:
        a._state = None
        a._turn_state = None


def set_persist_state(adapters):
    """Inject last turn-end state (recursively detached) into all adapters.
    turn2_start state == turn1_end state, detached value."""
    for a in adapters:
        ts = getattr(a, "_turn_state", None)
        a._state = _detach_state(ts) if ts is not None else None


def clear_all_state(adapters):
    for a in adapters:
        a._state = None
        a._turn_state = None
        a._last_state = None


def candidate_scores(model, tok, prompt_ids, cand_ids, pad_id, offset=0,
                      chunk_size=2):
    """Differentiable candidate scoring: mean log-prob of candidate tokens.
    Processes candidates in small chunks to avoid OOM on 8GB GPUs.
    position policy (P0I design 2.2):
      C-reset   -> offset=0, positions arange(L) inside each turn
      C-persist -> offset = cumulative previous-turn length
    """
    n = len(cand_ids)
    P = len(prompt_ids)
    L = P + max(len(c) for c in cand_ids)
    scores = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch = cand_ids[start:end]
        bs = len(batch)
        input_ids = torch.full((bs, L), pad_id, dtype=torch.long)
        attn = torch.zeros((bs, L), dtype=torch.long)
        pos = torch.zeros((bs, L), dtype=torch.long)
        for i, c in enumerate(batch):
            seq = prompt_ids + c
            sl = len(seq)
            input_ids[i, :sl] = torch.tensor(seq)
            attn[i, :sl] = 1
            pos[i, :sl] = torch.arange(sl, dtype=torch.long, device="cuda") + offset
        input_ids, attn, pos = input_ids.cuda(), attn.cuda(), pos.cuda()
        out = model.model(input_ids=input_ids, attention_mask=attn, use_cache=False,
                          position_ids=pos)
        if isinstance(out, tuple):
            out = out[0]
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        elif isinstance(out, tuple):
            out = out[0]
        logprobs = torch.log_softmax(model.lm_head(out).float(), dim=-1)
        for i, c in enumerate(batch):
            cc = torch.tensor(c, dtype=torch.long, device=input_ids.device)
            cpos = torch.arange(len(c), dtype=torch.long, device=input_ids.device) + (P - 1)
            lp = logprobs[i, cpos, cc].mean()
            scores.append(lp)
        del input_ids, attn, pos, out, logprobs
        torch.cuda.empty_cache()
    return torch.stack(scores)


def save_ckpt(step, model, adapters, train_params, final=False, extra=None):
    ck = {
        "variant": "p0i_pilot_%s_%s_s%d" % (INIT, MODE, SEED),
        "init": INIT, "mode": MODE,
        "replaced_layers": REPLACED,
        "retention_gamma": {str(REPLACED[i]): adapters[i].ret.gamma_logit.detach().cpu().clone()
                            for i in range(len(adapters))},
        "post_norm_weights": {
            str(li): {nm: getattr(model.model.layers[li], nm).weight.detach().cpu().clone()
                      for nm in POST_NORM_NAMES}
            for li in REPLACED},
        "step": step, "lr": LR, "seed": SEED,
        "init_ckpt": CKPT_MAP[INIT],
    }
    if extra:
        ck.update(extra)
    path = os.path.join(OUT_DIR, "best_model.pt" if final else "ckpt_step%04d.pt" % step)
    torch.save(ck, path)
    return path


def main():
    print("=== P0I pilot: init=%s mode=%s steps=%d lr=%g ===" % (
        INIT, MODE, N_STEPS, LR), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    PROGRESS_PATH = os.path.join(OUT_DIR, "progress.json")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model, tok, adapters, train_params = build_model(INIT)
    opt = torch.optim.Adam(train_params, lr=LR)

    # Generate training data (seed 42, same protocol)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    _ = generate_dataset(600)         # replay parity
    _ = generate_dataset(150)[:150]   # held-out placeholder (skip)
    n_train = 600
    corpus = generate_dataset(n_train)[:n_train]
    print("train trajectories: %d" % len(corpus), flush=True)

    cand_ids = [tok(r, add_special_tokens=False)["input_ids"] for r in RELATIONS]
    pad_id = tok.pad_token_id

    step = 0
    t0 = time.time()
    losses = []
    matched = 0
    skipped = 0
    epoch = 0
    while step < N_STEPS and epoch < 3:
        epoch += 1
        print("epoch %d (step %d)" % (epoch, step), flush=True)
        for traj in corpus:
            if step >= N_STEPS:
                break
            clear_all_state(adapters)
            offset = 0
            for turn_i, turn in enumerate(traj["turns"]):
                if step >= N_STEPS:
                    break
                # turn boundary state injection
                if turn_i == 0:
                    reset_turn_state(adapters)
                elif MODE == "reset":
                    reset_turn_state(adapters)
                else:
                    set_persist_state(adapters)
                prompt_ids = tok(turn["prompt"], truncation=True,
                                 max_length=MAX_TOKENS)["input_ids"]
                s_len = len(prompt_ids)
                correct_i = RELATIONS.index(turn["answer"])
                scores = candidate_scores(model, tok, prompt_ids, cand_ids, pad_id,
                                          offset=offset)
                loss = F.cross_entropy(scores.unsqueeze(0),
                                       torch.tensor([correct_i], device=scores.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
                matched += 1
                step += 1
                try:
                    with open(os.path.join(OUT_DIR, "step_marker.txt"), "w") as f:
                        f.write("step=%d\n" % step)
                except Exception:
                    pass
                # advance offset for persist
                if MODE == "persist":
                    offset += s_len
                else:
                    offset = 0
                if step % 50 == 0 or step == N_STEPS:
                    el = time.time() - t0
                    print("step %d/%d: loss=%.4f (%.0fs)" % (
                        step, N_STEPS, loss.item(), el), flush=True)
                    try:
                        with open(PROGRESS_PATH, "w") as f:
                            json.dump({"step": step, "n_steps": N_STEPS,
                                       "loss": loss.item(), "elapsed_s": el,
                                       "epoch": epoch, "matched": matched,
                                       "skipped": skipped,
                                       "last_losses": losses[-50:]}, f)
                    except Exception:
                        pass
                if step % 300 == 0 or step == N_STEPS:
                    p = save_ckpt(step, model, adapters, train_params,
                                  final=(step == N_STEPS),
                                  extra={"final_loss": float(loss.item())})
                    print("checkpoint: %s" % p, flush=True)

    final_path = save_ckpt(step, model, adapters, train_params, final=True,
                           extra={"final_loss": float(losses[-1]) if losses else None,
                                  "matched": matched, "skipped": skipped,
                                  "epochs": epoch})
    with open(os.path.join(OUT_DIR, "train_log.json"), "w") as f:
        json.dump({"init": INIT, "mode": MODE, "steps": step, "losses": losses,
                   "final_loss": float(losses[-1]) if losses else None,
                   "lr": LR, "seed": SEED, "matched": matched, "skipped": skipped,
                   "epochs": epoch, "ckpt_path": final_path}, f, indent=2)
    print("\nsaved to %s" % OUT_DIR, flush=True)
    print("P0I pilot %s/%s OK" % (INIT, MODE), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
