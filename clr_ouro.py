# -*- coding: utf-8 -*-
"""
CLR × Ouro — core modules (M1).

Self-contained port of the CLR FiLM/PlaaCore machinery onto Ouro-1.4B
(24 shared layers × total_ut_steps=4 universal-transformer loops).

Design (see DESIGN.md):
  - FiLM wraps OuroDecoderLayer at layers {3,7,11,15,19,23}; because layer
    weights are shared across UT loops, FiLM automatically applies at every
    loop; with per-loop S updates (v1) the modulation becomes loop-dependent.
  - v0: S updates once per TURN (after the full 4-loop forward)  — direct CLR port.
  - v1: S updates once per LOOP (after each UT step), via looped_forward_with_state.
  - S stays the ONLY cross-turn channel (turn prompts are truncated upstream).

Known port gotchas handled here:
  - OuroModel.forward reads `decoder_layer.attention_type` -> wrapper must expose it.
  - OuroModel.forward returns a 3-tuple (BaseModelOutputWithPast, hs_list, gate_list).
  - OuroDecoderLayer returns a bare tensor; wrapper asserts this (the tuple branch
    of the old CLR wrapper silently skipped the layer).
"""
import os
import torch
import torch.nn as nn

# 2026-07-10: moved OFF the H: drive after it bit-corrupted model.safetensors
# TWICE at the same offset (bad sectors). Weights live on D: now; verify SHA256
# after any copy: 58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af
MODEL_PATH = os.environ.get("OURO_MODEL_PATH", "D:/LoopLM_models/Ouro-1.4B")
DM = 2048           # Ouro hidden size
DS = 64             # validated on Qwen3.5-0.8B (16 insufficient, 64 sufficient)
FILM_LAYERS = [3, 7, 11, 15, 19, 23]   # same 24-layer geometry as the 0.8B study
ALPHA = 0.1


class FiLMWrapper(nn.Module):
    """Multiplicative FiLM on the layer input: h <- h * (1 + ALPHA*tanh(W S))."""

    def __init__(self, layer, d_model=DM, d_state=DS):
        super().__init__()
        self.layer = layer
        self._s = None
        self.scale_proj = nn.Linear(d_state, d_model, dtype=torch.bfloat16, device="cuda")
        nn.init.normal_(self.scale_proj.weight, mean=0.0, std=0.01)
        # OuroModel.forward reads this attribute off each layer object:
        self.attention_type = layer.attention_type

    def forward(self, h, *args, **kwargs):
        if self._s is not None:
            s = self._s
            if s.shape[0] != h.shape[0]:
                s = s.expand(h.shape[0], -1)
            scale = ALPHA * torch.tanh(self.scale_proj(s.to(torch.bfloat16)))
            h = h * (1 + (scale.unsqueeze(1) if h.dim() == 3 else scale))
        o = self.layer(h, *args, **kwargs)
        assert not isinstance(o, (tuple, list)), \
            "OuroDecoderLayer returned a tuple; FiLMWrapper needs the bare-tensor path"
        return o


class PlaaCore(nn.Module):
    def __init__(self, d_model=DM, d_state=DS):
        super().__init__()
        self.read_proj = nn.Linear(d_model, d_state, dtype=torch.bfloat16, device="cuda")
        self.gru = nn.GRUCell(d_state, d_state, dtype=torch.bfloat16, device="cuda")
        self.init_s = nn.Parameter(torch.zeros(1, d_state, dtype=torch.bfloat16))

    def forward(self, S, h_last):
        return self.gru(self.read_proj(h_last.to(torch.bfloat16)), S)

    def init_state(self, B, device):
        base = self.init_s.expand(B, -1)
        return (base + torch.randn_like(base) * 0.01).contiguous().to(device)

    def update(self, S, h_last):
        """StateModule alias of forward(): S_{t+1} = GRU(read_proj(h_last), S_t)."""
        return self.forward(S, h_last)

    def to_modulation(self, S):
        """StateModule alias: the vector state IS the FiLM modulation."""
        return S




class HaltGate(nn.Module):
    """S-conditioned bias (or replacement) for Ouro early_exit_gate logits.

    Paper 2 Direction C prototype:
      lambda_i = sigmoid(gate(h_i) + gain * halt_proj(S_i))   # default residual
      or, if replace=True: lambda_i = sigmoid(gain * halt_proj(S_i))

    Backbone early_exit_gate stays frozen. gain starts >1 so a small halt_proj
    can actually move the near-saturated pretrained early-loop gates.
    """

    def __init__(self, d_state=DS, gain_init=5.0, replace=False):
        super().__init__()
        self.replace = replace
        self.halt_proj = nn.Linear(d_state, 1, dtype=torch.bfloat16, device="cuda")
        nn.init.zeros_(self.halt_proj.weight)
        # slight positive bias -> mild early-exit prior once gain is applied
        nn.init.constant_(self.halt_proj.bias, 0.1)
        self.gain = nn.Parameter(torch.tensor([gain_init], dtype=torch.bfloat16, device="cuda"))

    def bias(self, S):
        """Return [B, 1, 1] bias for broadcasting onto gate logits [B, T, 1]."""
        b = self.halt_proj(S.to(torch.bfloat16))  # [B, 1]
        g = self.gain.to(b.dtype).clamp(min=0.0)
        return (g * b).unsqueeze(1)

def load_ouro(model_path=MODEL_PATH, four_bit=True):
    """Load frozen Ouro-1.4B (+tokenizer). Requires the two local
    transformers-5.x compat patches already applied to modeling_ouro.py."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    kwargs = dict(device_map={"": "cuda:0"}, trust_remote_code=True,
                  torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, tok


def inject_film(model):
    """Replace FILM_LAYERS of the (shared) stack with FiLM wrappers."""
    layers = model.model.layers
    film_layers = []
    for idx in FILM_LAYERS:
        w = FiLMWrapper(layers[idx])
        layers[idx] = w
        film_layers.append(w)
    return film_layers


def set_state(film_layers, S):
    for l in film_layers:
        l._s = S


# ── v0 forward: one S per whole (4-loop) turn forward ──
def turn_forward_v0(model, inp, film_layers, S):
    """Run a full Ouro forward (all UT loops) with fixed S.
    Returns (last_hidden_state, logits_at_last_pos, h_last_vec)."""
    set_state(film_layers, S)
    base_out, hs_list, gate_list = model.model(
        input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
        use_cache=False)
    h = base_out.last_hidden_state
    lens = inp["attention_mask"].sum(dim=1) - 1
    h_last = h[0, lens[0]]
    logits = model.lm_head(h[:, lens[0]])[0]
    return h, logits, h_last


# ── v1 forward: S updates after every UT loop ──
def _exit_pdf_from_gates(gate_list):
    """Geometric early-exit PDF from per-UT gate logits (same as OuroForCausalLM).

    gate_list[k]: [B, T, 1] pre-sigmoid logits.
    Returns stacked PDF [B, T, n_ut].
    """
    pdf_list = []
    remaining = torch.ones_like(gate_list[0].squeeze(-1))
    for idx, g in enumerate(gate_list):
        lam = torch.sigmoid(g.squeeze(-1))
        if idx < len(gate_list) - 1:
            p_i = lam * remaining
            remaining = remaining * (1.0 - lam)
        else:
            p_i = remaining
        pdf_list.append(p_i)
    return torch.stack(pdf_list, dim=2)


def turn_forward_v1(model, inp, film_layers, plaa_core, S, ablate_s=False,
                    state_log=None, inject_override=None, halt_gate=None,
                    use_weighted_exit=False, state_module=None, max_ut=None,
                    intervene=None, no_update=False, per_loop_logits=False):
    """Replicates OuroModel.forward's UT loop (use_cache=False path) with a
    per-loop state update: after UT step k, S <- GRU(S, h_k[last]); the FiLM
    for UT step k+1 sees the updated S. With ablate_s=True, zeros are injected
    at every loop (state evolution still tracked but never seen by FiLM).

    Optional halt_gate (HaltGate): adds S-conditioned bias to frozen
    early_exit_gate at each UT step (uses the S seen by FiLM at that loop).

    Sequence-state extension (S0 experiment, all optional, None-preserving):
      state_module : StateModule (VectorStateModule / SequenceStateModule).
                     When given, the modulation injected into FiLM is
                     state_module.to_modulation(state) and the state is
                     advanced with state_module.update(...) instead of
                     plaa_core(...); plaa_core is ignored in that case.
      max_ut       : run only the first max_ut UT loops and read logits from
                     that loop's post-norm hidden (eval-time depth scan).
      intervene    : callable state -> state applied right before
                     to_modulation (injection-only intervention; the state
                     chain still evolves from the true state).
      no_update    : skip the state update entirely (static/frozen state).
      per_loop_logits: if True, the non-gate return becomes
                     (logits, S, s_norms, per_loop_logits) with one logits
                     tensor per UT loop (early-readout curve).

    Returns
      if use_weighted_exit or halt_gate is not None:
        (logits_at_last_pos, S_final, per_loop_state_norms, aux)
        where aux = {exit_pdf_last, expected_exit_step, gate_means, per_step_logits optional}
      else (legacy):
        (logits_at_last_pos, S_final, per_loop_state_norms)
    """
    from transformers.masking_utils import create_causal_mask
    import inspect as _inspect

    m = model.model
    inputs_embeds = m.embed_tokens(inp["input_ids"])
    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
    # compat across transformers versions: 5.x uses "inputs_embeds";
    # 4.54.x uses "input_embeds" and requires "cache_position".
    _cm_params = set(_inspect.signature(create_causal_mask).parameters)
    _cm_kwargs = dict(config=m.config, attention_mask=inp["attention_mask"],
                      past_key_values=None, position_ids=position_ids)
    if "input_embeds" in _cm_params and "inputs_embeds" not in _cm_params:
        _cm_kwargs["input_embeds"] = inputs_embeds
        if "cache_position" in _cm_params:
            _cm_kwargs["cache_position"] = cache_position
    else:
        _cm_kwargs["inputs_embeds"] = inputs_embeds
    causal_mask = create_causal_mask(**_cm_kwargs)
    position_embeddings = m.rotary_emb(inputs_embeds, position_ids)
    lens = inp["attention_mask"].sum(dim=1) - 1
    last = int(lens[0].item())

    hidden = inputs_embeds
    s_norms = []
    gate_list = []
    hidden_list = []
    pll = [] if per_loop_logits else None
    need_gates = (halt_gate is not None) or use_weighted_exit
    if need_gates and (state_module is not None or max_ut is not None):
        raise NotImplementedError(
            "halt-gate / weighted-exit paths support only the vector state "
            "(state_module=None) at full UT depth")
    updater = state_module if state_module is not None else plaa_core
    ut_steps = m.total_ut_steps if max_ut is None else max_ut
    assert 1 <= ut_steps <= m.total_ut_steps
    for current_ut in range(ut_steps):
        # inject_override: {loop_idx: tensor} -- FiLM sees the override at that
        # loop while the state chain continues from the true state
        # (injection-only intervention, used by the swap causal test).
        if inject_override is not None and current_ut in inject_override:
            s_inj = inject_override[current_ut].to(S.device)
            s_for_gate = s_inj if state_module is None else \
                state_module.to_modulation(s_inj)
            set_state(film_layers, s_for_gate)
        elif state_module is None:
            # legacy vector path: inject the raw state (or literal zeros)
            s_for_gate = torch.zeros_like(S) if ablate_s else S
            set_state(film_layers, s_for_gate)
        elif ablate_s:
            # protocol-parity "without_S": literal zero modulation, bypassing
            # the sequence readout so the FiLM input matches the vector case
            s_for_gate = torch.zeros(S.shape[0], state_module.mod_dim,
                                     device=S.device, dtype=torch.bfloat16)
            set_state(film_layers, s_for_gate)
        else:
            z = intervene(S) if intervene is not None else S
            set_state(film_layers, state_module.to_modulation(z))
        for decoder_layer in m.layers[: m.config.num_hidden_layers]:
            hidden = decoder_layer(
                hidden, attention_mask=causal_mask, position_ids=position_ids,
                past_key_value=None, use_cache=False, cache_position=cache_position,
                position_embeddings=position_embeddings, current_ut=current_ut)
        hidden_normed = m.norm(hidden)
        if need_gates:
            if halt_gate is not None and getattr(halt_gate, "replace", False):
                # full replacement: ignore backbone gate logits
                b = halt_gate.bias(s_for_gate)  # [B,1,1]
                g = b.expand(hidden_normed.shape[0], hidden_normed.shape[1], 1)
            else:
                g = m.early_exit_gate(hidden_normed)  # [B, T, 1], frozen backbone
                if halt_gate is not None:
                    g = g + halt_gate.bias(s_for_gate)
            gate_list.append(g)
            hidden_list.append(hidden_normed)
        if not no_update:
            S = updater.update(S, hidden_normed[0, last].unsqueeze(0))
        s_norms.append(float(S.detach().float().norm()))
        if state_log is not None:
            state_log.append(S.detach().float().cpu().squeeze(0).clone())
        if pll is not None:
            pll.append(model.lm_head(hidden_normed[:, last])[0].detach().float().cpu())
        # Next UT step continues from the post-norm hidden, exactly as in the
        # original OuroModel.forward (which reassigns hidden_states = norm(...)).
        hidden = hidden_normed

    if not need_gates:
        logits = model.lm_head(hidden[:, last])[0]
        if pll is not None:
            return logits, S, s_norms, pll
        return logits, S, s_norms

    # Exit PDF + last-step / expected logits at the final non-pad token.
    exit_pdf = _exit_pdf_from_gates(gate_list)          # [B, T, n_ut]
    pdf_last = exit_pdf[0, last]                        # [n_ut]
    expected_exit = (pdf_last * torch.arange(
        pdf_last.numel(), device=pdf_last.device, dtype=pdf_last.dtype)).sum()

    last_logits = model.lm_head(hidden_list[-1][:, last])[0]
    if use_weighted_exit:
        expected_logits = None
        for k, h_k in enumerate(hidden_list):
            step_logits = model.lm_head(h_k[:, last])[0]
            w = pdf_last[k].to(step_logits.dtype)
            expected_logits = step_logits * w if expected_logits is None else expected_logits + step_logits * w
        logits = expected_logits
    else:
        logits = last_logits

    aux = {
        "exit_pdf_last": pdf_last,
        "expected_exit_step": expected_exit,
        "gate_means": [float(torch.sigmoid(g[0, last, 0]).float()) for g in gate_list],
        "last_logits": last_logits,
    }
    return logits, S, s_norms, aux
