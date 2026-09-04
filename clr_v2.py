# -*- coding: utf-8 -*-
"""
CLR-v2 — backbone-decoupled latent-state recurrent module.

Architecture (contrast with clr_ouro.py / v1):
    Backbone hidden  (2048, Ouro-specific)
        |  ReadAdapter  (2048 -> latent_slots*latent_dim)   <-- backbone-coupled
        v
    Latent State  [B, latent_slots, latent_dim]   (FIXED size, backbone-agnostic)
        |  CLR Recurrent Core  (per-slot GRU + slot mixing, latent_steps recursive)
        v
    Latent State  [B, latent_slots, latent_dim]
        |  WriteAdapter  (latent_slots*latent_dim -> 2048, FiLM scale)  <-- backbone-coupled
        v
    Backbone hidden modulation (h * (1 + alpha*tanh(scale)))

Design goals (CLR-v2 spec):
  * Latent state size fixed and decoupled from backbone hidden size.
    Swapping backbone only requires retraining Read/Write Adapters; the Core's
    parameter shapes never change.
  * Explicit latent update steps: latent_steps in {0,1,2,4,8,16} (NOT baked in).
  * Cross-turn state S_t preserved; supports reset / persist / detach.
  * Intervention-ready: state swap / corruption / interpolation operate on the
    [B, L, D] tensor before WriteAdapter (see eval helpers in _train_clr_v2.py).

Interfaces intentionally mirror clr_ouro.py's PlaaCore so eval/training loops
stay familiar:
    init_state(B, device) -> S
    update(S, h_last, latent_steps) -> S_new   (called per UT loop or per turn)
    to_modulation(S) -> scale [B, DM]          (fed to FiLMv2 wrapper)
"""
import os
import torch
import torch.nn as nn

# Ouro backbone facts (v1 parity)
DM = 2048          # Ouro hidden size
ALPHA = float(os.environ.get("CLR2_ALPHA", "0.1"))   # FiLM multiplier scale
WRITE_INIT_STD = float(os.environ.get("CLR2_WRITE_INIT", "0.01"))
READ_INIT_STD = float(os.environ.get("CLR2_READ_INIT", "0.02"))

# CLR-v2 latent geometry (FIXED across backbones)
LATENT_SLOTS = int(os.environ.get("CLR2_SLOTS", "4"))
LATENT_DIM = int(os.environ.get("CLR2_LATENT_DIM", "64"))
LATENT_STEPS = int(os.environ.get("CLR2_LATENT_STEPS", "1"))


class ReadAdapter(nn.Module):
    """Backbone-coupled: hidden [B, DM] -> latent [B, L, D]."""

    def __init__(self, d_model=DM, n_slots=LATENT_SLOTS, d_latent=LATENT_DIM,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.n_slots = n_slots
        self.d_latent = d_latent
        self.proj = nn.Linear(d_model, n_slots * d_latent,
                              dtype=dtype, device=device)
        nn.init.normal_(self.proj.weight, mean=0.0, std=READ_INIT_STD)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h_last):
        # h_last: [B, DM]
        z = self.proj(h_last.to(torch.bfloat16))          # [B, L*D]
        return z.view(z.shape[0], self.n_slots, self.d_latent)


class CLRCore(nn.Module):
    """Backbone-agnostic recurrent core operating on fixed [B, L, D] states.

    Per latent step k:
        step 0: x = z            (external read of current hidden)
        step k>0: x = S          (self-driven depth recursion)
        S = GRU_cell(x, S) + slot-mixing
    latent_steps=1 reproduces the v1 PlaaCore semantics
    (S_new = GRU(read(h_last), S_t)) while staying in slot space.
    latent_steps=0 freezes S (no update) — used for depth-scaling scans.
    """

    def __init__(self, n_slots=LATENT_SLOTS, d_latent=LATENT_DIM,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.n_slots = n_slots
        self.d_latent = d_latent
        self.gru = nn.GRUCell(d_latent, d_latent, dtype=dtype, device=device)
        # lightweight slot communication: linear mix over the slot axis
        self.slot_mix = nn.Linear(d_latent, d_latent, dtype=dtype, device=device)
        self.slot_gate = nn.Linear(d_latent, d_latent, dtype=dtype, device=device)
        nn.init.zeros_(self.slot_mix.weight)
        nn.init.zeros_(self.slot_mix.bias)
        nn.init.zeros_(self.slot_gate.weight)
        nn.init.zeros_(self.slot_gate.bias)

    def forward(self, S, z, latent_steps=LATENT_STEPS):
        # S, z: [B, L, D]
        if latent_steps <= 0:
            return S
        B, L, D = S.shape
        s = S
        for k in range(int(latent_steps)):
            x = z if k == 0 else s
            s_flat = s.reshape(B * L, D)
            x_flat = x.reshape(B * L, D)
            h = self.gru(x_flat, s_flat)                  # [B*L, D]
            h = h.view(B, L, D)
            # slot mixing with a learned gate (residual, gated)
            mix = torch.tanh(self.slot_mix(h))            # [B, L, D]
            gate = torch.sigmoid(self.slot_gate(h))
            h = h + gate * mix
            s = h
        return s


class WriteAdapter(nn.Module):
    """Backbone-coupled: latent [B, L, D] -> FiLM scale [B, DM].

    Scale is injected as h * (1 + ALPHA*tanh(scale)), identical form to the
    v1 scale_proj so the modulation semantics stay comparable.
    """

    def __init__(self, d_model=DM, n_slots=LATENT_SLOTS, d_latent=LATENT_DIM,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.proj = nn.Linear(n_slots * d_latent, d_model, dtype=dtype, device=device)
        nn.init.normal_(self.proj.weight, mean=0.0, std=WRITE_INIT_STD)
        nn.init.zeros_(self.proj.bias)

    def forward(self, S):
        # S: [B, L, D] -> scale [B, DM]
        scale = ALPHA * torch.tanh(self.proj(S.reshape(S.shape[0], -1)))
        return scale


class FiLMv2(nn.Module):
    """Multiplicative FiLM wrapper for OuroDecoderLayer, same contract as the
    v1 FiLMWrapper (attention_type passthrough + bare-tensor assert) but the
    scale is supplied externally by the WriteAdapter instead of an internal
    scale_proj. This keeps CLR-v2's modulation path backbone-agnostic."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self._scale = None
        # OuroModel.forward reads this attribute off each layer object:
        self.attention_type = layer.attention_type

    def forward(self, h, *args, **kwargs):
        if self._scale is not None:
            s = self._scale
            if s.shape[0] != h.shape[0]:
                s = s.expand(h.shape[0], -1)
            scale = s.to(h.dtype)
            h = h * (1 + (scale.unsqueeze(1) if h.dim() == 3 else scale))
        o = self.layer(h, *args, **kwargs)
        assert not isinstance(o, (tuple, list)), \
            "OuroDecoderLayer returned a tuple; FiLMv2 needs the bare-tensor path"
        return o


# ── v1-parity FiLM layer indices (24-layer Ouro stack) ──
FILM_LAYERS = [3, 7, 11, 15, 19, 23]


class CLRv2(nn.Module):
    """Top-level CLR-v2 module: ReadAdapter + CLRCore + WriteAdapter.

    S_t lives here; callers drive it with:
        S = module.init_state(B, device)
        ... backbone forward with module.to_modulation(S) ...
        S = module.update(S, h_last, latent_steps)   # detach handled by caller
    """

    def __init__(self, d_model=DM, n_slots=LATENT_SLOTS, d_latent=LATENT_DIM,
                 latent_steps=LATENT_STEPS):
        super().__init__()
        self.n_slots = n_slots
        self.d_latent = d_latent
        self.latent_steps = latent_steps
        self.read = ReadAdapter(d_model, n_slots, d_latent)
        self.core = CLRCore(n_slots, d_latent)
        self.write = WriteAdapter(d_model, n_slots, d_latent)
        self.init_s = nn.Parameter(torch.zeros(1, n_slots, d_latent,
                                               dtype=torch.bfloat16))

    def init_state(self, B, device, noise=0.01):
        base = self.init_s.expand(B, -1, -1)
        return (base + torch.randn_like(base) * noise).contiguous().to(device)

    def zero_state(self, B, device):
        return torch.zeros(B, self.n_slots, self.d_latent,
                           dtype=torch.bfloat16, device=device)

    def update(self, S, h_last, latent_steps=None):
        """S_{t+1} = CLRCore(ReadAdapter(h_last), S_t)."""
        if latent_steps is None:
            latent_steps = self.latent_steps
        z = self.read(h_last)
        return self.core(S, z, latent_steps)

    def to_modulation(self, S):
        """Scale [B, DM] for FiLMv2 injection."""
        return self.write(S)

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters())


def inject_film_v2(model):
    """Replace FILM_LAYERS of the (shared) stack with FiLMv2 wrappers."""
    layers = model.model.layers
    film_layers = []
    for idx in FILM_LAYERS:
        w = FiLMv2(layers[idx])
        layers[idx] = w
        film_layers.append(w)
    return film_layers


def set_state(film_layers, scale):
    for l in film_layers:
        l._scale = scale


# ── v1-parity UT-loop forward with per-loop latent update ──
def turn_forward_v2(model, inp, film_layers, clrv2, S, latent_steps=None,
                    ablate_s=False, no_update=False, max_ut=None,
                    per_loop_logits=False):
    """Replicates OuroModel.forward's UT loop (use_cache=False) with a
    per-loop CLR-v2 update: after UT step k, S <- CLRCore(Read(h_last), S);
    the FiLM of UT step k+1 sees the updated state (scale from WriteAdapter).

    ablate_s : inject literal zero scale at every loop (protocol-parity "no S").
    no_update: keep S frozen (never call update) — for static-state scans.
    Returns (logits_at_last_pos, S_final, per_loop_state_norms[, per_loop_logits]).
    """
    from transformers.masking_utils import create_causal_mask
    import inspect as _inspect

    m = model.model
    inputs_embeds = m.embed_tokens(inp["input_ids"])
    cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
    position_ids = cache_position.unsqueeze(0)
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
    pll = [] if per_loop_logits else None
    ut_steps = m.total_ut_steps if max_ut is None else max_ut
    assert 1 <= ut_steps <= m.total_ut_steps
    for current_ut in range(ut_steps):
        if ablate_s:
            scale = torch.zeros(S.shape[0], DM, device=S.device, dtype=torch.bfloat16)
        else:
            scale = clrv2.to_modulation(S)
        set_state(film_layers, scale)
        for decoder_layer in m.layers[: m.config.num_hidden_layers]:
            hidden = decoder_layer(
                hidden, attention_mask=causal_mask, position_ids=position_ids,
                past_key_value=None, use_cache=False, cache_position=cache_position,
                position_embeddings=position_embeddings, current_ut=current_ut)
        hidden_normed = m.norm(hidden)
        if not no_update:
            S = clrv2.update(S, hidden_normed[0, last].unsqueeze(0), latent_steps)
        s_norms.append(float(S.detach().float().norm()))
        if pll is not None:
            pll.append(model.lm_head(hidden_normed[:, last])[0].detach().float().cpu())
        hidden = hidden_normed

    logits = model.lm_head(hidden[:, last])[0]
    if pll is not None:
        return logits, S, s_norms, pll
    return logits, S, s_norms


def load_ouro(model_path=None, four_bit=True):
    """Load frozen Ouro-1.4B (+tokenizer). Same contract as clr_ouro.load_ouro."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if model_path is None:
        model_path = os.environ.get(
            "OURO_MODEL_PATH", "D:/training/Ouro-1.4B")
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
