# -*- coding: utf-8 -*-
"""
retention_ouro.py — Phase 0E minimal converted model (forward only).

Replaces the softmax mixing in selected OuroAttention layers with linear
retention (RetNet-style, RoPE preserved). Q/K/V/O projections are
exact-copied from the source attention; the only new trainable parameter is
per-head decay gamma (multi-scale init, Amendment 1). Runtime state is
(S, z) where z is a non-trainable normalization accumulator (Amendment 2).

Design constraints (P0_SUBSTRATE_DESIGN.md Amendments):
  1. multi-scale gamma init: gamma_h = sigmoid(theta_h), theta_h=logit(1-2^(-5-h))
  2. (S, z) runtime state; z non-trainable scale accumulator
  3. exact_copy = tensor copied exactly, NOT attention function preserved
  4. RoPE episode-global position counter across turns (C-persist); local per-turn (C-reset)
  5. retention state resets at each Ouro loop boundary
  7. parallel/recurrent equivalence test gate (this module exposes both paths)

NOT trained here. Forward only. No checkpoint migration in this file — that
is Phase 0F. Here we only build the converted module and verify forward.
"""
import math
import torch
import torch.nn as nn

# Reuse Ouro's RoPE application helpers from modeling_ouro
# (apply_rotary_pos_emb is a module-level function in modeling_ouro.py).
# We import lazily to avoid loading the full model here.

HEAD_DIM = 128
NUM_HEADS = 16  # Ouro MHA (num_attention_heads == num_key_value_heads)


def _rotate_half(x):
    """Standard GPT-NeoX/Llama-style RoPE half-rotation.
    Splits the last dim in two halves: (-x2, x1)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _multi_scale_gamma_init(num_heads: int) -> torch.Tensor:
    """Amendment 1 (original): gamma_h ~ 1 - 2^(-5-h), h=0..num_heads-1,
    parameterized as logit theta (gamma = sigmoid(theta)).
    NOTE: high heads saturate sigmoid -> gradient vanishes (see memory
    p0h-distill-gamma-vanish). Use gamma_mode='clamped_direct' for training."""
    h = torch.arange(num_heads, dtype=torch.float32)
    gamma = 1.0 - 2.0 ** (-(5.0 + h))      # h=0 -> 0.969, h=15 -> ~1.0
    gamma = gamma.clamp(1e-4, 1 - 1e-4)    # keep in open interval
    theta = torch.log(gamma / (1.0 - gamma))  # logit
    return theta  # shape [num_heads], sigmoid -> gamma


def _multi_scale_gamma_init_direct(num_heads: int) -> torch.Tensor:
    """Amendment 1 (technical revision): same multi-scale gamma values, but
    stored DIRECTLY as the parameter (gamma = clamp(theta, 1e-6, 1-1e-6)).
    Gradient no longer vanishes for high heads. Bounds wide enough that
    initial values are (almost) never clipped."""
    h = torch.arange(num_heads, dtype=torch.float32)
    gamma = 1.0 - 2.0 ** (-(5.0 + h))
    gamma = gamma.clamp(1e-6, 1 - 1e-6)
    return gamma


class RetentionAttention(nn.Module):
    """Drop-in replacement for OuroAttention.forward's mixing.

    Projections q/k/v/o_proj are EXACT copies (loaded externally, Phase 0F).
    Only the softmax(QK^T/√d)V mixing becomes retention. RoPE is preserved
    (applied to Q,K exactly as in OuroAttention).

    Forward returns (output, new_recurrent_state) so callers can persist or
    reset state across turns (Phase 0I reset/persist axis).

    Two compute paths are provided (Amendment 7 equivalence gate):
      - recurrent: token-by-token scan, maintains (S, z)
      - parallel:  chunkwise matrix form, mathematically equal
    """

    def __init__(self, layer_idx: int, num_heads: int = NUM_HEADS,
                 head_dim: int = HEAD_DIM, gamma_mode: str = "sigmoid_logit"):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.gamma_mode = gamma_mode
        # Class C new parameter (Amendment 1): per-head decay.
        # gamma_mode='clamped_direct' stores gamma values directly and avoids
        # sigmoid saturation (technical revision, see p0h-distill-gamma-vanish).
        if gamma_mode == "clamped_direct":
            self.gamma_logit = nn.Parameter(_multi_scale_gamma_init_direct(num_heads))
        else:
            self.gamma_logit = nn.Parameter(_multi_scale_gamma_init(num_heads))

    @property
    def gamma(self):
        if self.gamma_mode == "clamped_direct":
            return torch.clamp(self.gamma_logit, 1e-6, 1 - 1e-6)  # [num_heads]
        return torch.sigmoid(self.gamma_logit)  # [num_heads], in (0,1)

    def init_state(self, batch_size: int, device=None, dtype=torch.bfloat16):
        """Initial retention state: S=0, z=eps. Returns (S, z).
        S: [B,H,d_h,d_h]; z: [B,H,1,1] (4D to avoid 3D/4D broadcast bugs)."""
        device = device or self.gamma_logit.device
        S = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim,
                        device=device, dtype=dtype)
        z = torch.full((batch_size, self.num_heads, 1, 1), 1e-6,
                       device=device, dtype=dtype)
        return (S, z)

    def recurrent_forward(self, q, k, v, state=None):
        """q,k,v: [B, H, L, d_h] (already projected + RoPE'd).
        state: (S, z) or None. Returns (output [B,H,L,d_h], new_state).
        Token-by-token scan. O(L) memory."""
        B, H, L, d_h = q.shape
        gamma = self.gamma.to(q.dtype).view(1, H, 1, 1)       # [1,H,1,1]
        gamma_z = self.gamma.to(q.dtype).view(1, H, 1, 1)     # [1,H,1,1] (4D match)
        if state is None:
            state = self.init_state(B, device=q.device, dtype=q.dtype)
        S, z = state
        outs = []
        for n in range(L):
            kn = k[:, :, n:n+1, :]    # [B,H,1,d_h]
            vn = v[:, :, n:n+1, :]    # [B,H,1,d_h]
            qn = q[:, :, n:n+1, :]    # [B,H,1,d_h]
            # S_n = gamma * S_{n-1} + K_n^T V_n  (outer product per head)
            S = gamma * S + torch.matmul(kn.transpose(-1, -2), vn)  # [B,H,d_h,d_h]
            # z_n = gamma * z_{n-1} + ||K_n||^2  (4D: [B,H,1,1])
            z = gamma_z * z + (kn.pow(2).sum(dim=-1, keepdim=True))  # [B,H,1,1]
            # O_n = (Q_n . S_n) / sqrt(z_n)
            on = torch.matmul(qn, S) / torch.sqrt(z + 1e-6)   # [B,H,1,d_h]
            outs.append(on)
        out = torch.cat(outs, dim=2)  # [B,H,L,d_h]
        return out, (S, z)

    def parallel_forward(self, q, k, v, state=None):
        """Parallel (chunkwise matrix) retention. Mathematically equal to
        recurrent (Amendment 7 gate). O(L^2) but useful for verification.
        Does NOT consume/produce a running state across calls — used to
        check the recurrent path is correct on a single chunk."""
        B, H, L, d_h = q.shape
        gamma = self.gamma.to(q.dtype).view(1, H, 1, 1)
        # decay matrix D[i,j] = gamma^{i-j} for i>=j, else 0 (causal)
        idx = torch.arange(L, device=q.device)
        rel = idx[:, None] - idx[None, :]                      # [L,L], rel[i,j]=i-j
        # mask: keep only j<=i (rel>=0); set rest to 0 contribution via mask
        causal_mask = (rel >= 0)                                # [L,L] bool
        g = self.gamma.to(q.dtype)                              # [H]
        D = g.view(H, 1, 1) ** rel.clamp(min=0).view(1, L, L)    # [H,L,L], gamma^{rel}
        D = D * causal_mask.view(1, L, L).to(D.dtype)            # zero out j>i
        # retention: (Q K^T * D) V — normalized by z (no softmax scaling)
        qk = torch.matmul(q, k.transpose(-1, -2))                  # [B,H,L,L]
        qk = qk * D.unsqueeze(0)                                  # decay
        # z normalization: cumulative ||K||^2 with decay
        kk = k.pow(2).sum(dim=-1)                                 # [B,H,L]
        z = (kk.unsqueeze(2) * D.unsqueeze(0)).sum(dim=-1)       # [B,H,L]
        attn = qk / torch.sqrt(z.unsqueeze(-1) + 1e-6)            # [B,H,L,L]
        out = torch.matmul(attn, v)                               # [B,H,L,d_h]
        return out, state  # state unchanged (parallel is stateless for this chunk)

    def forward(self, hidden_states, q_proj, k_proj, v_proj, o_proj,
                position_embeddings=None, state=None, mode="recurrent"):
        """hidden_states: [B, L, hidden]. Projections passed in (exact-copied
        weights from source OuroAttention, loaded in Phase 0F).
        Returns (output [B, L, hidden], new_state)."""
        B, L, _ = hidden_states.shape
        q = q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # RoPE (preserve Ouro's rotary; position_embeddings = (cos, sin))
        # Inlined to avoid hard-importing modeling_ouro (it lives in HF cache).
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # cos/sin: [B, L, d_h] or [1, L, d_h]; broadcast to [B,1,L,d_h]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q = (q * cos) + (_rotate_half(q) * sin)
            k = (k * cos) + (_rotate_half(k) * sin)
        if mode == "parallel":
            out, state = self.parallel_forward(q, k, v, state)
        else:
            out, state = self.recurrent_forward(q, k, v, state)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)    # [B,L,hidden]
        out = o_proj(out)
        return out, state


def convert_ouro_to_retention(model, replaced_layers, gamma_mode="sigmoid_logit"):
    """Replace OuroAttention at given layer indices with RetentionAttention.
    Q/K/V/O projection weights are EXACT-copied from the source attention
    (they live in the same module — we keep q/k/v/o_proj, swap the forward).

    gamma_mode: 'sigmoid_logit' (original, saturated for high heads) or
    'clamped_direct' (technical revision, trainable at all heads).

    model: loaded OuroForCausalLM (4-bit frozen). Returns list of
    (retention_module, layer_idx, saved_projections) for manifest/verification.
    This does NOT save a checkpoint — Phase 0F does, with the manifest."""
    converted = []
    layers = model.model.layers
    for idx in replaced_layers:
        layer = layers[idx]
        src_attn = layer.self_attn
        # Build retention module
        ret = RetentionAttention(layer_idx=idx,
                                  num_heads=src_attn.num_key_value_groups
                                             and src_attn.config.num_attention_heads
                                             or NUM_HEADS,
                                  head_dim=src_attn.head_dim,
                                  gamma_mode=gamma_mode)
        # Reuse the EXACT projection weights from source (exact_copy).
        # We keep the source q/k/v/o_proj Linear modules in place and have
        # RetentionAttention.forward take them as args — so weights never
        # move, guaranteeing max_abs_diff==0 trivially.
        # Store projection refs on the retention module for convenience.
        ret.q_proj = src_attn.q_proj
        ret.k_proj = src_attn.k_proj
        ret.v_proj = src_attn.v_proj
        ret.o_proj = src_attn.o_proj
        ret.config = src_attn.config
        # Move retention params (gamma_logit) to the source attention's
        # device (cuda:0). Keep bf16 dtype — NF4-quantized source weights are
        # stored as uint8, but gamma is a float parameter.
        ret = ret.to(next(src_attn.parameters()).device).to(torch.bfloat16)
        if gamma_mode == "clamped_direct":
            with torch.no_grad():
                ret.gamma_logit = nn.Parameter(ret.gamma_logit.detach().to(torch.float32))
        # attention_type lives on OuroDecoderLayer, not OuroAttention;
        # OuroModel.forward reads decoder_layer.attention_type (intact).
        ret._src_attn = src_attn  # keep reference for manifest/verification
        converted.append((ret, idx))
        layer.self_attn = _RetentionAttentionAdapter(ret, src_attn)
    return converted


class _RetentionAttentionAdapter(nn.Module):
    """Adapter so OuroDecoderLayer.forward (which calls self.self_attn(
    hidden_states=..., attention_mask=..., position_ids=...,
    position_embeddings=...)) routes through retention, while preserving
    the q/k/v/o_proj weights exactly on the original attention module."""

    def __init__(self, ret, src_attn):
        super().__init__()
        self.ret = ret
        self.src_attn = src_attn
        # attention_type lives on OuroDecoderLayer (not OuroAttention);
        # OuroModel.forward reads decoder_layer.attention_type, still intact.
        self.layer_idx = src_attn.layer_idx
        self.config = src_attn.config
        self.head_dim = src_attn.head_dim
        self.num_key_value_groups = src_attn.num_key_value_groups

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False, cache_position=None,
                position_embeddings=None, **kwargs):
        # Drop attention_mask (retention is causal by construction of the
        # decay matrix / scan). RoPE position_embeddings passed through.
        out, state = self.ret.forward(
            hidden_states,
            q_proj=self.ret.q_proj, k_proj=self.ret.k_proj,
            v_proj=self.ret.v_proj, o_proj=self.ret.o_proj,
            position_embeddings=position_embeddings,
            state=getattr(self, "_state", None),
            mode=getattr(self, "_mode", "recurrent"))
        self._last_state = state  # caller can retrieve for persist/reset
        # OuroDecoderLayer expects (output, None) tuple from self_attn
        return out, None
