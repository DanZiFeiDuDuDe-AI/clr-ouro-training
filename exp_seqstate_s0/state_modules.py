# -*- coding: utf-8 -*-
"""
exp_seqstate_s0 / state_modules.py — StateModule abstraction (S0).

The whole S0 experiment only needs three operations from the latent state:

    class StateModule(nn.Module):
        def init_state(self, B, device) -> state       # [B, ...] initial latent
        def update(self, state, h_last) -> state       # h_last: [B, DM] post-norm
                                                       # hidden at final token
        def to_modulation(self, state) -> [B, 64]      # pooled FiLM modulation

Capacity contract: every state carries exactly DS=64 scalars.
  - Vector:   state = [B, 64]        (GRU update; identity modulation)
  - Sequence: state = [B, K, d]      (K*d == 64; tiny-transformer update;
                                      flatten+linear modulation by default)

Design decisions (rationale in README.md):
  * read_proj projects h_last to 64 scalars (NOT to d) and reshapes into K
    context tokens. Upstream summary bandwidth is therefore identical across
    every K/d config and identical to the vector baseline; only the latent
    layout and the (tiny) updater differ.
  * pool bias is zero-initialised so "zero slots" maps to near-zero
    modulation at init; the ablate_s protocol injects literal zeros in
    modulation space (bypassing W_p), byte-parity with the vector protocol.
  * learned position embeddings (init N(0, 0.02)) give slots distinct
    identities so K x d cannot collapse into an unordered bag by symmetry.
  * the updater is a single-head pre-norm transformer block (no dropout,
    fully deterministic); attention is bidirectional over context+slots.
"""
import torch
import torch.nn as nn

from clr_ouro import DM, DS, PlaaCore


class StateModule(nn.Module):
    """Interface only. mod_dim = width of the FiLM modulation vector."""
    mod_dim = DS

    def init_state(self, B, device):
        raise NotImplementedError

    def update(self, state, h_last):
        raise NotImplementedError

    def to_modulation(self, state):
        raise NotImplementedError


class VectorStateModule(PlaaCore):
    """Flat GRU latent state — the v0/v1 baseline, exposed through StateModule.

    Numerics are identical to PlaaCore (init_s + 0.01 noise, GRUCell read of
    h_last, identity modulation), and the state_dict keys match PlaaCore's,
    so existing v1 checkpoints load into this class unchanged.
    """
    mod_dim = DS


class _TinyTransformerBlock(nn.Module):
    """Single-head pre-norm transformer block (no dropout, deterministic)."""

    def __init__(self, d, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d, dtype=torch.bfloat16, device="cuda")
        self.wq = nn.Linear(d, d, dtype=torch.bfloat16, device="cuda")
        self.wk = nn.Linear(d, d, dtype=torch.bfloat16, device="cuda")
        self.wv = nn.Linear(d, d, dtype=torch.bfloat16, device="cuda")
        self.wo = nn.Linear(d, d, dtype=torch.bfloat16, device="cuda")
        self.ln2 = nn.LayerNorm(d, dtype=torch.bfloat16, device="cuda")
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d, dtype=torch.bfloat16, device="cuda"),
            nn.GELU(),
            nn.Linear(ff_mult * d, d, dtype=torch.bfloat16, device="cuda"),
        )
        self.scale = d ** -0.5

    def forward(self, x):
        h = self.ln1(x)
        q, k, v = self.wq(h), self.wk(h), self.wv(h)
        att = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        x = x + self.wo(att @ v)
        x = x + self.ff(self.ln2(x))
        return x


class SequenceStateModule(StateModule):
    """Fixed-length latent sequence Z in [B, K, d] with K*d == 64.

    update:  Z_{t+1} = F_theta(Z_t, P(h_t))  -- K summary tokens (reshaped
             64-dim read) prepended to K slots, one tiny transformer block.
    to_modulation: default flatten + learned linear W_p: vec(Z) -> [B, 64];
              optional attention pooling (pool="attention", non-default).
    """

    def __init__(self, n_slots=4, d_slot=16, n_layers=1, pool="flatten",
                 d_model=DM):
        super().__init__()
        assert n_slots * d_slot == DS, \
            f"K*d must equal latent capacity {DS}, got {n_slots}x{d_slot}"
        assert pool in ("flatten", "attention"), pool
        self.n_slots = n_slots
        self.d_slot = d_slot
        self.n_layers = n_layers
        self.pool = pool

        # summary read: h_last [B, DM] -> 64 scalars -> K context tokens of d
        self.read_proj = nn.Linear(d_model, DS, dtype=torch.bfloat16,
                                   device="cuda")
        self.init_z = nn.Parameter(
            torch.zeros(1, n_slots, d_slot, dtype=torch.bfloat16,
                       device="cuda"))
        seq_len = 2 * n_slots            # K context tokens + K slots
        self.pos_emb = nn.Parameter(
            torch.zeros(seq_len, d_slot, dtype=torch.bfloat16,
                       device="cuda"))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList(
            [_TinyTransformerBlock(d_slot) for _ in range(n_layers)])

        if pool == "flatten":
            self.pool_proj = nn.Linear(DS, DS, dtype=torch.bfloat16,
                                       device="cuda")
            nn.init.zeros_(self.pool_proj.bias)
            nn.init.normal_(self.pool_proj.weight, std=0.01)
        else:
            self.pool_query = nn.Parameter(
                torch.zeros(1, 1, d_slot, dtype=torch.bfloat16,
                           device="cuda"))
            self.pool_proj = nn.Linear(d_slot, DS, dtype=torch.bfloat16,
                                       device="cuda")
            nn.init.zeros_(self.pool_proj.bias)

    def init_state(self, B, device):
        base = self.init_z.expand(B, -1, -1)
        return (base + torch.randn_like(base) * 0.01).contiguous().to(device)

    def update(self, Z, h_last):
        ctx = self.read_proj(h_last.to(torch.bfloat16))               # [B, 64]
        ctx = ctx.reshape(ctx.shape[0], self.n_slots, self.d_slot)    # [B, K, d]
        toks = torch.cat([ctx, Z.to(torch.bfloat16)], dim=1)          # [B, 2K, d]
        toks = toks + self.pos_emb
        for blk in self.blocks:
            toks = blk(toks)
        return toks[:, self.n_slots:]                                 # slots only

    def to_modulation(self, Z):
        Z = Z.to(torch.bfloat16)
        B = Z.shape[0]
        if self.pool == "flatten":
            return self.pool_proj(Z.reshape(B, -1))                   # [B, 64]
        w = torch.softmax(
            (Z @ self.pool_query.reshape(self.d_slot, 1)) * (self.d_slot ** -0.5),
            dim=1)                                                    # [B, K, 1]
        pooled = (w * Z).sum(dim=1)                                   # [B, d]
        return self.pool_proj(pooled)                                 # [B, 64]
