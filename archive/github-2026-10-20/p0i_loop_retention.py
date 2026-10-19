# -*- coding: utf-8 -*-
"""
P0I core: loop-boundary-reset retention adapter + turn-state manager.

Phase 0I semantics (网友 spec):
  token recurrence      : retention 正常递归 (retention_ouro.RetentionAttention)
  Ouro loop boundary    : RESET   — current_ut>0 时强制从零开始 (不混入 cross-loop persistence)
  turn boundary         : C-reset → _state=None ; C-persist → _state=prev turn end
  episode boundary      : 始终 RESET (新 trajectory 前清空)
  position policy       : C-persist 用跨 turn 连续 position_ids ; C-reset 每 turn 从 0

We do NOT modify 网友's retention_ouro.py; we subclass the adapter so the
original file stays byte-identical.
"""
import os
import sys
sys.path.insert(0, "D:/training")
os.environ["OURO_MODEL_PATH"] = "D:/training/Ouro-1.4B"

import torch
from retention_ouro import _RetentionAttentionAdapter

REPLACED = [3, 7, 11, 15, 19, 23]


class P0ILoopRetentionAdapter(_RetentionAttentionAdapter):
    """Adapter with explicit Ouro-loop-boundary reset.

    - current_ut == 0: uses externally injected self._state (turn-start state).
    - current_ut  > 0: forces state=None (loop boundary reset).
    After the ut==0 forward, records self._turn_state = self._last_state
    (the turn-end state to be passed to the next turn under C-persist).
    """

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False, cache_position=None,
                position_embeddings=None, **kwargs):
        ut = kwargs.get("current_ut", 0)
        if ut > 0:
            # Ouro loop boundary: reset, no cross-loop persistence.
            self._state = None
        out, _ = super().forward(
            hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            use_cache=use_cache, cache_position=cache_position,
            position_embeddings=position_embeddings, **kwargs)
        if ut == 0:
            # turn-end state = state after the FIRST (ut==0) loop only.
            self._turn_state = self._last_state
        return out, None


def convert_p0i(model, replaced_layers=REPLACED, gamma_mode="sigmoid_logit"):
    """convert_ouro_to_retention + wrap each adapter in P0ILoopRetentionAdapter.

    Returns (adapters, rets):
      adapters: list of P0ILoopRetentionAdapter (one per replaced layer, aligned to REPLACED order)
      rets:     list of RetentionAttention
    """
    from retention_ouro import convert_ouro_to_retention
    converted = convert_ouro_to_retention(model, replaced_layers, gamma_mode=gamma_mode)
    adapters, rets = [], []
    for ret, idx in converted:
        layer = model.model.layers[idx]
        src_attn = layer.self_attn.src_attn
        adapter = P0ILoopRetentionAdapter(ret, src_attn)
        layer.self_attn = adapter
        adapters.append(adapter)
        rets.append(ret)
    return adapters, rets


def load_m2_canonical(ckpt_path="D:/training/checkpoints/ouro_retention_m2_d3/best_model.pt"):
    """Load frozen Ouro + retention from M2_CANONICAL (d3), freeze gamma,
    unfreeze post-norm weights. Returns (model, tok, adapters, rets)."""
    from clr_ouro import load_ouro
    model, tok = load_ouro()
    adapters, rets = convert_p0i(model, gamma_mode="sigmoid_logit")
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    for ret in rets:
        ret.gamma_logit.data.copy_(ck["retention_gamma"][str(ret.layer_idx)])
        ret.gamma_logit.requires_grad_(False)
    pnw = ck.get("post_norm_weights")
    assert pnw is not None, "M2_CANONICAL checkpoint must carry post_norm_weights"
    post_norm_params = []
    for li in REPLACED:
        layer = model.model.layers[li]
        for nm in ("input_layernorm_2", "post_attention_layernorm_2"):
            w = getattr(layer, nm).weight
            w.data.copy_(pnw[str(li)][nm].to("cuda"))
            w.requires_grad_(True)
            post_norm_params.append(w)
    return model, tok, adapters, rets, post_norm_params


# ── turn-state management ──
def reset_turn_state(adapters):
    """C-reset / episode boundary: each adapter starts from zero."""
    for a in adapters:
        a._state = None
        a._turn_state = None


def set_persist_state(adapters):
    """C-persist turn boundary: carry previous turn-end state."""
    for a in adapters:
        a._state = getattr(a, "_turn_state", None)


def get_turn_end_states(adapters):
    return [getattr(a, "_turn_state", None) for a in adapters]


def clear_all_state(adapters):
    for a in adapters:
        a._state = None
        a._turn_state = None
        a._last_state = None
