import math

import torch
import torch.nn as nn


class RetrievalController(nn.Module):
    """Minimal query-to-slot dot-product controller."""

    def __init__(self, query_dim=2048, slot_dim=64, proj_dim=64):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, proj_dim)
        self.slot_proj = nn.Linear(slot_dim, proj_dim)
        self.scale = math.sqrt(float(proj_dim))

    def forward(self, query_h, state, n_valid):
        q = self.query_proj(query_h.float())
        s = self.slot_proj(state.float())
        logits = (s * q[:, None, :]).sum(-1) / self.scale
        if n_valid < logits.shape[1]:
            logits = logits.clone()
            logits[:, n_valid:] = -torch.inf
        return logits


class AllSlotsVisibleBaseline(nn.Module):
    """Separate answer-only baseline; not part of the selective controller."""

    def __init__(self, query_dim=2048, state_dim=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(query_dim + state_dim, hidden), nn.ReLU(), nn.Linear(hidden, 64))

    def forward(self, query_h, state):
        return self.net(torch.cat([query_h.float(), state.flatten(1).float()], dim=1))
