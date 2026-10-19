import torch
import torch.nn as nn
import torch.nn.functional as F

DM, DS = 2560, 256  # Qwen3-4B hidden dim, latent dim


class FiLMLayer(nn.Module):
    """Multiplicative FiLM modulation layer for PLAA.
    
    Injects into Qwen3-4B layers 16-28.
    Forward: hs = hs * (1 + alpha * tanh(W_l * S_t))
    """
    def __init__(self, bl):
        super().__init__()
        self.bl = bl
        self.scale_proj = nn.Linear(DS, DM, dtype=torch.bfloat16)
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        self.alpha = 0.1

    def forward(self, h, **kw):
        o = self.bl(h, **kw)
        hs = o[0] if isinstance(o, tuple) else o
        if self._s is not None:
            scale = torch.tanh(self.scale_proj(self._s).unsqueeze(1))
            hs = hs * (1 + self.alpha * scale)
        return (hs,) + o[1:] if isinstance(o, tuple) else hs


class PlaaCore(nn.Module):
    """PLAA latent state core.
    
    S_t in R^256 evolves via GRU. Initialized as learned param + noise.
    """
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(DS, DS, dtype=torch.bfloat16).cuda()
        self.init_s = nn.Parameter(
            torch.zeros(1, DS, dtype=torch.bfloat16, device="cuda"))

    def forward(self, S, h_proj):
        return self.gru(h_proj, S)

    def init_state(self, B):
        base = self.init_s.expand(B, -1)
        noise = torch.randn_like(base) * 0.01
        return (base + noise).contiguous()
