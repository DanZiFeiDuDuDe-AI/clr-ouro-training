import torch
import torch.nn as nn


class OracleLocalCore(nn.Module):
    """Canonical zero-init write cell with explicit oracle slot allocation."""

    def __init__(self, hidden_dim=2048, slots=4, dim=64, device=None, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slots = slots
        self.dim = dim
        self.content_read = nn.Linear(hidden_dim, dim)
        self.shared_gru = nn.GRUCell(dim, dim)
        if device is not None:
            self.to(device=device, dtype=dtype)

    def zero_state(self, batch_size, device=None, dtype=None):
        p = next(self.parameters())
        return torch.zeros(batch_size, self.slots, self.dim,
                           device=device or p.device, dtype=dtype or p.dtype)

    def write_to_slot(self, state, h, target_slot):
        z = self.content_read(h.to(device=state.device, dtype=state.dtype))
        candidate = self.shared_gru(z, state[:, target_slot, :])
        before = state[:, [j for j in range(self.slots) if j != target_slot], :]
        new_state = state.clone()
        new_state[:, target_slot, :] = candidate
        after = new_state[:, [j for j in range(self.slots) if j != target_slot], :]
        if not torch.equal(before, after):
            raise AssertionError("non-target slots changed")
        return new_state, z, candidate

    def forward_facts(self, facts, slot_indices=None, keep_states=False):
        b, n, _ = facts.shape
        state = self.zero_state(b, device=facts.device, dtype=self.content_read.weight.dtype)
        if slot_indices is None:
            slot_indices = list(range(n))
        states = [state] if keep_states else None
        for t in range(n):
            state, _, _ = self.write_to_slot(state, facts[:, t], int(slot_indices[t]))
            if keep_states:
                states.append(state)
        return state, states

    def parameter_report(self):
        return {
            "content_read": int(sum(p.numel() for p in self.content_read.parameters())),
            "shared_gru": int(sum(p.numel() for p in self.shared_gru.parameters())),
            "core_total": int(sum(p.numel() for p in self.parameters())),
            "has_init_s": False,
        }
