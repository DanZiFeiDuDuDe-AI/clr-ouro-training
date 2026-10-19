import torch
import torch.nn as nn


class OracleAllocationCore(nn.Module):
    """Fresh 4x64 storage core with externally specified slot allocation."""
    def __init__(self, hidden_dim=2048, slots=4, dim=64, device=None, dtype=None):
        super().__init__()
        self.hidden_dim = hidden_dim; self.slots = slots; self.dim = dim
        self.content_read = nn.Linear(hidden_dim, dim)
        self.shared_gru = nn.GRUCell(dim, dim)
        self.init_s = nn.Parameter(torch.zeros(1, slots, dim))
        if device is not None:
            self.to(device=device, dtype=dtype or torch.float32)

    def init_state(self, batch_size, device=None, dtype=None):
        s = self.init_s.expand(batch_size, -1, -1)
        if device is not None: s = s.to(device=device)
        if dtype is not None: s = s.to(dtype=dtype)
        return s.clone()

    def write_to_slot(self, state, h, target_slot):
        z = self.content_read(h.to(device=state.device, dtype=state.dtype))
        old = state[:, target_slot, :]
        candidate = self.shared_gru(z, old)
        new_state = state.clone()
        new_state[:, target_slot, :] = candidate
        # This assertion is part of the oracle-allocation contract.
        if not torch.equal(new_state[:, [j for j in range(self.slots) if j != target_slot]],
                           state[:, [j for j in range(self.slots) if j != target_slot]]):
            raise AssertionError("non-target slots changed")
        return new_state

    def forward_facts(self, facts, slot_indices=None, keep_states=False):
        b, n, _ = facts.shape
        state = self.init_state(b, facts.device, self.init_s.dtype)
        states = [state] if keep_states else None
        if slot_indices is None: slot_indices = [i for i in range(n)]
        for t in range(n):
            state = self.write_to_slot(state, facts[:, t], int(slot_indices[t]))
            if keep_states: states.append(state)
        return state, states

    def parameter_report(self):
        return {
            "content_read": int(sum(p.numel() for p in self.content_read.parameters())),
            "shared_gru": int(sum(p.numel() for p in self.shared_gru.parameters())),
            "init_s": int(self.init_s.numel()),
            "core_total": int(sum(p.numel() for p in self.parameters())),
        }
