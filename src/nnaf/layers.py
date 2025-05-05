import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, p: float = 0):
        super().__init__()
        self.p = nn.Buffer(torch.tensor(p, dtype=torch.float))
    
    def forward(self, x: torch.Tensor):
        if abs(self.p) < 1e-6 or not self.training:
            return x
        keep = -self.p + 1
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        choice = x.new_empty(shape).bernoulli_(keep)
        if keep > 0:
            choice.div_(keep)
        return x * choice
    
    def extra_repr(self):
        return f"p={self.p.item():.1f}"
