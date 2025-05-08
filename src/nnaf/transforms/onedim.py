import torch
from typing import Any

class RandomSlice:
    def __init__(
        self,
        dim: int = 0,
        min_len: int = None,
        max_len: int = None,
    ):
        self.dim = dim
        self.min_len = min_len
        self.max_len = max_len
    
    def __call__(self, x: torch.Tensor):
        total_len = x.shape[self.dim]
        min_len = self.min_len if self.min_len is not None else 1
        max_len = self.max_len if self.max_len is not None else total_len
        length = torch.randint(min_len, max_len + 1, (1,)).item()
        start = torch.randint(0, total_len - length, (1,)).item()
        end = start + length
        _slice = torch.arange(start, end)
        _slice = x.index_select(self.dim, _slice)
        return _slice

class ToDtype:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
    
    def __call__(self, x: Any):
        x = torch.as_tensor(x)
        x = x.to(self.dtype)
        return x
