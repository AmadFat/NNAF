from nnaf_utils.pytype import *

from ..pttype import *


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
        return x.index_select(self.dim, torch.arange(start, end))


class ToDtype:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype

    def __call__(self, x: Any):
        x = torch.as_tensor(x)
        x = x.to(self.dtype)
        return x
