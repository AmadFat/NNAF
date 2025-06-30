from nnaf_utils.pytype import *

from .pttype import *


class GatedMlp(nn.Module):
    def __init__(
        self,
        in_chs: int,
        hid_chs: int = None,
        out_chs: int = None,
        activation: Callable = F.silu,
        bias: bool = False,
        multiple_of: int = 128,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        out_chs = out_chs if out_chs is not None else in_chs
        hid_chs = hid_chs if hid_chs is not None else int(8 * in_chs / 3)
        hid_chs = (hid_chs + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_chs, 2 * hid_chs, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hid_chs, out_chs, bias=bias, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y


class Mlp(nn.Module):
    def __init__(
        self,
        in_chs: int,
        hid_chs: int = None,
        out_chs: int = None,
        bias: bool = True,
        activation: nn.Module = nn.ReLU,
        depth: int = 2,
        norm: nn.Module = None,
        linear: nn.Module = nn.Linear,
        dropout: float = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        from functools import partial

        factory_kwargs = {"device": device, "dtype": dtype}
        out_chs = out_chs if out_chs is not None else in_chs
        hid_chs = hid_chs if hid_chs is not None else in_chs
        dropout = partial(nn.Dropout, p=dropout) if dropout > 0 else None
        self.layers = [linear(in_chs, hid_chs if depth > 1 else out_chs, bias=bias, **factory_kwargs)]
        for _ in range(depth - 2):
            seq = [activation()]
            if norm is not None:
                seq.append(norm(hid_chs, **factory_kwargs))
            if dropout is not None:
                seq.append(dropout())
            seq.append(linear(hid_chs, hid_chs, bias=bias, **factory_kwargs))
            self.layers.append(nn.Sequential(*seq))
        if depth > 1:
            seq = [activation()]
            if norm is not None:
                seq.append(norm(hid_chs, **factory_kwargs))
            if dropout is not None:
                seq.append(dropout())
            seq.append(linear(hid_chs, out_chs, bias=bias, **factory_kwargs))
            self.layers.append(nn.Sequential(*seq))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x
