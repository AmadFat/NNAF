def warmup_cosine_decay(
    optimizer,
    max_epochs: int,
    warmup_epochs: int | float = 0,
    warmup_min: float = 0,
    warmup_max: float = 1,
    cosine_min: float = 0,
    cosine_max: float = 1,
    loader_size: int = 1,
):
    import math
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = int(warmup_epochs * loader_size)
    warmup_amplitude = warmup_max - warmup_min
    cosine_steps = max_epochs * loader_size - warmup_steps
    cosine_amplitude = 0.5 * (cosine_max - cosine_min)
    cosine_mean = 0.5 * (cosine_max + cosine_min)
    def cosine_decay_with_warmup(i):
        if warmup_steps and i < warmup_steps:
            return warmup_amplitude * i / warmup_steps + warmup_min
        else:
            i = i - warmup_steps
            return cosine_amplitude * math.cos(i * math.pi / cosine_steps) + cosine_mean
    return LambdaLR(optimizer, cosine_decay_with_warmup)

def warmup_multistep_decay(
    optimizer,
    decay_epochs: list[int | float],
    decays: float | list[float] = 1,
    warmup_epochs: int | float = 0,
    warmup_min: float = 0,
    warmup_max: float = 1,
    loader_size: int = 1,
):
    assert not isinstance(decays, list) or len(decays) == len(decay_epochs)
    from torch.optim.lr_scheduler import LambdaLR
    from bisect import bisect_left
    from functools import reduce
    warmup_steps = int(warmup_epochs * loader_size)
    warmup_amplitude = warmup_max - warmup_min
    decay_steps = [int(e * loader_size) for e in decay_epochs]
    def multistep_decay_with_warmup(i):
        if warmup_steps and i < warmup_steps:
            return warmup_amplitude * i / warmup_steps + warmup_min
        elif isinstance(decays, list):
            return reduce(lambda x, y: x * y, [1] + decays[:bisect_left(decay_steps, i)])
        else:
            return decays ** bisect_left(decay_steps, i)
    return LambdaLR(optimizer, multistep_decay_with_warmup)
