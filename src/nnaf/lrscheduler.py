from torch.optim.lr_scheduler import LambdaLR
import math

def get_cosine_decay_with_warmup(
    optimizer,
    max_epochs: int,
    warmup_epochs: int = 0,
    warmup_min: float = 0,
    warmup_max: float = 0,
    cosine_min: float = 0,
    cosine_max: float = 0,
    loader_size: int = 0,
):
    warmup_steps = warmup_epochs * loader_size
    warmup_amplitude = warmup_max - warmup_min
    cosine_steps = (max_epochs - warmup_epochs) * loader_size
    cosine_amplitude = 0.5 * (cosine_max - cosine_min)
    cosine_mean = 0.5 * (cosine_max + cosine_min)
    def cosine_decay_with_warmup(i):
        if warmup_steps and i < warmup_steps:
            return warmup_amplitude * i / warmup_steps + warmup_min
        else:
            i = i - warmup_steps
            return cosine_amplitude * math.cos(i * math.pi / cosine_steps) + cosine_mean
    return LambdaLR(optimizer, cosine_decay_with_warmup)
