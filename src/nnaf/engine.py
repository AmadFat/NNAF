import torch
from .logger import Logger

_grad_scaler = None
_bs_accumulated = None

def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    accumulate_batch_size: int = None,
    logger: Logger = None,
    device: str = "cpu",
    epoch: int = 0,
    max_epochs: int = 0,
    use_amp: bool = False,
):
    global _grad_scaler, _bs_accumulated

    model.train()
    if use_amp and _grad_scaler is None:
            _grad_scaler = torch.GradScaler(device=device)
    if accumulate_batch_size is not None:
        _bs_accumulated = 0

    for step, (imgs, anns) in enumerate(loader, 1):
        imgs = imgs.to(device, non_blocking=True)
        anns = anns.to(device, non_blocking=True)

        with torch.autocast(device_type=device, enabled=use_amp):
            preds = model(imgs)
            loss = criterion(preds, anns)

        if use_amp:
            _grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if accumulate_batch_size is not None:
            _bs_accumulated += imgs.shape[0]

        if accumulate_batch_size is None or _bs_accumulated == accumulate_batch_size:
            if grad_clip is not None:
                if use_amp:
                    _grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if use_amp:
                _grad_scaler.step(optimizer)
                _grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if accumulate_batch_size is not None:
                _bs_accumulated = 0

        if device == "cuda":
            torch.cuda.synchronize()

        if logger is not None:
            loss = loss.item()
            lr = optimizer.param_groups[0]["lr"]
            logger.add(
                loss=(loss, dict(trace=True, fmt=".4f", tag="train")),
                lr=(lr, dict(trace=False, fmt=".3e"))
            )
            logger.commit(
                epoch=epoch, max_epochs=max_epochs,
                step=step, max_steps=len(loader),
            )

        if lr_scheduler is not None:
            lr_scheduler.step()
