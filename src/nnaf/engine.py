from nnaf_logger import Loggerv2
from nnaf_utils.pytype import *

from .pttype import *


class Trainer:
    def __init__(
        self,
        device: str | torch.device = "cpu",
        amp_dtype: torch.dtype = None,
        grad_accumulate_steps: int = None,
        grad_clip: float = None,
        logger: Loggerv2 = None,
    ):
        self.device = device
        self.enable_amp = amp_dtype is not None
        self.amp_dtype = amp_dtype
        self.grad_accumulate_steps = grad_accumulate_steps or 1
        self.grad_clip = grad_clip
        self.logger = logger

        self.grad_scaler = torch.GradScaler(device=device, enabled=self.enable_amp)
        self.cal_index = 0

    def fit_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module | Callable,
        optimizer: Optimizer,
        step_scheduler: LRScheduler = None,
        epoch_scheduler: LRScheduler = None,
        epoch: int = 0,
    ):
        model.train()

        for step, (x, y) in enumerate(loader, 1):
            # send to target device
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # forward and backward
            with torch.autocast(
                device_type=self.device,
                dtype=self.amp_dtype,
                enabled=self.enable_amp,
            ):
                pred = model(x)
                loss = criterion(pred, y)
            self.grad_scaler.scale(loss).backward()
            self.cal_index += 1

            if self.cal_index % self.grad_accumulate_steps == 0:
                if self.grad_clip:
                    self.grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
                optimizer.zero_grad()

                # Loggerv2: add and commit
                if self.logger:
                    metrics = dict(
                        loss=loss.item(),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                    self.logger.add(epoch=epoch, step=step, tag="train", **metrics)
                    self.logger.commit(epoch=epoch, step=step)

                # step_lr_scheduler
                if step_scheduler:
                    step_scheduler.step()

        # epoch_lr_scheduler
        if epoch_scheduler:
            epoch_scheduler.step()
