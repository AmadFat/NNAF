from collections import deque, defaultdict
import logging
import enum

class Mode(enum.Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARN
    CRITICAL = logging.CRITICAL

class Fmt(enum.Enum):
    DEBUG_TRACE = "[%(levelname)s] %(asctime)s Epoch:%(epoch)s Step:%(step)s %(filename)s:%(lineno)d: %(message)s"
    DEBUG_NOTRACE = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d: %(message)s"
    INFO_TRACE = "[%(levelname)s] %(asctime)s E:%(epoch)s S:%(step)s %(message)s"
    INFO_NOTRACE = "[%(levelname)s] %(asctime)s %(message)s"
    WARNING_TRACE = "%(levelname).1s %(asctime)s E%(epoch)s S%(step)s %(message)s"
    WARNING_NOTRACE = "%(levelname).1s %(asctime)s %(message)s"
    CRITICAL_TRACE = "%(levelname).1s %(asctime)s %(epoch)s %(message)s"
    CRITICAL_NOTRACE = "%(levelname).1s %(asctime)s %(message)s"

class DateFmt(enum.Enum):
    DEFAULT = "%m-%d-%y %H:%M:%S"
    SHORT = "%m%d-%H%M"

class Logger:
    def __init__(
        self,
        name: str = "Amadeus",
        step_interval: int = 0,
        file_path: str = None,
        tb_path: str = None,
        mode: Mode | str = Mode.DEBUG,
        datefmt: DateFmt | str = DateFmt.SHORT,
        set_logging_root: bool = True,
    ):
        self.name = "root" if set_logging_root else name
        self.step_interval = step_interval
        self.mode = getattr(Mode, mode.upper()) if isinstance(mode, str) else mode
        self.datefmt = (getattr(DateFmt, datefmt.upper()) if isinstance(datefmt, str) else datefmt).value
        self.tracer = defaultdict(lambda: deque(maxlen=step_interval))
        self.str_metrics = defaultdict(lambda: dict())
        
        class ESFormatter(logging.Formatter):
            _fmt_trace = getattr(Fmt, f"{self.mode.name}_TRACE").value
            _fmt_notrace = getattr(Fmt, f"{self.mode.name}_NOTRACE").value
            def format(self, log):
                has_es = hasattr(log, 'epoch') or hasattr(log, 'step')
                self._style._fmt = self._fmt_trace if has_es else self._fmt_notrace
                return super().format(log)
        
        self.logger = logging.getLogger(None if set_logging_root else name)
        self.logger.propagate = False
        self.logger.setLevel(getattr(Mode, self.mode.name.upper()).value)
        handlers = [logging.StreamHandler()]
        if file_path is not None:
            handlers.append(logging.FileHandler(file_path))
        for hdl in handlers:
            hdl.setLevel(self.mode.value)
            hdl.setFormatter(ESFormatter(datefmt=self.datefmt))
        self.logger.handlers = handlers

        self.tbrecorder = None
        if tb_path is not None:
            try:
                import torch.utils.tensorboard, pathlib
                self.debug(f"Tensorboard path: {tb_path}")
                pathlib.Path(tb_path).mkdir(parents=True, exist_ok=True)
                self.tbrecorder = torch.utils.tensorboard.SummaryWriter(tb_path)
                self.global_step = 1
            except ImportError:
                self.warning("module `tensorboard` not found: relative features disabled.")

        if self.mode.value <= Mode.DEBUG.value:
            import importlib, platform, torch
            env_info = {
                "Hostname": platform.node(),
                "OS System": platform.system(),
                "OS Release": platform.release(),
                "Python Version": platform.python_version(),
                "PyTorch Version": torch.__version__,
            }
            if importlib.util.find_spec("torchvision"):
                import torchvision
                env_info["TorchVision Version"] = torchvision.__version__
            if torch.cuda.is_available():
                env_info.update({
                    "GPU Current Device": f"({torch.cuda.current_device()}) {torch.cuda.get_device_name()}",
                    "GPU Memory": f"{torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1000**3:.2f} GB"
                })
            try:
                import tabulate
                env_info = tabulate.tabulate(env_info.items(), tablefmt="grid")
            except ImportError:
                self.warning("module `tabulate` not found: relative features disabled.")
            self.debug(f"Environment info:\n{env_info}")
        self.debug("Logger initialized.")

    @staticmethod
    def _postproc_kwargs(**kwargs):
        if "epoch" in kwargs or "step" in kwargs:
            kwargs["extra"] = {
                "epoch": kwargs.pop("epoch", ""),
                "step": kwargs.pop("step", ""),
            }
        return kwargs
    
    def debug(self, msg, **kwargs):
        self.logger.debug(msg, **self._postproc_kwargs(**kwargs))
    
    def info(self, msg, **kwargs):
        self.logger.info(msg, **self._postproc_kwargs(**kwargs))

    def warning(self, msg, **kwargs):
        self.logger.warning(msg, **self._postproc_kwargs(**kwargs))

    def error(self, msg, **kwargs):
        self.logger.error(msg, **self._postproc_kwargs(**kwargs))

    def critical(self, msg, **kwargs):
        self.logger.critical(msg, **self._postproc_kwargs(**kwargs))

    def add(self, **kwargs):
        trace_default = kwargs.pop("trace", False)
        fmt_default = kwargs.pop("fmt", ".4f")
        tag_default = kwargs.pop("tag", "")
        for k, v in kwargs.items():
            trace, fmt, tag = trace_default, fmt_default, tag_default
            if type(v) in [tuple, list]:
                trace = v[1].pop("trace", trace)
                fmt = v[1].pop("fmt", fmt)
                tag = v[1].pop("tag", tag)
                v = v[0]
            if trace:
                self.tracer[k].append(v)
                v_smooth = sum(self.tracer[k]) / len(self.tracer[k])
            if self.tbrecorder is not None and isinstance(v, (int, float)):
                self.tbrecorder.add_scalars(
                    main_tag=k,
                    tag_scalar_dict={tag: v},
                    global_step=self.global_step,
                )
            v = f"{v:{fmt}}({v_smooth:{fmt}})" if trace else f"{v:{fmt}}"
            self.str_metrics[k][tag] = v
    
    def commit(
        self,
        epoch: int = 0, max_epochs: int = 0,
        step: int = 0, max_steps: int = 0,
    ):
        epoch_info = f"{epoch}/{max_epochs}" if epoch and max_epochs else f"{epoch}" if epoch and not max_epochs else ""
        step_info = f"{step}/{max_steps}" if step and max_steps else f"{step}" if step and not max_steps else ""
        flat_metrics = [f"{k}:{v}" if tag == "" else f"{k}+{tag}:{v}" for k, s in self.str_metrics.items() for tag, v in s.items()]
        method = self.info if step in [1, max_steps] or not step % self.step_interval else self.debug
        method(" | ".join(flat_metrics), epoch=epoch_info, step=step_info)
        self.str_metrics.clear()
        if self.tbrecorder is not None:
            self.global_step += 1

    @property
    def ignore_functions(self):
        """For torch.compile"""
        return [
            self.add,
            self.commit,
            self.debug,
            self.info,
            self.warning,
            self.error,
            self.critical,
            self._postproc_kwargs,
        ]
