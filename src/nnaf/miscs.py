from ._types import *

def dict2str(**kwargs):
    return ', '.join([f"{k}={v}" for k, v in kwargs.items()])

def refresh_dir(dir: PathOrStr, leave_empty: bool = True):
    dir = Path(dir)
    if dir.exists():
        for item in dir.iterdir():
            if item.is_dir():
                refresh_dir(item, leave_empty=False)
            else:
                item.unlink()
    elif leave_empty:
        dir.mkdir(parents=True, exist_ok=True)
