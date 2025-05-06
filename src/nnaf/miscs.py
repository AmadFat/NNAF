def dict2str(**kwargs):
    return ', '.join([f"{k}={v}" for k, v in kwargs.items()])
