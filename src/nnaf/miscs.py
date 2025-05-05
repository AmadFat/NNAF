import torch

def autotune_num_workers(dataset, batch_size, num_tests=64, max_downgrade=3):
    import time, contextlib, os, logging
    logging.debug("Auto-tuning num_workers...")
    @contextlib.contextmanager
    def safeloader(num_workers):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        try:
            yield loader
        finally:
            loader._iterator = None
            del loader
    
    worker, set_workers = os.cpu_count(), []
    while worker not in set_workers:
        set_workers.append(worker)
        worker = worker // 2
    best_worker, best_time = 0, float("inf")
    for idx, worker in enumerate(set_workers):
        if idx > max_downgrade:
            break
        try:
            with safeloader(worker) as loader:
                start = time.perf_counter()
                cnt = 0
                while cnt < num_tests:
                    for _ in loader:
                        cnt += 1
                        if cnt >= num_tests:
                            break
                elapsed_time = time.perf_counter() - start
                logging.debug(f"Elapsed time with {worker} workers: {elapsed_time:.4f}s")
                if elapsed_time < best_time:
                    best_worker, best_time = worker, elapsed_time
        except Exception as e:
            print(f"Error with {worker} workers: {e}")
    return best_worker

def dict2str(**kwargs):
    return ', '.join([f"{k}={v}" for k, v in kwargs.items()])
