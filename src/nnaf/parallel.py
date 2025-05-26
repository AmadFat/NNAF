from ._types import *
import asyncio, inspect, threading
from concurrent.futures import ThreadPoolExecutor

def run_in_another_thread(
    func,
    *args,
    timeout: float = 5,
    success_callback: Callable[[Any], None] = None,
    error_callback: Callable[[Exception], None] = None,
    **kwargs,
) -> None:
    """
    Run a synchronous or asynchronous function in a non-blocking way using a separate
    thread with a limited timeout. If successful, calls the success callback with the
    result. If an error occurs, calls the error callback with the exception.
    """
    
    assert timeout is not None and timeout > 0

    def run():
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                if inspect.iscoroutinefunction(func):
                    def wrap_afunc():
                        return asyncio.run(asyncio.wait_for(func(*args, **kwargs), timeout=timeout))
                    future = executor.submit(wrap_afunc)
                else:
                    future = executor.submit(func, *args, **kwargs)
                result = future.result(timeout=timeout)
                
            if success_callback is not None:
                success_callback(result)

        except Exception as e:
            if error_callback is not None:
                error_callback(e)
    
    thread = threading.Thread(target=run)
    thread.start()
