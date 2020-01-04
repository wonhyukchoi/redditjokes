import time
import functools

def timer(func):
    @functools.wraps(func)
    def timer_wrapper(* args, **kwargs):
        now = time.time()
        _return = func(*args, **kwargs)
        print(f'{func.__name__} took {(time.time()-now):.2f} seconds \n')
        return _return
    return timer_wrapper
    