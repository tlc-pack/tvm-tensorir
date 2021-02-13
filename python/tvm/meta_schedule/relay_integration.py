import tvm
from .search import SearchTask
from .dispatcher import DispatchContext
from . import _ffi_api


@tvm._ffi.register_func("meta_schedule.relay_integration.auto_schedule_primfunc")
def auto_schedule_primfunc(func):
    target = tvm.target.Target.current()
    task = SearchTask(func, target=target)
    trace = DispatchContext.current.query(task)
    if trace is None:
        return None
    space = DispatchContext.current.space
    new_func = _ffi_api.ApplyTrace(trace, task, space)
    return new_func
