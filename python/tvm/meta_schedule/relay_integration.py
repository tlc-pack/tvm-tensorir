import tvm
from .search import SearchTask
from .dispatcher import DispatchContext
from tvm.ir.transform import PassContext
from tvm import meta_schedule as ms
from tvm import autotvm, transform
from . import _ffi_api
from tvm import auto_scheduler
import threading
from tvm import tir
from tvm.script import ty


@tvm._ffi.register_func("meta_schedule.relay_integration.auto_schedule_primfunc")
def auto_schedule_primfunc(func):
    env = TaskExtractionTracingEnvironment.current
    if env is None:
        target = tvm.target.Target.current()
        task = SearchTask(func, target=target, target_host=target)
        trace = DispatchContext.current.query(task)
        if trace is None:
            return None
        space = DispatchContext.current.space
        new_func = _ffi_api.ApplyTrace(trace, task, space)
        return new_func
    else:
        env.add_func(func)
        return None


def call_all_topi_funcs(mod, params, target):
    """Call all TOPI compute to extract auto_scheduler tasks in a Relay program"""
    # pylint: disable=import-outside-toplevel
    from tvm import relay
    from tvm.relay.backend import graph_runtime_codegen

    # Turn off AutoTVM config not found warnings
    old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
    autotvm.GLOBAL_SCOPE.silent = True

    with transform.PassContext(
            opt_level=3,
            config={
                "relay.with_tir_schedule": True,
                "relay.backend.use_meta_schedule": True,
                "relay.backend.disable_compile_engine_cache": True,
            },
            disabled_pass={"MetaSchedulerLayoutRewrite"},
    ):
        try:
            opt_mod, _ = relay.optimize(mod, target, params)
            grc = graph_runtime_codegen.GraphRuntimeCodegen(None, target)
            grc.codegen(opt_mod["main"])
        except tvm.TVMError:
            print(
                "Get errors with GraphRuntimeCodegen for task extraction. "
                "Fallback to VMCompiler."
            )
            compiler = relay.vm.VMCompiler()
            if params:
                compiler.set_params(params)
            mod = tvm.IRModule.from_expr(mod) if isinstance(mod, relay.Function) else mod
            compiler.lower(mod, target)

    autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent


def extract_tasks(mod, params, target, target_host=None):
    if isinstance(target, str):
        target = tvm.target.Target(target)
    if isinstance(target_host, str):
        target_host = tvm.target.Target(target_host)
    env = TaskExtractionTracingEnvironment()

    with env:
        # Wrap build call in a new thread to avoid the conflict
        # between python's multiprocessing and tvm's thread pool
        build_thread = threading.Thread(target=call_all_topi_funcs, args=(mod, params, target))
        build_thread.start()
        build_thread.join()

    return env.funcs.values();


def is_meta_schedule_enabled():
    """Return whether the meta-schedule is enabled.

    Parameters
    ----------
    enabled: bool
        Whether the meta-schedule is enabled
    """
    return PassContext.current().config.get("relay.backend.use_meta_schedule", False)


class TaskExtractionTracingEnvironment:
    """Global environment for tracing all topi function calls"""

    current = None

    def __init__(self):
        self.funcs = {}

    def __enter__(self):
        TaskExtractionTracingEnvironment.current = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        TaskExtractionTracingEnvironment.current = None

    def add_func(self, func):
        """Add the function of a search task

        Parameters
        ----------
        func: PrimFunc
            The function of a task
        """
        task = ms.SearchTask(func)
        if task.task_name not in self.funcs:
            self.funcs[task.task_name] = func
