# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-variable,invalid-name

"""
Integrate meta schedule into relay. It implements the following items:
1. Extract search tasks from a relay program
2. Get scheduled function from dispatcher context for compile_engine.
"""

import threading

import tvm
from tvm import autotvm
from tvm import meta_schedule as ms
from tvm import transform
from tvm.ir.transform import PassContext

from . import _ffi_api
from .dispatcher import DispatchContext
from .search import SearchTask


@tvm._ffi.register_func("meta_schedule.relay_integration.get_func_from_dispatcher")
def get_func_from_dispatcher(func):
    """
    If it is in TaskExtractionTracingEnvironmentQuery, add it into the function table.
    Otherwise, query the dispatcher for trace and apply it onto the function.
    Note: This is used internally for relay integration. Do
    not use this as a general user-facing API.

    Parameters
    ----------
    func: tvm.tir.PrimFunc
        The original function from compile engine

    Returns
    -------
    transformed_func: Optional[tvm.tir.PrimFunc]
        The new function after applying trace
    """
    env = TaskExtractionTracingEnvironment.current
    if env is None:
        target = tvm.target.Target.current()
        task = SearchTask(func, target=target, target_host=target)
        trace = DispatchContext.current.query(task)
        if trace is None:
            return None
        space = DispatchContext.current.space
        new_func = _ffi_api.ApplyTrace(trace, task, space)  # pylint: disable=no-member
        return new_func
    else:
        env.add_func(func)
        return None


def call_all_topi_funcs(mod, params, target):
    """Call all TOPI compute to extract meta schedule tasks in a Relay program"""
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
        disabled_pass={"MetaScheduleLayoutRewrite"},
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

    autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent


def extract_tasks(mod, params, target):
    """Extract tuning tasks from a relay program.

    Parameters
    ----------
    mod: tvm.IRModule or relay.function.Function
        The module or function to tune
    params: dict of str to numpy array
        The associated parameters of the program
    target: Union[tvm.target.Target, str]
        The compilation target

    Returns
    -------
    funcs: dict of str to tvm.tir.PrimFunc
        The extracted functions along with its name
    """
    if isinstance(target, str):
        target = tvm.target.Target(target)
    env = TaskExtractionTracingEnvironment()

    with env:
        # Wrap build call in a new thread to avoid the conflict
        # between python's multiprocessing and tvm's thread pool
        build_thread = threading.Thread(target=call_all_topi_funcs, args=(mod, params, target))
        build_thread.start()
        build_thread.join()
    return env.funcs


def is_meta_schedule_enabled():
    """Return whether the meta-schedule is enabled.

    Returns
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
