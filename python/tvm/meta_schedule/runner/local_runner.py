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
"""Local Runner"""
import concurrent.futures
import itertools
import os
import os.path as osp
import shutil
from contextlib import contextmanager
from typing import Callable, List, Optional, Union
import tvm

from ...contrib.popen_pool import PopenPoolExecutor
from ...runtime import Device, Module, NDArray
from ..arg_info import ArgInfo, Args, PyArgsInfo
from ..utils import get_global_func_with_default_on_worker
from .runner import EvaluatorConfig, PyRunner, RunnerFuture, RunnerInput, RunnerResult

class LocalRunnerFuture(RunnerFuture):
    future: concurrent.futures.Future
    timeout_sec: float
    
    def __init__(self, future: concurrent.futures.Future, timeout_sec: float) -> None:
        """Constructor
        Parameters
        ----------
        future: concurrent.futures.Future
            The concurrent function to check when the function is done and to return the result.
        timeout_sec: float
            The timeout in seconds.
        """
        super().__init__()
        self.future = future
        self.timeout_sec = timeout_sec
    
    def done(self) -> bool:
        return self.future.done()

    def result(self) -> RunnerResult:
        try:
            run_sec: List[float] = self.future.result()
        except TimeoutError as exception:
            return RunnerResult(
                None,
                error_msg=f"LocalRunner: Timeout, killed after {self.timeout_sec} seconds",
            )
        except Exception as exception:  # pylint: disable=broad-except
            return RunnerResult(
                None,
                error_msg="LocalRunner: An exception occurred\n" + str(exception),
            )
        return RunnerResult(run_sec, None)

class LocalRunner(PyRunner):
    """Local runner
    Parameters
    ----------
    evaluator_config: EvaluatorConfig
        The evaluator configuration.
    cooldown_sec: float
        The cooldown in seconds.
    alloc_repeat: int
        The number of times to repeat the allocation.
    f_alloc_argument: Optional[str, Callable]
        The function name to allocate the arguments or the function itself.
    f_run_evaluator: Optional[str, Callable]
        The function name to run the evaluator or the function itself.
    f_cleanup: Optional[str, Callable]
        The function name to cleanup the session or the function itself.
    pool: PopenPoolExecutor
        The popen pool executor.
    Note
    ----
    Does not support customizd function name passing for f_alloc_argument, f_run_evaluator, 
    f_cleanup. These functions must be passed directly.
    """
    timeout_sec: float
    evaluator_config: EvaluatorConfig
    cooldown_sec: float
    alloc_repeat: int

    f_alloc_argument: Optional[
        Union[
            str,
            Callable[
                [Device, int, PyArgsInfo],
                List[Args],
            ],
        ]
    ] = None
    f_run_evaluator: Optional[
        Union[
            str,
            Callable[
                [
                    Module,
                    Device,
                    EvaluatorConfig,
                    List[Args],
                ],
                List[float],
            ],
        ]
    ] = None
    f_cleanup: Optional[
        Union[str, Callable[[Optional[str], Optional[str]], None]]
    ] = None

    pool: PopenPoolExecutor
    
    def __init__(
		self,
        timeout_sec: float,
        evaluator_config: Optional[EvaluatorConfig] = None,
        cooldown_sec: float = 0.0,
        alloc_repeat: int = 1,
        f_alloc_argument: Optional[str] = None,
        f_run_evaluator: Optional[str] = None,
        f_cleanup: Optional[str] = None,
        initializer: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__()
        self.timeout_sec = timeout_sec
        self.evaluator_config = EvaluatorConfig._parse(evaluator_config)
        self.cooldown_sec = cooldown_sec
        self.alloc_repeat = alloc_repeat
        self.f_alloc_argument = f_alloc_argument
        self.f_run_evaluator = f_run_evaluator
        self.f_cleanup = f_cleanup
        
    
    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        results: List[RunnerFuture] = []
        for runner_input in runner_inputs:
            future = LocalRunnerFuture(
                future=self.pool.submit(
                    LocalRunner._worker_func,
                    self.f_alloc_argument,
                    self.f_run_evaluator,
                    self.f_cleanup,
                    self.evaluator_config,
                    self.alloc_repeat,
                    str(runner_input.artifact_path),
                    str(runner_input.device_type),
                    tuple(arg_info.as_python() for arg_info in runner_input.args_info),
                ),
                timeout_sec=self.timeout_sec,
            )
            results.append(future)
        return results

	@staticmethod
    def _worker_func(
        _f_alloc_argument: Optional[str],
        _f_run_evaluator: Optional[str],
        _f_cleanup: Optional[str],
        evaluator_config: EvaluatorConfig,
        alloc_repeat: int,
        artifact_path: str,
        device_type: str,
        args_info: PyArgsInfo,
    ) -> List[float]:
        f_alloc_argument: Callable[
            [Device, int, PyArgsInfo],
            List[Args],
        ] = get_global_func_with_default_on_worker(
            _f_alloc_argument,
            default_alloc_argument,
        )
        f_run_evaluator: Callable[
            [
                Module,
                Device,
                EvaluatorConfig,
                List[Args],
            ],
            List[float],
        ] = get_global_func_with_default_on_worker(
            _f_run_evaluator,
            default_run_evaluator,
        )
        f_cleanup: Callable[
            [Optional[str], Optional[str]], None
        ] = get_global_func_with_default_on_worker(
            _f_cleanup,
            default_cleanup,
        )
        
        @contextmanager
        def resource_handler():
            try:
                yield
            finally:
                # Step 5. Clean up
                f_cleanup()
        with resource_handler():
            # Step 1: create the local runtime module
            rt_mod = tvm.module.load_module(artifact_path)
            # Step 2: create the local device
            device = tvm.ndarray.device(dev_type=device_type, dev_id=0)
            # Step 3: Allocate input arguments
            repeated_args: List[Args] = f_alloc_argument(
                device,
                alloc_repeat,
                args_info,
            )
            # Step 4: Run time_evaluator
            costs: List[float] = f_run_evaluator(
                rt_mod,
                device,
                evaluator_config,
                repeated_args,
            )
        return costs


def default_alloc_argument(
    device: Device,
    alloc_repeat: int,
    args_info: PyArgsInfo,
) -> List[Args]:
    try:
        f_random_fill = tvm.get_global_func("tvm.contrib.random.random_fill", True)
    except AttributeError as error:
        raise AttributeError(
            'Unable to find function "tvm.contrib.random.random_fill" on local runner. '
            "Please make sure USE_RANDOM is turned ON in the config.cmake."
        ) from error
    repeated_args: List[Args] = []
    for _ in range(alloc_repeat):
        args: Args = []
        for arg_info in args_info:
            arg = ArgInfo.alloc(arg_info, device)
            if isinstance(arg, NDArray):
                f_random_fill(arg)
            args.append(arg)
        repeated_args.append(args)
    return repeated_args

def default_run_evaluator(
    rt_mod: Module,
    device: Device,
    evaluator_config: EvaluatorConfig,
    repeated_args: List[Args],
) -> List[float]:
    evaluator = rt_mod.time_evaluator(
        func_name=rt_mod.entry_name,
        dev=device,
        number=evaluator_config.number,
        repeat=evaluator_config.repeat,
        min_repeat_ms=evaluator_config.min_repeat_ms,
        f_preproc="cache_flush_cpu_non_first_arg"
        if evaluator_config.enable_cpu_cache_flush
        else "",
    )
    repeated_costs: List[List[float]] = []
    for args in repeated_args:
        device.sync()
        profile_result = evaluator(*args)
        repeated_costs.append(profile_result.results)
    costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
    return costs

def default_cleanup(
) -> None:
    pass