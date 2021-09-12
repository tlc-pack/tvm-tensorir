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
"""RPC Runner"""
import concurrent.futures
from contextlib import contextmanager
import itertools
import os.path as osp
from typing import Callable, List, Optional, Union

from ...contrib.popen_pool import PopenPoolExecutor
from ...rpc import RPCSession
from ...runtime import Device, Module, NDArray
from ..arg_info import ArgInfo, Args, PyArgsInfo
from ..utils import get_global_func_with_default_on_worker
from .rpc_config import RPCConfig
from .runner import EvaluatorConfig, PyRunner, RunnerFuture, RunnerInput, RunnerResult


class RPCRunnerFuture(RunnerFuture):
    """RPC based runner future

    Parameters
    ----------
    future: concurrent.futures.Future
        The concurrent function to check when the function is done and to return the result.
    timeout_sec: float
        The timeout in seconds.
    """

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
                error_msg=f"RPCRunner: Timeout, killed after {self.timeout_sec} seconds",
            )
        except Exception as exception:  # pylint: disable=broad-except
            return RunnerResult(
                None,
                error_msg="RPCRunner: An exception occurred\n" + str(exception),
            )
        return RunnerResult(run_sec, None)


class RPCRunner(PyRunner):
    """RPC based runner

    Parameters
    ----------
    rpc_config: RPCConfig
        The rpc configuration.
    evaluator_config: EvaluatorConfig
        The evaluator configuration.
    cooldown_sec: float
        The cooldown in seconds. TODO(@junrushao1994,@zxybazh): This is not used yet.
    alloc_repeat: int
        The number of times to repeat the allocation.
    f_create_session: Optional[str, Callable]
        The function name to create the session or the function itself.
    f_upload_module: Optional[str, Callable]
        The function name to upload the module or the function itself.
    f_alloc_argument: Optional[str, Callable]
        The function name to allocate the arguments or the function itself.
    f_run_evaluator: Optional[str, Callable]
        The function name to run the evaluator or the function itself.
    f_cleanup: Optional[str, Callable]
        The function name to cleanup the session or the function itself.
    pool: str
        The popen pool executor.

    Note
    ----
    Does not support customized function name passing for f_create_session, f_upload_module,
    f_alloc_argument, f_run_evaluator, f_cleanup. These functions must be passed directly.
    """

    rpc_config: RPCConfig
    evaluator_config: EvaluatorConfig
    cooldown_sec: float
    alloc_repeat: int

    f_create_session: Optional[
        Union[
            str,
            Callable[
                [RPCConfig],
                RPCSession,
            ],
        ]
    ] = None
    f_upload_module: Optional[Union[str, Callable[[RPCSession, str, str], Module]]] = None
    f_alloc_argument: Optional[
        Union[
            str,
            Callable[
                [RPCSession, Device, int, PyArgsInfo],
                List[Args],
            ],
        ]
    ] = None
    f_run_evaluator: Optional[
        Union[
            str,
            Callable[
                [
                    RPCSession,
                    Module,
                    Device,
                    EvaluatorConfig,
                    List[Args],
                ],
                List[float],
            ],
        ]
    ] = None
    f_cleanup: Optional[Union[str, Callable[[Optional[RPCSession], Optional[str]], None]]] = None

    pool: PopenPoolExecutor

    def __init__(
        self,
        rpc_config: Optional[RPCConfig] = None,
        evaluator_config: Optional[EvaluatorConfig] = None,
        cooldown_sec: float = 0.0,
        alloc_repeat: int = 1,
        f_create_session: Optional[str] = None,
        f_upload_module: Optional[str] = None,
        f_alloc_argument: Optional[str] = None,
        f_run_evaluator: Optional[str] = None,
        f_cleanup: Optional[str] = None,
        max_connections: Optional[int] = None,
        initializer: Optional[Callable[[], None]] = None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        rpc_config: RPCConfig
            The rpc configuration.
        evaluator_config: EvaluatorConfig
            The evaluator configuration.
        cooldown_sec: float
            The cooldown in seconds.
        alloc_repeat: int
            The number of times to random fill the allocation.
        f_create_session: Optional[str, Callable]
            The function name to create the session or the function itself.
        f_upload_module: Optional[str, Callable]
            The function name to upload the module or the function itself.
        f_alloc_argument: Optional[str, Callable]
            The function name to allocate the arguments or the function itself.
        f_run_evaluator: Optional[str, Callable]
            The function name to run the evaluator or the function itself.
        f_cleanup: Optional[str, Callable]
            The function name to cleanup the session or the function itself.
        max_connections: Optional[int]
            The maximum number of connections.
        initializer: Optional[Callable[[], None]]
            The initializer function.
        """
        super().__init__()
        self.rpc_config = RPCConfig._parse(rpc_config)
        self.evaluator_config = EvaluatorConfig._parse(evaluator_config)
        self.cooldown_sec = cooldown_sec
        self.alloc_repeat = alloc_repeat
        self.f_create_session = f_create_session
        self.f_upload_module = f_upload_module
        self.f_alloc_argument = f_alloc_argument
        self.f_run_evaluator = f_run_evaluator
        self.f_cleanup = f_cleanup

        num_servers = self.rpc_config.count_num_servers(allow_missing=False)
        if max_connections is None:
            max_connections = num_servers
        else:
            max_connections = min(max_connections, num_servers)

        self.pool = PopenPoolExecutor(
            max_workers=max_connections,
            timeout=rpc_config.session_timeout_sec,
            initializer=initializer,
        )
        self._sanity_check()

    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        results: List[RunnerFuture] = []
        for runner_input in runner_inputs:
            future = RPCRunnerFuture(
                future=self.pool.submit(
                    RPCRunner._worker_func,
                    self.f_create_session,
                    self.f_upload_module,
                    self.f_alloc_argument,
                    self.f_run_evaluator,
                    self.f_cleanup,
                    self.rpc_config,
                    self.evaluator_config,
                    self.alloc_repeat,
                    str(runner_input.artifact_path),
                    str(runner_input.device_type),
                    tuple(arg_info.as_python() for arg_info in runner_input.args_info),
                ),
                timeout_sec=self.rpc_config.session_timeout_sec,
            )
            results.append(future)
        return results

    def _sanity_check(self) -> None:
        def _check(
            f_create_session: Optional[str] = None,
            f_upload_module: Optional[str] = None,
            f_alloc_argument: Optional[str] = None,
            f_run_evaluator: Optional[str] = None,
            f_cleanup: Optional[str] = None,
        ) -> None:
            get_global_func_with_default_on_worker(name=f_create_session, default=None)
            get_global_func_with_default_on_worker(name=f_upload_module, default=None)
            get_global_func_with_default_on_worker(name=f_alloc_argument, default=None)
            get_global_func_with_default_on_worker(name=f_run_evaluator, default=None)
            get_global_func_with_default_on_worker(name=f_cleanup, default=None)

        value = self.pool.submit(
            _check,
            self.f_create_session,
            self.f_upload_module,
            self.f_alloc_argument,
            self.f_run_evaluator,
            self.f_cleanup,
        )
        value.result()

    @staticmethod
    def _worker_func(
        _f_create_session: Optional[str],
        _f_upload_module: Optional[str],
        _f_alloc_argument: Optional[str],
        _f_run_evaluator: Optional[str],
        _f_cleanup: Optional[str],
        rpc_config: RPCConfig,
        evaluator_config: EvaluatorConfig,
        alloc_repeat: int,
        artifact_path: str,
        device_type: str,
        args_info: PyArgsInfo,
    ) -> List[float]:
        # Step 0. Get the registered functions
        f_create_session: Callable[
            [RPCConfig],
            RPCSession,
        ] = get_global_func_with_default_on_worker(
            _f_create_session,
            default_create_session,
        )
        f_upload_module: Callable[
            [RPCSession, str, str],
            Module,
        ] = get_global_func_with_default_on_worker(
            _f_upload_module,
            default_upload_module,
        )
        f_alloc_argument: Callable[
            [RPCSession, Device, int, PyArgsInfo],
            List[Args],
        ] = get_global_func_with_default_on_worker(
            _f_alloc_argument,
            default_alloc_argument,
        )
        f_run_evaluator: Callable[
            [
                RPCSession,
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
            [Optional[RPCSession], Optional[str]], None
        ] = get_global_func_with_default_on_worker(
            _f_cleanup,
            default_cleanup,
        )
        session: Optional[RPCSession] = None
        remote_path: Optional[str] = None

        @contextmanager
        def resource_handler():
            try:
                yield
            finally:
                # Step 5. Clean up
                f_cleanup(session, remote_path)

        with resource_handler():
            # Step 1. Create session
            session = f_create_session(rpc_config)
            device = session.device(dev_type=device_type, dev_id=0)
            # Step 2. Upload the module
            _, remote_path = osp.split(artifact_path)
            local_path: str = artifact_path
            rt_mod: Module = f_upload_module(session, local_path, remote_path)

            # Step 3: Allocate input arguments
            repeated_args: List[Args] = f_alloc_argument(
                session,
                device,
                alloc_repeat,
                args_info,
            )
            # Step 4: Run time_evaluator
            costs: List[float] = f_run_evaluator(
                session,
                rt_mod,
                device,
                evaluator_config,
                repeated_args,
            )
        return costs


def default_create_session(rpc_config: RPCConfig) -> RPCSession:
    """Default function to create the session

    Parameters
    ----------
    rpc_config : RPCConfig
        The configuration of the RPC session

    Returns
    -------
    session : RPCSession
        The created rpc session
    """
    return rpc_config.connect_server()


def default_upload_module(
    session: RPCSession,
    local_path: str,
    remote_path: str,
) -> Module:
    """Default function to upload the module

    Parameters
    ----------
    session: RPCSession
        The session to upload the module
    local_path: str
        The local path of the module
    remote_path: str
        The remote path to place the module

    Returns
    -------
    rt_mod : Module
        The runtime module
    """
    session.upload(local_path, remote_path)
    rt_mod: Module = session.load_module(remote_path)
    return rt_mod


def default_alloc_argument(
    session: RPCSession,
    device: Device,
    alloc_repeat: int,
    args_info: PyArgsInfo,
) -> List[Args]:
    """Default function to allocate the arguments

    Parameters
    ----------
    session: RPCSession
        The session to allocate the arguments
    device: Device
        The device to allocate the arguments
    alloc_repeat: int
        The number of times to repeat the allocation
    args_info: PyArgsInfo
        The arguments info

    Returns
    -------
    repeated_args: List[Args]
        The allocation args
    """
    try:
        f_random_fill = session.get_function("tvm.contrib.random.random_fill")
    except AttributeError as error:
        raise AttributeError(
            'Unable to find function "tvm.contrib.random.random_fill" on remote RPC server. '
            "Please make sure USE_RANDOM is turned ON in the config.cmake on the RPC server."
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
    session: RPCSession,  # pylint: disable=unused-argument
    rt_mod: Module,
    device: Device,
    evaluator_config: EvaluatorConfig,
    repeated_args: List[Args],
) -> List[float]:
    """Default function to run the evaluator

    Parameters
    ----------
    session: RPCSession
        The session to run the evaluator
    rt_mod: Module
        The runtime module
    device: Device
        The device to run the evaluator
    evaluator_config: EvaluatorConfig
        The evaluator config
    repeated_args: List[Args]
        The repeated arguments

    Returns
    -------
    costs: List[float]
        The evaluator results
    """
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
    session: Optional[RPCSession],
    remote_path: Optional[str],
) -> None:
    """Default function to clean up the session

    Parameters
    ----------
    session: RPCSession
        The session to clean up
    remote_path: str
        The remote path to clean up
    """
    if session is not None and remote_path is not None:
        session.remove(remote_path)
        session.remove(remote_path + ".so")
        session.remove("")
