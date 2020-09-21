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
"""
Taken and modified from Ansor.

Distributed measurement infrastructure to measure the runtime costs of tensor programs.

These functions are responsible for building the tvm module, uploading it to
remote devices, recording the running time costs, and checking the correctness of the output.

We separate the measurement into two steps: build and run.
A builder builds the executable binary files and a runner runs the binary files to
get the measurement results. The flow of data structures is

                `ProgramBuilder`                 `ProgramRunner`
`MeasureInput` -----------------> `BuildResult` ----------------> `MeasureResult`

We implement these in python to utilize python's multiprocessing and error handling.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import List, Optional, Tuple

from tvm._ffi import register_func, register_object
from tvm.contrib import ndk as build_func_ndk
from tvm.contrib import tar as build_func_tar
from tvm.driver import build
from tvm.runtime import Object

from . import _ffi_api
from .measure_util import (
    NoDaemonPool,
    call_func_with_timeout,
    check_remote,
    cpu_count,
    make_error_msg,
    realize_arguments,
    request_remote,
    vprint,
)
from .schedule import Schedule
from .search_task import SearchTask

# The maximum possible cost used to indicate the cost for timeout
# We use 1e10 instead of sys.float_info.max for better readability in log
MAX_TIME_COST = 1e10


class MeasureErrorNo:
    """ Error type for MeasureResult. """

    NO_ERROR = 0  # No error
    INSTANTIATION_ERROR = 1  # Errors happen when apply transform steps from init state
    COMPILE_HOST = 2  # Errors happen when compiling code on host (e.g., tvm.build)
    COMPILE_DEVICE = 3  # Errors happen when compiling code on device
    # (e.g. OpenCL JIT on the device)
    RUNTIME_DEVICE = 4  # Errors happen when run program on device
    WRONG_ANSWER = 5  # Answer is wrong when compared to a reference output
    BUILD_TIMEOUT = 6  # Timeout during compilation
    RUN_TIMEOUT = 7  # Timeout during run
    UNKNOWN_ERROR = 8  # Unknown error


########## Three basic classes ##########


@register_object("meta_schedule.MeasureInput")
class MeasureInput(Object):
    """Store the input of a measurement.

    Parameters
    ----------
    sch : Schedule
        The meta schedule object
    """

    task: SearchTask
    sch: Schedule

    def __init__(self, task: SearchTask, sch: Schedule):
        self.__init_handle_by_constructor__(
            _ffi_api.MeasureInput, task, sch  # pylint: disable=no-member
        )


@register_object("meta_schedule.BuildResult")
class BuildResult(Object):
    """Store the result of a build.

    Parameters
    ----------
    filename : str
        The filename of built binary file.
    error_no : int
        The error code.
    error_msg : str
        The error message if there is any error.
    time_cost : float
        The time cost of build.
    """

    filename: str
    error_no: int
    error_msg: str
    time_cost: float

    def __init__(
        self,
        filename: str,
        error_no: int,
        error_msg: str,
        time_cost: float,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.BuildResult,  # pylint: disable=no-member
            filename,
            error_no,
            error_msg,
            time_cost,
        )


BuildResultType = Tuple[str, int, str, float]


@register_object("meta_schedule.MeasureResult")
class MeasureResult(Object):
    """Store the results of a measurement.

    Parameters
    ----------
    costs : List[float]
        The time costs of execution.
    error_no : int
        The error code.
    error_msg : str
        The error message if there is any error.
    all_cost : float
        The time cost of build and run.
    timestamp : float
        The time stamps of this measurement.
    """

    costs: List[float]
    error_no: int
    error_msg: str
    all_cost: float
    timestamp: float

    def __init__(
        self,
        costs: List[float],
        error_no: int,
        error_msg: str,
        all_cost: float,
        timestamp: float,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.MeasureResult,  # pylint: disable=no-member
            costs,
            error_no,
            error_msg,
            all_cost,
            timestamp,
        )


MeasureResultType = Tuple[List[float], int, str, float, float]

########## ProgramBuilder ##########


@register_object("meta_schedule.ProgramBuilder")
class ProgramBuilder(Object):
    """ The base class of ProgramBuilders. """

    n_parallel: int
    timeout: int

    @staticmethod
    def create(name: str) -> ProgramBuilder:
        if name == "local":
            return LocalBuilder()
        raise ValueError("Unknown name of program builder: " + name)

    def build(
        self, measure_inputs: List[MeasureInput], verbose: int = 1
    ) -> List[BuildResult]:
        """Build programs and return results.

        Parameters
        ----------
        measure_inputs : List[MeasureInput]
            A List of MeasureInput.
        verbose: int = 1
            Verbosity level. 0 for silent, 1 to output information during program building.

        Returns
        -------
        res : List[BuildResult]
        """
        return _ffi_api.ProgramBuilderBuild(  # pylint: disable=no-member
            self, measure_inputs, verbose
        )


########## ProgramRunner ##########


@register_object("meta_schedule.ProgramRunner")
class ProgramRunner(Object):
    """ The base class of ProgramRunners. """

    timeout: int
    number: int
    repeat: int
    min_repeat_ms: int
    cooldown_interval: float
    enable_cpu_cache_flush: bool

    @staticmethod
    def create(name: str) -> ProgramRunner:
        if name == "rpc":
            return RPCRunner()
        if name.startswith("rpc "):
            return RPCRunner.create(name[4:])
        raise ValueError("Unknown name of program builder: " + name)

    def run(
        self,
        measure_inputs: List[MeasureInput],
        build_results: List[BuildResult],
        verbose: int = 1,
    ) -> List[MeasureResult]:
        """Run measurement and return results.

        Parameters
        ----------
        measure_inputs : List[MeasureInput]
            A List of MeasureInput.
        build_results : List[BuildResult]
            A List of BuildResult to be ran.
        verbose: int = 1
            Verbosity level. 0 for silent, 1 to output information during program running.

        Returns
        -------
        res : List[MeasureResult]
        """
        return _ffi_api.ProgramRunnerRun(  # pylint: disable=no-member
            self, measure_inputs, build_results, verbose
        )


########## LocalBuilder ##########


@register_object("meta_schedule.LocalBuilder")
class LocalBuilder(ProgramBuilder):
    """LocalBuilder use local CPU cores to build programs in parallel."""

    build_func: str

    def __init__(
        self,
        timeout: int = 15,
        n_parallel: Optional[int] = None,
        build_func: str = "tar",
    ):
        if n_parallel is None:
            n_parallel = cpu_count()
        self.__init_handle_by_constructor__(
            _ffi_api.LocalBuilder,  #  pylint: disable=no-member
            timeout,
            n_parallel,
            build_func,
        )


########## RPCRunner ##########


@register_object("meta_schedule.RPCRunner")
class RPCRunner(ProgramRunner):
    """RPCRunner that uses RPC call to measures the time cost of programs on remote devices.
    Or sometime we may need to use RPC even in local running to insulate the thread environment.
    (e.g. running CUDA programs)"""

    tracker: str
    priority: int
    n_parallel: int

    def __init__(
        self,
        tracker: Optional[str] = None,
        priority: int = 1,
        n_parallel: int = 1,
        timeout: int = 10,
        number: int = 3,
        repeat: int = 1,
        min_repeat_ms: int = 0,
        cooldown_interval: float = 0.0,
        enable_cpu_cache_flush: bool = False,
    ):
        if tracker is None:
            tracker = os.environ.get("TVM_RPC_TRACKER", None)
        if check_remote(tracker, priority, timeout):
            print("Get devices for measurement successfully!")
        else:
            raise RuntimeError(
                "Cannot get remote devices from the tracker. "
                "Please check the status of tracker via "
                "'python -m tvm.exec.query_rpc_tracker --port [THE PORT YOU USE]' "
                "and make sure you have free devices on the queue status. "
                "You may also specify the RPC tracker info by setting "
                "environment variable 'TVM_RPC_TRACKER', "
                "e.g. 'TVM_RPC_TRACKER=0.0.0.0:9089:local'"
            )
        self.__init_handle_by_constructor__(
            _ffi_api.RPCRunner,  #  pylint: disable=no-member
            tracker,
            priority,
            n_parallel,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
        )

    @staticmethod
    def create(name: str) -> RPCRunner:
        if "*" not in name:
            return RPCRunner(tracker=name)
        split_result = name.split("*")
        if len(split_result) != 2:
            raise ValueError(f"Cannot create RPCRunner from string: {name}")
        tracker, n_parallel = map(str.strip, split_result)
        n_parallel = int(n_parallel)
        return RPCRunner(tracker=tracker, n_parallel=n_parallel)


########## MeasureCallback ##########


@register_object("meta_schedule.MeasureCallback")
class MeasureCallback(Object):
    """ The base class of measurement callback functions. """


########## Worker of LocalBuilder ##########


# We use fork and a global variable to copy arguments between processes.
# This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
LOCAL_BUILDER_WORKER_ARGS = None


@register_func("meta_schedule.local_builder.build")
def local_builder_build(
    measure_inputs: List[MeasureInput],
    timeout: int,
    n_parallel: int,
    build_func: str = "tar",
    verbose: int = 1,
) -> List[BuildResult]:
    """
    Build function of LocalBuilder to build the MeasureInputs to runnable modules.

    Parameters
    ----------
    measure_inputs : List[MeasureInput]
        The MeasureInputs to be built.
    timeout : int
        The timeout limit (in second) for each build process.
        This is used in a wrapper of the multiprocessing.Process.join().
    n_parallel : int
        Number of processes used to build in parallel.
    build_func : str = 'tar'
        The name of build function to process the built module.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program building.

    Returns
    -------
    results : List[BuildResult]
        The build results of these MeasureInputs.
    """
    global LOCAL_BUILDER_WORKER_ARGS
    LOCAL_BUILDER_WORKER_ARGS = (measure_inputs, build_func, timeout, verbose)
    pool = NoDaemonPool(n_parallel)
    indices = [i for i, _ in enumerate(measure_inputs)]
    results = pool.map(local_builder_wrapped_worker, indices)
    pool.terminate()
    pool.join()
    del pool
    results = [BuildResult(*item) for item in results]
    return results


def local_builder_worker(
    index: int,
    measure_inputs: List[MeasureInput],
    build_func: str,
    timeout: int,
    verbose: int,
) -> BuildResultType:
    """ Local worker for ProgramBuilder """
    # deal with build_func
    build_func = {
        "tar": build_func_tar.tar,  # export to tar
        "ndk": build_func_ndk.create_shared,  # export to ndk
    }.get(build_func, build_func)
    if isinstance(build_func, str):
        raise ValueError("Invalid build_func: " + build_func)
    # deal with measure_input
    measure_input = measure_inputs[index]

    def timed_func() -> BuildResultType:
        tic = time.time()
        # return values
        filename = ""
        error_no = MeasureErrorNo.NO_ERROR
        error_msg = ""
        time_cost = 1e9
        # create temporary path
        filename = os.path.join(
            tempfile.mkdtemp(), "tmp_func." + build_func.output_format
        )
        try:
            func = build(
                measure_input.sch.sch.func,
                target=measure_input.task.target,
                target_host=measure_input.task.target_host,
            )
            func.export_library(filename, build_func)
        except Exception:  # pylint: disable=broad-except
            vprint(verbose, ".E", end="")  # Build error
            error_no = MeasureErrorNo.COMPILE_HOST
            error_msg = make_error_msg()
        else:
            vprint(verbose, ".", end="")  # Build success
        time_cost = time.time() - tic
        return filename, error_no, error_msg, time_cost

    try:
        return call_func_with_timeout(timeout, timed_func)
    except TimeoutError:
        vprint(verbose, ".T", end="")  # Build timeout
        return "", MeasureErrorNo.BUILD_TIMEOUT, "", timeout


def local_builder_wrapped_worker(
    index: int,
) -> BuildResultType:
    """
    Build function of LocalBuilder to be ran in the Builder thread pool.

    Parameters
    ----------
    index : int
        The MeasureInput index to be processed by the current Builder thread.

    Returns
    -------
    res : BuildResult
        The build result of this Builder thread.
    """
    global LOCAL_BUILDER_WORKER_ARGS
    return local_builder_worker(index, *LOCAL_BUILDER_WORKER_ARGS)


########## Worker of RPCRunner ##########

# We use fork and a global variable to copy arguments between processes.
# This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
RPC_RUNNER_WORKER_ARGS = None


@register_func("meta_schedule.rpc_runner.run")
def rpc_runner_run(
    measure_inputs: List[MeasureInput],
    build_results: List[BuildResult],
    tracker: str,
    priority: int = 1,
    n_parallel: int = 1,  # TODO(@junrushao1994): perhaps auto detect?
    timeout: int = 10,
    number: int = 3,
    repeat: int = 1,
    min_repeat_ms: int = 0,
    cooldown_interval: float = 0.0,
    enable_cpu_cache_flush: bool = False,
    verbose: int = 1,
) -> List[MeasureResult]:
    """Run function of RPCRunner to test the performance of the input BuildResults.

    Parameters
    ----------
    inputs : List[MeasureInput]
        The MeasureInputs to be measured.
    build_results : List[BuildResult]
        The BuildResults to be measured.
    key : str
        The key of the device registered in the RPC tracker.
    host : str
        The host address of the RPC Tracker.
    port : int
        The port of RPC Tracker.
    priority : int = 1
        The priority of this run request, larger is more prior.
    n_parallel : int = 1
        The number of tasks run in parallel.
    timeout : int = 10
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 0
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool = False
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program measuring.

    Returns
    -------
    res : List[MeasureResult]
        The measure results of these MeasureInputs.
    """
    global RPC_RUNNER_WORKER_ARGS
    RPC_RUNNER_WORKER_ARGS = (
        measure_inputs,
        build_results,
        tracker,
        priority,
        timeout,
        number,
        repeat,
        min_repeat_ms,
        cooldown_interval,
        enable_cpu_cache_flush,
        verbose,
    )

    assert len(measure_inputs) == len(
        build_results
    ), "Measure input size should be equal to build results"
    pool = NoDaemonPool(n_parallel)
    indices = [i for i, _ in enumerate(measure_inputs)]
    results = pool.map(rpc_runner_wrapped_worker, indices)
    pool.terminate()
    pool.join()
    del pool
    results = [MeasureResult(*item) for item in results]
    vprint(verbose, "", end="\n")
    return results


def rpc_runner_worker(
    index: int,
    measure_inputs: List[MeasureInput],
    build_results: List[BuildResult],
    tracker: str,
    priority: int,
    timeout: int,
    number: int,
    repeat: int,
    min_repeat_ms: int,
    cooldown_interval: float,
    _enable_cpu_cache_flush: bool,
    verbose: int,
) -> MeasureResultType:
    """ RPC worker for ProgramRunner """
    measure_input = measure_inputs[index]
    build_result = build_results[index]

    if build_result.error_no != MeasureErrorNo.NO_ERROR:
        return (
            (MAX_TIME_COST,),
            build_result.error_no,
            build_result.error_msg,
            build_result.time_cost,
            time.time(),
        )

    def timed_func():
        tic = time.time()

        costs = (MAX_TIME_COST,)
        error_no = 0
        error_msg = ""
        all_cost = MAX_TIME_COST
        timestamp = -1.0

        try:
            try:
                # upload built module
                remote = request_remote(tracker, priority, timeout)
                remote.upload(build_result.filename)
                func = remote.load_module(os.path.split(build_result.filename)[1])
                # TODO(@junrushao1994): rebase and fix this
                ctx = remote.context(str(measure_input.task.target), 0)
                time_f = func.time_evaluator(
                    func_name=func.entry_name,
                    ctx=ctx,
                    number=number,
                    repeat=repeat,
                    min_repeat_ms=min_repeat_ms,
                    # TODO(@junrushao1994): rebase and enable this
                    # f_preproc="cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else "",
                )
            except Exception:  # pylint: disable=broad-except
                vprint(verbose, "*E", end="")  # Device compilation error
                error_no = MeasureErrorNo.COMPILE_DEVICE
                error_msg = make_error_msg()
                raise
            try:
                args = realize_arguments(remote, ctx, measure_input.task.func)
                ctx.sync()
                costs = time_f(*args).results
                # clean up remote files
                remote.remove(build_result.filename)
                remote.remove(os.path.splitext(build_result.filename)[0] + ".so")
                remote.remove("")
            except Exception:  # pylint: disable=broad-except
                vprint(verbose, "*E", end="")  # Runtime Error
                error_no = MeasureErrorNo.RUNTIME_DEVICE
                error_msg = make_error_msg()
                raise
        except Exception:  # pylint: disable=broad-except
            pass
        else:
            vprint(verbose, "*", end="")
        shutil.rmtree(os.path.dirname(build_result.filename))
        timestamp = time.time()
        all_cost = timestamp - tic + build_result.time_cost
        time.sleep(cooldown_interval)
        return costs, error_no, error_msg, all_cost, timestamp

    try:
        return call_func_with_timeout(timeout, timed_func)
    except TimeoutError:
        vprint(verbose, "*T", end="")  # Run timeout
        return (
            (MAX_TIME_COST,),
            MeasureErrorNo.RUN_TIMEOUT,
            "",
            build_result.time_cost + timeout,
            time.time(),
        )


def rpc_runner_wrapped_worker(
    index: int,
) -> MeasureResultType:
    """Function to be ran in the RPCRunner thread pool.

    Parameters
    ----------
    index : int
        The MeasureInput and BuildResult index to be processed by the current Runner thread.

    Returns
    -------
    res : MeasureResult
        The measure result of this Runner thread.
    """
    global RPC_RUNNER_WORKER_ARGS
    return rpc_runner_worker(index, *RPC_RUNNER_WORKER_ARGS)
