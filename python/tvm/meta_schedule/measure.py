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
from typing import Optional, List, Tuple, Callable
from tvm._ffi import register_object, register_func
from tvm.driver import build
from tvm.runtime import Object, ndarray
from .schedule import Schedule
from tvm.contrib import tar as build_func_tar, ndk as build_func_ndk
import time
import tempfile
from .measure_util import (
    make_error_msg,
    NoDaemonPool,
    call_func_with_timeout,
    request_remote,
)
import os.path as osp
import shutil


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


########## Five basic classes ##########


@register_object("meta_schedule.MeasureInput")
class MeasureInput(Object):

    # TODO

    """Store the input of a measurement.

    Parameters
    ----------
    sch : Schedule
        The meta schedule object
    """

    def __init__(self, sch: Schedule):
        self.__init_handle_by_constructor__(
            _ffi_api.MeasureInput, sch  # pylint: disable=undefined-variable
        )


@register_object("meta_schedule.BuildResult")
class BuildResult(Object):
    """Store the result of a build.

    Parameters
    ----------
    filename : Optional[str]
        The filename of built binary file.
    error_no : int
        The error code.
    error_msg : Optional[str]
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
        filename: Optional[str],
        error_no: int,
        error_msg: Optional[str],
        time_cost: float,
    ):
        filename = filename if filename is not None else ""
        error_msg = error_msg if error_msg is not None else ""

        self.__init_handle_by_constructor__(
            _ffi_api.BuildResult,  # pylint: disable=undefined-variable
            filename,
            error_no,
            error_msg,
            time_cost,
        )


@register_object("meta_schedule.MeasureResult")
class MeasureResult(Object):
    """Store the results of a measurement.

    Parameters
    ----------
    costs : List[float]
        The time costs of execution.
    error_no : int
        The error code.
    error_msg : Optional[str]
        The error message if there is any error.
    all_cost : float
        The time cost of build and run.
    timestamp : float
        The time stamps of this measurement.
    """

    costs: List[float]
    error_no: int
    error_msg: Optional[str]
    all_cost: float
    timestamp: float

    def __init__(
        self,
        costs: List[float],
        error_no: int,
        error_msg: Optional[str],
        all_cost: float,
        timestamp: float,
    ):
        error_msg = error_msg if error_msg is None else ""
        self.__init_handle_by_constructor__(
            _ffi_api.MeasureResult,  # pylint: disable=undefined-variable
            costs,
            error_no,
            error_msg,
            all_cost,
            timestamp,
        )


@register_object("meta_schedule.ProgramBuilder")
class ProgramBuilder(Object):
    """ The base class of ProgramBuilders. """

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
        return _ffi_api.ProgramBuilderBuild(  # pylint: disable=undefined-variable
            self, measure_inputs, verbose
        )


@register_object("meta_schedule.ProgramRunner")
class ProgramRunner(Object):
    """ The base class of ProgramRunners. """

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
        return _ffi_api.ProgramRunnerRun(  # pylint: disable=undefined-variable
            self, measure_inputs, build_results, verbose
        )


########## LocalBuilder ##########


# We use fork and a global variable to copy arguments between processes.
# This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
LOCAL_BUILDER_WORKER_ARGS = None


@register_func("meta_schedule.local_builder.build")
def local_builder_build(
    measure_inputs: List[MeasureInput],
    timeout: int,
    n_parallel: int,
    build_func: str = "default",
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
    build_func : str = 'default'
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
    results = pool.map(local_builder_worker, indices)
    pool.terminate()
    pool.join()
    del pool
    results = [BuildResult(*item) for item in results]
    return results


def local_builder_worker(
    index: int,
) -> Tuple[
    Optional[str],  # filename
    int,  # error_no
    Optional[str],  # error_msg
    float,  # time_cost
]:
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

    measure_inputs: List[MeasureInput]
    build_func: str
    timeout: int
    verbose: int
    measure_inputs, build_func, timeout, verbose = LOCAL_BUILDER_WORKER_ARGS

    if build_func == "default":
        build_func = build_func_tar.tar
    elif build_func == "ndk":
        build_func = build_func_ndk.create_shared
    else:
        raise ValueError("Invalid build_func" + build_func)

    def timed_func():
        tic = time.time()
        measure_input = measure_inputs[index]

        error_no = MeasureErrorNo.NO_ERROR
        error_msg = None

        if error_no == 0:
            dirname = tempfile.mkdtemp()
            filename = osp.join(dirname, "tmp_func." + build_func.output_format)

            try:
                func = build(measure_input.sch, target="llvm")
                func.export_library(filename, build_func)
            except Exception:  # pylint: disable=broad-except
                error_no = MeasureErrorNo.COMPILE_HOST
                error_msg = make_error_msg()
        else:
            filename = ""

        if verbose >= 1:
            if error_no == MeasureErrorNo.NO_ERROR:
                print(".", end="")
            else:
                print(".E", end="")  # Build error
        tok = time.time()
        return filename, error_no, error_msg, tok - tic

    res = call_func_with_timeout(timeout, timed_func)
    if isinstance(res, TimeoutError):
        if verbose >= 1:
            print(".T", end="")  # Build timeout
        res = None, [], MeasureErrorNo.BUILD_TIMEOUT, None, timeout

    return res


########## RPCRunner ##########

# We use fork and a global variable to copy arguments between processes.
# This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
RPC_RUNNER_WORKER_ARGS = None


@register_func("auto_scheduler.rpc_runner.run")
def rpc_runner_run(
    measure_inputs: List[MeasureInput],
    build_results: List[BuildResult],
    key: str,
    host: str,
    port: int,
    priority: int = 1,
    n_parallel: int = 1,
    timeout: int = 10,
    number: int = 3,
    repeat: int = 1,
    min_repeat_ms: int = 0,
    cooldown_interval: float = 0.0,
    enable_cpu_cache_flush: bool = False,
    verbose: int = 1,
):
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
        key,
        host,
        port,
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
    results = pool.map(rpc_runner_worker, indices)
    pool.terminate()
    pool.join()
    del pool
    results = [MeasureResult(*item) for item in results]
    if verbose >= 1:
        print("")
    return results


def rpc_runner_worker(
    index: int,
) -> Tuple[
    List[float],  # costs
    int,  # error_no
    Optional[str],  # error_msg
    float,  # all_cost
    float,  # timestamp
]:
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
    measure_inputs: List[MeasureInput]
    build_results: List[BuildResult]
    key: str
    host: str
    port: int
    priority: int
    timeout: int
    number: int
    repeat: int
    min_repeat_ms: int
    cooldown_interval: float
    enable_cpu_cache_flush: bool
    verbose: int
    (
        measure_inputs,
        build_results,
        key,
        host,
        port,
        priority,
        timeout,
        number,
        repeat,
        min_repeat_ms,
        cooldown_interval,
        enable_cpu_cache_flush,
        verbose,
    ) = RPC_RUNNER_WORKER_ARGS

    max_float = (
        1e10  # We use 1e10 instead of sys.float_info.max for better readability in log
    )
    measure_input = measure_inputs[index]
    build_result = build_results[index]

    if build_result.error_no != MeasureErrorNo.NO_ERROR:
        return (
            (max_float,),
            build_result.error_no,
            build_result.error_msg,
            build_result.time_cost,
            time.time(),
        )

    def timed_func():
        tic = time.time()
        error_no = 0
        error_msg = None
        try:
            # upload built module
            remote = request_remote(key, host, port, priority, timeout)
            remote.upload(build_result.filename)
            func = remote.load_module(osp.split(build_result.filename)[1])
            ctx = remote.context(str(measure_input.task.target), 0)
            # Limitation:
            # We can not get PackFunction directly in the remote mode as it is wrapped
            # under the std::function. We could lift the restriction later once we fold
            # the PackedFunc as an object. Currently, we pass function name to work
            # around it.
            f_prepare = (
                "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
            )
            time_f = func.time_evaluator(
                func.entry_name,
                ctx,
                number=number,
                repeat=repeat,
                min_repeat_ms=min_repeat_ms,
                f_preproc=f_prepare,
            )
        # pylint: disable=broad-except
        except Exception:
            costs = (max_float,)
            error_no = MeasureErrorNo.COMPILE_DEVICE
            error_msg = make_error_msg()

        if error_no == 0:
            try:
                # TODO(@junrushao1994): rework this
                args = [ndarray.empty(x.shape, x.dtype, ctx) for x in build_result.args]
                try:
                    random_fill = remote.get_function("tvm.contrib.random.random_fill")
                except AttributeError:
                    raise AttributeError(
                        "Please make sure USE_RANDOM is ON in the config.cmake "
                        "on the remote devices"
                    )
                for arg in args:
                    random_fill(arg)
                ctx.sync()

                costs = time_f(*args).results
                # clean up remote files
                remote.remove(build_result.filename)
                remote.remove(osp.splitext(build_result.filename)[0] + ".so")
                remote.remove("")
            # pylint: disable=broad-except
            except Exception:
                costs = (max_float,)
                error_no = MeasureErrorNo.RUNTIME_DEVICE
                error_msg = make_error_msg()

        shutil.rmtree(osp.dirname(build_result.filename))
        toc = time.time()

        time.sleep(cooldown_interval)
        if verbose >= 1:
            if error_no == MeasureErrorNo.NO_ERROR:
                print("*", end="")
            else:
                print("*E", end="")  # Run error

        return costs, error_no, error_msg, toc - tic + build_result.time_cost, toc

    res = call_func_with_timeout(timeout, timed_func)

    if isinstance(res, TimeoutError):
        if verbose >= 1:
            print("*T", end="")  # Run timeout
        res = (
            (max_float,),
            MeasureErrorNo.RUN_TIMEOUT,
            None,
            build_result.time_cost + timeout,
            time.time(),
        )
    return res
