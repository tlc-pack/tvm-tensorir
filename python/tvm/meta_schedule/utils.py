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
""" Utility functions used for measuring """
import json
import multiprocessing
import multiprocessing.pool
import os
import shutil
import signal
import tempfile
import time
import traceback
from threading import Thread
from typing import Any, Callable, List, Tuple

import psutil
from tvm import ir, rpc
from tvm._ffi import register_func
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
from tvm.contrib import ndk as build_func_ndk
from tvm.contrib import tar as build_func_tar
from tvm.driver import build as tvm_build
from tvm.runtime import NDArray, Device, ndarray
from tvm.tir import FloatImm, IntImm, PrimFunc

from .measure_record import BuildResult, MeasureErrorNo, MeasureInput, MeasureResult

MAX_ERROR_MSG_LEN = int(1e9)
# The maximum possible cost used to indicate the cost for timeout
# We use 1e10 instead of sys.float_info.max for better readability in log
MAX_TIME_COST = 1e10


def make_error_msg() -> str:
    """ Get the error message from traceback. """
    error_msg = str(traceback.format_exc())
    if len(error_msg) > MAX_ERROR_MSG_LEN:
        error_msg = (
            error_msg[: MAX_ERROR_MSG_LEN // 2] + "\n...\n" + error_msg[-MAX_ERROR_MSG_LEN // 2 :]
        )
    return error_msg


@register_func("meta_schedule._serialize_json")
def serialize_json(record: Any) -> str:
    """Serialize the record to JSON"""

    def to_native_py(obj):
        if isinstance(obj, ir.Array):
            return list(to_native_py(item) for item in obj)
        if isinstance(obj, ir.Map):
            return {
                to_native_py(k): to_native_py(v) for k, v in obj.items()
            }  # pylint: disable=unnecessary-comprehension)
        if isinstance(obj, (IntImm, FloatImm)):
            return obj.value
        return obj

    record = to_native_py(record)
    return json.dumps(record).strip()


@register_func("meta_schedule._deserialize_tuning_records")
def _deserialize_tuning_records(records: str) -> Any:
    """Deserialize the record from JSON"""
    return json.loads(records)  # pylint: disable=c-extension-no-member


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class NoDaemonPool(multiprocessing.pool.Pool):
    """A no daemon pool version of multiprocessing.Pool.
    This allows us to start new processes inside the worker function"""

    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super().__init__(*args, **kwargs)

    def __reduce__(self):
        pass


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
    """kill all child processes recursively"""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in children:
        try:
            process.send_signal(sig)
        except psutil.NoSuchProcess:
            return


def call_func_with_timeout(timeout, func, args=(), kwargs=None):
    """Call a function with timeout"""

    def func_wrapper(que):
        if kwargs:
            que.put(func(*args, **kwargs))
        else:
            que.put(func(*args))

    que = multiprocessing.Queue(2)
    process = multiprocessing.Process(target=func_wrapper, args=(que,))
    process.start()
    process.join(timeout)

    try:
        res = que.get(block=False)
    except multiprocessing.queues.Empty:
        res = TimeoutError()

    # clean queue and process
    kill_child_processes(process.pid)
    process.terminate()
    process.join()
    que.close()
    que.join_thread()
    del process
    del que

    if isinstance(res, TimeoutError):
        raise TimeoutError

    return res


def request_remote(
    key: str,
    host: str,
    port: int,
    priority: int = 1,
    timeout: int = 60,
) -> Tuple[rpc.TrackerSession, rpc.RPCSession]:
    """Request a remote session.

    Parameters
    ----------
    key : str
        The key of the device registered in the RPC tracker.
    host : str
        The host address of the RPC Tracker.
    port : int
        The port of RPC Tracker.
    priority : int = 1
        The priority of this request, larger is more prior.
    timeout : int = 60
        The timeout of this session in second.

    Returns
    -------
    tracker : TrackerSession
        The tracker session
    remote : RPCSession
        The connected remote RPCSession.
    """
    # connect to the tracker
    tracker = rpc.connect_tracker(host, port)
    remote = tracker.request(key, priority=priority, session_timeout=timeout)
    return tracker, remote


def check_remote_servers(
    key: str,
    host: str,
    port: int,
    priority: int = 100,
    timeout: int = 10,
) -> int:
    """Check the availability of remote servers.

    Parameters
    ----------
    key : str
        The key of the device registered in the RPC tracker.
    host : str
        The host address of the RPC Tracker.
    port : int
        The port of RPC Tracker.
    priority: int = 100
        The priority of this request, larger is more prior.
    timeout: int = 10
        The timeout of this check in seconds.

    Returns
    -------
    server_count: int
        True if can find available device.
    """

    tracker: rpc.TrackerSession = None

    def _check():
        nonlocal tracker
        tracker = request_remote(key, host, port, priority)[0]

    t = Thread(target=_check)
    t.start()
    t.join(timeout)
    if t.is_alive() or tracker is None:
        raise RuntimeError(
            "Cannot get remote devices from the tracker. "
            "Please check the status of tracker via "
            "'python -m tvm.exec.query_rpc_tracker --port [THE PORT YOU USE]' "
            "and make sure you have free devices on the queue status. "
            "Besides hard coding in the program, you may also specify it by setting "
            "environment variables TVM_TRACKER_HOST, TVM_TRACKER_PORT and TVM_TRACKER_KEY"
        )
    tracker_summary = tracker.summary()
    server_count = 0
    for item in tracker_summary["server_info"]:
        _, item_key = item["key"].split(":")  # 'server:rasp3b` -> 'rasp3b'
        if item_key == key:
            server_count += 1
    print(f"Get {server_count} RPC servers for measurement!")
    return server_count


def check_remote(key: str, host: str, port: int, priority: int = 100, timeout: int = 10) -> bool:
    """
    Check the availability of a remote device.

    Parameters
    ----------
    key : str
        The key of the device registered in the RPC tracker.
    host : str
        The host address of the RPC Tracker.
    port : int
        The port of RPC Tracker.
    priority: int = 100
        The priority of this request, larger is more prior.
    timeout: int = 10
        The timeout of this check in seconds.

    Returns
    -------
    available: bool
        True if can find available device.
    """

    def _check():
        request_remote(key, host, port, priority)

    t = Thread(target=_check)
    t.start()
    t.join(timeout)
    return not t.is_alive()


def realize_arguments(
    remote: rpc.RPCSession,
    ctx: Device,
    func: PrimFunc,
) -> List[NDArray]:
    """
    Check the availability of a remote device.

    Parameters
    ----------
    _remote: RPCSession
        The connected remote RPCSession
    ctx: Device
        The context that ndarrays to be created on the remote
    func: PrimFunc
        The PrimFunc to be run on the remote

    Returns
    -------
    args: List[NDArray]
        A list of arguments fed to the TVM runtime module built
    """
    args = []
    ndarrays = []

    for arg in func.params:
        if arg.dtype == "handle":
            buffer = func.buffer_map[arg]
            array = ndarray.empty(shape=buffer.shape, dtype=buffer.dtype, device=ctx)
            args.append(array)
            ndarrays.append(array)
        else:
            raise NotImplementedError("Unsupported type in realize_arguments: " + str(arg.dtype))
    try:
        f_random_fill = remote.get_function("tvm.contrib.random.random_fill")
    except AttributeError as error:
        raise AttributeError(
            "Please make sure USE_RANDOM is ON in the config.cmake " "on the remote devices"
        ) from error
    for array in ndarrays:
        f_random_fill(array)
    return args


@register_func("meta_schedule._cpu_count")
def cpu_count(logical=True) -> int:
    """
    Check the number of cpus available on the local device

    Returns
    -------
    cpu_count: int
        The number of cpus available on the local device
    """
    return psutil.cpu_count(logical=logical)


def vprint(verbose: int, content: str, end: str) -> None:
    """
    Print the content if verbose level >= 1

    Parameters
    ----------
    verbose: int
        The verbosity level
    content: str
        The content to be printed
    end: str
        The end of the print function used in python print
    """
    if verbose >= 1:
        print(content, end=end)


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
) -> BuildResult.TYPE:
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
    if measure_input.task.target.kind.name == "cuda":
        set_cuda_target_arch(measure_input.task.target.attrs["arch"])

    def timed_func() -> BuildResult.TYPE:
        tic = time.time()
        # return values
        filename = ""
        error_no = MeasureErrorNo.NO_ERROR
        error_msg = ""
        time_cost = 1e9
        # create temporary path
        filename = os.path.join(tempfile.mkdtemp(), "tmp_func." + build_func.output_format)
        try:
            func = tvm_build(
                measure_input.sch.mod["main"],
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
) -> BuildResult.TYPE:
    """
    Build function of LocalBuilder to be ran in the Builder thread pool.

    Parameters
    ----------
    index : int
        The MeasureInput index to be processed by the current Builder thread.

    Returns
    -------
    res : BuildResult.TYPE
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
    f_create_args=None,
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
    f_create_args: Callable[[Device], List[NDArray]] = None
        Optional callback to create arguments for functions to measure. This can be used for sparse
        workloads when we cannot use random tensors for measurement.
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
        f_create_args,
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
    key: str,
    host: str,
    port: int,
    priority: int,
    timeout: int,
    number: int,
    repeat: int,
    min_repeat_ms: int,
    cooldown_interval: float,
    enable_cpu_cache_flush: bool,
    f_create_args: Callable[[Device], List[NDArray]],
    verbose: int,
) -> MeasureResult.TYPE:
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
                _, remote = request_remote(key, host, port, priority, timeout)
                remote.upload(build_result.filename)
                func = remote.load_module(os.path.split(build_result.filename)[1])
                dev = remote.device(measure_input.task.target.kind.name, 0)
                time_f = func.time_evaluator(
                    func_name=func.entry_name,
                    dev=dev,
                    number=number,
                    repeat=repeat,
                    min_repeat_ms=min_repeat_ms,
                    f_preproc="cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else "",
                )
            except Exception:  # pylint: disable=broad-except
                vprint(verbose, "*E", end="")  # Device compilation error
                error_no = MeasureErrorNo.COMPILE_DEVICE
                error_msg = make_error_msg()
                raise
            try:
                # TODO(@junrushao1994): remove the hardcode
                mtriple = str(measure_input.task.target.attrs.get("mtriple", ""))
                if mtriple in ["aarch64-linux-gnu", "armv8l-linux-gnueabihf"]:
                    rpc_eval_repeat = 3
                else:
                    rpc_eval_repeat = 1
                if f_create_args is not None:
                    args_set = [f_create_args(dev) for _ in range(rpc_eval_repeat)]
                else:
                    args_set = [
                        realize_arguments(remote, dev, measure_input.sch.mod["main"])
                        for _ in range(rpc_eval_repeat)
                    ]
                dev.sync()
                costs = sum([time_f(*args).results for args in args_set], ())
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
) -> MeasureResult.TYPE:
    """Function to be ran in the RPCRunner thread pool.

    Parameters
    ----------
    index : int
        The MeasureInput and BuildResult index to be processed by the current Runner thread.

    Returns
    -------
    result : MeasureResult.TYPE
        The measure result of this Runner thread.
    """
    global RPC_RUNNER_WORKER_ARGS
    return rpc_runner_worker(index, *RPC_RUNNER_WORKER_ARGS)
