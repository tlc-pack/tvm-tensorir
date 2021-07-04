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
# pylint: disable=missing-function-docstring
import multiprocessing
import multiprocessing.pool
import signal
import traceback
from threading import Thread
from typing import Tuple

import psutil

from tvm import rpc
from tvm.runtime import ndarray
from tvm.tir import PrimFunc

MAX_ERROR_MSG_LEN = int(1e9)


def make_error_msg() -> str:
    """ Get the error message from traceback. """
    error_msg = str(traceback.format_exc())
    if len(error_msg) > MAX_ERROR_MSG_LEN:
        error_msg = (
            error_msg[: MAX_ERROR_MSG_LEN // 2]
            + "\n...\n"
            + error_msg[-MAX_ERROR_MSG_LEN // 2 :]
        )
    return error_msg


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


def parse_tracker_key(tracker: str) -> Tuple[str, int, str]:
    result = tracker.split(":")
    if len(result) != 3:
        raise ValueError(
            f"Unable to parse tracker: {tracker}. Please use the correct format "
            "'host:port:device_key', e.g. '0.0.0.0:9089:local'"
        )
    host, port, device_key = map(str.strip, result)
    try:
        port = int(port)
    except ValueError:
        raise ValueError(
            f"Unable to parse tracker: {tracker}. The port should be an integer. "
            "Please use the correct format 'host:port:device_key', e.g. '0.0.0.0:9089:local'"
        )
    return host, port, device_key


def request_remote(
    tracker: str,
    priority: int = 1,
    timeout: int = 60,
) -> rpc.RPCSession:
    """Request a remote session.

    Parameters
    ----------
    device_key : str
        The device key of registered device in tracker.
    host : Optional[str]
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST".
    port : Optional[int]
        The port of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT".
    priority : int = 1
        The priority of this request, larger is more prior.
    timeout : int = 60
        The timeout of this session in second.

    Returns
    -------
    remote : RPCSession
        The connected remote RPCSession.
    """
    # connect to the tracker
    host, port, device_key = parse_tracker_key(tracker)
    tracker = rpc.connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority, session_timeout=timeout)
    return remote


def check_remote(tracker, priority=100, timeout=10):
    """
    Check the availability of a remote device.

    Parameters
    ----------
    device_key: str
        device key of registered device in tracker.
    host: Optional[str]
        The host address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_HOST".
    port: Optional[int]
        The port address of rpc tracker.
        If is none, will use environment variable "TVM_TRACKER_PORT".
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
        request_remote(tracker, priority)

    t = Thread(target=_check)
    t.start()
    t.join(timeout)
    return not t.is_alive()


def realize_arguments(_remote, ctx, func: PrimFunc):
    args = []
    for arg in func.params:
        if arg.dtype == "handle":
            buffer = func.buffer_map[arg]
            args.append(ndarray.empty(shape=buffer.shape, dtype=buffer.dtype, ctx=ctx))
        else:
            raise NotImplementedError(
                "Unsupported type in realize_arguments: " + str(arg.dtype)
            )
    # TODO(@junrushao1994): rebase and enable this
    # try:
    #     f_random_fill = remote.get_function("tvm.contrib.random.random_fill")
    # except AttributeError:
    #     raise AttributeError(
    #         "Please make sure USE_RANDOM is ON in the config.cmake "
    #         "on the remote devices"
    #     )
    # for array in ndarrays:
    #     f_random_fill(array)
    return args


def cpu_count():
    return multiprocessing.cpu_count()


def vprint(verbose, content, end):
    if verbose >= 1:
        print(content, end=end)
