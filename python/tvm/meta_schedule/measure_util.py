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
from typing import Optional

import traceback
import multiprocessing
import multiprocessing.pool
import signal
import psutil
import os

from tvm import rpc

MAX_ERROR_MSG_LEN = 512


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

    return res


def request_remote(
    device_key: str,
    host: Optional[str] = None,
    port: Optional[int] = None,
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
    host = host or os.environ["TVM_TRACKER_HOST"]
    port = port or int(os.environ["TVM_TRACKER_PORT"])
    tracker = rpc.connect_tracker(host, port)
    remote = tracker.request(device_key, priority=priority, session_timeout=timeout)
    return remote
