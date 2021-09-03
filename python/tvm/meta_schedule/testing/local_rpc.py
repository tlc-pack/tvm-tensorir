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
import time
from typing import Optional

import tvm.rpc
import tvm.rpc.tracker


class Tracker:
    """Tracker

    Parameters
    ----------
    tracker : tvm.rpc.tracker.Tracker
        The tracker
    host : str
        The host url
    port : int
        The port number
    """

    tracker: tvm.rpc.tracker.Tracker
    host: str
    port: int

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9190,
        port_end: int = 9999,
        silent: bool = False,
    ) -> None:
        """Constructor

        Parameters
        ----------
        host : str
            The host url
        port : int
            The port number
        port_end : int
            The end port number
        silent : bool
            Whether run in silent mode
        """
        self.tracker = tvm.rpc.tracker.Tracker(
            host,
            port=port,
            port_end=port_end,
            silent=silent,
        )
        self.host = self.tracker.host
        self.port = self.tracker.port
        time.sleep(0.5)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.tracker.terminate()
        del self.tracker


class Server:
    """Server

    Parameters
    ----------
    server : tvm.rpc.Server
    """

    server: tvm.rpc.Server
    key: str

    def __init__(
        self,
        tracker: Tracker,
        key: str = "rpc.testing",
        host: str = "0.0.0.0",
        port: int = 9190,
        port_end: int = 9999,
        load_library: Optional[str] = None,
        custom_addr: Optional[str] = None,
        silent: bool = False,
        no_fork: bool = False,
    ) -> None:
        """Constructor

        Parameters
        ----------
        tracker : Tracker
            The tracker
        key : str
            The tracker key
        host : str
            The host url
        port : int
            The port number
        port_end : int
            The end port number
        load_library : str, optional
            Additional library to load
        custom_addr : str, optional
            custom rpc address
        silent : bool
            Whether run in silent mode
        no_fork : bool
            Whether use fork instead of spawn
        """
        self.server = tvm.rpc.Server(
            host=host,
            port=port,
            port_end=port_end,
            is_proxy=False,
            tracker_addr=(tracker.host, tracker.port),
            key=key,
            load_library=load_library,
            custom_addr=custom_addr,
            silent=silent,
            no_fork=no_fork,
        )
        self.key = key
        time.sleep(0.5)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.server.terminate()
        del self.server
