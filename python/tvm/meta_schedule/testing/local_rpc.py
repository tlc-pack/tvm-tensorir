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

import tvm.rpc
import tvm.rpc.tracker


class Tracker:

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
        self.tracker = tvm.rpc.tracker.Tracker(
            host,
            port=port,
            port_end=port_end,
            silent=silent,
        )
        self.host = self.tracker.host
        self.port = self.tracker.port


class Server:

    server: tvm.rpc.Server

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
