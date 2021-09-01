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
"""RPC-related utils"""

import os
from threading import Thread
from typing import NamedTuple, Optional

from tvm import rpc


class RPCConfig(NamedTuple):
    """RPC configuration

    Parameters
    ----------
    tracker_host: str
        Host of the RPC Tracker
    tracker_port: int
        Port of the RPC Tracker
    tracker_key: str
        Key of the Tracker
    session_timeout_sec: float
        Timeout of the RPC session
    session_priority: int
        Priority of the RPC session
    """

    tracker_host: Optional[str] = None
    tracker_port: Optional[str] = None
    tracker_key: Optional[str] = None
    session_priority: int = 1
    session_timeout_sec: int = 10

    def _sanity_check(self) -> None:
        err_str = (
            "RPCConfig.{0} is not provided. Please provide it explicitly,"
            "or set environment variable {1}"
        )
        if self.tracker_host is None:
            raise ValueError(err_str.format("tracker_host", "TVM_TRACKER_HOST"))
        if self.tracker_port is None:
            raise ValueError(err_str.format("tracker_port", "TVM_TRACKER_PORT"))
        if self.tracker_key is None:
            raise ValueError(err_str.format("tracker_key", "TVM_TRACKER_KEY"))

    @staticmethod
    def _parse(config: Optional["RPCConfig"]) -> "RPCConfig":
        if config is None:
            config = RPCConfig()
        config = RPCConfig(
            tracker_host=config.tracker_host or os.environ.get("TVM_TRACKER_HOST", None),
            tracker_port=config.tracker_host or os.environ.get("TVM_TRACKER_PORT", None),
            tracker_key=config.tracker_host or os.environ.get("TVM_TRACKER_KEY", None),
            session_priority=config.session_priority,
            session_timeout_sec=config.session_timeout_sec,
        )
        config._sanity_check()  # pylint: disable=protected-access
        return config

    def connect_tracker(self) -> rpc.TrackerSession:
        """Connect to the tracker

        Returns
        -------
        tracker : TrackerSession
            The connected tracker session
        """
        tracker: Optional[rpc.TrackerSession] = None

        def _connect():
            nonlocal tracker
            tracker = rpc.connect_tracker(self.tracker_host, self.tracker_port)

        t = Thread(target=_connect)
        t.start()
        t.join(self.session_timeout_sec)
        if t.is_alive() or tracker is None:
            raise ValueError(
                "Unable to connect to the tracker using the following configuration:\n"
                f"    tracker host: {self.tracker_host}\n"
                f"    tracker port: {self.tracker_port}\n"
                f"    timeout (sec): {self.session_timeout_sec}\n"
                "Please check the tracker status via the following command:\n"
                "     python3 -m tvm.exec.query_rpc_tracker "
                f"--host {self.tracker_host} --port {self.tracker_port}"
            )
        return tracker

    def connect_server(self) -> rpc.RPCSession:
        """Connect to the server

        Returns
        -------
        session : RPCSession
            The connected rpc session
        """
        tracker = self.connect_tracker()
        session: rpc.RPCSession = tracker.request(
            key=self.tracker_key,
            priority=self.session_priority,
            session_timeout=self.session_timeout_sec,
        )
        return session

    def count_num_servers(self, allow_missing=True) -> int:
        """Count the number of servers available in the tracker

        Parameters
        ----------
        allow_missing : bool
            Whether to allow no server to be found.

        Returns
        -------
        num_servers : int
            The number of servers
        """
        tracker = self.connect_tracker()
        tracker_summary = tracker.summary()
        result: int = 0
        for item in tracker_summary["server_info"]:
            _, item_key = item["key"].split(":")
            if item_key == self.tracker_key:
                result += 1
        if result == 0 and not allow_missing:
            raise ValueError(
                "Unable to find servers with the specific key using the following configuration:\n"
                f"    tracker host: {self.tracker_host}\n"
                f"    tracker port: {self.tracker_port}\n"
                f"    tracker key: {self.tracker_key}\n"
                f"    timeout (sec): {self.session_timeout_sec}\n"
                "Please check the tracker status via the following command:\n"
                "     python3 -m tvm.exec.query_rpc_tracker "
                f"--host {self.tracker_host} --port {self.tracker_port}\n"
                f'and look for key: "{self.tracker_key}"'
            )
        return result
