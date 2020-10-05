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
import os
from typing import List, Optional

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api
from .measure_record import BuildResult, MeasureInput, MeasureResult
from .utils import check_remote, cpu_count
from .schedule import Schedule

########## ProgramBuilder ##########


@register_object("meta_schedule.ProgramBuilder")
class ProgramBuilder(Object):
    """ The base class of ProgramBuilders. """

    n_parallel: int
    timeout: int

    @staticmethod
    def create(name: str) -> "ProgramBuilder":
        if name == "local":
            return LocalBuilder()
        raise ValueError("Unknown name of program builder: " + name)

    def build(
        self,
        measure_inputs: List[MeasureInput],
        verbose: int = 1,
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
            self,
            measure_inputs,
            verbose,
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
    def create(name: str) -> "ProgramRunner":
        if name == "rpc" or name.startswith("rpc://"):
            return RPCRunner.create(name)
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
            self,
            measure_inputs,
            build_results,
            verbose,
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
    def create(name: str) -> "RPCRunner":
        RPC_PREFIX = "rpc://"  # pylint: disable=invalid-name
        if name == "rpc":
            return RPCRunner()
        if not name.startswith(RPC_PREFIX):
            raise ValueError("Invalid RPC config: " + name)
        name = name[len(RPC_PREFIX) :].strip()
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
    """The base class of measurement callback functions."""


########## ProgramMeasurer ##########


@register_object("meta_schedule.ProgramMeasurer")
class ProgramMeasurer(Object):
    """Measurer that measures the time costs of tvm programs.

    Parameters
    ----------
    builder : ProgramBuilder
        The ProgramBuilder to build each program
    runner : ProgramRunner
        The ProgramRunner to measure each program
    callbacks : List[MeasureCallback]
        MeasureCallback to be called after each measure batch
    num_measured : int
        Number of samples that have been measured
    best_time_cost : float
        The best running time (the smaller the better)
    best_index : int
        The index of the samples that the best schedule is in
    best_sch : Optional[Schedule]
        The best schedule found so far
    """

    builder: ProgramBuilder
    runner: ProgramRunner
    callbacks: List[MeasureCallback]
    num_measured: int
    best_time_cost: float
    best_index: int
    best_sch: Optional[Schedule]

    def __init__(
        self,
        builder: ProgramBuilder,
        runner: ProgramRunner,
        callbacks: Optional[List[MeasureCallback]] = None,
    ):
        if callbacks is None:
            callbacks = []
        self.__init_handle_by_constructor__(
            _ffi_api.ProgramMeasurer,  # pylint: disable=no-member
            builder,
            runner,
            callbacks,
        )
