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
from typing import TYPE_CHECKING, Callable, List, Optional, Union

from tvm._ffi import register_object
from tvm.runtime import Object

from ..runtime import NDArray, TVMContext, ndarray
from . import _ffi_api
from .measure_record import BuildResult, MeasureInput, MeasureResult
from .schedule import Schedule
from .utils import check_remote_servers, cpu_count

if TYPE_CHECKING:
    import numpy as np

    from ..target import Target
    from ..tir import PrimFunc


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
        if name == "rpc":
            return RPCRunner()
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

    key: str
    host: str
    port: int
    priority: int
    n_parallel: int

    def __init__(
        self,
        key: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        priority: int = 1,
        n_parallel: Optional[int] = None,
        timeout: int = 10,
        number: int = 3,
        repeat: int = 1,
        min_repeat_ms: int = 40,
        cooldown_interval: float = 0.0,
        enable_cpu_cache_flush: bool = False,
        f_create_args: Optional[Callable[[TVMContext], List[NDArray]]] = None
    ):
        if key is None:
            key = os.environ.get("TVM_TRACKER_KEY", None)
            if key is None:
                raise ValueError(
                    "RPC device key is not provided. Please provide 'key' explicitly, "
                    "or set environment variable TVM_TRACKER_KEY"
                )
        if host is None:
            host = os.environ.get("TVM_TRACKER_HOST", None)
            if host is None:
                raise ValueError(
                    "RPC tracker's host address is not provided. Please provide 'host' explicitly, "
                    "or set environment variable TVM_TRACKER_HOST"
                )
        if port is None:
            port = os.environ.get("TVM_TRACKER_PORT", None)
            if port is None:
                raise ValueError(
                    "RPC tracker's host address is not provided. Please provide 'port' explicitly, "
                    "or set environment variable TVM_TRACKER_PORT"
                )
            port = int(port)

        n_servers = check_remote_servers(key, host, port, priority, timeout)

        if n_parallel is None:
            n_parallel = n_servers

        self.__init_handle_by_constructor__(
            _ffi_api.RPCRunner,  #  pylint: disable=no-member
            key,
            host,
            port,
            priority,
            n_parallel,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
            f_create_args,
        )


########## MeasureCallback ##########


@register_object("meta_schedule.MeasureCallback")
class MeasureCallback(Object):
    """The base class of measurement callback functions."""


@register_object("meta_schedule.RecordToFile")
class RecordToFile(Object):
    """The base class of measurement callback functions."""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.RecordToFile)  # pylint: disable=no-member


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
        builder: Union[str, ProgramBuilder] = "local",
        runner: Union[str, ProgramRunner] = "rpc",
        measure_callbacks: Optional[List[MeasureCallback]] = None,
    ):
        if not isinstance(builder, ProgramBuilder):
            builder = ProgramBuilder.create(builder)
        if not isinstance(runner, ProgramRunner):
            runner = ProgramRunner.create(runner)
        if measure_callbacks is None:
            measure_callbacks = []
        self.__init_handle_by_constructor__(
            _ffi_api.ProgramMeasurer,  # pylint: disable=no-member
            builder,
            runner,
            measure_callbacks,
        )


########## ProgramTester ##########


class ProgramTester:
    """A utility function to run a specific PrimFunc on RPC"""

    target: Target
    target_host: Optional[Target]
    build_func: str
    rpc_key: str
    rpc_host: str
    rpc_port: int

    def __init__(
        self,
        target: Target,
        target_host: Optional[Target] = None,
        build_func: str = "tar",
        rpc_key: Optional[str] = None,
        rpc_host: Optional[str] = None,
        rpc_port: Optional[int] = None,
    ):
        super().__init__()
        if rpc_key is None:
            rpc_key = os.environ.get("TVM_TRACKER_KEY", None)
            if rpc_key is None:
                raise ValueError(
                    "RPC device key is not provided. Please provide 'rpc_key' explicitly, "
                    "or set environment variable TVM_TRACKER_KEY"
                )
        if rpc_host is None:
            rpc_host = os.environ.get("TVM_TRACKER_HOST", None)
            if rpc_host is None:
                raise ValueError(
                    "RPC tracker's host address is not provided. Please provide "
                    "'rpc_host' explicitly, or set environment variable TVM_TRACKER_HOST"
                )
        if rpc_port is None:
            rpc_port = os.environ.get("TVM_TRACKER_PORT", None)
            if rpc_port is None:
                raise ValueError(
                    "RPC tracker's host address is not provided. Please provide 'port' explicitly, "
                    "or set environment variable TVM_TRACKER_PORT"
                )
            rpc_port = int(rpc_port)

        self.target = target
        self.target_host = target_host
        self.build_func = build_func
        self.rpc_key = rpc_key
        self.rpc_host = rpc_host
        self.rpc_port = rpc_port

    def __call__(self, func: PrimFunc, args: List[np.ndarray]):
        """Build and run the PrimFunc with the specific arguments

        Parameters
        ----------
        func : PrimFunc
            The TIR func to be built and run
        args : List[np.ndarray]
            The list of arguments to run

        Returns
        -------
        filename : str
            The path to the exported library
        """
        filename = self._build_prim_func(func)
        args = self._run_exported_library(filename, args)
        return args

    def _build_prim_func(self, func: PrimFunc) -> str:
        """Build a PrimFunc

        Parameters
        ----------
        func : PrimFunc
            The TIR func to be built

        Returns
        -------
        filename : str
            The path to the exported library
        """
        # pylint: disable=import-outside-toplevel
        import tempfile

        from tvm.driver import build as tvm_build

        from ..autotvm.measure.measure_methods import set_cuda_target_arch
        from ..contrib import ndk as build_func_ndk
        from ..contrib import tar as build_func_tar

        # pylint: enable=import-outside-toplevel

        build_func = self.build_func
        build_func = {
            "tar": build_func_tar.tar,  # export to tar
            "ndk": build_func_ndk.create_shared,  # export to ndk
        }.get(build_func, build_func)
        if isinstance(build_func, str):
            raise ValueError("Invalid build_func: " + build_func)
        if self.target.kind.name == "cuda":
            set_cuda_target_arch(self.target.attrs["arch"])
        func = tvm_build(
            func,
            target=self.target,
            target_host=self.target_host,
        )
        filename = os.path.join(tempfile.mkdtemp(), "tmp_func." + build_func.output_format)
        func.export_library(filename, build_func)
        return filename

    def _run_exported_library(
        self,
        filename: str,
        args: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Run an exported library

        Parameters
        ----------
        filename : str
            The exported library to be uploaded
        args : List[np.ndarray]
            The list of arguments to run

        Returns
        -------
        args : List[np.ndarray]
            The result after executing the exported library
        """

        from .utils import request_remote  # pylint: disable=import-outside-toplevel

        # upload built module
        _, remote = request_remote(self.rpc_key, self.rpc_host, self.rpc_port)
        remote.upload(filename)
        func = remote.load_module(os.path.split(filename)[1])
        ctx = remote.context(dev_type=self.target.kind.name, dev_id=0)
        args = [ndarray.array(arg, ctx=ctx) for arg in args]
        func(*args)
        args = [arg.asnumpy() for arg in args]
        return args
