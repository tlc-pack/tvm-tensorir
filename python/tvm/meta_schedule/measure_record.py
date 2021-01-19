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
from typing import List, Tuple

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api
from .schedule import Schedule
from .search import SearchTask


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

    TYPE = Tuple[str, int, str, float]

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

    TYPE = Tuple[List[float], int, str, float, float]

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

    @property
    def mean_cost(self) -> float:
        return _ffi_api.MeanResultMeanCost(self)  # pylint: disable=no-member
