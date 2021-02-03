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
"""Cost model that estimates the performance of tensor programs"""
import ctypes
from typing import List, Optional

import numpy as np
from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api
from .measure_record import MeasureInput, MeasureResult
from .schedule import Schedule
from .search import SearchTask


@register_object("meta_schedule.CostModel")
class CostModel(Object):
    """The base class for cost model"""

    def update(
        self,
        inputs: List[MeasureInput],
        results: List[MeasureResult],
    ) -> None:
        """Update the cost model according to new measurement results (training data).

        Parameters
        ----------
        inputs : List[MeasureInput]
            The measurement inputs
        results : List[MeasureResult]
            The measurement results
        """
        _ffi_api.CostModelUpdate(self, inputs, results)  # pylint: disable=no-member

    def predict(
        self,
        task: SearchTask,
        schedules: List[Schedule],
    ) -> np.ndarray:
        """Predict the scores of schedules

        Parameters
        ----------
        task : SearchTask
            The search task
        schedules : List[Schedule]
            The input schedules

        Returns
        -------
        scores: numpy.ndarray
            An NDArray of shape (n, ), where n is the number of schedules passed in.
            The predicted scores for all schedules
        """
        n = len(schedules)
        result = np.zeros(shape=(n,), dtype="float64")
        _ffi_api.CostModelPredict(  # pylint: disable=no-member
            self,
            task,
            schedules,
            result.ctypes.data_as(ctypes.c_void_p),
        )
        return result


@register_object("meta_schedule.RandCostModel")
class RandCostModel(CostModel):
    """A model returns random estimation for all inputs"""

    def __init__(self, seed: Optional[int] = None):
        self.__init_handle_by_constructor__(
            _ffi_api.RandCostModel, seed  # pylint: disable=no-member
        )


@register_object("meta_schedule.PyCostModel")
class PyCostModel(CostModel):
    """Base class for cost models implemented in python"""

    def __init__(self):
        def update_func(inputs: List[MeasureInput], results: List[MeasureResult]):
            self.update(inputs, results)

        def predict_func(task: SearchTask, schedules: List[Schedule], return_ptr):
            n = len(schedules)
            return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_double))
            array_wrapper = np.ctypeslib.as_array(return_ptr, shape=(n,))
            array_wrapper[:] = self.predict(task, schedules)

        self.__init_handle_by_constructor__(
            _ffi_api.PyCostModel, update_func, predict_func  # pylint: disable=no-member
        )

    def update(
        self,
        inputs: List[MeasureInput],
        results: List[MeasureResult],
    ) -> None:
        """Update the cost model according to new measurement results (training data).

        Parameters
        ----------
        inputs : List[MeasureInput]
            The measurement inputs
        results : List[MeasureResult]
            The measurement results
        """
        raise NotImplementedError

    def predict(
        self,
        task: SearchTask,
        schedules: List[Schedule],
    ) -> np.ndarray:
        """Predict the scores of schedules

        Parameters
        ----------
        task : SearchTask
            The search task
        schedules : List[Schedule]
            The input schedules

        Returns
        -------
        scores: numpy.ndarray
            An NDArray of shape (n, ), where n is the number of schedules passed in.
            The predicted scores for all schedules
        """
        raise NotImplementedError
