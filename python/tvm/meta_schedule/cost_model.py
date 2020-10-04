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
from typing import List

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api
from .measure import MeasureInput, MeasureResult
from .schedule import Schedule
from .search_task import SearchTask


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
    ) -> List[float]:
        """Predict the scores of schedules

        Parameters
        ----------
        task : SearchTask
            The search task
        schedules : List[Schedule]
            The input schedules

        Returns
        -------
        scores: List[float]
            The predicted scores for all schedules
        """
        result = _ffi_api.CostModelPredict(  # pylint: disable=no-member
            self,
            task,
            schedules,
        )
        return [x.value for x in result]


@register_object("meta_schedule.RandomModel")
class RandomModel(CostModel):
    """A model returns random estimation for all inputs"""

    def __init__(self):
        self.__init_handle_by_constructor__(
            _ffi_api.RandomModel  # pylint: disable=no-member
        )
