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
"""Meta Schedule CostModel."""

from typing import List, TYPE_CHECKING
import ctypes

import numpy as np

from tvm._ffi import register_object
from tvm.runtime import Object

from .. import _ffi_api
from ..runner import RunnerResult
from ..tune_context import TuneContext
from ..search_strategy import MeasureCandidate
from ..utils import _get_hex_address, check_override


@register_object("meta_schedule.CostModel")
class CostModel(Object):
    """Cost model for estimation of running time, thus reducing search space."""

    def load(self, file_location: str) -> bool:
        """Load the cost model from given file location.

        Parameters
        ----------
        file_location : str
            The file location.

        Return
        ------
        result : bool
            Whether cost model was loaded successfully.
        """
        _ffi_api.CostModelLoad(self, file_location)  # type: ignore # pylint: disable=no-member

    def save(self, file_location: str) -> bool:
        """Save the cost model to given file location.

        Parameters
        ----------
        file_location : str
            The file location.

        Return
        ------
        result : bool
            Whether cost model was saved successfully.
        """
        _ffi_api.CostModelSave(self, file_location)  # type: ignore # pylint: disable=no-member

    def update(
        self,
        tune_context: TuneContext,
        candidates: List[MeasureCandidate],
        results: List[RunnerResult],
    ) -> bool:
        """Update the cost model given running results.

        Parameters
        ----------
        tune_context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.
        results : List[RunnerResult]
            The running results of the measure candidates.

        Return
        ------
        result : bool
            Whether cost model was updated successfully.
        """
        _ffi_api.CostModelUpdate(self, tune_context, candidates, results)  # type: ignore # pylint: disable=no-member

    def predict(self, tune_context: TuneContext, candidates: List[MeasureCandidate]) -> np.ndarray:
        """Update the cost model given running results.

        Parameters
        ----------
        tune_context : TuneContext,
            The tuning context.
        candidates : List[MeasureCandidate]
            The measure candidates.


        Return
        ------
        result : bool
            The predicted running results.
        """
        n = len(candidates)
        results = np.zeros(shape=(n,), dtype="float64")
        _ffi_api.CostModelPredict(
            self,
            tune_context,
            candidates,
            results.ctypes.data_as(ctypes.c_void_p),
        )  # type: ignore # pylint: disable=no-member
        return results


@register_object("meta_schedule.PyCostModel")
class PyCostModel(CostModel):
    """An abstract CostModel with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        @check_override(self.__class__, CostModel)
        def f_load(file_location: str) -> bool:
            self.load(file_location)

        @check_override(self.__class__, CostModel)
        def f_save(file_location: str) -> bool:
            self.save(file_location)

        @check_override(self.__class__, CostModel)
        def f_update(
            self,
            tune_context: TuneContext,
            candidates: List[MeasureCandidate],
            results: List[RunnerResult],
        ) -> bool:
            self.update(tune_context, candidates, results)

        @check_override(self.__class__, CostModel)
        def f_predict(tune_context: TuneContext, candidates: List[MeasureCandidate], return_ptr):
            n = len(candidates)
            return_ptr = ctypes.cast(return_ptr, ctypes.POINTER(ctypes.c_double))
            array_wrapper = np.ctypeslib.as_array(return_ptr, shape=(n,))
            array_wrapper[:] = self.predict(tune_context, candidates)

        def f_as_string() -> str:
            return str(self)

        self.__init_handle_by_constructor__(
            _ffi_api.CostModelPyCostModel,  # type: ignore # pylint: disable=no-member
            f_load,
            f_save,
            f_update,
            f_predict,
            f_as_string,
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({_get_hex_address(self.handle)})"
