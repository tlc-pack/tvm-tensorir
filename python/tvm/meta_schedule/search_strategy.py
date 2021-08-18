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
"""Search Strategy"""

import random

from typing import List, TYPE_CHECKING

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Trace

from . import _ffi_api

if TYPE_CHECKING:
    from .tune_context import TuneContext


@register_object("meta_schedule.SearchStrategy")
class SearchStrategy(Object):
    """Description and abstraction of a search strategy class."""

    def initialize_with_tune_context(
        self,
        context: "TuneContext",
    ) -> None:
        return _ffi_api.SearchStrategyInitializeWithTuneContext(  # pylint: disable=no-member
            context
        )

    def generate_measure_candidates(self) -> List["BuilderInput"]:
        return _ffi_api.SearchStrategyGenerateMeasureCandidates()  # pylint: disable=no-member

    def notify_measure_results(self, results: List["MeasureResult"]) -> None:
        return _ffi_api.SearchStrategyNotifyMeasureResults(results)  # pylint: disable=no-member

    def pre_tuning(self, design_spaces: List[Trace]) -> None:
        return _ffi_api.SearchStrategyPreTuning(design_spaces)  # pylint: disable=no-member

    def post_tuning_func(self) -> None:
        return _ffi_api.SearchStrategyPostTuning()  # pylint: disable=no-member


@register_object("meta_schedule.PySearchStrategy")
class PySearchStrategy(SearchStrategy):
    """Search strategy that is implemented in python"""

    def __init__(self):
        def initialize_with_tune_context_func(context: "TuneContext") -> None:
            self.initialize_with_tune_context(context)

        def generate_measure_candidates_func() -> List["BuilderInput"]:
            return self.generate_measure_candidates()

        def notify_measure_results_func(results: List["MeasureResult"]) -> None:
            self.notify_measure_results(results)

        def pre_tuning_func(design_spaces: List[Trace]) -> None:
            self.pre_tuning(design_spaces)

        def post_tuning_func() -> None:
            self.post_tuning()

        self.__init_handle_by_constructor__(
            _ffi_api.PySearchStrategy,  # pylint: disable=no-member
            initialize_with_tune_context_func,
            generate_measure_candidates_func,
            notify_measure_results_func,
            pre_tuning_func,
            post_tuning_func,
        )

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the search strategy with a given context

        Parameters
        ----------
        context : TuneContext
            The auto tuning context
        """
        raise NotImplementedError

    def generate_measure_candidates(self) -> List["BuilderInput"]:
        """generate candidates for autotuning measurement according to the tune context

        Parameters
        ----------
        context : TuneContext
            The auto tuning context
        """
        raise NotImplementedError

    def notify_measure_results(self, results: List["MeasureResult"]) -> None:
        """Update the search srategy status accoding to the measurement results

        Returns
        -------
        results: List[Schedule]
            A list of schedules
        """
        raise NotImplementedError

    def pre_tuning(self, design_spaces: List[Trace]) -> None:
        """Initiate the search strategy status before tuning"""
        raise NotImplementedError

    def post_tuning(self) -> None:
        """Finish the search strategy process after tuning"""
        raise NotImplementedError


class ReplaySearchStrategy(PySearchStrategy):
    """Random search strategy"""

    def __init__(self, trials, batch_size):
        super().__init__()
        self.trials = trials
        self.batch_size = batch_size

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        pass

    def pre_tuning(self, design_spaces: List["Trace"]) -> None:
        self.design_spaces = design_spaces
        self.count = 0

    def post_tuning(self) -> None:
        pass

    def generate_measure_candidates(self) -> List["BuilderInput"]:
        """generate candidates for autotuning measurement according to the space generator"""

        if self.count >= self.trials:
            return None
        candidates = []
        for _ in range(self.count, min(self.count + self.batch_size, self.trials)):
            trace = Trace(random.choice(self.design_spaces).trace.insts, None)
            candidates.append(trace)
        return candidates

    def notify_measure_results(self, results: List["MeasureResult"]) -> None:
        """Update the search strategy status accoding to the measurement results

        Returns
        -------
        results: List[Schedule]
            A list of schedules
        """
        self.count += len(results)
