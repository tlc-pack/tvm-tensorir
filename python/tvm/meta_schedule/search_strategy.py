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
"""Space Generator"""

from typing import List, TYPE_CHECKING, Any

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.ir import IRModule

from . import _ffi_api
from .space_generator import SpaceGenerator

if TYPE_CHECKING:
    from .tune_context import TuneContext


@register_object("meta_schedule.SearchStrategy")
class SearchStrategy(Object):
    """Description and abstraction of a search strategy class."""

    def GenerateMeasureCandidates(  # pylint: disable=invalid-name
        self,
        context: "TuneContext",
    ) -> List[Any]:
        return _ffi_api.SearchStrategyGenerateMeasureCandidates(  # pylint: disable=no-member
            self, context
        )

    def UpdateResults(self, results: List[Any]):  # pylint: disable=invalid-name
        return _ffi_api.SearchStrategyGenerate(self, results)  # pylint: disable=no-member

    def initialize(self, **kwargs):
        raise NotImplementedError


@register_object("meta_schedule.PySearchStrategy")
class PySearchStrategy(SearchStrategy):
    """Search strategy that is implemented in python"""

    def __init__(self):
        def generate_measure_candidates_func(context: "TuneContext"):
            return self.generate_measure_candidates(context)

        def update_results_func(results: List[Object]):
            self.update_results(results)

        self.__init_handle_by_constructor__(
            _ffi_api.PySearchStrategyNew,  # pylint: disable=no-member
            generate_measure_candidates_func,
            update_results_func,
        )

    def generate_measure_candidates(self, context: "TuneContext") -> List[Object]:
        """generate candidates for autotuning measurement according to the tune context

        Parameters
        ----------
        context : TuneContext
            The auto tuning context
        """
        raise NotImplementedError

    def update_results(self, results: List[Object]):
        """Update the search srategy status accoding to the measurement results

        Returns
        -------
        results: List[Schedule]
            A list of schedules
        """
        raise NotImplementedError


class ReplaySearchStrategy(PySearchStrategy):
    """Random search strategy"""

    def __init__(self, trails, batch_size):
        super().__init__()
        self.trails = trails
        self.batch_size = batch_size
        self.count = 0

    def generate_measure_candidates(self, context: "TuneContext") -> List[Any]:
        """generate candidates for autotuning measurement according to the tune context

        Parameters
        ----------
        context : TuneContext
            The auto tuning context
        """
        raise NotImplementedError

    def generate_measure_candidates_sg(
        self, space_gen: SpaceGenerator, workload: IRModule
    ) -> List[Any]:
        """generate candidates for autotuning measurement according to the space generator

        Parameters
        ----------
        context : TuneContext
            The auto tuning context
        """
        if self.count >= self.trails:
            return []
        candidates = []
        for _ in range(self.count, min(self.count + self.batch_size, self.trails)):
            (sch,) = space_gen.generate(workload)
            candidates.append(sch)
        return candidates

    def update_results(self, results: List[Any]):
        """Update the search srategy status accoding to the measurement results

        Returns
        -------
        results: List[Schedule]
            A list of schedules
        """
        self.count += len(results)
