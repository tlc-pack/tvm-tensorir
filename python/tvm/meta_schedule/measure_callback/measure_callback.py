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
"""Meta Schedule MeasureCallback."""

from typing import TYPE_CHECKING, List

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.search_strategy import MeasureCandidate
from tvm.meta_schedule.builder import BuilderResult
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.utils import _get_hex_address

from .. import _ffi_api

if TYPE_CHECKING:
    from ..tune_context import TuneContext
    from ..task_scheduler import TaskScheduler


@register_object("meta_schedule.MeasureCallback")
class MeasureCallback(Object):
    """Rules to apply after measure results is available."""

    def apply(
        self,
        task_scheduler: "TaskScheduler",
        tasks: List["TuneContext"],
        measure_candidates: List[MeasureCandidate],
        builds: List[BuilderResult],
        results: List[RunnerResult],
    ) -> bool:
        """Apply a measure callback to the given schedule.

        Parameters
        ----------
        task_scheduler: TaskScheduler
            The task scheduler.
        tasks: List[TuneContext]
            The list of tune context to process.
        measure_candidats: List[MeasureCandidate]
            The measure candidates.
        builds: List[BuilderResult]
            The builder results by building the measure candidates.
        results: List[RunnerResult]
            The runner results by running the built measure candidates.

        Returns
        -------
        result : bool
            Whether the measure callback was successfully applied.
        """
        return _ffi_api.MeasureCallbackApply(
            self, task_scheduler, tasks, measure_candidates, builds, results
        )


@register_object("meta_schedule.PyMeasureCallback")
class PyMeasureCallback(MeasureCallback):
    """An abstract MeasureCallback with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        def f_apply(
            task_scheduler: "TaskScheduler",
            tasks: List[TuneContext],
            measure_candidates: List[MeasureCandidate],
            builds: List[BuilderResult],
            results: List[RunnerResult],
        ) -> bool:
            return self.apply(task_scheduler, tasks, measure_candidates, builds, results)

        def f_as_string() -> str:
            return str(self)

        self.__init_handle_by_constructor__(
            _ffi_api.MeasureCallbackPyMeasureCallback,  # type: ignore # pylint: disable=no-member
            f_apply,
            f_as_string,
        )

    def __str__(self) -> str:
        return f"PyMeasureCallback({_get_hex_address(self.handle)})"
