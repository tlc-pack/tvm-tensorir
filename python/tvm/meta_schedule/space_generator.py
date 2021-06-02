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

from typing import Callable, List, Optional, Union
from tvm._ffi import register_object
from tvm.runtime import Object
from .schedule import Schedule
from .tune_context import TuneContext
import tvm

from . import _ffi_api


@register_object("meta_schedule.SpaceGenerator")
class SpaceGenerator(Object):
    """Description and abstraction of a design space generator.
    The design space generator could be specified by manually written schedule function,
    generated via loop analysis, ansor-like rules that apply to each block, etc."""

    def initialize_with_tune_context(self, context: TuneContext) -> None:
        return _ffi_api.SpaceGeneratorInitializeWithTuneContext(self, context)

    def generate(self) -> List[Schedule]:
        return _ffi_api.SpaceGeneratorGenerate(self)  # pylint: disable=no-member

    def initialize(self, **kwargs):
        raise NotImplementedError


@register_object("meta_schedule.PySpaceGenerator")
class PySpaceGenerator(SpaceGenerator):
    """Design space generator that is implemented in python"""

    def __init__(self):
        def init_with_tune_context_func(context: TuneContext):
            self.init_with_tune_context(context)

        def generate_func():
            return self.generate()

        self.__init_handle_by_constructor__(
            _ffi_api.PySpaceGeneratorNew,
            init_with_tune_context_func,
            generate_func,  # pylint: disable=no-member
        )

    def init_with_tune_context(self, context: TuneContext) -> None:
        """initialize the space generator according to the tune context

        Parameters
        ----------
        context : TuneContext
            The auto tuning context
        """
        raise NotImplementedError

    def generate(self) -> List[Schedule]:
        """Generate a list of schdules from generator

        Returns
        -------
        results: List[Schedule]
            A list of schedules
        """
        raise NotImplementedError


class ScheduleFn(PySpaceGenerator):
    """Design space that is specified by a schedule function"""

    SCH_FN_TYPE = Union[
        Callable[[Schedule], None],
        Callable[[Schedule], Schedule],
        Callable[[Schedule], List[Schedule]],
    ]

    def __init__(self, sch_fn: SCH_FN_TYPE):
        super().__init__()
        self.sch_fn = sch_fn

    def init_with_tune_context(self, context: TuneContext) -> None:
        self.workload = context.workload

    def generate(self) -> List[Schedule]:
        sch = Schedule(self.workload)
        result = self.sch_fn(sch)
        if result is None:
            return [sch]
        elif isinstance(result, (list, tvm.ir.Array)):
            for res in result:
                if not isinstance(res, Schedule):
                    raise TypeError(
                        "Wrong type of element in the list, expected Schedule got " + str(type(res))
                    )
            return result
        else:
            return [result]

    def initialize(self, *, workload=None, **kwargs):
        self.workload = workload
        if workload is None:
            raise ValueError("Workload is not given.")
