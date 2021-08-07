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

from typing import Callable, List, Union, TYPE_CHECKING

import tvm
from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.ir import IRModule
from tvm.tir import PrimFunc

from . import _ffi_api
from .schedule import Schedule

if TYPE_CHECKING:
    from .tune_context import TuneContext


@register_object("meta_schedule.SpaceGenerator")
class SpaceGenerator(Object):
    """Description and abstraction of a design space generator.
    The design space generator could be specified by manually written schedule function,
    generated via loop analysis, ansor-like rules that apply to each block, etc."""

    def initialize_with_tune_context(
        self,
        context: "TuneContext",
    ) -> None:
        return _ffi_api.SpaceGeneratorInitializeWithTuneContext(  # pylint: disable=no-member
            context
        )

    def generate(self, workload: IRModule) -> List[Schedule]:
        return _ffi_api.SpaceGeneratorGenerate(workload)  # pylint: disable=no-member

    def initialize(self, **kwargs) -> None:
        raise NotImplementedError


@register_object("meta_schedule.PySpaceGenerator")
class PySpaceGenerator(SpaceGenerator):
    """Design space generator that is implemented in python"""

    def __init__(self):
        def initialize_with_tune_context_func(context: "TuneContext") -> None:
            self.initialize_with_tune_context(context)

        def generate_func(workload: IRModule) -> List[Schedule]:
            return self.generate(workload)

        self.__init_handle_by_constructor__(
            _ffi_api.PySpaceGenerator,  # pylint: disable=no-member
            initialize_with_tune_context_func,
            generate_func,
        )

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        """initialize the space generator according to the tune context

        Parameters
        ----------
        context : TuneContext
            The auto tuning context
        """
        raise NotImplementedError

    def generate(self, workload: IRModule) -> List[Schedule]:
        """Generate a list of schedules from generator

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

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        raise NotImplementedError

    def generate(self, workload: IRModule) -> List[Schedule]:
        sch = Schedule(workload)
        result = self.sch_fn(sch)
        if result is None:
            return [sch]
        if isinstance(result, (list, tvm.ir.Array)):
            for ret in result:
                if not isinstance(ret, Schedule):
                    raise TypeError(
                        "Wrong type of element in the list, expected Schedule got "
                        + f"'{type(ret)}': {ret}"
                    )
            return result
        return [result]


@register_object("meta_schedule.SpaceGeneratorUnion")
class SpaceGeneratorUnion(SpaceGenerator):
    """Design space generator union that is implemented in python"""

    def __init__(self, space_generators: List[SpaceGenerator]):
        self.__init_handle_by_constructor__(
            _ffi_api.SpaceGeneratorUnionNew, space_generators  # pylint: disable=no-member
        )

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        return _ffi_api.SpaceGeneratorUnionInitializeWithTuneContext(  # pylint: disable=no-member
            self, context
        )

    def generate(self, workload: IRModule) -> List[Schedule]:
        if isinstance(workload, PrimFunc):
            workload = IRModule({"main": workload})
        return _ffi_api.SpaceGeneratorUnionGenerate(self, workload)  # pylint: disable=no-member

    def initialize(self, **kwargs) -> None:
        raise NotImplementedError
