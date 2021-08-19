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
from tvm.tir.schedule import Schedule, Trace

from . import _ffi_api

if TYPE_CHECKING:
    from .tune_context import TuneContext


@register_object("meta_schedule.SpaceGenerator")
class SpaceGenerator(Object):
    """
    The abstract design space generator class.

    The design space generator could be specified by manually written schedule function,
    generated via loop analysis, ansor-like rules that apply to each block, etc.

    Here, the SpaceGenerator class is the base class for all space generators.
    It comes with `initialize_with_tune_context` and `generate` methods.

    The space generator is expected to generate design spaces (i.e., traces) given certain workload.
    """

    def initialize_with_tune_context(
        self,
        context: "TuneContext",
    ) -> None:
        """Initialize a space generator with a given tune context.
        Parameters
        ----------
        context : TuneContext
            The tunning context for the space generator, also allowing access to all other classes
            in the same tunning context.
        """
        return _ffi_api.SpaceGeneratorInitializeWithTuneContext(  # pylint: disable=no-member
            context
        )

    def generate(self, workload: IRModule) -> List[Trace]:
        """Generate design spaces given a workload.
        Parameters
        ----------
        workload : IRModule
            The workload to be used to generate the design spaces.
        Returns
        -------
        results : List[Trace]
            The generated design spaces, i.e., traces.
        """
        return _ffi_api.SpaceGeneratorGenerate(workload)  # pylint: disable=no-member

    def initialize(self, **kwargs) -> None:
        """Custom initialization"""
        raise NotImplementedError


@register_object("meta_schedule.PySpaceGenerator")
class PySpaceGenerator(SpaceGenerator):
    """
    The PySpaceGenerator is defined for cutomizable design space generation from python side.
    With PySpaceGenerator, you can define your own initilization and generate function.
    """

    def __init__(self):
        """Construct a PySpaceGenerator.
        No parameters but uses the function `initialize_with_tune_context` and `generate`
        defined in class body.

        Note
        ----
        The PyTaskScheduler will use the `initialize_with_tune_context` and `generate` functions
        defined in the class body as the function call.
        """

        def initialize_with_tune_context_func(context: "TuneContext") -> None:
            """Pass the initialize_with_tune_context function to constructor."""
            self.initialize_with_tune_context(context)

        def generate_func(workload: IRModule) -> List[Trace]:
            """Pass the generate function to constructor."""
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

    def generate(self, workload: IRModule) -> List[Trace]:
        """Generate a list of schedules from generator
        Parameters
        ----------
        workload : TuneContext
            The auto tuning context

        Returns
        -------
        results: List[Schedule]
            A list of schedules
        """
        raise NotImplementedError


class ScheduleFn(PySpaceGenerator):
    """
    ScheduleFn is a special case of SpaceGenerator where design space is specified by a schedule
    function.
    """

    SCH_FN_TYPE = Union[
        # Multiple cases of schedule function could be supported here.
        Callable[[IRModule], None],  # No output
        Callable[[IRModule], Trace],  # Single output
        Callable[[IRModule], List[Trace]],  # Multiple outputs
    ]

    def __init__(self, sch_fn: SCH_FN_TYPE):
        """Construct a ScheduleFn.
        Parameters
        ----------
        sch_fn : Callable[[IRModule], Trace]
            The schedule function.
        """
        super().__init__()
        self.sch_fn = sch_fn

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the schedule function with a given tune context.
        Parameters
        ----------
        context : TuneContext
            The tunning context for the space generator, also allowing access to all other classes
            in the same tunning context.
        """
        raise NotImplementedError

    def generate(self, workload: IRModule) -> List[Trace]:
        """Generate a list of schedules from generator
        Parameters
        ----------
        workload : IRModule
            The workload to be used to generate the design spaces.

        Returns
        -------
        results: List[Trace]
            The generated design spaces, i.e., traces.
        """
        sch = Schedule(workload, traced=True)  # Make sure the schedule is traced
        result = self.sch_fn(sch)  # Call the schedule function
        if result is None:  # The case of no output
            return [sch.trace]
        if isinstance(result, (list, tvm.ir.Array)):  # enumerate the outputs
            for ret in result:
                if not isinstance(ret, Trace):
                    raise TypeError(
                        "Wrong type of element in the list, expected Trace got "
                        + f"'{type(ret)}': {ret}"
                    )
            return result  # the case of multiple outputs
        return [result]  # the case of single output


@register_object("meta_schedule.SpaceGeneratorUnion")
class SpaceGeneratorUnion(SpaceGenerator):
    """
    Union of space generators implmented in python.

    The union space generator is used to generate the union of multiple space generators. It is
    a derived class of the SpaceGenerator.
    """

    def __init__(self, space_generators: List[SpaceGenerator]):
        """
        Parameters
        ----------
        space_generators : List[SpaceGenerator]
            The list of space generators to be unioned.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.SpaceGeneratorUnion, space_generators  # pylint: disable=no-member
        )

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        """Initialize the union space generator with a given tune context.
        Parameters
        ----------
        context : TuneContext
            The tunning context for the union space generator.
        """
        _ffi_api.SpaceGeneratorUnionInitializeWithTuneContext(  # pylint: disable=no-member
            self, context
        )

    def generate(self, workload: IRModule) -> List[Trace]:
        """Generate design spaces from the space generator.

        Parameters
        ----------
        workload : IRModule
            The workload to be used for design space generation.
        Returns
        -------
        results: List[Trace]
            The design spaces generated, i.e., a list of traces.
        """
        # Process the case of PrimFunc given as workload
        if isinstance(workload, PrimFunc):
            workload = IRModule({"main": workload})
        return _ffi_api.SpaceGeneratorUnionGenerate(self, workload)  # pylint: disable=no-member

    def initialize(self, **kwargs) -> None:
        """Custom initialization"""
        raise NotImplementedError
