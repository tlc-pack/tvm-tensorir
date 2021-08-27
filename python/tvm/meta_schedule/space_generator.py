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

from typing import Callable, List, Union

import tvm
from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.ir import IRModule
from tvm.tir.schedule import Schedule, Trace

from . import _ffi_api


@register_object("meta_schedule.SpaceGenerator")
class SpaceGenerator(Object):
    """The abstract design space generator interface."""

    def initialize_with_tune_context(
        self,
        tune_context: "TuneContext",
    ) -> None:
        """Initialize the design space generator with tuning context.

        Parameters
        ----------
        tune_context : TuneContext
            The tuning context for initializing the design space generator.
        """
        _ffi_api.SpaceGeneratorInitializeWithTuneContext(  # pylint: disable=no-member
            self, tune_context
        )

    def generate_design_space(self, mod: IRModule) -> List[Trace]:
        """Generate design spaces given a module.

        Parameters
        ----------
        mod : IRModule
            The module used for design space generation.

        Returns
        -------
        design_spaces : List[Trace]
            The generated design spaces, i.e., traces.
        """
        return _ffi_api.SpaceGeneratorGenerateDesignSpace(self, mod)  # pylint: disable=no-member


@register_object("meta_schedule.PySpaceGenerator")
class PySpaceGenerator(SpaceGenerator):
    """An abstract design space generator with customized methods on the python-side."""

    def __init__(self):
        """Constructor."""

        def initialize_with_tune_context_func(tune_context: "TuneContext") -> None:
            self.initialize_with_tune_context(tune_context)

        def generate_design_space_func(mod: IRModule) -> List[Trace]:
            return self.generate_design_space(mod)

        self.__init_handle_by_constructor__(
            _ffi_api.PySpaceGenerator,  # pylint: disable=no-member
            initialize_with_tune_context_func,
            generate_design_space_func,
        )

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        raise NotImplementedError

    def generate_design_space(self, mod: IRModule) -> List[Trace]:
        raise NotImplementedError


class ScheduleFn(PySpaceGenerator):
    """A design space generator with design spaces specified by a schedule function."""

    # Multiple cases of schedule functions supported
    SCH_FN_TYPE = Union[
        Callable[[IRModule], None],  # No output
        Callable[[IRModule], Trace],  # Single output
        Callable[[IRModule], List[Trace]],  # Multiple outputs
    ]

    def __init__(self, sch_fn: SCH_FN_TYPE):
        """Constructor.

        Parameters
        ----------
        sch_fn : SCH_FN_TYPE
            The schedule function.
        """
        super().__init__()
        self.sch_fn = sch_fn

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        """Initialize the design space generator with tuning context.

        Parameters
        ----------
        tune_context : TuneContext
            The tuning context for initializing the design space generator.
        """
        raise NotImplementedError

    def generate_design_space(self, mod: IRModule) -> List[Trace]:
        """Generate design spaces given a module.

        Parameters
        ----------
        mod : IRModule
            The module used for design space generation.

        Returns
        -------
        design_spaces : List[Trace]
            The generated design spaces, i.e., traces.
        """
        sch = Schedule(mod, traced=True)  # Make sure the schedule is traced
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
    """Union of design space generators."""

    def __init__(self, space_generators: List[SpaceGenerator]):
        """
        Constructor.

        Parameters
        ----------
        space_generators : List[SpaceGenerator]
            The list of design space generators to be unioned.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.SpaceGeneratorUnion, space_generators  # pylint: disable=no-member
        )

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        """Initialize the design space generator with tuning context.

        Parameters
        ----------
        tune_context : TuneContext
            The tuning context for initializing the design space generator.
        """
        _ffi_api.SpaceGeneratorUnionInitializeWithTuneContext(  # pylint: disable=no-member
            self, tune_context
        )

    def generate_design_space(self, mod: IRModule) -> List[Trace]:
        """Generate design spaces from the space generator.

        Parameters
        ----------
        mod : IRModule
            The mod to be used for design space generation.
        Returns
        -------
        results: List[Trace]
            The design spaces generated, i.e., a list of traces.
        """
        return _ffi_api.SpaceGeneratorUnionGenerateDesignSpace(  # pylint: disable=no-member
            self, mod
        )
