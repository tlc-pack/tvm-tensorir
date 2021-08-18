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
"""Schedule Rule"""

from typing import Callable, List, Union, TYPE_CHECKING

import tvm
from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import Schedule, BlockRV

from . import _ffi_api

if TYPE_CHECKING:
    from .tune_context import TuneContext


@register_object("meta_schedule.ScheduleRule")
class ScheduleRule(Object):
    """Description and abstraction of a schedule rule."""

    def initialize_with_tune_context(
        self,
        context: "TuneContext",
    ) -> None:
        return _ffi_api.ScheduleRuleInitializeWithTuneContext(  # pylint: disable=no-member
            self, context
        )

    def apply(self, schedule: Schedule, block: BlockRV) -> List[Schedule]:
        return _ffi_api.ScheduleRuleApply(self, schedule, block)  # pylint: disable=no-member

    def initialize(self, **kwargs):
        raise NotImplementedError


@register_object("meta_schedule.PyScheduleRule")
class PyScheduleRule(ScheduleRule):
    """Schedule rule that is implemented in python"""

    RULE_FN_TYPE = Union[
        Callable[[Schedule], None],
        Callable[[Schedule], Schedule],
        Callable[[Schedule], List[Schedule]],
    ]

    def __init__(self, name, rule_func: RULE_FN_TYPE):
        def initialize_with_tune_context_func(context: "TuneContext"):
            self.initialize_with_tune_context(context)

        def apply_func(schedule: Schedule, block: BlockRV):
            return self.apply(schedule, block)

        self.__init_handle_by_constructor__(
            _ffi_api.PyScheduleRule,  # pylint: disable=no-member
            name,
            initialize_with_tune_context_func,
            apply_func,
        )
        self.rule_func = rule_func

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        """initialize the space generator according to the tune context

        Parameters
        ----------
        context : TuneContext
            The auto tuning context
        """
        raise NotImplementedError

    def apply(self, schedule: Schedule, block: BlockRV) -> List[Schedule]:
        """apply a list of schedules from schedule rule

        Returns
        -------
        results: List[Schedule]
            A list of schedules
        """
        result = self.rule_func(schedule, block)
        if result is None:
            return [schedule]
        if isinstance(result, (list, tvm.ir.Array)):
            for ret in result:
                if not isinstance(ret, Schedule):
                    raise TypeError(
                        "Wrong type of element in the list, expected Schedule got "
                        + f"'{type(ret)}': {ret}"
                    )
            return result
        return [result]


def as_schedule_rule(name) -> PyScheduleRule:
    """A decorate that wraps a python function into ScheduleRule

    Parameters
    ----------
    name : str
        Name of the schedule rule

    Returns
    -------
    rule : PyScheduleRule
        The schedule rule
    """

    def wrap(func):
        return PyScheduleRule(name, func)

    return wrap
