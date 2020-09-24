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
""" Search API """
from typing import Callable, List, Optional, Union

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir import PrimFunc

from . import _ffi_api
from .measure import MeasureCallback, ProgramBuilder, ProgramRunner
from .random_variable import BlockRV
from .schedule import Schedule
from .search_task import SearchTask

########## RulePackedArgs ##########


@register_object("meta_schedule.RulePackedArgs")
class RulePackedArgs(Object):
    """Input/output arguments of a SearchRule

    Parameters
    ----------
    proceed: List[Schedule]
        The arguments the rule should apply to
    skipped: List[Schedule]
        The arguments the rule should skip
    """

    proceed: List[Schedule]
    skipped: List[Schedule]

    def __init__(
        self,
        proceed: List[Schedule],
        skipped: List[Schedule],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.RulePackedArgs,  # pylint: disable=no-member
            proceed,
            skipped,
        )


########## SearchRule ##########


@register_object("meta_schedule.SearchRule")
class SearchRule(Object):
    """A rule that applies to a block and generates a snippet of schedule on it"""

    name: str

    def __init__(self, name: str, apply: Callable[[Schedule, BlockRV], RulePackedArgs]):
        self.__init_handle_by_constructor__(
            _ffi_api.SearchRule,  # pylint: disable=no-member
            name,
            apply,
        )

    def __call__(self, sch: Schedule, block: BlockRV) -> RulePackedArgs:
        """Apply the rule to a block

        Parameters
        ----------
        sch: Schedule
            Where the schedule snippets should be generated
        block: BlockRV
            The block the rule applies on

        Returns
        ----------
        result: RulePackedArgs
            The new schedules generated
        """
        return _ffi_api.SearchRuleCall(self, sch, block)  # pylint: disable=no-member

    @staticmethod
    def compose(name: str, rules: List["SearchRule"]) -> "SearchRule":
        """Composing search rules sequentially into a single rule

        Parameters
        ----------
        name: str
            Name of the new composite search rule
        rules: List[SearchRule]
            The rules provided sequentially

        Returns
        ----------
        rule: ScheduleRule
            The composite rule
        """
        return _ffi_api.SearchRuleCompose(  # pylint: disable=no-member
            name,
            rules,
        )


def register_rule(name) -> SearchRule:
    """Register a search rule by wrapping the decorated function to SearchRule

    Parameters
    ----------
    name : str
        Name of the rule

    Returns
    -------
    rule : SearchRule
        The search rule
    """

    def wrap(func):
        def apply(sch: Schedule, block: BlockRV) -> RulePackedArgs:
            result = func(sch, block)
            if isinstance(result, RulePackedArgs):
                return result
            if isinstance(result, Schedule):
                return RulePackedArgs(proceed=[result], skipped=[])
            if isinstance(result, list):
                return RulePackedArgs(proceed=result, skipped=[])
            assert isinstance(
                result, dict
            ), "SearchRule does not support return type: " + str(type(result))
            assert {"proceed", "skipped"}.issuperset(
                set(result.keys())
            ), "Only the following keys are allowed: 'proceed', 'skipped'"
            proceed = result.get("proceed", [])
            skipped = result.get("skipped", [])
            return RulePackedArgs(proceed=proceed, skipped=skipped)

        return SearchRule(name, apply)

    return wrap


########## SearchSpace ##########


@register_object("meta_schedule.SearchSpace")
class SearchSpace(Object):
    """Description and abstraction of a search space.
    The search space could be specified by manually written schedule function,
    generated via loop analysis, ansor-like rules that apply to each block, etc."""

    def sample_schedule(self, task: SearchTask) -> Schedule:
        return _ffi_api.SearchSpaceSampleSchedule(  # pylint: disable=no-member
            self, task
        )

    def get_support(self, task: SearchTask) -> List[Schedule]:
        return _ffi_api.SearchSpaceGetSupport(self, task)  # pylint: disable=no-member


@register_object("meta_schedule.ScheduleFn")
class ScheduleFn(SearchSpace):
    """Search space that is specified by a schedule function"""

    TYPE = Callable[[Schedule], None]

    def __init__(self, func: TYPE):
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleFn,  # pylint: disable=no-member
            func,
        )


@register_object("meta_schedule.PostOrderApply")
class PostOrderApply(SearchSpace):
    """Search space that is specified by applying rules in post-DFS order"""

    rule: SearchRule

    def __init__(self, rule: SearchRule):
        self.__init_handle_by_constructor__(
            _ffi_api.PostOrderApply,  # pylint: disable=no-member
            rule,
        )


########## SearchStrategy ##########


@register_object("meta_schedule.SearchStrategy")
class SearchStrategy(Object):
    """The search strategy for exploring the search space.
    It could be always replay the sampling function, or saving several traces
    from the sample function and then do lightweight-metropolis-hastings, or integrate those with
    evolutionary search, etc.
    """

    @staticmethod
    def create(strategy: str) -> "SearchStrategy":
        if strategy == "replay":
            return Replay()
        raise ValueError("Cannot create search strategy from: " + strategy)


@register_object("meta_schedule.Replay")
class Replay(SearchStrategy):
    """ defined in src/meta_schedule/search.h """

    def __init__(
        self,
        batch_size: int = 16,
        num_iterations: int = 128,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Replay,  # pylint: disable=no-member
            batch_size,
            num_iterations,
        )


########## Search API ##########


def autotune(
    task: Union[PrimFunc, SearchTask],
    space: Union[ScheduleFn.TYPE, SearchSpace],
    strategy: Union[str, SearchStrategy],
    builder: Union[str, ProgramBuilder] = "local",
    runner: Union[str, ProgramRunner] = "rpc",
    measure_callbacks: Optional[List[MeasureCallback]] = None,
    verbose: int = 1,
) -> Optional[Schedule]:
    """The entry function for auto tuning.

    Parameters
    ----------
    task: Union[PrimFunc, SearchTask]
        The search task
    space: Union[ScheduleFn.TYPE, SearchSpace]
        The search space
    strategy: Union[str, SearchStrategy]
        The search strategy
    builder: Union[str, ProgramBuilder]
        Program builder used to run TIR build process
    runner: Union[str, ProgramRunner]
        Program runner used to run the TIR profiling process, or interact with RPC tracker
    measure_callbacks: Optional[List[MeasureCallback]]
        The callbacks to be triggered after each batch of meansuring
    verbose: int
        Flag for the verbose mode

    Returns
    -------
    best_schedule : Optional[Schedule]
        The best schedule found, None if no valid schedule is found in the search space
    """
    if not isinstance(task, SearchTask):
        task = SearchTask(task)
    if not isinstance(space, SearchSpace):
        space = ScheduleFn(space)
    if not isinstance(strategy, SearchStrategy):
        strategy = SearchStrategy.create(strategy)
    if not isinstance(builder, ProgramBuilder):
        builder = ProgramBuilder.create(builder)
    if not isinstance(runner, ProgramRunner):
        runner = ProgramRunner.create(runner)
    if measure_callbacks is None:
        measure_callbacks = []
    return _ffi_api.AutoTune(  # pylint: disable=no-member
        task,
        space,
        strategy,
        builder,
        runner,
        measure_callbacks,
        verbose,
    )
