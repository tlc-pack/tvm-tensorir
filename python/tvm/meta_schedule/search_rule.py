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
"""Search rules in meta schedule"""
from typing import Callable, List

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api, _ffi_api_rule
from .instruction import BlockRV
from .schedule import Schedule

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


def always_inline() -> SearchRule:
    """Create a rule that inlines all possible blocks

    Returns
    ----------
    rule: SearchRule
        A search rule that does inlining
    """
    return _ffi_api_rule.AlwaysInline()  # pylint: disable=no-member


def add_cache_write() -> SearchRule:
    """Create a rule that adds a cache write stage after multi-level tiling

    Returns
    ----------
    rule: SearchRule
        A search rule that does cache write
    """
    return _ffi_api_rule.AddCacheWrite()  # pylint: disable=no-member


def multi_level_tiling_with_fusion(tiling_structure: str) -> SearchRule:
    """Create a rule that does multi-level tiling and fusion together
    if there is sufficient amount of data reuse

    Parameters
    ----------
    tiling_structure: str
        Structure of tiling. On CPU, recommended to use 'SSRSRS';
        On GPU, recommended to use 'SSSRRSRS'

    Returns
    ----------
    rule: SearchRule
        A search rule that does multi-level tiling with fusion
    """
    return _ffi_api_rule.MultiLevelTilingWithFusion(  # pylint: disable=no-member
        tiling_structure
    )


def multi_level_tiling(tiling_structure: str) -> SearchRule:
    """Create a rule that does multi-level tiling if there is sufficient amount of data reuse

    Parameters
    ----------
    tiling_structure: str
        Structure of tiling. On CPU, recommended to use 'SSRSRS';
        On GPU, recommended to use 'SSSRRSRS'

    Returns
    ----------
    rule: SearchRule
        A search rule that does multi-level tiling
    """
    return _ffi_api_rule.MultiLevelTiling(tiling_structure)  # pylint: disable=no-member
