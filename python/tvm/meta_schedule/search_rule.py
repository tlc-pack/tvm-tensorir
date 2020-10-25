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
from typing import Any, Callable, Dict, List, Optional

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir import TensorIntrin

from . import _ffi_api, _ffi_api_search_rule
from .instruction import BlockRV
from .schedule import Schedule
from .search import SearchTask

########## SearchRule ##########


@register_object("meta_schedule.SearchRule")
class SearchRule(Object):
    """A rule that applies to a block and generates a snippet of schedule on it"""

    CONTEXT_INFO_TYPE = Optional[Dict[str, Any]]
    RETURN_TYPE = Dict[Schedule, CONTEXT_INFO_TYPE]

    name: str

    def __init__(
        self,
        name: str,
        apply: Callable[[SearchTask, Schedule, BlockRV, CONTEXT_INFO_TYPE], RETURN_TYPE],
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.SearchRule,  # pylint: disable=no-member
            name,
            apply,
        )

    def __call__(
        self,
        task: SearchTask,
        sch: Schedule,
        block: BlockRV,
        info: CONTEXT_INFO_TYPE = None,
    ) -> RETURN_TYPE:
        """Apply the rule to a block

        Parameters
        ----------
        task: SearchTask
            The search task
        sch: Schedule
            Where the schedule snippets should be generated
        block: BlockRV
            The block the rule applies on
        info: CONTEXT_INFO_TYPE
            The context info about the schedule

        Returns
        ----------
        result: RETURN_TYPE
            The new schedules generated
        """
        ret = _ffi_api.SearchRuleApply(self, task, sch, block, info)  # pylint: disable=no-member
        return {k: v for k, v in ret.items()}  # pylint: disable=unnecessary-comprehension


def compose(name: str, rules: List[SearchRule]) -> SearchRule:
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
        def apply(
            task: SearchTask,
            sch: Schedule,
            block: BlockRV,
            info: SearchRule.CONTEXT_INFO_TYPE,
        ) -> SearchRule.RETURN_TYPE:
            ret = func(task, sch, block, info)
            if isinstance(ret, Schedule):
                return {ret: info}
            if isinstance(ret, dict):
                for k, v in ret.items():
                    assert isinstance(k, Schedule)
                    assert v is None or isinstance(v, dict)
                return ret
            raise TypeError(
                "SearchRule shoud return Dict<Schedule, ContextInfo>, "
                + f"but gets type '{type(ret)}': {ret}"
            )

        return SearchRule(name, apply)

    return wrap


def inline_pure_spatial(strict_mode: bool) -> SearchRule:
    """Create a rule that inlines all possible pure spatial block

    Parameters
    ----------
    strict_mode : bool
        Requires the block to be strictly inlineable

    Returns
    ----------
    rule: SearchRule
        A search rule that does inlining
    """
    return _ffi_api_search_rule.InlinePureSpatial(strict_mode)  # pylint: disable=no-member


def multi_level_tiling_and_fusion(
    structure: str,
    add_read_cache: bool,
    add_write_cache: bool,
    fusion_levels: List[int],
) -> SearchRule:
    """Create a rule that does multi-level tiling if there is sufficient amount of data reuse.
    Optionally add read cache and write cache, do fusion if possible.

    Parameters
    ----------
    structure : str
        Structure of tiling. On CPU, recommended to use 'SSRSRS';
        On GPU, recommended to use 'SSSRRSRS'
    add_read_cache : bool
        Add cache_read before the multi-level tiling
    add_write_cache : bool
        Add cache_write after the multi-level tiling
    fusion_levels : List[int]
        The possible tile levels that a single elementwise consumer is fused at

    Returns
    ----------
    rule: SearchRule
        The rule created
    """
    return _ffi_api_search_rule.MultiLevelTilingAndFusion(  # pylint: disable=no-member
        structure,
        add_read_cache,
        add_write_cache,
        fusion_levels,
    )


def mark_parallelize_outer(max_extent: int) -> SearchRule:
    """Create a rule that parallelizes the outer loops

    Parameters
    ----------
    max_extent : int
        The maximum extent of loops to be parallelized together

    Returns
    ----------
    rule: SearchRule
        The search rule created
    """
    return _ffi_api_search_rule.MarkParallelizeOuter(max_extent)  # pylint: disable=no-member


def mark_vectorize_inner(max_extent: int) -> SearchRule:
    """Create a rule that vectorizes the inner loops

    Parameters
    ----------
    max_extent : int
        The maximum extent of loops to be vectorized together

    Returns
    ----------
    rule: SearchRule
        The search rule created
    """
    return _ffi_api_search_rule.MarkVectorizeInner(max_extent)  # pylint: disable=no-member


def mark_tensorize(tensor_intrins: List[TensorIntrin]) -> SearchRule:
    """Rewrite block and its surrounding loops to match the tensor intrinsics if possible

    Parameters
    ----------
    tensor_intrins : List[TensorIntrin]
        The tensor intrinsics to be matched

    Returns
    -------
    rule: SearchRule
        The rule created
    """
    return _ffi_api_search_rule.MarkTensorize(tensor_intrins)  # pylint: disable=no-member
