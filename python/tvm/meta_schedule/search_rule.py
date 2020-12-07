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
                "SearchRule should return Dict<Schedule, ContextInfo>, "
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


def multi_level_tiling(
    structure: str,
    must_cache_read: bool,
    cache_read_scope: str,
    can_cache_write: bool,
    must_cache_write: bool,
    cache_write_scope: str,
    fusion_levels: List[int],
    vector_load_max_len: Optional[int] = None,
    tile_marks: Optional[List[str]] = None,
) -> SearchRule:
    """Create a rule that does multi-level tiling if there is sufficient amount of data reuse.
    Optionally add read cache and write cache, do fusion if possible.

    Parameters
    ----------
    structure : str
        Structure of tiling. On CPU, recommended to use 'SSRSRS';
        On GPU, recommended to use 'SSSRRSRS'
    must_cache_read : bool
        Add cache_read before the multi-level tiling
    can_cache_write : bool
        Add cache_write after the multi-level tiling
    must_cache_write : bool
        Must add cache_write after the multi-level tiling
    fusion_levels : List[int]
        The possible tile levels that a single elementwise consumer is fused at
    vector_load_max_len : Optional[int]
        For cache_read, if vectorized load is used, the max length of the vectorized load
    tile_marks : Optional[List[str]]
        The marks to be used on each tile

    Returns
    ----------
    rule: SearchRule
        The rule created
    """
    return _ffi_api_search_rule.MultiLevelTiling(  # pylint: disable=no-member
        structure,
        must_cache_read,
        cache_read_scope,
        can_cache_write,
        must_cache_write,
        cache_write_scope,
        fusion_levels,
        vector_load_max_len,
        tile_marks,
    )


def random_compute_location() -> SearchRule:
    """A rule that randomly select a compute-at location for a free block

    Returns
    ----------
    rule: SearchRule
        The search rule created
    """
    return _ffi_api_search_rule.RandomComputeLocation()  # pylint: disable=no-member


def mark_parallelize_outer(max_jobs_per_core: int) -> SearchRule:
    """Create a rule that parallelizes the outer loops

    Parameters
    ----------
    max_jobs_per_core : int
        The maximum number of jobs to be run each core

    Returns
    ----------
    rule: SearchRule
        The search rule created
    """
    return _ffi_api_search_rule.MarkParallelizeOuter(max_jobs_per_core)  # pylint: disable=no-member


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


def mark_auto_unroll(max_steps: List[int], unroll_explicit: bool) -> SearchRule:
    """Create a rule that marks the loops to be auto-unrolled

    Parameters
    ----------
    max_steps : List[int]
        The candidate of max_steps in auto_unroll
    unroll_explicit : bool
        Whether to unroll explicitly

    Returns
    ----------
    rule: SearchRule
        The search rule created
    """
    return _ffi_api_search_rule.MarkAutoUnroll(  # pylint: disable=no-member
        max_steps, unroll_explicit
    )


def parallelize_vectorize_unroll(
    max_jobs_per_core: int = 16,
    maximize_parallel: bool = True,
    max_vectorize_extent: int = 32,
    unroll_max_steps: Optional[List[int]] = None,
    unroll_explicit: bool = True,
):
    """Mark parallelize, vectorize and unroll to each block correspondingly

    Parameters
    ----------
    max_jobs_per_core: int
        The maximum number of jobs to be launched per CPU core. It sets the uplimit of CPU
        parallism, i.e. `num_cores * max_jobs_per_core`.
        Use -1 to disable parallism.
    maximize_parallel: bool
        Whether to maximize the parallelism in decision making. If true, we
        deterministically parallelize the outer loops to maximum; Otherwise, we randomly pick a
        parallelism extent
    max_vectorize_extent: int
        The maximum extent to be vectorized. It sets the uplimit of the CPU vectorization.
        Use -1 to disable vectorization.
    unroll_max_steps: Optional[List[int]]
        The maximum number of unroll steps to be done.
        Use None to disable unroll
    unroll_explicit: bool
        Whether to explicitly unroll the loop, or just add a unroll pragma

    Returns
    ----------
    rule: SearchRule
        The search rule created
    """
    if unroll_max_steps is None:
        unroll_max_steps = []
    return _ffi_api_search_rule.ParallelizeVectorizeUnroll(  # pylint: disable=no-member
        max_jobs_per_core,
        maximize_parallel,
        max_vectorize_extent,
        unroll_max_steps,
        unroll_explicit,
    )


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
