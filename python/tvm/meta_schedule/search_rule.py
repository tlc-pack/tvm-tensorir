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
from typing import Any, Callable, List, Optional

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

    name: str

    def __init__(
        self,
        name: str,
        apply: Callable[[SearchTask, Schedule, BlockRV], List[Schedule]],
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
    ) -> List[Schedule]:
        """Apply the rule to a block

        Parameters
        ----------
        task: SearchTask
            The search task
        sch: Schedule
            Where the schedule snippets should be generated
        block: BlockRV
            The block the rule applies on

        Returns
        ----------
        result: List[Schedule]
            The new schedules generated
        """
        return _ffi_api.SearchRuleApply(self, task, sch, block)  # pylint: disable=no-member


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
        ) -> List[Schedule]:
            ret = func(task, sch, block)
            if isinstance(ret, Schedule):
                return [ret]
            if isinstance(ret, (list, tuple)):
                for item in ret.items():
                    assert isinstance(item, Schedule)
                return ret
            raise TypeError(
                "SearchRule should return List[Schedule], " + f"but gets type '{type(ret)}': {ret}"
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
    tile_binds: Optional[List[str]] = None,
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
    tile_binds : Optional[List[str]]
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
        tile_binds,
    )


def random_compute_location() -> SearchRule:
    """A rule that randomly select a compute-at location for a free block

    Returns
    ----------
    rule: SearchRule
        The search rule created
    """
    return _ffi_api_search_rule.RandomComputeLocation()  # pylint: disable=no-member


def parallelize_vectorize_unroll(
    max_jobs_per_core: int = 16,
    max_vectorize_extent: int = 32,
    unroll_max_steps: Optional[List[int]] = None,
    unroll_explicit: bool = True,
) -> SearchRule:
    """Mark parallelize, vectorize and unroll to each block correspondingly

    Parameters
    ----------
    max_jobs_per_core: int
        The maximum number of jobs to be launched per CPU core. It sets the uplimit of CPU
        parallism, i.e. `num_cores * max_jobs_per_core`.
        Use -1 to disable parallism.
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
