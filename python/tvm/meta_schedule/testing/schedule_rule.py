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
"""Default schedule rules"""
from typing import List

from tvm.meta_schedule.schedule_rule import (
    AutoInline,
    MultiLevelTiling,
    ParallelizeVectorizeUnroll,
    RandomComputeLocation,
    ReuseType,
    ScheduleRule,
)
from tvm.target import Target


def get(target: Target) -> List[ScheduleRule]:
    """Default schedule rules"""
    if target.kind.name == "llvm":
        return [
            auto_inline(target),
            multi_level_tiling(target),
        ]
    if target.kind.name == "cuda":
        return [
            auto_inline(target),
            multi_level_tiling(target),
            auto_inline_after_tiling(target),
        ]
    raise NotImplementedError(f"{target.kind.name} is not supported")


def auto_inline(target: Target) -> ScheduleRule:
    """Default schedule rules for auto inline"""
    if target.kind.name == "llvm":
        return AutoInline(
            into_producer=False,
            into_consumer=True,
            into_cache_only=False,
            inline_const_tensor=True,
            disallow_if_then_else=True,
            require_injective=True,
            require_ordered=True,
            disallow_op=["tir.exp"],
        )
    if target.kind.name == "cuda":
        return AutoInline(
            into_producer=False,
            into_consumer=True,
            into_cache_only=False,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        )
    raise NotImplementedError(f"{target.kind.name} is not supported")


def auto_inline_after_tiling(target: Target) -> ScheduleRule:
    """Default schedule rules for auto inline after tiling"""
    if target.kind.name == "llvm":
        return AutoInline(
            into_producer=True,
            into_consumer=True,
            into_cache_only=True,
            inline_const_tensor=True,
            disallow_if_then_else=True,
            require_injective=True,
            require_ordered=True,
            disallow_op=["tir.exp"],
        )
    if target.kind.name == "cuda":
        return AutoInline(
            into_producer=True,
            into_consumer=True,
            into_cache_only=True,
            inline_const_tensor=True,
            disallow_if_then_else=False,
            require_injective=False,
            require_ordered=False,
            disallow_op=None,
        )
    raise NotImplementedError(f"{target.kind.name} is not supported")


def multi_level_tiling(target: Target) -> ScheduleRule:
    """Default schedule rules for with multi-level tiling and reuse"""
    if target.kind.name == "llvm":
        return MultiLevelTiling(
            structure="SSRSRS",
            tile_binds=None,
            max_innermost_factor=64,
            vector_load_max_len=None,
            reuse_read=None,
            reuse_write=ReuseType(
                req="may",
                levels=[1, 2],
                scope="global",
            ),
        )
    if target.kind.name == "cuda":
        return MultiLevelTiling(
            structure="SSSRRSRS",
            tile_binds=["blockIdx.x", "vthread.x", "threadIdx.x"],
            max_innermost_factor=64,
            vector_load_max_len=4,
            reuse_read=ReuseType(
                req="must",
                levels=[4],
                scope="shared",
            ),
            reuse_write=ReuseType(
                req="must",
                levels=[3],
                scope="local",
            ),
        )
    raise NotImplementedError(f"{target.kind.name} is not supported")


def parallel_vectorize_unroll(target: Target) -> ScheduleRule:
    """Default schedule rules for with parallel-vectorize-unroll"""
    if target.kind.name == "llvm" or target.kind.name == "cuda":
        return ParallelizeVectorizeUnroll(
            max_jobs_per_core=16,
            max_vectorize_extent=32,
            unroll_max_steps=[0, 16, 64, 512],
            unroll_explicit=True,
        )
    raise NotImplementedError(f"{target.kind.name} is not supported")


def random_compute_location(target: Target) -> ScheduleRule:
    """Default schedule rules for with random-compute-location"""
    if target.kind.name == "llvm":
        return RandomComputeLocation()
    raise NotImplementedError(f"{target.kind.name} is not supported")