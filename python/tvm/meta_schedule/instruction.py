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
""" Meta Schedule Instructions """
from typing import List, Union

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir.schedule import BlockRV, ExprRV, LoopRV

RAND_VAR_TYPE = Union[ExprRV, BlockRV, LoopRV]  # pylint: disable=invalid-name


########## Instruction ##########


@register_object("meta_schedule.InstAttrs")
class InstAttrs(Object):
    """Attributes of an instruction"""


@register_object("meta_schedule.Instruction")
class Instruction(Object):
    """An instruction in meta scheduling
    Parameters
    ----------
    inputs : List[Object]
        The input random variables it consumers
    outputs : List[Object]
        The output random variables it produces
    inst_attrs : Attrs
        The attributes of the instruction
    """

    inputs: List[Object]
    outputs: List[Object]
    inst_attrs: InstAttrs


########## Sampling ##########


@register_object("meta_schedule.attrs.SamplePerfectTileAttrs")
class SamplePerfectTileAttrs(InstAttrs):
    """Attrs of the instruction to sample perfect tile factors"""

    n_splits: int
    max_innermost_factor: int


@register_object("meta_schedule.attrs.SampleCategoricalAttrs")
class SampleCategoricalAttrs(InstAttrs):
    """Attrs of the instruction to sample from a categorical distribution"""

    candidates: List[int]
    probs: List[float]


@register_object("meta_schedule.attrs.SampleComputeLocationAttrs")
class SampleComputeLocationAttrs(InstAttrs):
    """Attrs of the instruction to sample a compute-at location from a block"""


########## Block/Loop Relationship ##########


@register_object("meta_schedule.attrs.GetProducersAttrs")
class GetProducersAttrs(InstAttrs):
    """Attrs of the instruction that gets the producers of a specific block"""


@register_object("meta_schedule.attrs.GetConsumersAttrs")
class GetConsumersAttrs(InstAttrs):
    """Attrs of the instruction that gets the consumers of a specific block"""


@register_object("meta_schedule.attrs.GetBlockAttrs")
class GetBlockAttrs(InstAttrs):
    """Attrs of the instruction that gets a specific block by its name"""

    name: str


@register_object("meta_schedule.attrs.GetAxesAttrs")
class GetAxesAttrs(InstAttrs):
    """Attrs of the instruction that gets loop axes on top of a specifc block"""


########## Scheduling Primitives ##########


@register_object("meta_schedule.attrs.MarkLoopAttrs")
class MarkLoopAttrs(InstAttrs):
    """Attrs of the instruction that marks specific loops"""


@register_object("meta_schedule.attrs.MarkBlockAttrs")
class MarkBlockAttrs(InstAttrs):
    """Attrs of the instruction that marks a specific block"""


@register_object("meta_schedule.attrs.FuseAttrs")
class FuseAttrs(InstAttrs):
    """Attrs of the instruction that applies fusion"""


@register_object("meta_schedule.attrs.SplitAttrs")
class SplitAttrs(InstAttrs):
    """Attrs of the instruction that applies loop splitting"""


@register_object("meta_schedule.attrs.ReorderAttrs")
class ReorderAttrs(InstAttrs):
    """Attrs of the instruction that applies loop reordering"""


@register_object("meta_schedule.attrs.ComputeAtAttrs")
class ComputeAtAttrs(InstAttrs):
    """Attrs of the instruction that applies compute_at"""


@register_object("meta_schedule.attrs.ReverseComputeAtAttrs")
class ReverseComputeAtAttrs(InstAttrs):
    """Attrs of the instruction that applies reverse_compute_at"""


@register_object("meta_schedule.attrs.ComputeInlineAttrs")
class ComputeInlineAttrs(InstAttrs):
    """Attrs of the instruction that applies compute_inline"""


@register_object("meta_schedule.attrs.CacheReadAttrs")
class CacheReadAttrs(InstAttrs):
    """Attrs of the instruction that applies cache_read"""

    storage_scope: str


@register_object("meta_schedule.attrs.CacheWriteAttrs")
class CacheWriteAttrs(InstAttrs):
    """Attrs of the instruction that applies cache_write"""

    storage_scope: str


@register_object("meta_schedule.attrs.DecomposeReductionAttrs")
class DecomposeReductionAttrs(InstAttrs):
    """Attrs of the instruction that applies decompose_reduction"""


@register_object("meta_schedule.attrs.TensorizeAttrs")
class TensorizeAttrs(InstAttrs):
    """Attrs of the instruction that applies tensorize"""


@register_object("meta_schedule.attrs.EnterPostProcAttrs")
class EnterPostProcAttrs(InstAttrs):
    """Attrs of an NOP that indicates entrance of post processing"""


@register_object("meta_schedule.attrs.ParallelAttrs")
class ParallelAttrs(InstAttrs):
    """Attrs of the instruction that applies parallel"""


@register_object("meta_schedule.attrs.VectorizeAttrs")
class VectorizeAttrs(InstAttrs):
    """Attrs of the instruction that applies vectorize"""


@register_object("meta_schedule.attrs.RFactorAttrs")
class RFactorAttrs(InstAttrs):
    """Attrs of the instruction that applies rfactor"""


@register_object("meta_schedule.attrs.SetScopeAttrs")
class SetScopeAttrs(InstAttrs):
    """Attrs of the instruction that applies set_scope"""
