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
from typing import List

from tvm._ffi import register_object
from tvm.ir import Attrs
from tvm.runtime import Object


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
    inst_attrs: Attrs


########## Sampling ##########


@register_object("meta_schedule.attrs.SamplePerfectTileAttrs")
class SamplePerfectTileAttrs(Attrs):
    """Attrs of the instruction to sample perfect tile factors"""

    n_splits: int
    max_innermost_factor: int


@register_object("meta_schedule.attrs.SampleTileFactorAttrs")
class SampleTileFactorAttrs(Attrs):
    """Attrs of the instruction to sample tiling factors"""

    n_splits: int
    where: List[int]


########## Block/Loop Relationship ##########


@register_object("meta_schedule.attrs.GetOnlyConsumerAttrs")
class GetOnlyConsumerAttrs(Attrs):
    """Attrs of the instruction that gets the only consumer of a specific block"""


@register_object("meta_schedule.attrs.GetBlockAttrs")
class GetBlockAttrs(Attrs):
    """Attrs of the instruction that gets a specific block by its name"""

    name: str


@register_object("meta_schedule.attrs.GetAxesAttrs")
class GetAxesAttrs(Attrs):
    """Attrs of the instruction that gets loop axes on top of a specifc block"""


########## Scheduling Primitives ##########


@register_object("meta_schedule.attrs.SplitAttrs")
class SplitAttrs(Attrs):
    """Attrs of the instruction that applies loop splitting"""


@register_object("meta_schedule.attrs.ReorderAttrs")
class ReorderAttrs(Attrs):
    """Attrs of the instruction that applies loop reordering"""


@register_object("meta_schedule.attrs.ComputeInlineAttrs")
class ComputeInlineAttrs(Attrs):
    """Attrs of the instruction that applies compute_inline"""


@register_object("meta_schedule.attrs.CacheWriteAttrs")
class CacheWriteAttrs(Attrs):
    """Attrs of the instruction that applies cache_write"""

    storage_scope: str


@register_object("meta_schedule.attrs.DecomposeReductionAttrs")
class DecomposeReductionAttrs(Attrs):
    """Attrs of the instruction that applies decompose_reduction"""
