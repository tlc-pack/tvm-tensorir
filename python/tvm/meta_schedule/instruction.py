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
from tvm._ffi import register_object
from tvm.runtime import Object


@register_object("meta_schedule.Instruction")
class Instruction(Object):
    """Base class for all meta scheduling instrructions"""


@register_object("meta_schedule.SamplePerfectTileInst")
class SamplePerfectTileInst(Instruction):
    """An instruction to sample possible perfect tiling factors"""


@register_object("meta_schedule.SampleTileFactorInst")
class SampleTileFactorInst(Instruction):
    """An instruction to sample possible tiling factors"""


@register_object("meta_schedule.GetBlockInst")
class GetBlockInst(Instruction):
    """An instruction to retrieve a block using its name"""


@register_object("meta_schedule.GetAxesInst")
class GetAxesInst(Instruction):
    """An instruction to retrieve nested loop axes on top of a block"""


@register_object("meta_schedule.SplitInst")
class SplitInst(Instruction):
    """An instruction to split a loop by a set of factors"""


@register_object("meta_schedule.ReorderInst")
class ReorderInst(Instruction):
    """An instruction to reorder the given axes"""


@register_object("meta_schedule.DecomposeReductionInst")
class DecomposeReductionInst(Instruction):
    """An instruction for decompose_reduction in TIR"""
