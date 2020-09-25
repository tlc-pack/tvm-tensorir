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
"""Meta schedule analysis API """
from typing import List

from tvm.tir import IterVar, BufferLoad

from . import _ffi_api_analysis
from .random_variable import BlockRV
from .schedule import Schedule


def is_trivial_binding(sch: Schedule, block: BlockRV) -> bool:
    return _ffi_api_analysis.IsTrivialBinding(sch, block)  # pylint: disable=no-member


def get_iter_type(sch: Schedule, block: BlockRV) -> List[str]:
    table = {
        IterVar.DataPar: "spatial",
        IterVar.CommReduce: "reduce",
    }
    iter_types = _ffi_api_analysis.GetIterType(sch, block)  # pylint: disable=no-member
    return [table.get(iter_type, "opaque") for iter_type in iter_types]


def is_leaf(sch: Schedule, block: BlockRV) -> bool:
    return _ffi_api_analysis.IsLeaf(sch, block)  # pylint: disable=no-member


def is_body_single_stmt(sch: Schedule, block: BlockRV) -> bool:
    return _ffi_api_analysis.IsBodySingleStmt(sch, block)  # pylint: disable=no-member


def get_buffer_store(sch: Schedule, block: BlockRV) -> BufferLoad:
    return _ffi_api_analysis.GetBufferStore(sch, block)  # pylint: disable=no-member


def get_buffer_load(sch: Schedule, block: BlockRV) -> List[BufferLoad]:
    return _ffi_api_analysis.GetBufferLoad(sch, block)  # pylint: disable=no-member
