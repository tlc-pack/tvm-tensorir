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
# pylint: disable=missing-function-docstring,missing-module-docstring
import pytest
import tvm
from tvm import tir
from tvm.script import ty

import util

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name


@tvm.script.tir
def element_wise_storage_align(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        B = tir.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
        for i0 in tir.serial(0, 128):
            for ax1 in tir.serial(0, 128):
                with tir.block([128, 128], "B") as [vi, vj]:
                    tir.bind(vi, i0)
                    tir.bind(vj, ax1)
                    tir.reads([A[vi, vj]])
                    tir.writes([B[vi, vj]])
                    tir.block_attr({"buffer_dim_align":[[[0, 128, 127]]]})
                    B[vi, vj] = (A[vi, vj]*tir.float32(2))
            for i1 in tir.serial(0, 128):
                with tir.block([128, 128], "C") as [vi_1, vj_1]:
                    tir.bind(vi_1, i0)
                    tir.bind(vj_1, i1)
                    tir.reads([B[vi_1, vj_1]])
                    tir.writes([C[vi_1, vj_1]])
                    C[vi_1, vj_1] = (B[vi_1, vj_1] + tir.float32(1))


def test_storage_align():
    func = util.element_wise_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, _ = s.get_axes(C)
    s.compute_at(B, outer)
    s.storage_align(B, 0, axis=0, factor=128, offset=127)
    tvm.ir.assert_structural_equal(element_wise_storage_align, s.mod["main"])


if __name__ == "__main__":
    test_storage_align()
