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

import tvm
from tvm import tir
from tvm.script import ty


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_reduction_roundtrip():
    func_rt = tvm.script.from_source(tvm.script.asscript(matmul))
    tvm.ir.assert_structural_equal(matmul, func_rt)


@tvm.script.tir
def matmul_decompose0(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([128, 128], "init") as [vi, vj]:
        C[vi, vj] = 0.0

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_reduction_decompose():
    s = tir.create_schedule(matmul)
    C = s.get_block("update")
    i, j, k = s.get_axes(C)
    s.decompose_reduction(C, i)
    tvm.ir.assert_structural_equal(matmul_decompose0, s.func)


def test_reduction_merge():
    s = tir.create_schedule(matmul_decompose0)
    init = s.get_block("init")
    update = s.get_block("update")
    s.merge_reduction(init, update)
    tvm.ir.assert_structural_equal(matmul, s.func)


@tvm.script.tir
def matmul_blockzied(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 128):
            with tir.block([128, 1, tir.reduce_axis(0, 1)], "blockized_update") as [vi, vjo, vko]:
                tir.bind(vi, i0)
                tir.bind(vjo, 0)
                tir.bind(vko, 0)
                with tir.init():
                    for i1 in range(0, 128):
                        C[vi, i1] = tir.float32(0)
                with tir.block([128, tir.reduce_axis(0, 128)], "update") as [vj, vk]:
                    C[vi, vj] = (C[vi, vj] + (A[vi, vk]*B[vj, vk]))


def test_reduction_blockize():
    s = tir.create_schedule(matmul)
    C = s.get_block("update")
    i, j, k = s.get_axes(C)
    s.blockize(j)
    tvm.ir.assert_structural_equal(matmul_blockzied, s.func)


@tvm.script.tir
def matmul_scale(a: ty.handle, b: ty.handle, e: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    E = tir.match_buffer(e, [128, 128])

    C = tir.buffer_allocate([128, 128])
    D = tir.buffer_allocate([128, 128])
    with tir.block([128, 128], "D") as [vi, vj]:
        D[vi, vj] = A[vi, vj] * 2.0

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + D[vi, vk] * B[vj, vk]

    with tir.block([128, 128], "E") as [vi, vj]:
        E[vi, vj] = C[vi, vj] + 1.0


@tvm.script.tir
def matmul_scale_inline(a: ty.handle, b: ty.handle, e: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    E = tir.match_buffer(e, [128, 128])

    C = tir.buffer_allocate([128, 128])
    with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + (A[vi, vk]*2.0) * B[vj, vk]

    with tir.block([128, 128], "E") as [vi, vj]:
        E[vi, vj] = C[vi, vj] + 1.0


def test_reduction_compute_inline():
    s = tir.create_schedule(matmul_scale)
    D = s.get_block("D")
    s.compute_inline(D)
    tvm.ir.assert_structural_equal(s.func, matmul_scale_inline)


if __name__ == "__main__":
    test_reduction_roundtrip()
    test_reduction_decompose()
    test_reduction_merge()
    test_reduction_blockize()
    test_reduction_compute_inline()
