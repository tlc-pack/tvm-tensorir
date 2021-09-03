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
def fused_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))

    for i in range(0, 16384):
        with tir.block([128, 128], "B") as [vi, vj]:
            tir.bind(vi, i // 128)
            tir.bind(vj, i % 128)
            B[vi, vj] = A[vi, vj] * 2.0

    for j in range(0, 16384):
        with tir.block([128, 128], "C") as [vi, vj]:
            tir.bind(vi, j // 128)
            tir.bind(vj, j % 128)
            C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def split_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))

    for io, ii, j in tir.grid(8, 16, 128):
        with tir.block([128, 128], "B") as [vi, vj]:
            tir.bind(vi, io * 16 + ii)
            tir.bind(vj, j)
            B[vi, vj] = A[vi, vj] * 2.0

    for i, jo, ji in tir.grid(128, 10, 13):
        with tir.block([128, 128], "C") as [vi, vj]:
            tir.where(jo * 13 + ji < 128)
            tir.bind(vi, i)
            tir.bind(vj, jo * 13 + ji)
            C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def compute_at_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    B = tir.alloc_buffer((128, 128))
    for i in range(0, 128):
        for j in range(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(0, 128):
            with tir.block([128, 128], "C") as [vi, vj]:
                C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def reverse_compute_at_element_wise(a: ty.handle, c: ty.handle) -> None:
    # function attr dict
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)

    # body
    for i0_outer in range(0, 8):
        for i1_outer in range(0, 8):
            for i0_inner in range(0, 16):
                for i1_inner in range(0, 16):
                    with tir.block([128, 128], "B") as [vi, vj]:
                        tir.bind(vi, ((i0_outer * 16) + i0_inner))
                        tir.bind(vj, ((i1_outer * 16) + i1_inner))
                        B[vi, vj] = A[vi, vj] * tir.float32(2)
                for ax1 in range(0, 16):
                    with tir.block([128, 128], "C") as [vi, vj]:
                        tir.bind(vi, ((i0_outer * 16) + i0_inner))
                        tir.bind(vj, ((i1_outer * 16) + ax1))
                        C[vi, vj] = B[vi, vj] + tir.float32(1)


@tvm.script.tir
def predicate_fuse(b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (16, 16), "float32")
    B = tir.match_buffer(b, (16, 16), "float32")
    for i in range(0, 256):
        with tir.block([16, 16], "update") as [vi, vj]:
            tir.bind(vi, tir.floordiv(i, 16))
            tir.bind(vj, (tir.floormod(tir.floordiv(i, 4), 4) * 4) + tir.floormod(i, 4))
            C[vi, vj] = B[vi, vj] + tir.float32(1)


@tvm.script.tir
def matmul_reorder(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.match_buffer(b, (128, 128), "float32")

    for i0, j0 in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)
    for k, i in tir.grid(128, 16384):
        with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
            tir.bind(vi, tir.floordiv(i, 128))
            tir.bind(vj, tir.floormod(i, 128))
            tir.bind(vk, k)
            C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@tvm.script.tir
def inline_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = A[vi, vj] * 2.0 + 1.0


@tvm.script.tir
def blockize(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    for i, j in tir.grid(8, 8):
        with tir.block([8, 8], "blockized_B") as [vi, vj]:
            tir.bind(vi, i)
            tir.bind(vj, j)
            for ii, jj in tir.grid(16, 16):
                with tir.block([128, 128], "B") as [vii, vjj]:
                    tir.bind(vii, vi * 16 + ii)
                    tir.bind(vjj, vj * 16 + jj)
                    B[vii, vjj] = A[vii, vjj] * tir.float32(2)
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = B[vi, vj] + tir.float32(1)


@tvm.script.tir
def element_wise_reverse_inline(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")

    with tir.block([128, 128], "B") as [vi, vj]:
        C[vi, vj] = (A[vi, vj] * 2.0) + 1.0


@tvm.script.tir
def matmul_reduction(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.match_buffer(b, (128, 128), "float32")

    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)
        for k in range(0, 128):
            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@tvm.script.tir
def compute_at_case(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")

    B = tir.alloc_buffer((128, 128))
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "B0") as [vi, vj]:
            A[vi, vj] = 2.0
        for k in range(0, 128):
            with tir.block([128, 128], "B1") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                B[vi, vj] = A[vi, vj] * 2.0
            with tir.block([128, 128], "C") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                C[vi, vj] = B[vi, vj] * 2.0


@tvm.script.tir
def test_func_cache_rw(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    D = tir.alloc_buffer((128, 128), "float32")

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "A") as [vi, vj, vk]:
        A[vi, vj] = A[vi, vj] + B[vi, vk] * C[vj, vk]

    with tir.block([128, 128], "D") as [vi, vj]:
        D[vi, vj] = A[vi, vj]


@tvm.script.tir
def test_func_cache_read(a: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    B = tir.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
    D = tir.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
    A_local = tir.alloc_buffer(
        [128, 128], elem_offset=0, scope="local", align=128, offset_factor=1
    )
    with tir.block([128, 128, tir.reduce_axis(0, 128)], "A") as [vi, vj, vk]:
        A[vi, vj] = A[vi, vj] + (B[vi, vk] * C[vj, vk])
    with tir.block([128, 128], "") as [v0, v1]:
        A_local[v0, v1] = A[v0, v1]
    with tir.block([128, 128], "D") as [vi_1, vj_1]:
        D[vi_1, vj_1] = A_local[vi_1, vj_1]


@tvm.script.tir
def test_func_cache_write(a: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    B = tir.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
    D = tir.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
    A_local = tir.alloc_buffer(
        [128, 128], elem_offset=0, scope="local", align=128, offset_factor=1
    )
    with tir.block([128, 128, tir.reduce_axis(0, 128)], "A") as [vi, vj, vk]:
        A_local[vi, vj] = A_local[vi, vj] + (B[vi, vk] * C[vj, vk])
    with tir.block([128, 128], "") as [v0, v1]:
        A[v0, v1] = A_local[v0, v1]
    with tir.block([128, 128], "D") as [vi_1, vj_1]:
        D[vi_1, vj_1] = A[vi_1, vj_1]


@tvm.script.tir
def cache_read(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    AA = tir.alloc_buffer((128, 128), "float32", scope="local")
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "AA") as [vi, vj]:
            AA[vi, vj] = A[vi, vj]
    for i in range(0, 128):
        for j in range(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                B[vi, vj] = AA[vi, vj] * tir.float32(2)
        for j in range(0, 128):
            with tir.block([128, 128], "C") as [vi, vj]:
                C[vi, vj] = B[vi, vj] + tir.float32(1)

@tvm.script.tir
def cache_write(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    CC = tir.alloc_buffer((128, 128), "float32", scope="local")
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "B") as [vi, vj]:
            B[vi, vj] = A[vi, vj] * tir.float32(2)
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "CC") as [vi, vj]:
            CC[vi, vj] = B[vi, vj] + tir.float32(1)
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = CC[vi, vj]


@tvm.script.tir
def blockize_schedule_1(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        B = tir.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
        for i0_outer in range(0, 8):
            for i1_outer in range(0, 8):
                with tir.block([8, 8], "blockized_B") as [vio, vjo]:
                    tir.bind(vio, i0_outer)
                    tir.bind(vjo, i1_outer)
                    tir.reads([A[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]])
                    tir.writes([B[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]])
                    for i0_inner in range(0, 16):
                        for i1_inner in range(0, 16):
                            with tir.block([128, 128], "B") as [vi, vj]:
                                tir.bind(vi, ((vio * 16) + i0_inner))
                                tir.bind(vj, ((vjo * 16) + i1_inner))
                                tir.reads([A[vi : (vi + 1), vj : (vj + 1)]])
                                tir.writes([B[vi : (vi + 1), vj : (vj + 1)]])
                                B[vi, vj] = A[vi, vj] * tir.float32(2)
                with tir.block([8, 8], "blockized_C") as [vio, vjo]:
                    tir.bind(vio, i0_outer)
                    tir.bind(vjo, i1_outer)
                    tir.reads([B[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]])
                    tir.writes([C[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]])
                    for ax0 in range(0, 16):
                        for ax1 in range(0, 16):
                            with tir.block([128, 128], "C") as [vi, vj]:
                                tir.bind(vi, ((vio * 16) + ax0))
                                tir.bind(vj, ((vjo * 16) + ax1))
                                tir.reads([B[vi : (vi + 1), vj : (vj + 1)]])
                                tir.writes([C[vi : (vi + 1), vj : (vj + 1)]])
                                C[vi, vj] = B[vi, vj] + tir.float32(1)


@tvm.script.tir
def blockize_schedule_2(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        B = tir.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
        for i0_outer in range(0, 4):
            for i1_outer in range(0, 4):
                for ax0 in range(0, 2):
                    for ax1 in range(0, 2):
                        with tir.block([8, 8], "blockized_B") as [vio, vjo]:
                            tir.bind(vio, ((i0_outer * 2) + ax0))
                            tir.bind(vjo, ((i1_outer * 2) + ax1))
                            tir.reads(
                                [A[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]]
                            )
                            tir.writes(
                                [B[(vio * 16) : ((vio * 16) + 16), (vjo * 16) : ((vjo * 16) + 16)]]
                            )
                            for i0_inner in range(0, 16):
                                for i1_inner in range(0, 16):
                                    with tir.block([128, 128], "B") as [vi, vj]:
                                        tir.bind(vi, ((vio * 16) + i0_inner))
                                        tir.bind(vj, ((vjo * 16) + i1_inner))
                                        tir.reads([A[vi : (vi + 1), vj : (vj + 1)]])
                                        tir.writes([B[vi : (vi + 1), vj : (vj + 1)]])
                                        B[vi, vj] = A[vi, vj] * tir.float32(2)
                for i0_inner_1 in range(0, 32):
                    for i1_inner_1 in range(0, 32):
                        with tir.block([128, 128], "C") as [vi, vj]:
                            tir.bind(vi, ((i0_outer * 32) + i0_inner_1))
                            tir.bind(vj, ((i1_outer * 32) + i1_inner_1))
                            tir.reads([B[vi : (vi + 1), vj : (vj + 1)]])
                            tir.writes([C[vi : (vi + 1), vj : (vj + 1)]])
                            C[vi, vj] = B[vi, vj] + tir.float32(1)

@tvm.script.tir
def matmul_pragma(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 128, annotations = {"pragma_auto_unroll_max_step":16, "pragma_unroll_explicit":False}):
            for i1 in range(0, 128):
                for i2 in range(0, 128):
                    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                        tir.bind(vi, i0)
                        tir.bind(vj, i1)
                        tir.bind(vk, i2)
                        tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vj:(vj + 1), vk:(vk + 1)]])
                        tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                        with tir.init():
                            C[vi, vj] = 0.0
                        C[vi, vj] = C[vi, vj] + (A[vi, vk]*B[vj, vk])


@tvm.script.tir
def element_wise_double_buffer(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))

    with tir.block([128, 128], "B") as [vi, vj]:
        tir.block_attr({"double_buffer_scope": 1})
        B[vi, vj] = A[vi, vj] * 2.0

    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def predicate_consumer_block(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [1])
    B = tir.alloc_buffer([1])
    C = tir.match_buffer(c, [1])

    for i0 in tir.serial(0, 1):
        with tir.block([1], "B") as [i]:
            tir.bind(i, i0)
            B[i] = A[i]

    for i1_outer, i1_inner in tir.grid(1, 32):
        with tir.block([1], "C") as [i_1]:
            tir.where((((i1_outer*32) + i1_inner) < 1))
            tir.bind(i_1, i1_inner)
            C[i_1] = B[i_1]


@tvm.script.tir
def predicate_producer_block(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [1])
    B = tir.alloc_buffer([1])
    C = tir.match_buffer(c, [1])

    for i0_outer, i0_inner in tir.grid(1, 32):
        with tir.block([1], "B") as [i]:
            tir.where((((i0_outer*32) + i0_inner) < 1))
            tir.bind(i, i0_inner)
            B[i] = A[i]

    for i1 in tir.serial(0, 1):
        with tir.block([1], "C") as [i_1]:
            tir.bind(i_1, i1)
            C[i_1] = B[i_1]


@tvm.script.tir
def compute_at_with_consumer_predicate(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [1], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        B = tir.alloc_buffer([1], elem_offset=0, align=128, offset_factor=1)
        for i1_outer in tir.serial(0, 1):
            with tir.block([1], "B") as [i]:
                tir.bind(i, 0)
                tir.reads([A[i:(i + 1)]])
                tir.writes([B[i:(i + 1)]])
                B[i] = A[i]
            for i1_inner in tir.serial(0, 32):
                with tir.block([1], "C") as [i_1]:
                    tir.where((((i1_outer*32) + i1_inner) < 1))
                    tir.bind(i_1, i1_inner)
                    tir.reads([B[i_1:(i_1 + 1)]])
                    tir.writes([C[i_1:(i_1 + 1)]])
                    C[i_1] = B[i_1]


@tvm.script.tir
def reverse_compute_at_with_producer_predicate(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [1], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        B = tir.alloc_buffer([1], elem_offset=0, align=128, offset_factor=1)
        for i0_outer in tir.serial(0, 1):
            for i0_inner in tir.serial(0, 32):
                with tir.block([1], "B") as [i]:
                    tir.where((((i0_outer*32) + i0_inner) < 1))
                    tir.bind(i, i0_inner)
                    tir.reads([A[i:(i + 1)]])
                    tir.writes([B[i:(i + 1)]])
                    B[i] = A[i]
            with tir.block([1], "C") as [i_1]:
                tir.bind(i_1, 0)
                tir.reads([B[i_1:(i_1 + 1)]])
                tir.writes([C[i_1:(i_1 + 1)]])
                C[i_1] = B[i_1]


@tvm.script.tir
def element_wise_set_scope(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [128, 128])
    A = tir.match_buffer(a, [128, 128])
    B_shared = tir.alloc_buffer([128, 128], scope="shared")
    for i0 in tir.thread_binding(0, 128, thread = "blockIdx.x"):
        for ax1 in tir.thread_binding(0, 128, thread = "threadIdx.x"):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i0)
                tir.bind(vj, ax1)
                B_shared[vi, vj] = A[vi, vj] * tir.float32(2)
        for i1 in tir.thread_binding(0, 128, thread = "threadIdx.x"):
            with tir.block([128, 128], "C") as [vi_1, vj_1]:
                tir.bind(vi_1, i0)
                tir.bind(vj_1, i1)
                C[vi_1, vj_1] = B_shared[vi_1, vj_1] + tir.float32(1)


@tvm.script.tir
def ewise_arg_inlined(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        B = tir.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
        for i0, i1 in tir.grid(128, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i0)
                tir.bind(vj, i1)
                tir.reads([A[vi, vj]])
                tir.writes([B[vi, vj]])
                B[vi, vj] = (A[vi, vj]*tir.float32(2))
        for i0_1, i1_1 in tir.grid(128, 128):
            with tir.block([128, 128], "C") as [vi_1, vj_1]:
                tir.bind(vi_1, i0_1)
                tir.bind(vj_1, i1_1)
                tir.reads([B[vi_1, vj_1]])
                tir.writes([C[vi_1, vj_1]])
                C[vi_1, vj_1] = (B[vi_1, vj_1] + tir.float32(1))


@tvm.script.tir
def rowsum(a: ty.handle, b:ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128,])
    with tir.block([tir.reduce_axis(0, 128), 128], "B") as [vk, vi]:
        with tir.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vk]


@tvm.script.tir
def rowsum_blockized(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128])
    with tir.block([tir.reduce_axis(0, 1), 1], "blockized_B") as [vko, vio]:
        tir.bind(vko, 0)
        tir.bind(vio, 0)
        with tir.init():
            for i1 in tir.serial(0, 128):
                with tir.block([128], "B_init") as [vi_init]:
                    tir.bind(vi_init, i1)
                    B[vi_init] = tir.float32(0)
        for i0, i1_1 in tir.grid(128, 128):
            with tir.block([tir.reduce_axis(0, 128), 128], "B") as [vk, vi]:
                tir.bind(vk, i0)
                tir.bind(vi, i1_1)
                B[vi] = (B[vi] + A[vi, vk])


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name
# fmt: on

# pylint: disable=invalid-name


def test_fuse():
    func = util.element_wise_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_loops(B)
    s.fuse(outer, inner)
    outer, inner = s.get_loops(C)
    s.fuse(outer, inner)
    mod = tvm.script.create_module({"fused_element_wise": fused_element_wise})
    fused_func = mod["fused_element_wise"]
    tvm.ir.assert_structural_equal(fused_func, s.mod["main"])


def test_split_fuse():
    func = util.element_wise_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_loops(B)
    s.split(outer, factors=[None, 16])
    outer, inner = s.get_loops(C)
    s.split(inner, factors=[10, None])
    mod = tvm.script.create_module({"split_element_wise": split_element_wise})
    split_func = mod["split_element_wise"]
    tvm.ir.assert_structural_equal(split_func, s.mod["main"])


def test_compute_at():
    func = util.element_wise_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, _ = s.get_loops(C)
    s.compute_at(B, outer)
    mod = tvm.script.create_module({"compute_at_element_wise": compute_at_element_wise})
    split_func = mod["compute_at_element_wise"]
    tvm.ir.assert_structural_equal(split_func, s.mod["main"])


def test_reverse_compute_at():
    func = util.element_wise_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    i, j = s.get_loops(B)
    i1, i2 = s.split(i, factors=[None, 16])
    j1, j2 = s.split(j, factors=[None, 16])
    s.reorder(i1, j1, i2, j2)
    s.reverse_compute_at(C, i2)
    tvm.ir.assert_structural_equal(reverse_compute_at_element_wise, s.mod["main"])


def test_compute_at_with_consumer_predicate():
    func = predicate_consumer_block
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    i1_o, i1_i = s.get_loops(C)
    s.compute_at(B, i1_o)
    tvm.ir.assert_structural_equal(compute_at_with_consumer_predicate, s.mod["main"])


def test_reverse_compute_at_with_producer_predicate():
    func = predicate_producer_block
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    i0_o, i0_i = s.get_loops(B)
    s.reverse_compute_at(C, i0_o)
    tvm.ir.assert_structural_equal(reverse_compute_at_with_producer_predicate, s.mod["main"])


def test_fuse_loop_sref():
    func = util.predicate_stmt()

    # schedule
    s = tir.Schedule(func, debug_mode=True)
    update = s.get_block("update")
    i, jo, ji = s.get_loops(update)
    ijo = s.fuse(i, jo)
    s.fuse(ijo, ji)

    mod = tvm.script.create_module({"predicate_fuse": predicate_fuse})
    predicate_fuse_func = mod["predicate_fuse"]

    tvm.ir.assert_structural_equal(s.mod["main"], predicate_fuse_func)


def test_reorder_normal():
    func = util.matmul_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    update = s.get_block("update")
    i, j, k = s.get_loops(update)
    s.reorder(k, i)
    s.reorder(i, j)
    s.decompose_reduction(update, k)
    s.fuse(i, j)
    mod = tvm.script.create_module({"matmul_reorder": matmul_reorder})
    matmul_reorder_func = mod["matmul_reorder"]
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_reorder_func)


def test_compute_inline():
    func = util.element_wise_stmt()

    # schedule
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    s.compute_inline(B)

    tvm.ir.assert_structural_equal(inline_element_wise, s.mod["main"])


def test_reverse_compute_inline():
    func = util.element_wise_stmt()

    # schedule
    s = tir.Schedule(func, debug_mode=True)
    C = s.get_block("C")
    s.reverse_compute_inline(C)
    tvm.ir.assert_structural_equal(element_wise_reverse_inline, s.mod["main"])


def test_compute_at_fail():
    mod = tvm.script.create_module({"compute_at_case": compute_at_case})
    func = mod["compute_at_case"]
    s = tir.Schedule(func, debug_mode=True)
    B1 = s.get_block("B1")
    C = s.get_block("C")
    i, j, _ = s.get_loops(C)
    # TODO(@junrushao1994)
    try:
        s.compute_at(C, j)
        assert False
    except:  # pylint: disable=bare-except
        pass
    with pytest.raises(ValueError):
        s.compute_at(B1, i)


def test_reduction():
    func = util.matmul_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    update = s.get_block("update")
    i, _, k = s.get_loops(update)
    init = s.decompose_reduction(update, i)
    i, j_i = s.get_loops(init)
    s.split(j_i, factors=[None, 4])
    s.merge_reduction(init, update)
    s.decompose_reduction(update, k)
    mod = tvm.script.create_module({"matmul_reduction": matmul_reduction})
    matmul_reduction_func = mod["matmul_reduction"]
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_reduction_func)


def test_cache_read():
    func = util.element_wise_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, _ = s.get_loops(C)
    s.compute_at(B, outer)
    _ = s.cache_read(B, 0, "local")
    mod = tvm.script.create_module({"cache_read": cache_read})
    cached_func = mod["cache_read"]
    tvm.ir.assert_structural_equal(cached_func, s.mod["main"])


def test_cache_write():
    func = util.element_wise_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    C = s.get_block("C")
    _ = s.cache_write(C, 0, "local")
    mod = tvm.script.create_module({"cache_write": cache_write})
    cached_func = mod["cache_write"]
    tvm.ir.assert_structural_equal(cached_func, s.mod["main"])


def test_blockize():
    func = util.element_wise_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    _ = s.get_block("C")
    x, y = s.get_loops(B)
    xo, xi = s.split(x, factors=[None, 16])
    yo, yi = s.split(y, factors=[None, 16])
    s.reorder(xo, yo, xi, yi)
    s.blockize(xi)
    mod = tvm.script.create_module({"blockize": blockize})
    blockized_func = mod["blockize"]
    tvm.ir.assert_structural_equal(blockized_func, s.mod["main"])


def test_cache_read_write():
    func = test_func_cache_rw
    # schedule cache read
    s = tir.Schedule(func, debug_mode=True)
    blockA = s.get_block("A")
    s.cache_read(blockA, 0, "local")
    tvm.ir.assert_structural_equal(test_func_cache_read, s.mod["main"])

    # schedule cache write
    s = tir.Schedule(func, debug_mode=True)
    blockA = s.get_block("A")
    s.cache_write(blockA, 0, "local")
    tvm.ir.assert_structural_equal(test_func_cache_write, s.mod["main"])


def test_blockize_schedule():
    func = util.element_wise_stmt()
    # test 1
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    x, y = s.get_loops(B)
    xo, xi = s.split(x, factors=[None, 16])
    yo, yi = s.split(y, factors=[None, 16])
    s.reorder(xo, yo, xi, yi)
    s.blockize(xi)
    s.reverse_compute_at(C, yo)
    s.blockize(s.get_loops(C)[-2])
    tvm.ir.assert_structural_equal(s.mod["main"], blockize_schedule_1)
    # test 2
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    x, y = s.get_loops(C)
    xo, xi = s.split(x, factors=[None, 16])
    yo, yi = s.split(y, factors=[None, 16])
    s.reorder(xo, yo, xi, yi)
    s.blockize(xi)
    s.compute_at(B, yo)
    s.blockize(s.get_loops(B)[-2])
    tvm.ir.assert_structural_equal(s.mod["main"], blockize_schedule_1)
    # test 3
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    C = s.get_block("C")
    x, y = s.get_loops(B)
    xo, xi = s.split(x, factors=[None, 16])
    yo, yi = s.split(y, factors=[None, 16])
    s.reorder(xo, yo, xi, yi)
    b_outer = s.blockize(xi)
    xC, yC = s.get_loops(C)
    xCo, xCi = s.split(xC, factors=[None, 32])
    yCo, yCi = s.split(yC, factors=[None, 32])
    s.reorder(xCo, yCo, xCi, yCi)
    s.compute_at(b_outer, yCo)
    tvm.ir.assert_structural_equal(s.mod["main"], blockize_schedule_2)


def test_blockize_init_loops():
    s = tir.Schedule(rowsum, debug_mode=True)
    k, _ = s.get_loops(s.get_block("B"))
    s.blockize(k)
    tvm.ir.assert_structural_equal(s.mod["main"], rowsum_blockized)


def test_pragma():
    func = util.matmul_stmt()
    s = tir.Schedule(func, debug_mode=True)
    C = s.get_block("update")
    i, _, _ = s.get_loops(C)
    s.pragma(i, "auto_unroll_max_step", 16)
    s.pragma(i, "unroll_explicit", False)
    tvm.ir.assert_structural_equal(matmul_pragma, s.mod["main"])


def test_double_buffer():
    func = util.element_wise_stmt()
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("B")
    s.double_buffer(B)
    tvm.ir.assert_structural_equal(element_wise_double_buffer, s.mod["main"])


def test_set_scope():
    func = util.element_wise_stmt()
    s = tir.Schedule(func, debug_mode=True)
    B, C = s.get_block("B"), s.get_block("C")
    ci, cj = s.get_loops(C)
    s.compute_at(B, ci)
    bi, bj = s.get_loops(B)
    s.bind(bj, "threadIdx.x")
    s.bind(cj, "threadIdx.x")
    s.bind(bi, "blockIdx.x")
    s.set_scope(B, 0, "shared")
    tvm.ir.assert_structural_equal(element_wise_set_scope, s.mod["main"])


def test_inline_argument():
    s = tir.Schedule(util.ewise2, debug_mode=True)
    s.inline_argument(1)
    expected = util.element_wise_stmt()
    tvm.ir.assert_structural_equal(expected, s.mod["main"])
    s.compute_inline(s.get_block("B"))
    tvm.ir.assert_structural_equal(inline_element_wise, s.mod["main"])


if __name__ == "__main__":
    # test_fuse()
    # test_split_fuse()
    # test_fuse_loop_sref()
    # test_reorder_normal()
    # test_compute_at()
    # test_reverse_compute_at()
    test_compute_at_with_consumer_predicate()
    # test_reverse_compute_at_with_producer_predicate()
    # test_compute_inline()
    # test_reverse_compute_inline()
    # test_compute_at_fail()
    # test_reduction()
    # test_cache_read()
    # test_cache_write()
    # test_cache_read_write()
    # test_blockize()
    test_blockize_schedule()
    # test_blockize_init_loops()
    # test_pragma()
    # test_double_buffer()
    # test_set_scope()
    test_inline_argument()
