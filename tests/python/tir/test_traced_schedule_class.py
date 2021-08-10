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
# pylint: disable=missing-function-docstring
"""Test for traced schedule class"""
import pytest
from collections import defaultdict
from typing import Union

import tvm
from tvm import tir
from tvm.ir import IRModule
from tvm.script import ty
from tvm.tir import ForKind, PrimFunc
from tvm.tir.schedule import Trace

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks
# fmt: off

@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
def matmul_relu(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    D = tir.match_buffer(d, (1024, 1024), "float32")
    C = tir.alloc_buffer((1024, 1024), "float32")
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    with tir.block([1024, 1024], "relu") as [vi, vj]:
        D[vi, vj] = tir.max(C[vi, vj], 0.0)


@tvm.script.tir
def plus_one_matmul(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    D = tir.match_buffer(d, (1024, 1024), "float32")
    C = tir.alloc_buffer((1024, 1024), "float32")
    with tir.block([1024, 1024], "plus_one") as [vi, vj]:
        C[vi, vj] = A[vi, vj] + 1.0
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        with tir.init():
            D[vi, vj] = 0.0
        D[vi, vj] = D[vi, vj] + C[vi, vk] * B[vk, vj]


@tvm.script.tir
def plus_one_matmul_fused(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    D = tir.match_buffer(d, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.alloc_buffer([1024, 1024], elem_offset=0, align=128, offset_factor=1)
        for i0 in range(0, 1024):
            for i1 in range(0, 1024):
                for i2 in range(0, 1024):
                    for ax0, ax1 in tir.grid(1, 1):  # pylint: disable=unused-variable
                        with tir.block([1024, 1024], "plus_one") as [vi, vj]:
                            tir.bind(vi, i0)
                            tir.bind(vj, i2)
                            tir.reads([A[vi:(vi + 1), vj:(vj + 1)]])
                            tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                            C[vi, vj] = (A[vi, vj] + tir.float32(1))
                    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi_1, vj_1, vk]:
                        tir.bind(vi_1, i0)
                        tir.bind(vj_1, i1)
                        tir.bind(vk, i2)
                        tir.reads([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)], C[vi_1:(vi_1 + 1), vk:(vk + 1)], B[vk:(vk + 1), vj_1:(vj_1 + 1)]])
                        tir.writes([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                        with tir.init():
                            D[vi_1, vj_1] = 0.0
                        D[vi_1, vj_1] = D[vi_1, vj_1] + C[vi_1, vk] * B[vk, vj_1]


@tvm.script.tir
def matmul_blockized(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 1024):
            for i1 in range(0, 1024):
                with tir.block([1024, 1024, tir.reduce_axis(0, 1)], "blockized_C") as [vi, vj, vk]:
                    tir.bind(vi, i0)
                    tir.bind(vj, i1)
                    tir.bind(vk, 0)
                    tir.reads([
                        C[vi : (vi + 1), vj : (vj + 1)],
                        A[vi : (vi + 1), 0:1024],
                        B[0:1024, vj : (vj + 1)],
                    ])
                    tir.writes([C[vi : (vi + 1), vj : (vj + 1)]])
                    with tir.init():
                        with tir.block([], "matmul_init") as []:
                            tir.reads([])
                            tir.writes([
                                C[vi : (vi + 1), vj : (vj + 1)]
                            ])
                            C[vi, vj] = tir.float32(0)
                    for i2 in range(0, 1024):
                        with tir.block([tir.reduce_axis(0, 1024)], "C") as [vk]:
                            tir.bind(vk, i2)
                            tir.reads([
                                C[vi : (vi + 1), vj : (vj + 1)],
                                A[vi : (vi + 1), vk : (vk + 1)],
                                B[vk : (vk + 1), vj : (vj + 1)],
                            ])
                            tir.writes([C[vi : (vi + 1), vj : (vj + 1)]])
                            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
def matmul_decomposed(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 1024):
            for i1 in range(0, 1024):
                with tir.block([1024, 1024], "matmul_init") as [vi_init, vj_init]:
                    tir.bind(vi_init, i0)
                    tir.bind(vj_init, i1)
                    tir.reads([])
                    tir.writes([C[vi_init:(vi_init + 1), vj_init:(vj_init + 1)]])
                    C[vi_init, vj_init] = tir.float32(0)
                for i2 in range(0, 1024):
                    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul_update") as [vi, vj, vk]:
                        tir.bind(vi, i0)
                        tir.bind(vj, i1)
                        tir.bind(vk, i2)
                        tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                        tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                        C[vi, vj] = (C[vi, vj] + (A[vi, vk]*B[vk, vj]))


@tvm.script.tir
def matmul_relu_fused(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    D = tir.match_buffer(d, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.alloc_buffer([1024, 1024], elem_offset=0, align=128, offset_factor=1)
        for i0 in range(0, 1024):
            for i1 in range(0, 1024):
                for i2 in range(0, 1024):
                    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
                        tir.bind(vi, i0)
                        tir.bind(vj, i1)
                        tir.bind(vk, i2)
                        tir.reads([
                            C[vi : (vi + 1), vj : (vj + 1)],
                            A[vi : (vi + 1), vk : (vk + 1)],
                            B[vk : (vk + 1), vj : (vj + 1)],
                        ])
                        tir.writes([C[vi : (vi + 1), vj : (vj + 1)]])
                        with tir.init():
                            C[vi, vj] = 0.0
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
                for ax0 in range(0, 1):  # pylint: disable=unused-variable
                    for ax1 in range(0, 1):  # pylint: disable=unused-variable
                        with tir.block([1024, 1024], "relu") as [vi_1, vj_1]:
                            tir.bind(vi_1, i0)
                            tir.bind(vj_1, i1)
                            tir.reads([C[vi_1 : (vi_1 + 1), vj_1 : (vj_1 + 1)]])
                            tir.writes([D[vi_1 : (vi_1 + 1), vj_1 : (vj_1 + 1)]])
                            D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))


@tvm.script.tir
def matmul_cache_read(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        A_local = tir.alloc_buffer([1024, 1024], elem_offset=0, scope="local", align=128, offset_factor=1)
        B_local = tir.alloc_buffer([1024, 1024], elem_offset=0, scope="local", align=128, offset_factor=1)
        for ax0 in range(0, 1024):
            for ax1 in range(0, 1024):
                with tir.block([1024, 1024], "") as [v0, v1]:
                    tir.bind(v0, ax0)
                    tir.bind(v1, ax1)
                    tir.reads([B[v0:(v0 + 1), v1:(v1 + 1)]])
                    tir.writes([B_local[v0:(v0 + 1), v1:(v1 + 1)]])
                    B_local[v0, v1] = B[v0, v1]
        for ax0_1 in range(0, 1024):
            for ax1_1 in range(0, 1024):
                with tir.block([1024, 1024], "") as [v0_1, v1_1]:
                    tir.bind(v0_1, ax0_1)
                    tir.bind(v1_1, ax1_1)
                    tir.reads([A[v0_1:(v0_1 + 1), v1_1:(v1_1 + 1)]])
                    tir.writes([A_local[v0_1:(v0_1 + 1), v1_1:(v1_1 + 1)]])
                    A_local[v0_1, v1_1] = A[v0_1, v1_1]
        for i0 in range(0, 1024):
            for i1 in range(0, 1024):
                for i2 in range(0, 1024):
                    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
                        tir.bind(vi, i0)
                        tir.bind(vj, i1)
                        tir.bind(vk, i2)
                        tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A_local[vi:(vi + 1), vk:(vk + 1)], B_local[vk:(vk + 1), vj:(vj + 1)]])
                        tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                        with tir.init():
                            C[vi, vj] = 0.0
                        C[vi, vj] = C[vi, vj] + A_local[vi, vk]*B_local[vk, vj]


@tvm.script.tir
def matmul_cache_write(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.alloc_buffer([1024, 1024], elem_offset=0, scope="local", align=128, offset_factor=1)
        for i0 in range(0, 1024):
            for i1 in range(0, 1024):
                for i2 in range(0, 1024):
                    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "C") as [vi, vj, vk]:
                        tir.bind(vi, i0)
                        tir.bind(vj, i1)
                        tir.bind(vk, i2)
                        tir.reads([
                            C_local[vi : (vi + 1), vj : (vj + 1)],
                            A[vi : (vi + 1), vk : (vk + 1)],
                            B[vk : (vk + 1), vj : (vj + 1)],
                        ])
                        tir.writes([C_local[vi : (vi + 1), vj : (vj + 1)]])
                        with tir.init():
                            C_local[vi, vj] = 0.0
                        C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
        for ax0 in range(0, 1024):
            for ax1 in range(0, 1024):
                with tir.block([1024, 1024], "") as [v0, v1]:
                    tir.bind(v0, ax0)
                    tir.bind(v1, ax1)
                    tir.reads([C_local[v0 : (v0 + 1), v1 : (v1 + 1)]])
                    tir.writes([C[v0 : (v0 + 1), v1 : (v1 + 1)]])
                    C[v0, v1] = C_local[v0, v1]


@tvm.script.tir
def elementwise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.alloc_buffer((1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    with tir.block([1024, 1024], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] + 1.0
    with tir.block([1024, 1024], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] * 2.0


@tvm.script.tir
def elementwise_inlined(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    with tir.block([1024, 1024], "C") as [vi, vj]:
        C[vi, vj] = (A[vi, vj] + 1.0) * 2.0



@tvm.script.tir
def tensorize_desc(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = tir.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = tir.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        for i, j, k in tir.grid(16, 16, 16):
            with tir.block([16, 16, tir.reduce_axis(0, 16)], "update") as [vii, vjj, vkk]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                tir.bind(vkk, vk + k)
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vkk, vjj]


@tvm.script.tir
def tensorize_impl(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = tir.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = tir.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        for i, j, k in tir.grid(16, 16, 16):
            with tir.block([16, 16, tir.reduce_axis(0, 16)], "update") as [vii, vjj, vkk]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                tir.bind(vkk, vk + k)
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vkk, vjj]


@tvm.script.tir
def matmul_tensorized(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0_outer in range(0, 64):
            for i1_outer in range(0, 64):
                for i0_inner_init in range(0, 16):
                    for i1_inner_init in range(0, 16):
                        with tir.block([1024, 1024], "matmul_init") as [vi_init, vj_init]:
                            tir.bind(vi_init, ((i0_outer*16) + i0_inner_init))
                            tir.bind(vj_init, ((i1_outer*16) + i1_inner_init))
                            tir.reads([])
                            tir.writes([C[vi_init:(vi_init + 1), vj_init:(vj_init + 1)]])
                            C[vi_init, vj_init] = tir.float32(0)
                for i2_outer in range(0, 64):
                    with tir.block([64, 64, tir.reduce_axis(0, 64)], "blockized_matmul_update") as [vio, vjo, vko]:
                        tir.bind(vio, i0_outer)
                        tir.bind(vjo, i1_outer)
                        tir.bind(vko, i2_outer)
                        tir.reads([C[(vio*16):((vio*16) + 16), (vjo*16):((vjo*16) + 16)], A[(vio*16):((vio*16) + 16), (vko*16):((vko*16) + 16)], B[(vko*16):((vko*16) + 16), (vjo*16):((vjo*16) + 16)]])
                        tir.writes([C[(vio*16):((vio*16) + 16), (vjo*16):((vjo*16) + 16)]])
                        for i in range(0, 16):
                            for j in range(0, 16):
                                for k in range(0, 16):
                                    with tir.block([16, 16, tir.reduce_axis(0, 16)], "update") as [vii, vjj, vkk]:
                                        tir.bind(vii, ((vio*16) + i))
                                        tir.bind(vjj, ((vjo*16) + j))
                                        tir.bind(vkk, ((vko*16) + k))
                                        tir.reads([C[vii:(vii + 1), vjj:(vjj + 1)], A[vii:(vii + 1), vkk:(vkk + 1)], B[vkk:(vkk + 1), vjj:(vjj + 1)]])
                                        tir.writes([C[vii:(vii + 1), vjj:(vjj + 1)]])
                                        C[vii, vjj] = (C[vii, vjj] + (A[vii, vkk]*B[vkk, vjj]))


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


def _check_serialization(sch: tir.Schedule, mod: Union[PrimFunc, IRModule]) -> tir.Schedule:
    record = sch.trace.as_json()
    new_sch = tir.Schedule(mod=mod, traced=True)
    Trace.apply_json_to_schedule(record, sch=new_sch)
    assert tvm.ir.structural_equal(new_sch.mod, sch.mod)
    py_repr = "\n".join(sch.trace.as_python())
    new_py_repr = "\n".join(new_sch.trace.as_python())
    assert py_repr == new_py_repr
    # print(py_repr)
    return new_sch


##########  Utility  ##########


def test_traced_schedule_copy():
    sch = tir.Schedule(mod=matmul, traced=True)
    i, j, k = sch.get_loops(sch.get_block("matmul"))
    sch_copy = sch.copy(seed=42)
    assert not sch.get_sref(i).same_as(sch_copy.get_sref(i))
    assert not sch.get_sref(j).same_as(sch_copy.get_sref(j))
    assert not sch.get_sref(k).same_as(sch_copy.get_sref(k))
    assert sch.get_sref(i).stmt.same_as(sch_copy.get_sref(i).stmt)
    assert sch.get_sref(j).stmt.same_as(sch_copy.get_sref(j).stmt)
    assert sch.get_sref(k).stmt.same_as(sch_copy.get_sref(k).stmt)
    i_0, i_1 = sch.split(i, factors=[None, 512])
    j_0, j_1 = sch_copy.split(j, factors=[None, 256])

    assert sch.get_sref(i_0).stmt.extent == 2
    assert sch.get_sref(i_1).stmt.extent == 512
    with pytest.raises(IndexError):
        sch_copy.get_sref(i_0)
    with pytest.raises(IndexError):
        sch_copy.get_sref(i_1)

    with pytest.raises(IndexError):
        sch.get_sref(j_0)
    with pytest.raises(IndexError):
        sch.get_sref(j_1)
    assert sch_copy.get_sref(j_0).stmt.extent == 4
    assert sch_copy.get_sref(j_1).stmt.extent == 256
    _check_serialization(sch, mod=matmul)
    _check_serialization(sch_copy, mod=matmul)


##########  Sampling  ##########


def test_traced_schedule_sample_perfect_tile():
    sch = tir.Schedule(matmul, traced=True)
    i, _, _ = sch.get_loops(sch.get_block("matmul"))
    factors = sch.sample_perfect_tile(i, n=4)
    factors = [sch.get(i) for i in factors]
    prod = factors[0] * factors[1] * factors[2] * factors[3]
    assert prod == 1024
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_sample_categorical():
    n = 1000
    sch = tir.Schedule(mod=matmul, traced=True)
    counter = defaultdict(int)
    candidates = [5, 2, 7, 1]
    probs = [0.15, 0.55, 0.05, 0.25]
    for _ in range(n):
        v = sch.get(sch.sample_categorical(candidates, probs))
        counter[v] += 1
    for i, prob in enumerate(probs):
        assert (prob - 0.07) * n <= counter[candidates[i]] <= (prob + 0.07) * n
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_sample_compute_location():
    counter = defaultdict(int)
    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    for _ in range(100):
        loop = sch.sample_compute_location(block=block)
        loop = sch.get_sref(loop)
        counter[str(loop)] += 1
        new_sch = _check_serialization(sch, mod=matmul)
        old_decision = int(sch.trace.decisions[sch.trace.insts[-1]])
        new_decision = int(new_sch.trace.decisions[new_sch.trace.insts[-1]])
        assert old_decision == new_decision
    assert len(counter) == 5
    assert str(tir.schedule.StmtSRef.root_mark()) in counter
    assert str(tir.schedule.StmtSRef.inline_mark()) in counter


##########  Get blocks & loops  ##########


def test_traced_schedule_get_block():
    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    assert tvm.ir.structural_equal(
        sch.get_sref(block).stmt,
        matmul.body.block.body.body.body.body.block,
    )
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_get_loops():
    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    axes = sch.get_loops(block)
    i_0, i_1, i_2 = [sch.get_sref(i).stmt for i in axes]
    assert tvm.ir.structural_equal(i_0, matmul.body.block.body)
    assert tvm.ir.structural_equal(i_1, matmul.body.block.body.body)
    assert tvm.ir.structural_equal(i_2, matmul.body.block.body.body.body)
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_get_producers():
    sch = tir.Schedule(mod=matmul_relu, traced=True)
    block = sch.get_block("relu")
    (producer,) = sch.get_producers(block)
    assert tvm.ir.structural_equal(
        sch.get_sref(producer).stmt,
        sch.get_sref(sch.get_block("matmul")).stmt,
    )
    _check_serialization(sch, mod=matmul_relu)


def test_traced_schedule_get_consumers():
    sch = tir.Schedule(mod=matmul_relu, traced=True)
    block = sch.get_block("matmul")
    (consumer,) = sch.get_consumers(block)
    assert tvm.ir.structural_equal(
        sch.get_sref(consumer).stmt,
        sch.get_sref(sch.get_block("relu")).stmt,
    )
    _check_serialization(sch, mod=matmul_relu)


##########  Transform loops  ##########


def test_traced_schedule_fuse():
    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    i, j, _ = sch.get_loops(block)
    sch.fuse(i, j)
    assert len(sch.get_loops(block)) == 2
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_split():
    sch = tir.Schedule(mod=matmul, traced=True)
    i, _, _ = sch.get_loops(sch.get_block("matmul"))
    i_0, i_1, i_2 = [sch.get_sref(i).stmt for i in sch.split(i, factors=[None, 8, 32])]
    assert tvm.ir.structural_equal(i_0, sch.mod["main"].body.block.body)
    assert tvm.ir.structural_equal(i_1, sch.mod["main"].body.block.body.body)
    assert tvm.ir.structural_equal(i_2, sch.mod["main"].body.block.body.body.body)
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_reorder():
    sch = tir.Schedule(mod=matmul, traced=True)
    i_0, i_1, i_2 = sch.get_loops(sch.get_block("matmul"))
    sch.reorder(i_2, i_1, i_0)
    i_0, i_1, i_2 = [sch.get_sref(i).stmt for i in [i_0, i_1, i_2]]
    tir_sch = tir.Schedule(matmul, debug_mode=True)
    ti_0, ti_1, ti_2 = tir_sch.get_loops(tir_sch.get_block("matmul"))
    tir_sch.reorder(ti_2, ti_1, ti_0)
    assert tvm.ir.structural_equal(i_0, tir_sch.get(ti_0))
    assert tvm.ir.structural_equal(i_1, tir_sch.get(ti_1))
    assert tvm.ir.structural_equal(i_2, tir_sch.get(ti_2))
    _check_serialization(sch, mod=matmul)


##########  Manipulate ForKind  ##########


def test_traced_schedule_parallel():
    def check_annotation(sch, loop):
        loop = sch.get_sref(loop).stmt
        assert loop.kind == ForKind.PARALLEL

    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    i, _, _ = sch.get_loops(block)
    sch.parallel(i)
    check_annotation(sch, i)
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_vectorize():
    def check_annotation(sch, loop):
        loop = sch.get_sref(loop).stmt
        assert loop.kind == ForKind.VECTORIZED

    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    i, _, _ = sch.get_loops(block)
    sch.vectorize(i)
    check_annotation(sch, i)
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_unroll():
    def check_annotation(sch, loop):
        loop = sch.get_sref(loop).stmt
        assert loop.kind == ForKind.UNROLLED

    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    _, _, k = sch.get_loops(block)
    sch.unroll(k)
    check_annotation(sch, k)
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_bind():
    def check_annotation(sch, loop):
        loop = sch.get_sref(loop).stmt
        assert loop.kind == ForKind.THREAD_BINDING
        assert loop.thread_binding.thread_tag == "threadIdx.x"

    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    i, _, _ = sch.get_loops(block)
    sch.bind(i, "threadIdx.x")
    check_annotation(sch, i)
    _check_serialization(sch, mod=matmul)


##########  Insert cache stages  ##########


def test_traced_schedule_cache_read():
    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    sch.cache_read(block, i=1, storage_scope="local")
    sch.cache_read(block, i=2, storage_scope="local")
    assert tvm.ir.structural_equal(sch.mod["main"], matmul_cache_read)
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_cache_write():
    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    sch.cache_write(block, i=0, storage_scope="local")
    assert tvm.ir.structural_equal(sch.mod["main"], matmul_cache_write)
    _check_serialization(sch, mod=matmul)


##########  Compute location  ##########


def test_traced_schedule_compute_at():
    sch = tir.Schedule(mod=plus_one_matmul, traced=True)
    plus_one_block = sch.get_block("plus_one")
    matmul_block = sch.get_block("matmul")
    _, _, i_2 = sch.get_loops(matmul_block)
    sch.compute_at(plus_one_block, i_2, preserve_unit_loop=True)
    assert tvm.ir.structural_equal(sch.mod["main"], plus_one_matmul_fused)
    _check_serialization(sch, mod=plus_one_matmul)


def test_traced_schedule_reverse_compute_at():
    sch = tir.Schedule(mod=matmul_relu, traced=True)
    relu_block = sch.get_block("relu")
    matmul_block = sch.get_block("matmul")
    _, i_1, _ = sch.get_loops(matmul_block)
    sch.reverse_compute_at(relu_block, i_1, preserve_unit_loop=True)
    assert tvm.ir.structural_equal(sch.mod["main"], matmul_relu_fused)
    _check_serialization(sch, mod=matmul_relu)


def test_traced_schedule_compute_inline():
    sch = tir.Schedule(mod=elementwise, traced=True)
    block = sch.get_block(name="B")
    sch.compute_inline(block=block)
    assert tvm.ir.structural_equal(sch.mod["main"], elementwise_inlined)
    _check_serialization(sch, mod=elementwise)


##########  Reduction  ##########


def test_traced_schedule_decompose_reduction():
    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    _, _, k = sch.get_loops(block)
    sch.decompose_reduction(block, k)
    assert tvm.ir.structural_equal(sch.mod["main"], matmul_decomposed)
    _check_serialization(sch, mod=matmul)


##########  Blockize & Tensorize  ##########


def test_traced_schedule_blockize():
    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    _, _, k = sch.get_loops(block)
    sch.blockize(k)
    assert tvm.ir.structural_equal(sch.mod["main"], matmul_blockized)
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_tensorize():
    tir.TensorIntrin.register("tir_test.tensor_intrin", tensorize_desc, tensorize_impl)
    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block)
    i_o, i_i = sch.split(i, factors=[None, 16])
    j_o, j_i = sch.split(j, factors=[None, 16])
    k_o, k_i = sch.split(k, factors=[None, 16])
    sch.reorder(i_o, j_o, k_o, i_i, j_i, k_i)
    sch.decompose_reduction(block, k_o)
    sch.tensorize(i_i, "tir_test.tensor_intrin")
    assert tvm.ir.structural_equal(sch.mod["main"], matmul_tensorized)
    _check_serialization(sch, mod=matmul)


##########  Annotation  ##########


def test_traced_schedule_mark_loop():
    def check_annotation(sch, loop):
        loop = sch.get_sref(loop).stmt
        assert len(loop.annotations) == 1
        attr_key, value = loop.annotations.items()[0]
        assert attr_key == "ann_key"
        assert value == "ann_val"

    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    i, _, _ = sch.get_loops(block)
    sch.mark_loop(i, "ann_key", "ann_val")
    check_annotation(sch, i)
    _check_serialization(sch, mod=matmul)


def test_traced_schedule_mark_block():
    def check_annotation(sch, block):
        block = sch.get_sref(block).stmt
        assert len(block.annotations) == 1
        attr_key, value = block.annotations.items()[0]
        assert attr_key == "ann_key"
        assert value == "1"

    sch = tir.Schedule(mod=matmul, traced=True)
    block = sch.get_block("matmul")
    sch.mark_block(block, "ann_key", 1)
    check_annotation(sch, block)
    _check_serialization(sch, mod=matmul)


if __name__ == "__main__":
    ##########  Utility  ##########
    test_traced_schedule_copy()
    ##########  Sampling  ##########
    test_traced_schedule_sample_perfect_tile()
    test_traced_schedule_sample_categorical()
    test_traced_schedule_sample_compute_location()
    ##########  Get blocks & loops  ##########
    test_traced_schedule_get_block()
    test_traced_schedule_get_loops()
    # test_traced_schedule_get_child_blocks() TODO
    test_traced_schedule_get_producers()
    test_traced_schedule_get_consumers()
    ##########  Transform loops  ##########
    test_traced_schedule_fuse()
    test_traced_schedule_split()
    test_traced_schedule_reorder()
    ##########  Manipulate ForKind  ##########
    test_traced_schedule_parallel()
    test_traced_schedule_vectorize()
    test_traced_schedule_unroll()
    test_traced_schedule_bind()
    ##########  Insert cache stages  ##########
    test_traced_schedule_cache_read()
    test_traced_schedule_cache_write()
    ##########  Compute location  ##########
    test_traced_schedule_compute_at()
    test_traced_schedule_reverse_compute_at()
    test_traced_schedule_compute_inline()
    # test_traced_schedule_reverse_compute_inline() TODO
    ##########  Reduction  ##########
    # test_traced_schedule_rfactor() TODO
    test_traced_schedule_decompose_reduction()
    # test_traced_schedule_merge_reduction() TODO
    ##########  Blockize & Tensorize  ##########
    test_traced_schedule_blockize()
    test_traced_schedule_tensorize()
    ##########  Annotation  ##########
    test_traced_schedule_mark_loop()
    test_traced_schedule_mark_block()
    # test_traced_schedule_pragma() TODO
    ##########  Misc  ##########
    # test_traced_schedule_enter_postproc() TODO
    # test_traced_schedule_double_buffer() TODO
    # test_traced_schedule_set_scope() TODO
    # test_traced_schedule_storage_align() TODO
    # test_traced_schedule_inline_argument() TODO
