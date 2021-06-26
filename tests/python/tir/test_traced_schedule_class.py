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
"""Test for traced schedule class"""
# pylint: disable=missing-function-docstring
from typing import Union

import tvm
from tvm import tir
from tvm.ir import IRModule
from tvm.tir import PrimFunc
from tvm.script import ty
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


def _check_serialization(sch: tir.Schedule, mod: Union[PrimFunc, IRModule]) -> str:
    record = sch._trace.as_json()
    new_sch = tir.Schedule(mod=mod, traced=True)
    Trace.apply_json_to_schedule(json=record, sch=new_sch)
    assert tvm.ir.structural_equal(new_sch.mod, sch.mod)
    py_repr = "\n".join(sch._trace.as_python())
    new_py_repr = "\n".join(new_sch._trace.as_python())
    assert py_repr == new_py_repr
    return py_repr


def test_traced_schedule_sample_perfect_tile():
    sch = tir.Schedule(matmul, traced=True)
    i, _, _ = sch.get_loops(sch.get_block("matmul"))
    factors = sch.sample_perfect_tile(i, n=4)
    factors = [sch.get(i) for i in factors]
    prod = factors[0] * factors[1] * factors[2] * factors[3]
    assert prod == 1024
    _check_serialization(sch, mod=matmul)


if __name__ == "__main__":
    ##########  Sampling  ##########
    test_traced_schedule_sample_perfect_tile()
    # test_traced_schedule_sample_categorical()
    # test_traced_schedule_sample_compute_location()
    ##########  Get blocks & loops  ##########
    # test_traced_schedule_get_block()
    # test_traced_schedule_get_loops()
    # test_traced_schedule_get_child_blocks()
    # test_traced_schedule_get_producers()
    # test_traced_schedule_get_consumers()
    ##########  Transform loops  ##########
    # test_traced_schedule_fuse()
    # test_traced_schedule_split()
    # test_traced_schedule_reorder()
    ##########  Manipulate ForKind  ##########
    # test_traced_schedule_parallel()
    # test_traced_schedule_vectorize()
    # test_traced_schedule_unroll()
    # test_traced_schedule_bind()
    ##########  Insert cache stages  ##########
    # test_traced_schedule_cache_read()
    # test_traced_schedule_cache_write()
    ##########  Compute location  ##########
    # test_traced_schedule_compute_at()
    # test_traced_schedule_reverse_compute_at()
    # test_traced_schedule_compute_inline()
    # test_traced_schedule_reverse_compute_inline()
    ##########  Reduction  ##########
    # test_traced_schedule_rfactor()
    # test_traced_schedule_decompose_reduction()
    # test_traced_schedule_merge_reduction()
    ##########  Blockize & Tensorize  ##########
    # test_traced_schedule_blockize()
    # test_traced_schedule_tensorize()
    ##########  Annotation  ##########
    # test_traced_schedule_mark_loop()
    # test_traced_schedule_mark_block()
    # test_traced_schedule_pragma()
    ##########  Misc  ##########
    # test_traced_schedule_enter_postproc()
    # test_traced_schedule_double_buffer()
    # test_traced_schedule_set_scope()
    # test_traced_schedule_storage_align()
    # test_traced_schedule_inline_argument()
