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
""" Test meta schedule PostOrderApply + SearchRule """
# pylint: disable=missing-function-docstring
from typing import List

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.hybrid import ty

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks

# fmt: off

@tvm.hybrid.script
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])


@tvm.hybrid.script
def _matmul_sketch_0(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0_outer_outer_outer in range(0, 16):
            for i1_outer_outer_outer in range(0, 256):
                for i0_outer_outer_inner in range(0, 2):
                    for i1_outer_outer_inner in range(0, 2):
                        for i2_outer in range(0, 64):
                            for i0_outer_inner in range(0, 4):
                                for i1_outer_inner in range(0, 2):
                                    for i2_inner in range(0, 16):
                                        for i0_inner in range(0, 8):
                                            for i1_inner in range(0, 1):
                                                with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, ((((((i0_outer_outer_outer*2) + i0_outer_outer_inner)*4) + i0_outer_inner)*8) + i0_inner))
                                                    tir.bind(vj, (((((i1_outer_outer_outer*2) + i1_outer_outer_inner)*2) + i1_outer_inner) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*16) + i2_inner))
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C[vi, vj], (A[vi, vk]*B[vk, vj]))


@tvm.hybrid.script
def _matmul_sketch_1(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.buffer_allocate([1024, 1024], elem_offset=0, scope="local", align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 16):
            for i1_outer_outer_outer in range(0, 256):
                for i0_outer_outer_inner in range(0, 2):
                    for i1_outer_outer_inner in range(0, 2):
                        for i2_outer in range(0, 64):
                            for i0_outer_inner in range(0, 4):
                                for i1_outer_inner in range(0, 2):
                                    for i2_inner in range(0, 16):
                                        for i0_inner in range(0, 8):
                                            for i1_inner in range(0, 1):
                                                with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, ((((((i0_outer_outer_outer*2) + i0_outer_outer_inner)*4) + i0_outer_inner)*8) + i0_inner))
                                                    tir.bind(vj, (((((i1_outer_outer_outer*2) + i1_outer_outer_inner)*2) + i1_outer_inner) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*16) + i2_inner))
                                                    tir.reads([C_local[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C_local[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C_local[vi, vj], (A[vi, vk]*B[vk, vj]))
                        for ax0 in range(0, 32):
                            for ax1 in range(0, 2):
                                with tir.block([1024, 1024], "") as [v0, v1]:
                                    tir.bind(v0, (((i0_outer_outer_outer*64) + (i0_outer_outer_inner*32)) + ax0))
                                    tir.bind(v1, (((i1_outer_outer_outer*4) + (i1_outer_outer_inner*2)) + ax1))
                                    tir.reads([C_local[v0:(v0 + 1), v1:(v1 + 1)]])
                                    tir.writes([C[v0:(v0 + 1), v1:(v1 + 1)]])
                                    C[v0, v1] = C_local[v0, v1]


@tvm.hybrid.script
def _matmul_sketch_2(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.buffer_allocate([1024, 1024], elem_offset=0, scope="local", align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 16):
            for i1_outer_outer_outer in range(0, 256):
                for i0_outer_outer_inner in range(0, 2):
                    for i1_outer_outer_inner in range(0, 2):
                        for i2_outer in range(0, 64):
                            for i0_outer_inner in range(0, 4):
                                for i1_outer_inner in range(0, 2):
                                    for i2_inner in range(0, 16):
                                        for i0_inner in range(0, 8):
                                            for i1_inner in range(0, 1):
                                                with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, ((((((i0_outer_outer_outer*2) + i0_outer_outer_inner)*4) + i0_outer_inner)*8) + i0_inner))
                                                    tir.bind(vj, (((((i1_outer_outer_outer*2) + i1_outer_outer_inner)*2) + i1_outer_inner) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*16) + i2_inner))
                                                    tir.reads([C_local[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C_local[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C_local[vi, vj], (A[vi, vk]*B[vk, vj]))
                                        for ax0 in range(0, 8):
                                            with tir.block([1024, 1024], "") as [v0, v1]:
                                                tir.bind(v0, ((((i0_outer_outer_outer*64) + (i0_outer_outer_inner*32)) + (i0_outer_inner*8)) + ax0))
                                                tir.bind(v1, (((i1_outer_outer_outer*4) + (i1_outer_outer_inner*2)) + i1_outer_inner))
                                                tir.reads([C_local[v0:(v0 + 1), v1:(v1 + 1)]])
                                                tir.writes([C[v0:(v0 + 1), v1:(v1 + 1)]])
                                                C[v0, v1] = C_local[v0, v1]


@tvm.hybrid.script
def matmul_relu(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    D = tir.match_buffer(d, (1024, 1024), "float32")
    C = tir.buffer_allocate((1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])
    with tir.block([1024, 1024], "relu") as [vi, vj]:
        D[vi, vj] = tir.max(C[vi, vj], 0.0)


@tvm.hybrid.script
def _matmul_relu_sketch_0(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    D = tir.match_buffer(d, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.buffer_allocate([1024, 1024], elem_offset=0, align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 16):
            for i1_outer_outer_outer in range(0, 1):
                for i0_outer_outer_inner in range(0, 1):
                    for i1_outer_outer_inner in range(0, 512):
                        for i2_outer in range(0, 512):
                            for i0_outer_inner in range(0, 4):
                                for i1_outer_inner in range(0, 1):
                                    for i2_inner in range(0, 2):
                                        for i0_inner in range(0, 16):
                                            for i1_inner in range(0, 2):
                                                with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, (((((i0_outer_outer_outer + i0_outer_outer_inner)*4) + i0_outer_inner)*16) + i0_inner))
                                                    tir.bind(vj, (((((i1_outer_outer_outer*512) + i1_outer_outer_inner) + i1_outer_inner)*2) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*2) + i2_inner))
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C[vi, vj], (A[vi, vk]*B[vk, vj]))
        for i0 in range(0, 1024):
            for i1 in range(0, 1024):
                with tir.block([1024, 1024], "relu") as [vi_1, vj_1]:
                    tir.bind(vi_1, i0)
                    tir.bind(vj_1, i1)
                    tir.reads([C[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                    tir.writes([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                    D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))


@tvm.hybrid.script
def _matmul_relu_sketch_1(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    D = tir.match_buffer(d, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.buffer_allocate([1024, 1024], elem_offset=0, align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 16):
            for i1_outer_outer_outer in range(0, 1):
                for i0_outer_outer_inner in range(0, 1):
                    for i1_outer_outer_inner in range(0, 512):
                        for i2_outer in range(0, 512):
                            for i0_outer_inner in range(0, 4):
                                for i1_outer_inner in range(0, 1):
                                    for i2_inner in range(0, 2):
                                        for i0_inner in range(0, 16):
                                            for i1_inner in range(0, 2):
                                                with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, (((((i0_outer_outer_outer + i0_outer_outer_inner)*4) + i0_outer_inner)*16) + i0_inner))
                                                    tir.bind(vj, (((((i1_outer_outer_outer*512) + i1_outer_outer_inner) + i1_outer_inner)*2) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*2) + i2_inner))
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C[vi, vj], (A[vi, vk]*B[vk, vj]))
                        for ax0 in range(0, 64):
                            for ax1 in range(0, 2):
                                with tir.block([1024, 1024], "relu") as [vi_1, vj_1]:
                                    tir.bind(vi_1, (((i0_outer_outer_outer*64) + (i0_outer_outer_inner*64)) + ax0))
                                    tir.bind(vj_1, (((i1_outer_outer_outer*1024) + (i1_outer_outer_inner*2)) + ax1))
                                    tir.reads([C[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                                    tir.writes([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                                    D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))


@tvm.hybrid.script
def _matmul_relu_sketch_2(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    D = tir.match_buffer(d, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.buffer_allocate([1024, 1024], elem_offset=0, align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 16):
            for i1_outer_outer_outer in range(0, 1):
                for i0_outer_outer_inner in range(0, 1):
                    for i1_outer_outer_inner in range(0, 512):
                        for i2_outer in range(0, 512):
                            for i0_outer_inner in range(0, 4):
                                for i1_outer_inner in range(0, 1):
                                    for i2_inner in range(0, 2):
                                        for i0_inner in range(0, 16):
                                            for i1_inner in range(0, 2):
                                                with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, (((((i0_outer_outer_outer + i0_outer_outer_inner)*4) + i0_outer_inner)*16) + i0_inner))
                                                    tir.bind(vj, (((((i1_outer_outer_outer*512) + i1_outer_outer_inner) + i1_outer_inner)*2) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*2) + i2_inner))
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C[vi, vj], (A[vi, vk]*B[vk, vj]))
                                        for ax0 in range(0, 16):
                                            for ax1 in range(0, 2):
                                                with tir.block([1024, 1024], "relu") as [vi_1, vj_1]:
                                                    tir.bind(vi_1, ((((i0_outer_outer_outer*64) + (i0_outer_outer_inner*64)) + (i0_outer_inner*16)) + ax0))
                                                    tir.bind(vj_1, ((((i1_outer_outer_outer*1024) + (i1_outer_outer_inner*2)) + (i1_outer_inner*2)) + ax1))
                                                    tir.reads([C[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                                                    tir.writes([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                                                    D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


def _check_sketch(results: List[ms.Schedule], expected: List[tvm.tir.PrimFunc]):
    assert len(results) == len(expected)
    results = [x.sch.func for x in results]
    for x in expected:
        found = False
        for y in results:
            if tvm.ir.structural_equal(x, y):
                found = True
                break
        if not found:
            print(tvm.hybrid.ashybrid(x))
        assert found


def test_meta_schedule_sketch_cpu_matmul():
    space = ms.space.PostOrderApply(
        stages=[
            ms.rule.always_inline(),
            ms.rule.compose(
                name="tiling",
                rules=[
                    ms.rule.add_cache_write(),
                    ms.rule.multi_level_tiling(structure="SSRSRS"),
                    ms.rule.fusion(levels=[1, 2]),
                ],
            ),
        ]
    )
    support = space.get_support(task=ms.SearchTask(func=matmul))
    assert len(support) == 3


def test_meta_schedule_sketch_cpu_matmul_relu():
    space = ms.space.PostOrderApply(
        stages=[
            ms.rule.always_inline(),
            ms.rule.compose(
                name="tiling",
                rules=[
                    ms.rule.add_cache_write(),
                    ms.rule.multi_level_tiling(structure="SSRSRS"),
                    ms.rule.fusion(levels=[1, 2]),
                ],
            ),
        ]
    )
    support = space.get_support(task=ms.SearchTask(func=matmul_relu))
    assert len(support) == 3


if __name__ == "__main__":
    test_meta_schedule_sketch_cpu_matmul()
    test_meta_schedule_sketch_cpu_matmul_relu()
