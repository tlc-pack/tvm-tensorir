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
"""Test Ansor-like sketch generation in subgraphs in meta schedule"""
# pylint: disable=missing-function-docstring
from typing import List

import tvm
from tvm import te, tir, meta_schedule as ms
from tvm.script import ty
import workload


def _fix_sampling_tile_size(
    sch: ms.Schedule,
    decisions: List[List[int]],
    expected: List[tir.PrimFunc],
):
    insts = [
        inst
        for inst in sch.trace
        if isinstance(inst.inst_attrs, ms.instruction.SamplePerfectTileAttrs)
    ]
    assert len(insts) == len(decisions)
    for inst, decision in zip(insts, decisions):
        sch.mutate_decision(inst, decision)
    sch.replay_decision()
    results = [tvm.ir.structural_equal(sch.sch.func, i) for i in expected]
    assert sum(results) >= 1


def _get_support(func: tir.PrimFunc, task_name: str):
    return ms.space.PostOrderApply(
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
    ).get_support(task=ms.SearchTask(func=func, task_name=task_name))


def _debug(support: List[ms.Schedule]):
    for i, sch in enumerate(support):
        print(f"###### {i}")
        print(tvm.script.asscript(sch.sch.func))
        for inst in sch.trace:
            if inst in sch.decisions:
                print(sch.decisions[inst])


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks
# fmt: off

@tvm.script.tir
def _matmul_sketch_0(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    C = tir.match_buffer(c, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0_outer_outer_outer in range(0, 128):
            for i1_outer_outer_outer in range(0, 16):
                for i0_outer_outer_inner in range(0, 2):
                    for i1_outer_outer_inner in range(0, 1):
                        for i2_outer in range(0, 64):
                            for i0_outer_inner in range(0, 1):
                                for i1_outer_inner in range(0, 2):
                                    for i2_inner in range(0, 8):
                                        for i0_inner in range(0, 2):
                                            for i1_inner in range(0, 16):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "C") as [i, j, k]:
                                                    tir.bind(i, (((((i0_outer_outer_outer*2) + i0_outer_outer_inner) + i0_outer_inner)*2) + i0_inner))
                                                    tir.bind(j, (((((i1_outer_outer_outer + i1_outer_outer_inner)*2) + i1_outer_inner)*16) + i1_inner))
                                                    tir.bind(k, ((i2_outer*8) + i2_inner))
                                                    tir.reads([C[i:(i + 1), j:(j + 1)], A[i:(i + 1), k:(k + 1)], B[k:(k + 1), j:(j + 1)]])
                                                    tir.writes([C[i:(i + 1), j:(j + 1)]])
                                                    reducer.step(C[i, j], (A[i, k]*B[k, j]))

@tvm.script.tir
def _matmul_sketch_1(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.buffer_allocate([512, 512], elem_offset=0, scope="local", align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 2):
            for i1_outer_outer_outer in range(0, 8):
                for i0_outer_outer_inner in range(0, 4):
                    for i1_outer_outer_inner in range(0, 8):
                        for i2_outer in range(0, 128):
                            for i0_outer_inner in range(0, 32):
                                for i1_outer_inner in range(0, 2):
                                    for i2_inner in range(0, 4):
                                        for i0_inner in range(0, 2):
                                            for i1_inner in range(0, 4):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, ((((((i0_outer_outer_outer*4) + i0_outer_outer_inner)*32) + i0_outer_inner)*2) + i0_inner))
                                                    tir.bind(vj, ((((((i1_outer_outer_outer*8) + i1_outer_outer_inner)*2) + i1_outer_inner)*4) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*4) + i2_inner))
                                                    tir.reads([C_local[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C_local[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C_local[vi, vj], (A[vi, vk]*B[vk, vj]))
                        for ax0 in range(0, 64):
                            for ax1 in range(0, 8):
                                with tir.block([512, 512], "") as [v0, v1]:
                                    tir.bind(v0, (((i0_outer_outer_outer*256) + (i0_outer_outer_inner*64)) + ax0))
                                    tir.bind(v1, (((i1_outer_outer_outer*64) + (i1_outer_outer_inner*8)) + ax1))
                                    tir.reads([C_local[v0:(v0 + 1), v1:(v1 + 1)]])
                                    tir.writes([C[v0:(v0 + 1), v1:(v1 + 1)]])
                                    C[v0, v1] = C_local[v0, v1]

@tvm.script.tir
def _matmul_sketch_2(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    C = tir.match_buffer(c, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.buffer_allocate([512, 512], elem_offset=0, scope="local", align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 2):
            for i1_outer_outer_outer in range(0, 64):
                for i0_outer_outer_inner in range(0, 32):
                    for i1_outer_outer_inner in range(0, 1):
                        for i2_outer in range(0, 512):
                            for i0_outer_inner in range(0, 1):
                                for i1_outer_inner in range(0, 2):
                                    for i2_inner in range(0, 1):
                                        for i0_inner in range(0, 8):
                                            for i1_inner in range(0, 4):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, (((((i0_outer_outer_outer*32) + i0_outer_outer_inner) + i0_outer_inner)*8) + i0_inner))
                                                    tir.bind(vj, (((((i1_outer_outer_outer + i1_outer_outer_inner)*2) + i1_outer_inner)*4) + i1_inner))
                                                    tir.bind(vk, (i2_outer + i2_inner))
                                                    tir.reads([C_local[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C_local[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C_local[vi, vj], (A[vi, vk]*B[vk, vj]))
                                        for ax0 in range(0, 8):
                                            for ax1 in range(0, 4):
                                                with tir.block([512, 512], "") as [v0, v1]:
                                                    tir.bind(v0, ((((i0_outer_outer_outer*256) + (i0_outer_outer_inner*8)) + (i0_outer_inner*8)) + ax0))
                                                    tir.bind(v1, ((((i1_outer_outer_outer*8) + (i1_outer_outer_inner*8)) + (i1_outer_inner*4)) + ax1))
                                                    tir.reads([C_local[v0:(v0 + 1), v1:(v1 + 1)]])
                                                    tir.writes([C[v0:(v0 + 1), v1:(v1 + 1)]])
                                                    C[v0, v1] = C_local[v0, v1]

# TODO(@junrushao1994): remove it and use workload.matmul
@tvm.script.tir
def workload_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (512, 512), "float32")
    B = tir.match_buffer(b, (512, 512), "float32")
    C = tir.match_buffer(c, (512, 512), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([512, 512, tir.reduce_axis(0, 512)], "C") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


def test_meta_schedule_sketch_cpu_matmul():
    # func = te.create_func(workload.matmul(n=512, m=512, k=512))
    func = workload_matmul
    support = _get_support(func=func, task_name="matmul")
    expected = [_matmul_sketch_0, _matmul_sketch_1, _matmul_sketch_2]
    assert len(support) == 3
    _fix_sampling_tile_size(
        sch=support[0],
        decisions=[
            [128, 2, 1, 2],
            [16, 1, 2, 16],
            [64, 8],
        ],
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[1],
        decisions=[
            [2, 4, 32, 2],
            [8, 8, 2, 4],
            [128, 4],
        ],
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[2],
        decisions=[
            [2, 32, 1, 8],
            [64, 1, 2, 4],
            [512, 1],
        ],
        expected=expected,
    )


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks
# fmt: off

@tvm.script.tir
def _matmul_relu_sketch_0(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    D = tir.match_buffer(d, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.buffer_allocate([512, 512], elem_offset=0, align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 1):
            for i1_outer_outer_outer in range(0, 32):
                for i0_outer_outer_inner in range(0, 16):
                    for i1_outer_outer_inner in range(0, 4):
                        for i2_outer in range(0, 256):
                            for i0_outer_inner in range(0, 2):
                                for i1_outer_inner in range(0, 2):
                                    for i2_inner in range(0, 2):
                                        for i0_inner in range(0, 16):
                                            for i1_inner in range(0, 2):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, ((((((i0_outer_outer_outer*16) + i0_outer_outer_inner)*2) + i0_outer_inner)*16) + i0_inner))
                                                    tir.bind(vj, ((((((i1_outer_outer_outer*4) + i1_outer_outer_inner)*2) + i1_outer_inner)*2) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*2) + i2_inner))
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C[vi, vj], (A[vi, vk]*B[vk, vj]))
        for i0 in range(0, 512):
            for i1 in range(0, 512):
                with tir.block([512, 512], "relu") as [vi_1, vj_1]:
                    tir.bind(vi_1, i0)
                    tir.bind(vj_1, i1)
                    tir.reads([C[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                    tir.writes([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                    D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))

@tvm.script.tir
def _matmul_relu_sketch_1(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    D = tir.match_buffer(d, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.buffer_allocate([512, 512], elem_offset=0, align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 1):
            for i1_outer_outer_outer in range(0, 32):
                for i0_outer_outer_inner in range(0, 16):
                    for i1_outer_outer_inner in range(0, 4):
                        for i2_outer in range(0, 256):
                            for i0_outer_inner in range(0, 2):
                                for i1_outer_inner in range(0, 2):
                                    for i2_inner in range(0, 2):
                                        for i0_inner in range(0, 16):
                                            for i1_inner in range(0, 2):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, ((((((i0_outer_outer_outer*16) + i0_outer_outer_inner)*2) + i0_outer_inner)*16) + i0_inner))
                                                    tir.bind(vj, ((((((i1_outer_outer_outer*4) + i1_outer_outer_inner)*2) + i1_outer_inner)*2) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*2) + i2_inner))
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C[vi, vj], (A[vi, vk]*B[vk, vj]))
                        for ax0 in range(0, 32):
                            for ax1 in range(0, 4):
                                with tir.block([512, 512], "relu") as [vi_1, vj_1]:
                                    tir.bind(vi_1, (((i0_outer_outer_outer*512) + (i0_outer_outer_inner*32)) + ax0))
                                    tir.bind(vj_1, (((i1_outer_outer_outer*16) + (i1_outer_outer_inner*4)) + ax1))
                                    tir.reads([C[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                                    tir.writes([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                                    D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))

@tvm.script.tir
def _matmul_relu_sketch_2(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    D = tir.match_buffer(d, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.buffer_allocate([512, 512], elem_offset=0, align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 1):
            for i1_outer_outer_outer in range(0, 32):
                for i0_outer_outer_inner in range(0, 16):
                    for i1_outer_outer_inner in range(0, 4):
                        for i2_outer in range(0, 256):
                            for i0_outer_inner in range(0, 2):
                                for i1_outer_inner in range(0, 2):
                                    for i2_inner in range(0, 2):
                                        for i0_inner in range(0, 16):
                                            for i1_inner in range(0, 2):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, ((((((i0_outer_outer_outer*16) + i0_outer_outer_inner)*2) + i0_outer_inner)*16) + i0_inner))
                                                    tir.bind(vj, ((((((i1_outer_outer_outer*4) + i1_outer_outer_inner)*2) + i1_outer_inner)*2) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*2) + i2_inner))
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    reducer.step(C[vi, vj], (A[vi, vk]*B[vk, vj]))
                                        for ax0 in range(0, 16):
                                            for ax1 in range(0, 2):
                                                with tir.block([512, 512], "relu") as [vi_1, vj_1]:
                                                    tir.bind(vi_1, ((((i0_outer_outer_outer*512) + (i0_outer_outer_inner*32)) + (i0_outer_inner*16)) + ax0))
                                                    tir.bind(vj_1, ((((i1_outer_outer_outer*16) + (i1_outer_outer_inner*4)) + (i1_outer_inner*2)) + ax1))
                                                    tir.reads([C[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                                                    tir.writes([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                                                    D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))

@tvm.script.tir
def workload_matmul_relu(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (512, 512), "float32")
    B = tir.match_buffer(b, (512, 512), "float32")
    C = tir.buffer_allocate((512, 512), "float32")
    D = tir.match_buffer(d, (512, 512), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([512, 512, tir.reduce_axis(0, 512)], "matmul") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])
    with tir.block([512, 512], "relu") as [vi, vj]:
        D[vi, vj] = tir.max(C[vi, vj], 0.0)

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


def test_meta_schedule_sketch_cpu_matmul_relu():
    # func = te.create_func(workload.matmul_relu(n=512, m=512, k=512))
    func = workload_matmul_relu
    support = _get_support(func=func, task_name="matmul_relu")
    assert len(support) == 3
    expected = [_matmul_relu_sketch_0, _matmul_relu_sketch_1, _matmul_relu_sketch_2]
    _fix_sampling_tile_size(
        sch=support[0],
        decisions=[
            [1, 16, 2, 16],
            [32, 4, 2, 2],
            [256, 2],
        ],
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[1],
        decisions=[
            [1, 16, 2, 16],
            [32, 4, 2, 2],
            [256, 2],
        ],
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[2],
        decisions=[
            [1, 16, 2, 16],
            [32, 4, 2, 2],
            [256, 2],
        ],
        expected=expected,
    )


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unexpected-keyword-arg
# fmt: off

@tvm.script.tir
def _conv2d_nchw_sketch_0(var_X: ty.handle, var_W: ty.handle, var_compute: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.buffer_allocate([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
        for i0 in range(0, 1):
            for i1 in range(0, 512):
                for i2 in range(0, 58):
                    for i3 in range(0, 58):
                        with tir.block([1, 512, 58, 58], "pad_temp") as [i0_1, i1_1, i2_1, i3_1]:
                            tir.bind(i0_1, i0)
                            tir.bind(i1_1, i1)
                            tir.bind(i2_1, i2)
                            tir.bind(i3_1, i3)
                            tir.reads([X[i0_1:(i0_1 + 1), i1_1:(i1_1 + 1), (i2_1 - 1):((i2_1 - 1) + 1), (i3_1 - 1):((i3_1 - 1) + 1)]])
                            tir.writes([pad_temp[i0_1:(i0_1 + 1), i1_1:(i1_1 + 1), i2_1:(i2_1 + 1), i3_1:(i3_1 + 1)]])
                            pad_temp[i0_1, i1_1, i2_1, i3_1] = tir.if_then_else(((((i2_1 >= 1) and (i2_1 < 57)) and (i3_1 >= 1)) and (i3_1 < 57)), X[i0_1, i1_1, (i2_1 - 1), (i3_1 - 1)], tir.float32(0), dtype="float32")
        for i0_2_outer_outer_outer, i1_2_outer_outer_outer, i2_2_outer_outer_outer, i3_2_outer_outer_outer, i0_2_outer_outer_inner, i1_2_outer_outer_inner, i2_2_outer_outer_inner, i3_2_outer_outer_inner, i4_outer, i5_outer, i6_outer, i0_2_outer_inner, i1_2_outer_inner, i2_2_outer_inner, i3_2_outer_inner, i4_inner, i5_inner, i6_inner, i0_2_inner, i1_2_inner, i2_2_inner, i3_2_inner in tir.grid(1, 8, 1, 14, 1, 1, 2, 1, 256, 1, 3, 1, 16, 7, 2, 2, 3, 1, 1, 4, 4, 2):
            with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                tir.bind(nn, (((i0_2_outer_outer_outer + i0_2_outer_outer_inner) + i0_2_outer_inner) + i0_2_inner))
                tir.bind(ff, (((((i1_2_outer_outer_outer + i1_2_outer_outer_inner)*16) + i1_2_outer_inner)*4) + i1_2_inner))
                tir.bind(yy, ((((((i2_2_outer_outer_outer*2) + i2_2_outer_outer_inner)*7) + i2_2_outer_inner)*4) + i2_2_inner))
                tir.bind(xx, (((((i3_2_outer_outer_outer + i3_2_outer_outer_inner)*2) + i3_2_outer_inner)*2) + i3_2_inner))
                tir.bind(rc, ((i4_outer*2) + i4_inner))
                tir.bind(ry, ((i5_outer*3) + i5_inner))
                tir.bind(rx, (i6_outer + i6_inner))
                tir.reads([compute[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                tir.writes([compute[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                reducer.step(compute[nn, ff, yy, xx], (pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]))

@tvm.script.tir
def _conv2d_nchw_sketch_1(var_X: ty.handle, var_W: ty.handle, var_compute: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.buffer_allocate([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
        compute_local = tir.buffer_allocate([1, 512, 56, 56], elem_offset=0, scope="local", align=128, offset_factor=1)
        for i0 in range(0, 1):
            for i1 in range(0, 512):
                for i2 in range(0, 58):
                    for i3 in range(0, 58):
                        with tir.block([1, 512, 58, 58], "pad_temp") as [i0_1, i1_1, i2_1, i3_1]:
                            tir.bind(i0_1, i0)
                            tir.bind(i1_1, i1)
                            tir.bind(i2_1, i2)
                            tir.bind(i3_1, i3)
                            tir.reads([X[i0_1:(i0_1 + 1), i1_1:(i1_1 + 1), (i2_1 - 1):((i2_1 - 1) + 1), (i3_1 - 1):((i3_1 - 1) + 1)]])
                            tir.writes([pad_temp[i0_1:(i0_1 + 1), i1_1:(i1_1 + 1), i2_1:(i2_1 + 1), i3_1:(i3_1 + 1)]])
                            pad_temp[i0_1, i1_1, i2_1, i3_1] = tir.if_then_else(((((i2_1 >= 1) and (i2_1 < 57)) and (i3_1 >= 1)) and (i3_1 < 57)), X[i0_1, i1_1, (i2_1 - 1), (i3_1 - 1)], tir.float32(0), dtype="float32")
        for i0_2_outer_outer_outer in range(0, 1):
            for i1_2_outer_outer_outer in range(0, 1):
                for i2_2_outer_outer_outer in range(0, 1):
                    for i3_2_outer_outer_outer in range(0, 1):
                        for i0_2_outer_outer_inner, i1_2_outer_outer_inner, i2_2_outer_outer_inner, i3_2_outer_outer_inner, i4_outer, i5_outer, i6_outer, i0_2_outer_inner, i1_2_outer_inner in tir.grid(1, 8, 1, 1, 32, 3, 3, 1, 4):
                            for i2_2_outer_inner, i3_2_outer_inner, i4_inner, i5_inner, i6_inner, i0_2_inner, i1_2_inner, i2_2_inner, i3_2_inner in tir.grid(7, 4, 16, 1, 1, 1, 16, 8, 14):
                                with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                    tir.bind(nn, (((i0_2_outer_outer_outer + i0_2_outer_outer_inner) + i0_2_outer_inner) + i0_2_inner))
                                    tir.bind(ff, ((((((i1_2_outer_outer_outer*8) + i1_2_outer_outer_inner)*4) + i1_2_outer_inner)*16) + i1_2_inner))
                                    tir.bind(yy, (((((i2_2_outer_outer_outer + i2_2_outer_outer_inner)*7) + i2_2_outer_inner)*8) + i2_2_inner))
                                    tir.bind(xx, (((((i3_2_outer_outer_outer + i3_2_outer_outer_inner)*4) + i3_2_outer_inner)*14) + i3_2_inner))
                                    tir.bind(rc, ((i4_outer*16) + i4_inner))
                                    tir.bind(ry, (i5_outer + i5_inner))
                                    tir.bind(rx, (i6_outer + i6_inner))
                                    tir.reads([compute_local[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                                    tir.writes([compute_local[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                                    reducer.step(compute_local[nn, ff, yy, xx], (pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]))
                        for ax0, ax1, ax2, ax3 in tir.grid(1, 512, 56, 56):  # pylint: disable=unused-variable
                            with tir.block([1, 512, 56, 56], "") as [v0, v1, v2, v3]:
                                tir.bind(v0, i0_2_outer_outer_outer)
                                tir.bind(v1, ((i1_2_outer_outer_outer*512) + ax1))
                                tir.bind(v2, ((i2_2_outer_outer_outer*56) + ax2))
                                tir.bind(v3, ((i3_2_outer_outer_outer*56) + ax3))
                                tir.reads([compute_local[v0:(v0 + 1), v1:(v1 + 1), v2:(v2 + 1), v3:(v3 + 1)]])
                                tir.writes([compute[v0:(v0 + 1), v1:(v1 + 1), v2:(v2 + 1), v3:(v3 + 1)]])
                                compute[v0, v1, v2, v3] = compute_local[v0, v1, v2, v3]

@tvm.script.tir
def _conv2d_nchw_sketch_2(var_X: ty.handle, var_W: ty.handle, var_compute: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.buffer_allocate([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
        compute_local = tir.buffer_allocate([1, 512, 56, 56], elem_offset=0, scope="local", align=128, offset_factor=1)
        for i0 in range(0, 1):
            for i1 in range(0, 512):
                for i2 in range(0, 58):
                    for i3 in range(0, 58):
                        with tir.block([1, 512, 58, 58], "pad_temp") as [i0_1, i1_1, i2_1, i3_1]:
                            tir.bind(i0_1, i0)
                            tir.bind(i1_1, i1)
                            tir.bind(i2_1, i2)
                            tir.bind(i3_1, i3)
                            tir.reads([X[i0_1:(i0_1 + 1), i1_1:(i1_1 + 1), (i2_1 - 1):((i2_1 - 1) + 1), (i3_1 - 1):((i3_1 - 1) + 1)]])
                            tir.writes([pad_temp[i0_1:(i0_1 + 1), i1_1:(i1_1 + 1), i2_1:(i2_1 + 1), i3_1:(i3_1 + 1)]])
                            pad_temp[i0_1, i1_1, i2_1, i3_1] = tir.if_then_else(((((i2_1 >= 1) and (i2_1 < 57)) and (i3_1 >= 1)) and (i3_1 < 57)), X[i0_1, i1_1, (i2_1 - 1), (i3_1 - 1)], tir.float32(0), dtype="float32")
        for i0_2_outer_outer_outer, i1_2_outer_outer_outer, i2_2_outer_outer_outer, i3_2_outer_outer_outer, i0_2_outer_outer_inner, i1_2_outer_outer_inner, i2_2_outer_outer_inner, i3_2_outer_outer_inner in tir.grid(1, 1, 1, 1, 1, 8, 1, 1):
            for i4_outer in range(0, 32):
                for i5_outer in range(0, 3):
                    for i6_outer in range(0, 3):
                        for i0_2_outer_inner in range(0, 1):
                            for i1_2_outer_inner in range(0, 4):
                                for i2_2_outer_inner in range(0, 7):
                                    for i3_2_outer_inner in range(0, 4):
                                        for i4_inner in range(0, 16):
                                            for i5_inner in range(0, 1):
                                                for i6_inner in range(0, 1):
                                                    for i0_2_inner in range(0, 1):
                                                        for i1_2_inner in range(0, 16):
                                                            for i2_2_inner in range(0, 8):
                                                                for i3_2_inner in range(0, 14):
                                                                    with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                                                        tir.bind(nn, (((i0_2_outer_outer_outer + i0_2_outer_outer_inner) + i0_2_outer_inner) + i0_2_inner))
                                                                        tir.bind(ff, ((((((i1_2_outer_outer_outer*8) + i1_2_outer_outer_inner)*4) + i1_2_outer_inner)*16) + i1_2_inner))
                                                                        tir.bind(yy, (((((i2_2_outer_outer_outer + i2_2_outer_outer_inner)*7) + i2_2_outer_inner)*8) + i2_2_inner))
                                                                        tir.bind(xx, (((((i3_2_outer_outer_outer + i3_2_outer_outer_inner)*4) + i3_2_outer_inner)*14) + i3_2_inner))
                                                                        tir.bind(rc, ((i4_outer*16) + i4_inner))
                                                                        tir.bind(ry, (i5_outer + i5_inner))
                                                                        tir.bind(rx, (i6_outer + i6_inner))
                                                                        tir.reads([compute_local[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                                                                        tir.writes([compute_local[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                                                                        reducer.step(compute_local[nn, ff, yy, xx], (pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]))
            for ax0, ax1, ax2, ax3 in tir.grid(1, 64, 56, 56):  # pylint: disable=unused-variable
                with tir.block([1, 512, 56, 56], "") as [v0, v1, v2, v3]:
                    tir.bind(v0, (i0_2_outer_outer_outer + i0_2_outer_outer_inner))
                    tir.bind(v1, (((i1_2_outer_outer_outer*512) + (i1_2_outer_outer_inner*64)) + ax1))
                    tir.bind(v2, (((i2_2_outer_outer_outer*56) + (i2_2_outer_outer_inner*56)) + ax2))
                    tir.bind(v3, (((i3_2_outer_outer_outer*56) + (i3_2_outer_outer_inner*56)) + ax3))
                    tir.reads([compute_local[v0:(v0 + 1), v1:(v1 + 1), v2:(v2 + 1), v3:(v3 + 1)]])
                    tir.writes([compute[v0:(v0 + 1), v1:(v1 + 1), v2:(v2 + 1), v3:(v3 + 1)]])
                    compute[v0, v1, v2, v3] = compute_local[v0, v1, v2, v3]


@tvm.script.tir
def workload_conv2d_nchw(var_X: ty.handle, var_W: ty.handle, var_compute: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.buffer_allocate([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
        for i0 in range(0, 1):
            for i1 in range(0, 512):
                for i2 in range(0, 58):
                    for i3 in range(0, 58):
                        with tir.block([1, 512, 58, 58], "pad_temp") as [i0_1, i1_1, i2_1, i3_1]:
                            tir.bind(i0_1, i0)
                            tir.bind(i1_1, i1)
                            tir.bind(i2_1, i2)
                            tir.bind(i3_1, i3)
                            tir.reads([X[i0_1:(i0_1 + 1), i1_1:(i1_1 + 1), (i2_1 - 1):((i2_1 - 1) + 1), (i3_1 - 1):((i3_1 - 1) + 1)]])
                            tir.writes([pad_temp[i0_1:(i0_1 + 1), i1_1:(i1_1 + 1), i2_1:(i2_1 + 1), i3_1:(i3_1 + 1)]])
                            pad_temp[i0_1, i1_1, i2_1, i3_1] = tir.if_then_else(((((i2_1 >= 1) and (i2_1 < 57)) and (i3_1 >= 1)) and (i3_1 < 57)), X[i0_1, i1_1, (i2_1 - 1), (i3_1 - 1)], tir.float32(0), dtype="float32")
        for i0_2 in range(0, 1):
            for i1_2 in range(0, 512):
                for i2_2 in range(0, 56):
                    for i3_2 in range(0, 56):
                        for i4 in range(0, 512):
                            for i5 in range(0, 3):
                                for i6 in range(0, 3):
                                    with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                        tir.bind(nn, i0_2)
                                        tir.bind(ff, i1_2)
                                        tir.bind(yy, i2_2)
                                        tir.bind(xx, i3_2)
                                        tir.bind(rc, i4)
                                        tir.bind(ry, i5)
                                        tir.bind(rx, i6)
                                        tir.reads([compute[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                                        tir.writes([compute[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                                        reducer.step(compute[nn, ff, yy, xx], (pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]))

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unexpected-keyword-arg


def test_meta_schedule_sketch_cpu_conv2d_nchw():
    # func = te.create_func(
    #     workload.conv2d_nchw(
    #         n=1,
    #         h=56,
    #         w=56,
    #         ci=512,
    #         co=512,
    #         kh=3,
    #         kw=3,
    #         stride=1,
    #         padding=1,
    #     )
    # )
    func = workload_conv2d_nchw
    support = _get_support(func=func, task_name="conv2d_nchw")
    assert len(support) == 3
    expected = [_conv2d_nchw_sketch_0, _conv2d_nchw_sketch_1, _conv2d_nchw_sketch_2]
    _fix_sampling_tile_size(
        sch=support[0],
        decisions=[
            [1, 1, 1, 1],
            [8, 1, 16, 4],
            [1, 2, 7, 4],
            [14, 1, 2, 2],
            [256, 2],
            [1, 3],
            [3, 1],
        ],
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[1],
        decisions=[
            [1, 1, 1, 1],
            [1, 8, 4, 16],
            [1, 1, 7, 8],
            [1, 1, 4, 14],
            [32, 16],
            [3, 1],
            [3, 1],
        ],
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[2],
        decisions=[
            [1, 1, 1, 1],
            [1, 8, 4, 16],
            [1, 1, 7, 8],
            [1, 1, 4, 14],
            [32, 16],
            [3, 1],
            [3, 1],
        ],
        expected=expected,
    )


def test_meta_schedule_sketch_cpu_conv2d_nchw_bias():
    func = te.create_func(
        workload.conv2d_nchw_bias(
            n=1,
            h=56,
            w=56,
            ci=512,
            co=512,
            kh=3,
            kw=3,
            stride=1,
            padding=1,
        )
    )
    print(tvm.script.asscript(func))


def test_meta_schedule_sketch_cpu_conv2d_nchw_bias_bn_relu():  # pylint: disable=invalid-name
    func = te.create_func(
        workload.conv2d_nchw_bias_bn_relu(
            n=1,
            h=56,
            w=56,
            ci=512,
            co=512,
            kh=3,
            kw=3,
            stride=1,
            padding=1,
        )
    )
    print(tvm.script.asscript(func))


def test_meta_schedule_sketch_cpu_max_pool2d_nchw():
    func = te.create_func(workload.max_pool2d_nchw(n=1, h=56, w=56, ci=512, padding=1))
    print(tvm.script.asscript(func))


if __name__ == "__main__":
    test_meta_schedule_sketch_cpu_matmul()
    test_meta_schedule_sketch_cpu_matmul_relu()
    test_meta_schedule_sketch_cpu_conv2d_nchw()
    # test_meta_schedule_sketch_cpu_conv2d_nchw_bias()
    # test_meta_schedule_sketch_cpu_conv2d_nchw_bias_bn_relu()
    # test_meta_schedule_sketch_cpu_max_pool2d_nchw()
