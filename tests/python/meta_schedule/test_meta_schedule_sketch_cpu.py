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

import te_workload
import tvm
from tvm import meta_schedule as ms
from tvm import te, tir
from tvm.script import ty

SPACE = ms.space.PostOrderApply(
    stages=[
        ms.rule.inline_pure_spatial(strict_mode=True),
        ms.rule.multi_level_tiling(
            structure="SSRSRS",
            must_cache_read=False,
            cache_read_scope="global",
            can_cache_write=True,
            must_cache_write=False,
            cache_write_scope="global",
            consumer_inline_strict=True,
            fusion_levels=[1, 2],
        ),
    ],
    postprocs=[],
)


def _fix_sampling_tile_size(
    sch: tir.Schedule,
    func: tir.PrimFunc,
    possible_decisions: List[List[List[int]]],
    expected: List[tir.PrimFunc],
):
    insts = [inst for inst in sch.trace.insts if inst.kind.name == "SamplePerfectTile"]
    for decisions in possible_decisions:
        if len(insts) != len(decisions):
            continue
        new_decisions = {
            k: v
            for k, v in sch.trace.decisions.items()  # pylint: disable=unnecessary-comprehension
        }
        for inst, decision in zip(insts, decisions):
            new_decisions[inst] = decision
        trace = tir.schedule.Trace(sch.trace.insts, new_decisions)
        new_sch = tir.Schedule(func, traced=True)
        trace.apply_to_schedule(new_sch, remove_postproc=True)
        results = [tvm.ir.structural_equal(new_sch.mod["main"], i) for i in expected]
        if sum(results) >= 1:
            return
    assert False


def _get_support(func: tir.PrimFunc, task_name: str):
    return SPACE.get_support(task=ms.SearchTask(workload=func, task_name=task_name))


def _debug(support: List[tir.Schedule]):
    for i, sch in enumerate(support):
        print(f"###### {i}")
        print(tvm.script.asscript(sch.mod["main"]))
        for inst in sch.trace.insts:
            if inst in sch.trace.decisions:
                print(sch.trace.decisions[inst], ",")


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unused-variable
# fmt: off

@tvm.script.tir
def _matmul_sketch_0(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.alloc_buffer([512, 512], elem_offset=0, scope="global", align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 32):
            for i1_outer_outer_outer in range(0, 2):
                for i0_outer_outer_inner in range(0, 1):
                    for i1_outer_outer_inner in range(0, 8):
                        for i2_outer in range(0, 256):
                            for i0_outer_inner in range(0, 1):
                                for i1_outer_inner in range(0, 8):
                                    for i2_inner in range(0, 2):
                                        for i0_inner in range(0, 16):
                                            for i1_inner in range(0, 4):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "C") as [vi, vj, vk]:
                                                    tir.bind(vi, (i0_outer_outer_outer*16) + i0_inner)
                                                    tir.bind(vj, ((((i1_outer_outer_outer*256) + (i1_outer_outer_inner*32)) + (i1_outer_inner*4)) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*2) + i2_inner))
                                                    tir.reads([C_local[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C_local[vi:(vi + 1), vj:(vj + 1)]])
                                                    with tir.init():
                                                        C_local[vi, vj] = 0.0
                                                    C_local[vi, vj] = C_local[vi, vj] + A[vi, vk]*B[vk, vj]
                        for ax0 in range(0, 16):
                            for ax1 in range(0, 32):
                                with tir.block([512, 512], "") as [v0, v1]:
                                    tir.bind(v0, ((i0_outer_outer_outer*16) + ax0))
                                    tir.bind(v1, (((i1_outer_outer_outer*256) + (i1_outer_outer_inner*32)) + ax1))
                                    tir.reads([C_local[v0:(v0 + 1), v1:(v1 + 1)]])
                                    tir.writes([C[v0:(v0 + 1), v1:(v1 + 1)]])
                                    C[v0, v1] = C_local[v0, v1]


@tvm.script.tir
def _matmul_sketch_1(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.alloc_buffer([512, 512], elem_offset=0, scope="global", align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 32):
            for i1_outer_outer_outer in range(0, 2):
                for i0_outer_outer_inner in range(0, 1):
                    for i1_outer_outer_inner in range(0, 8):
                        for i2_outer in range(0, 256):
                            for i0_outer_inner in range(0, 1):
                                for i1_outer_inner in range(0, 8):
                                    for i2_inner in range(0, 2):
                                        for i0_inner in range(0, 16):
                                            for i1_inner in range(0, 4):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "C") as [vi, vj, vk]:
                                                    tir.bind(vi, ((i0_outer_outer_outer*16) + i0_inner))
                                                    tir.bind(vj, ((((i1_outer_outer_outer*256) + (i1_outer_outer_inner*32)) + (i1_outer_inner*4)) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*2) + i2_inner))
                                                    tir.reads([C_local[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C_local[vi:(vi + 1), vj:(vj + 1)]])
                                                    with tir.init():
                                                        C_local[vi, vj] = 0.0
                                                    C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
                for ax0 in range(0, 16):
                    for ax1 in range(0, 256):
                        with tir.block([512, 512], "") as [v0, v1]:
                            tir.bind(v0, ((i0_outer_outer_outer*16) + ax0))
                            tir.bind(v1, ((i1_outer_outer_outer*256) + ax1))
                            tir.reads([C_local[v0:(v0 + 1), v1:(v1 + 1)]])
                            tir.writes([C[v0:(v0 + 1), v1:(v1 + 1)]])
                            C[v0, v1] = C_local[v0, v1]


@tvm.script.tir
def _matmul_sketch_2(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0_outer_outer_outer in range(0, 8):
            for i1_outer_outer_outer in range(0, 64):
                for i0_outer_outer_inner in range(0, 32):
                    for i1_outer_outer_inner in range(0, 1):
                        for i2_outer in range(0, 256):
                            for i0_outer_inner in range(0, 1):
                                for i1_outer_inner in range(0, 4):
                                    for i2_inner in range(0, 2):
                                        for i0_inner in range(0, 2):
                                            for i1_inner in range(0, 2):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "C") as [vi, vj, vk]:
                                                    tir.bind(vi, (((i0_outer_outer_outer*64) + (i0_outer_outer_inner*2)) + i0_inner))
                                                    tir.bind(vj, (((i1_outer_outer_outer*8) + (i1_outer_inner*2)) + i1_inner))
                                                    tir.bind(vk, ((i2_outer*2) + i2_inner))
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    with tir.init():
                                                        C[vi, vj] = 0.0
                                                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unused-variable


def test_meta_schedule_sketch_cpu_matmul():
    func = te.create_prim_func(te_workload.matmul(n=512, m=512, k=512))
    support = _get_support(func=func, task_name="matmul")
    expected = [_matmul_sketch_0, _matmul_sketch_1, _matmul_sketch_2]
    possible_decisions = [
        [
            [32, 1, 1, 16],
            [2, 8, 8, 4],
            [256, 2],
        ],
        [
            [32, 1, 1, 16],
            [2, 8, 8, 4],
            [256, 2],
        ],
        [
            [8, 32, 1, 2],
            [64, 1, 4, 2],
            [256, 2],
        ],
    ]
    assert len(support) == 3
    _fix_sampling_tile_size(
        sch=support[0],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[1],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[2],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unused-variable
# fmt: off

@tvm.script.tir
def _matmul_relu_sketch_0(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    D = tir.match_buffer(d, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.alloc_buffer([512, 512], elem_offset=0, align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 1):
            for i1_outer_outer_outer in range(0, 4):
                for i0_outer_outer_inner in range(0, 2):
                    for i1_outer_outer_inner in range(0, 2):
                        for i2_outer in range(0, 512):
                            for i0_outer_inner in range(0, 128):
                                for i1_outer_inner in range(0, 32):
                                    for i2_inner in range(0, 1):
                                        for i0_inner in range(0, 2):
                                            for i1_inner in range(0, 2):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, (((i0_outer_outer_inner*256) + (i0_outer_inner*2)) + i0_inner))
                                                    tir.bind(vj, ((((i1_outer_outer_outer*128) + (i1_outer_outer_inner*64)) + (i1_outer_inner*2)) + i1_inner))
                                                    tir.bind(vk, i2_outer)
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    with tir.init():
                                                        C[vi, vj] = 0.0
                                                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
                        for ax0 in range(0, 256):
                            for ax1 in range(0, 64):
                                with tir.block([512, 512], "relu") as [vi_1, vj_1]:
                                    tir.bind(vi_1, ((i0_outer_outer_inner*256) + ax0))
                                    tir.bind(vj_1, (((i1_outer_outer_outer*128) + (i1_outer_outer_inner*64)) + ax1))
                                    tir.reads([C[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                                    tir.writes([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                                    D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))


@tvm.script.tir
def _matmul_relu_sketch_1(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    D = tir.match_buffer(d, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.alloc_buffer([512, 512], elem_offset=0, align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 1):
            for i1_outer_outer_outer in range(0, 4):
                for i0_outer_outer_inner in range(0, 2):
                    for i1_outer_outer_inner in range(0, 2):
                        for i2_outer in range(0, 512):
                            for i0_outer_inner in range(0, 128):
                                for i1_outer_inner in range(0, 32):
                                    for i2_inner in range(0, 1):
                                        for i0_inner in range(0, 2):
                                            for i1_inner in range(0, 2):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, (((i0_outer_outer_inner*256) + (i0_outer_inner*2)) + i0_inner))
                                                    tir.bind(vj, ((((i1_outer_outer_outer*128) + (i1_outer_outer_inner*64)) + (i1_outer_inner*2)) + i1_inner))
                                                    tir.bind(vk, i2_outer)
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    with tir.init():
                                                        C[vi, vj] = 0.0
                                                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
                for ax0 in range(0, 512):
                    for ax1 in range(0, 128):
                        with tir.block([512, 512], "relu") as [vi_1, vj_1]:
                            tir.bind(vi_1, ax0)
                            tir.bind(vj_1, ((i1_outer_outer_outer*128) + ax1))
                            tir.reads([C[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                            tir.writes([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                            D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))


@tvm.script.tir
def _matmul_relu_sketch_2(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    D = tir.match_buffer(d, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.alloc_buffer([512, 512], elem_offset=0, align=128, offset_factor=1)
        for i0_outer_outer_outer in range(0, 1):
            for i1_outer_outer_outer in range(0, 4):
                for i0_outer_outer_inner in range(0, 2):
                    for i1_outer_outer_inner in range(0, 2):
                        for i2_outer in range(0, 512):
                            for i0_outer_inner in range(0, 128):
                                for i1_outer_inner in range(0, 32):
                                    for i2_inner in range(0, 1):
                                        for i0_inner in range(0, 2):
                                            for i1_inner in range(0, 2):
                                                with tir.block([512, 512, tir.reduce_axis(0, 512)], "matmul") as [vi, vj, vk]:
                                                    tir.bind(vi, (((i0_outer_outer_inner*256) + (i0_outer_inner*2)) + i0_inner))
                                                    tir.bind(vj, ((((i1_outer_outer_outer*128) + (i1_outer_outer_inner*64)) + (i1_outer_inner*2)) + i1_inner))
                                                    tir.bind(vk, i2_outer)
                                                    tir.reads([C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                                                    tir.writes([C[vi:(vi + 1), vj:(vj + 1)]])
                                                    with tir.init():
                                                        C[vi, vj] = 0.0
                                                    C[vi, vj] = C[vi, vj] + A[vi, vk]*B[vk, vj]
        for i0 in range(0, 512):
            for i1 in range(0, 512):
                with tir.block([512, 512], "relu") as [vi_1, vj_1]:
                    tir.bind(vi_1, i0)
                    tir.bind(vj_1, i1)
                    tir.reads([C[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                    tir.writes([D[vi_1:(vi_1 + 1), vj_1:(vj_1 + 1)]])
                    D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unused-variable


def test_meta_schedule_sketch_cpu_matmul_relu():
    func = te.create_prim_func(te_workload.matmul_relu(n=512, m=512, k=512))
    support = _get_support(func=func, task_name="matmul_relu")
    assert len(support) == 3
    expected = [_matmul_relu_sketch_0, _matmul_relu_sketch_1, _matmul_relu_sketch_2]
    possible_decisions = [
        [
            [1, 2, 128, 2],
            [4, 2, 32, 2],
            [512, 1],
        ],
        [
            [1, 2, 128, 2],
            [4, 2, 32, 2],
            [512, 1],
        ],
        [
            [1, 2, 128, 2],
            [4, 2, 32, 2],
            [512, 1],
        ],
    ]
    _fix_sampling_tile_size(
        sch=support[0],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[1],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[2],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unexpected-keyword-arg,unused-variable
# fmt: off

@tvm.script.tir
def _conv2d_nchw_sketch_0(var_X: ty.handle, var_W: ty.handle, var_compute: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.alloc_buffer([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
        compute_local = tir.alloc_buffer([1, 512, 56, 56], elem_offset=0, scope="global", align=128, offset_factor=1)
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
        for i0_2_outer_outer_outer, i1_2_outer_outer_outer, i2_2_outer_outer_outer, i3_2_outer_outer_outer in tir.grid(1, 2, 2, 7):
            for i0_2_outer_outer_inner, i1_2_outer_outer_inner, i2_2_outer_outer_inner, i3_2_outer_outer_inner in tir.grid(1, 8, 2, 4):
                for i4_outer, i5_outer, i6_outer in tir.grid(256, 1, 3):
                    for i0_2_outer_inner, i1_2_outer_inner, i2_2_outer_inner, i3_2_outer_inner in tir.grid(1, 16, 1, 1):
                        for i4_inner, i5_inner, i6_inner in tir.grid(2, 3, 1):
                            for i0_2_inner, i1_2_inner, i2_2_inner, i3_2_inner in tir.grid(1, 2, 14, 2):
                                with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                    tir.bind(nn, 0)
                                    tir.bind(ff, ((((i1_2_outer_outer_outer*256) + (i1_2_outer_outer_inner*32)) + (i1_2_outer_inner*2)) + i1_2_inner))
                                    tir.bind(yy, (((i2_2_outer_outer_outer*28) + (i2_2_outer_outer_inner*14)) + i2_2_inner))
                                    tir.bind(xx, (((i3_2_outer_outer_outer*8) + (i3_2_outer_outer_inner*2)) + i3_2_inner))
                                    tir.bind(rc, ((i4_outer*2) + i4_inner))
                                    tir.bind(ry, i5_inner)
                                    tir.bind(rx, i6_outer)
                                    tir.reads([compute_local[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                                    tir.writes([compute_local[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                                    with tir.init():
                                        compute_local[nn, ff, yy, xx] = 0.0
                                    compute_local[nn, ff, yy, xx] = compute_local[nn, ff, yy, xx] + pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]
                for ax0 in range(0, 1):
                    for ax1 in range(0, 32):
                        for ax2 in range(0, 14):
                            for ax3 in range(0, 2):
                                with tir.block([1, 512, 56, 56], "") as [v0, v1, v2, v3]:
                                    tir.bind(v0, 0)
                                    tir.bind(v1, (((i1_2_outer_outer_outer*256) + (i1_2_outer_outer_inner*32)) + ax1))
                                    tir.bind(v2, (((i2_2_outer_outer_outer*28) + (i2_2_outer_outer_inner*14)) + ax2))
                                    tir.bind(v3, (((i3_2_outer_outer_outer*8) + (i3_2_outer_outer_inner*2)) + ax3))
                                    tir.reads([compute_local[v0:(v0 + 1), v1:(v1 + 1), v2:(v2 + 1), v3:(v3 + 1)]])
                                    tir.writes([compute[v0:(v0 + 1), v1:(v1 + 1), v2:(v2 + 1), v3:(v3 + 1)]])
                                    compute[v0, v1, v2, v3] = compute_local[v0, v1, v2, v3]


@tvm.script.tir
def _conv2d_nchw_sketch_1(var_X: ty.handle, var_W: ty.handle, var_compute: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.alloc_buffer([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
        compute_local = tir.alloc_buffer([1, 512, 56, 56], elem_offset=0, scope="global", align=128, offset_factor=1)
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
        for i0_2_outer_outer_outer, i1_2_outer_outer_outer, i2_2_outer_outer_outer, i3_2_outer_outer_outer in tir.grid(1, 2, 2, 7):
            for i0_2_outer_outer_inner, i1_2_outer_outer_inner, i2_2_outer_outer_inner, i3_2_outer_outer_inner in tir.grid(1, 8, 2, 4):
                for i4_outer, i5_outer, i6_outer in tir.grid(256, 1, 3):
                    for i0_2_outer_inner, i1_2_outer_inner, i2_2_outer_inner, i3_2_outer_inner in tir.grid(1, 16, 1, 1):
                        for i4_inner, i5_inner, i6_inner in tir.grid(2, 3, 1):
                            for i0_2_inner, i1_2_inner, i2_2_inner, i3_2_inner in tir.grid(1, 2, 14, 2):
                                with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                    tir.bind(nn, 0)
                                    tir.bind(ff, ((((i1_2_outer_outer_outer*256) + (i1_2_outer_outer_inner*32)) + (i1_2_outer_inner*2)) + i1_2_inner))
                                    tir.bind(yy, (((i2_2_outer_outer_outer*28) + (i2_2_outer_outer_inner*14)) + i2_2_inner))
                                    tir.bind(xx, (((i3_2_outer_outer_outer*8) + (i3_2_outer_outer_inner*2)) + i3_2_inner))
                                    tir.bind(rc, ((i4_outer*2) + i4_inner))
                                    tir.bind(ry, i5_inner)
                                    tir.bind(rx, i6_outer)
                                    tir.reads([compute_local[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                                    tir.writes([compute_local[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                                    with tir.init():
                                        compute_local[nn, ff, yy, xx] = 0.0
                                    compute_local[nn, ff, yy, xx] = compute_local[nn, ff, yy, xx] + pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]
            for ax0 in range(0, 1):
                for ax1 in range(0, 256):
                    for ax2 in range(0, 28):
                        for ax3 in range(0, 8):
                            with tir.block([1, 512, 56, 56], "") as [v0, v1, v2, v3]:
                                tir.bind(v0, 0)
                                tir.bind(v1, ((i1_2_outer_outer_outer*256) + ax1))
                                tir.bind(v2, ((i2_2_outer_outer_outer*28) + ax2))
                                tir.bind(v3, ((i3_2_outer_outer_outer*8) + ax3))
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
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.alloc_buffer([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
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
        for i0_2_outer_outer_outer, i1_2_outer_outer_outer, i2_2_outer_outer_outer, i3_2_outer_outer_outer in tir.grid(1, 2, 1, 1):
            for i0_2_outer_outer_inner, i1_2_outer_outer_inner, i2_2_outer_outer_inner, i3_2_outer_outer_inner in tir.grid(1, 16, 7, 1):
                for i4_outer, i5_outer, i6_outer in tir.grid(32, 1, 1):
                    for i0_2_outer_inner, i1_2_outer_inner, i2_2_outer_inner, i3_2_outer_inner in tir.grid(1, 2, 1, 4):
                        for i4_inner, i5_inner, i6_inner in tir.grid(16, 3, 3):
                            for i0_2_inner, i1_2_inner, i2_2_inner, i3_2_inner in tir.grid(1, 8, 8, 14):
                                with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                    tir.bind(nn, 0)
                                    tir.bind(ff, ((((i1_2_outer_outer_outer*256) + (i1_2_outer_outer_inner*16)) + (i1_2_outer_inner*8)) + i1_2_inner))
                                    tir.bind(yy, ((i2_2_outer_outer_inner*8) + i2_2_inner))
                                    tir.bind(xx, ((i3_2_outer_inner*14) + i3_2_inner))
                                    tir.bind(rc, ((i4_outer*16) + i4_inner))
                                    tir.bind(ry, i5_inner)
                                    tir.bind(rx, i6_inner)
                                    tir.reads([compute[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                                    tir.writes([compute[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                                    with tir.init():
                                        compute[nn, ff, yy, xx] = 0.0
                                    compute[nn, ff, yy, xx] = compute[nn, ff, yy, xx] + pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unexpected-keyword-arg,unused-variable


def test_meta_schedule_sketch_cpu_conv2d_nchw():
    func = te.create_prim_func(
        te_workload.conv2d_nchw(
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
    support = _get_support(func=func, task_name="conv2d_nchw")
    assert len(support) == 3
    expected = [_conv2d_nchw_sketch_0, _conv2d_nchw_sketch_1, _conv2d_nchw_sketch_2]
    possible_decisions = [
        [
            [1, 1, 1, 1],
            [2, 8, 16, 2],
            [2, 2, 1, 14],
            [7, 4, 1, 2],
            [256, 2],
            [1, 3],
            [3, 1],
        ],
        [
            [1, 1, 1, 1],
            [2, 8, 16, 2],
            [2, 2, 1, 14],
            [7, 4, 1, 2],
            [256, 2],
            [1, 3],
            [3, 1],
        ],
        [
            [1, 1, 1, 1],
            [2, 16, 2, 8],
            [1, 7, 1, 8],
            [1, 1, 4, 14],
            [32, 16],
            [1, 3],
            [1, 3],
        ],
    ]
    _fix_sampling_tile_size(
        sch=support[0],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[1],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[2],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unexpected-keyword-arg,unused-variable
# fmt: off

@tvm.script.tir
def _conv2d_nchw_bias_bn_relu_sketch_0(var_X: ty.handle, var_W: ty.handle, var_B: ty.handle, var_bn_scale: ty.handle, var_bn_offset: ty.handle, var_compute: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    B = tir.match_buffer(var_B, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    bn_offset = tir.match_buffer(var_bn_offset, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    bn_scale = tir.match_buffer(var_bn_scale, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.alloc_buffer([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
        compute_1 = tir.alloc_buffer([1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
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
        for i0_2_outer_outer_outer, i1_2_outer_outer_outer, i2_2_outer_outer_outer, i3_2_outer_outer_outer in tir.grid(1, 4, 1, 1):
            for i0_2_outer_outer_inner, i1_2_outer_outer_inner, i2_2_outer_outer_inner, i3_2_outer_outer_inner in tir.grid(1, 8, 1, 2):
                for i4_outer, i5_outer, i6_outer in tir.grid(64, 1, 1):
                    for i0_2_outer_inner, i1_2_outer_inner, i2_2_outer_inner, i3_2_outer_inner in tir.grid(1, 2, 28, 7):
                        for i4_inner, i5_inner, i6_inner in tir.grid(8, 3, 3):
                            for i0_2_inner, i1_2_inner, i2_2_inner, i3_2_inner in tir.grid(1, 8, 2, 4):
                                with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                    tir.bind(nn, 0)
                                    tir.bind(ff, ((((i1_2_outer_outer_outer*128) + (i1_2_outer_outer_inner*16)) + (i1_2_outer_inner*8)) + i1_2_inner))
                                    tir.bind(yy, ((i2_2_outer_inner*2) + i2_2_inner))
                                    tir.bind(xx, (((i3_2_outer_outer_inner*28) + (i3_2_outer_inner*4)) + i3_2_inner))
                                    tir.bind(rc, ((i4_outer*8) + i4_inner))
                                    tir.bind(ry, i5_inner)
                                    tir.bind(rx, i6_inner)
                                    tir.reads([compute_1[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                                    tir.writes([compute_1[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                                    with tir.init():
                                        compute_1[nn, ff, yy, xx] = 0.0
                                    compute_1[nn, ff, yy, xx] = compute_1[nn, ff, yy, xx] + pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]
                for ax0 in range(0, 1):
                    for ax1 in range(0, 16):
                        for ax2 in range(0, 56):
                            for ax3 in range(0, 28):
                                with tir.block([1, 512, 56, 56], "compute_2") as [i0_7, i1_7, i2_7, i3_7]:
                                    tir.bind(i0_7, 0)
                                    tir.bind(i1_7, (((i1_2_outer_outer_outer*128) + (i1_2_outer_outer_inner*16)) + ax1))
                                    tir.bind(i2_7, ax2)
                                    tir.bind(i3_7, ((i3_2_outer_outer_inner*28) + ax3))
                                    tir.reads([compute_1[i0_7:(i0_7 + 1), i1_7:(i1_7 + 1), i2_7:(i2_7 + 1), i3_7:(i3_7 + 1)], B[i1_7:(i1_7 + 1), 0:1, 0:1], bn_scale[i1_7:(i1_7 + 1), 0:1, 0:1], bn_offset[i1_7:(i1_7 + 1), 0:1, 0:1]])
                                    tir.writes([compute[i0_7:(i0_7 + 1), i1_7:(i1_7 + 1), i2_7:(i2_7 + 1), i3_7:(i3_7 + 1)]])
                                    compute[i0_7, i1_7, i2_7, i3_7] = tir.max((((compute_1[i0_7, i1_7, i2_7, i3_7] + B[i1_7, 0, 0])*bn_scale[i1_7, 0, 0]) + bn_offset[i1_7, 0, 0]), tir.float32(0))


@tvm.script.tir
def _conv2d_nchw_bias_bn_relu_sketch_1(var_X: ty.handle, var_W: ty.handle, var_B: ty.handle, var_bn_scale: ty.handle, var_bn_offset: ty.handle, var_compute: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    B = tir.match_buffer(var_B, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    bn_offset = tir.match_buffer(var_bn_offset, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    bn_scale = tir.match_buffer(var_bn_scale, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.alloc_buffer([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
        compute_1 = tir.alloc_buffer([1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
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
        for i0_2_outer_outer_outer, i1_2_outer_outer_outer, i2_2_outer_outer_outer, i3_2_outer_outer_outer in tir.grid(1, 4, 1, 1):
            for i0_2_outer_outer_inner, i1_2_outer_outer_inner, i2_2_outer_outer_inner, i3_2_outer_outer_inner in tir.grid(1, 8, 1, 2):
                for i4_outer, i5_outer, i6_outer in tir.grid(64, 1, 1):
                    for i0_2_outer_inner, i1_2_outer_inner, i2_2_outer_inner, i3_2_outer_inner in tir.grid(1, 2, 28, 7):
                        for i4_inner, i5_inner, i6_inner in tir.grid(8, 3, 3):
                            for i0_2_inner, i1_2_inner, i2_2_inner, i3_2_inner in tir.grid(1, 8, 2, 4):
                                with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                    tir.bind(nn, 0)
                                    tir.bind(ff, ((((i1_2_outer_outer_outer*128) + (i1_2_outer_outer_inner*16)) + (i1_2_outer_inner*8)) + i1_2_inner))
                                    tir.bind(yy, ((i2_2_outer_inner*2) + i2_2_inner))
                                    tir.bind(xx, (((i3_2_outer_outer_inner*28) + (i3_2_outer_inner*4)) + i3_2_inner))
                                    tir.bind(rc, ((i4_outer*8) + i4_inner))
                                    tir.bind(ry, i5_inner)
                                    tir.bind(rx, i6_inner)
                                    tir.reads([compute_1[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                                    tir.writes([compute_1[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                                    with tir.init():
                                        compute_1[nn, ff, yy, xx] = 0.0
                                    compute_1[nn, ff, yy, xx] = compute_1[nn, ff, yy, xx] + pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]
            for ax0 in range(0, 1):
                for ax1 in range(0, 128):
                    for ax2 in range(0, 56):
                        for ax3 in range(0, 56):
                            with tir.block([1, 512, 56, 56], "compute_2") as [i0_7, i1_7, i2_7, i3_7]:
                                tir.bind(i0_7, 0)
                                tir.bind(i1_7, ((i1_2_outer_outer_outer*128) + ax1))
                                tir.bind(i2_7, ax2)
                                tir.bind(i3_7, ax3)
                                tir.reads([compute_1[i0_7:(i0_7 + 1), i1_7:(i1_7 + 1), i2_7:(i2_7 + 1), i3_7:(i3_7 + 1)], B[i1_7:(i1_7 + 1), 0:1, 0:1], bn_scale[i1_7:(i1_7 + 1), 0:1, 0:1], bn_offset[i1_7:(i1_7 + 1), 0:1, 0:1]])
                                tir.writes([compute[i0_7:(i0_7 + 1), i1_7:(i1_7 + 1), i2_7:(i2_7 + 1), i3_7:(i3_7 + 1)]])
                                compute[i0_7, i1_7, i2_7, i3_7] = tir.max((((compute_1[i0_7, i1_7, i2_7, i3_7] + B[i1_7, 0, 0])*bn_scale[i1_7, 0, 0]) + bn_offset[i1_7, 0, 0]), tir.float32(0))


@tvm.script.tir
def _conv2d_nchw_bias_bn_relu_sketch_2(var_X: ty.handle, var_W: ty.handle, var_B: ty.handle, var_bn_scale: ty.handle, var_bn_offset: ty.handle, var_compute: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    B = tir.match_buffer(var_B, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    bn_offset = tir.match_buffer(var_bn_offset, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    bn_scale = tir.match_buffer(var_bn_scale, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        pad_temp = tir.alloc_buffer([1, 512, 58, 58], elem_offset=0, align=128, offset_factor=1)
        compute_1 = tir.alloc_buffer([1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
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
        for i0_2_outer_outer_outer, i1_2_outer_outer_outer, i2_2_outer_outer_outer, i3_2_outer_outer_outer in tir.grid(1, 4, 1, 1):
            for i0_2_outer_outer_inner, i1_2_outer_outer_inner, i2_2_outer_outer_inner, i3_2_outer_outer_inner in tir.grid(1, 8, 1, 2):
                for i4_outer, i5_outer, i6_outer in tir.grid(64, 1, 1):
                    for i0_2_outer_inner, i1_2_outer_inner, i2_2_outer_inner, i3_2_outer_inner in tir.grid(1, 2, 28, 7):
                        for i4_inner, i5_inner, i6_inner in tir.grid(8, 3, 3):
                            for i0_2_inner, i1_2_inner, i2_2_inner, i3_2_inner in tir.grid(1, 8, 2, 4):
                                with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                    tir.bind(nn, 0)
                                    tir.bind(ff, ((((i1_2_outer_outer_outer*128) + (i1_2_outer_outer_inner*16)) + (i1_2_outer_inner*8)) + i1_2_inner))
                                    tir.bind(yy, ((i2_2_outer_inner*2) + i2_2_inner))
                                    tir.bind(xx, (((i3_2_outer_outer_inner*28) + (i3_2_outer_inner*4)) + i3_2_inner))
                                    tir.bind(rc, ((i4_outer*8) + i4_inner))
                                    tir.bind(ry, i5_inner)
                                    tir.bind(rx, i6_inner)
                                    tir.reads([compute_1[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)], pad_temp[nn:(nn + 1), rc:(rc + 1), (yy + ry):((yy + ry) + 1), (xx + rx):((xx + rx) + 1)], W[ff:(ff + 1), rc:(rc + 1), ry:(ry + 1), rx:(rx + 1)]])
                                    tir.writes([compute_1[nn:(nn + 1), ff:(ff + 1), yy:(yy + 1), xx:(xx + 1)]])
                                    with tir.init():
                                        compute_1[nn, ff, yy, xx] = 0.0
                                    compute_1[nn, ff, yy, xx] = compute_1[nn, ff, yy, xx] + pad_temp[nn, rc, (yy + ry), (xx + rx)]*W[ff, rc, ry, rx]
        for i0_6 in range(0, 1):
            for i1_6 in range(0, 512):
                for i2_6 in range(0, 56):
                    for i3_6 in range(0, 56):
                        with tir.block([1, 512, 56, 56], "compute_2") as [i0_7, i1_7, i2_7, i3_7]:
                            tir.bind(i0_7, i0_6)
                            tir.bind(i1_7, i1_6)
                            tir.bind(i2_7, i2_6)
                            tir.bind(i3_7, i3_6)
                            tir.reads([compute_1[i0_7:(i0_7 + 1), i1_7:(i1_7 + 1), i2_7:(i2_7 + 1), i3_7:(i3_7 + 1)], B[i1_7:(i1_7 + 1), 0:1, 0:1], bn_scale[i1_7:(i1_7 + 1), 0:1, 0:1], bn_offset[i1_7:(i1_7 + 1), 0:1, 0:1]])
                            tir.writes([compute[i0_7:(i0_7 + 1), i1_7:(i1_7 + 1), i2_7:(i2_7 + 1), i3_7:(i3_7 + 1)]])
                            compute[i0_7, i1_7, i2_7, i3_7] = tir.max((((compute_1[i0_7, i1_7, i2_7, i3_7] + B[i1_7, 0, 0])*bn_scale[i1_7, 0, 0]) + bn_offset[i1_7, 0, 0]), tir.float32(0))

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unexpected-keyword-arg,unused-variable


def test_meta_schedule_sketch_cpu_conv2d_nchw_bias_bn_relu():  # pylint: disable=invalid-name
    func = te.create_prim_func(
        te_workload.conv2d_nchw_bias_bn_relu(
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
    support = _get_support(func=func, task_name="conv2d_nchw_bias_bn_relu")
    assert len(support) == 3
    expected = [
        _conv2d_nchw_bias_bn_relu_sketch_0,
        _conv2d_nchw_bias_bn_relu_sketch_1,
        _conv2d_nchw_bias_bn_relu_sketch_2,
    ]
    possible_decisions = [
        [
            [1, 1, 1, 1],
            [4, 8, 2, 8],
            [1, 1, 28, 2],
            [1, 2, 7, 4],
            [64, 8],
            [1, 3],
            [1, 3],
        ]
    ]
    _fix_sampling_tile_size(
        sch=support[0],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[1],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )
    _fix_sampling_tile_size(
        sch=support[2],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )


def test_meta_schedule_sketch_cpu_max_pool2d_nchw():
    func = te.create_prim_func(
        te_workload.max_pool2d_nchw(
            n=1,
            h=56,
            w=56,
            ci=512,
            padding=1,
        )
    )
    support = _get_support(func=func, task_name="max_pool2d_nchw")
    assert len(support) == 1
    expected = [func]
    possible_decisions = [[]]
    _fix_sampling_tile_size(
        sch=support[0],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )


if __name__ == "__main__":
    test_meta_schedule_sketch_cpu_matmul()
    test_meta_schedule_sketch_cpu_matmul_relu()
    test_meta_schedule_sketch_cpu_conv2d_nchw()
    test_meta_schedule_sketch_cpu_conv2d_nchw_bias_bn_relu()
    test_meta_schedule_sketch_cpu_max_pool2d_nchw()
