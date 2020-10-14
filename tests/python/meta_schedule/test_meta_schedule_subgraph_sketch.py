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
    with tir.block([512, 512, tir.reduce_axis(0, 512)], "matmul") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


def test_meta_schedule_sketch_cpu_matmul():
    # func = te.create_func(workload.matmul(n=512, m=512, k=512))
    func = workload_matmul
    support = ms.space.PostOrderApply(
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
    ).get_support(task=ms.SearchTask(func=func, task_name="matmul"))
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


def test_meta_schedule_sketch_cpu_matmul_relu():
    func = te.create_func(workload.matmul_relu(n=512, m=512, k=512))
    print(tvm.script.asscript(func))


def test_meta_schedule_sketch_cpu_conv2d_nchw():
    func = te.create_func(
        workload.conv2d_nchw(
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
    # test_meta_schedule_sketch_cpu_matmul_relu()
    # test_meta_schedule_sketch_cpu_conv2d_nchw()
    # test_meta_schedule_sketch_cpu_conv2d_nchw_bias()
    # test_meta_schedule_sketch_cpu_conv2d_nchw_bias_bn_relu()
    # test_meta_schedule_sketch_cpu_max_pool2d_nchw()
