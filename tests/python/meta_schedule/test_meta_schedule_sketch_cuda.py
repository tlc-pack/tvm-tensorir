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

TARGET = tvm.target.Target("cuda")
SPACE = ms.space.PostOrderApply(
    stages=[
        ms.rule.inline_pure_spatial(strict_mode=False),
        ms.rule.multi_level_tiling(
            structure="SSSRRSRS",
            must_cache_read=True,
            cache_read_scope="shared",
            can_cache_write=True,
            must_cache_write=True,
            cache_write_scope="local",
            fusion_levels=[3],
            vector_load_max_len=4,
            tile_marks=["lazy_blockIdx.x", "lazy_vthread", "lazy_threadIdx.x"],
        ),
    ],
    postprocs=[
        ms.postproc.rewrite_parallel_vectorize_unroll(),
        ms.postproc.rewrite_cuda_thread_bind(),
    ],
)


def _fix_sampling_tile_size(
    sch: ms.Schedule,
    possible_decisions: List[List[List[int]]],
    expected: List[tir.PrimFunc],
):
    insts = [
        inst
        for inst in sch.trace
        if isinstance(inst.inst_attrs, ms.instruction.SamplePerfectTileAttrs)
    ]
    for decisions in possible_decisions:
        if len(insts) != len(decisions):
            continue
        for inst, decision in zip(insts, decisions):
            sch.mutate_decision(inst, decision)
        sch.replay_decision()
        print(tvm.script.asscript(sch.sch.func))
        results = [tvm.ir.structural_equal(sch.sch.func, i) for i in expected]
        if sum(results) >= 1:
            return
    assert False


def _get_support(func: tir.PrimFunc, task_name: str):
    return SPACE.get_support(
        task=ms.SearchTask(
            workload=func,
            task_name=task_name,
            target=TARGET,
            target_host="llvm",
        )
    )


def _debug(support: List[ms.Schedule]):
    for i, sch in enumerate(support):
        print(f"###### {i}")
        print(tvm.script.asscript(sch.sch.func))
        for inst in sch.trace:
            if inst in sch.decisions:
                print(sch.decisions[inst], ",")


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unused-variable
# fmt: off

@tvm.script.tir
def _matmul_sketch_0(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [512, 512], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [512, 512], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: (x + y), tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.buffer_allocate([512, 512], elem_offset=0, scope="local", align=128, offset_factor=1)
        B_shared = tir.buffer_allocate([512, 512], elem_offset=0, scope="shared", align=128, offset_factor=1)
        A_shared = tir.buffer_allocate([512, 512], elem_offset=0, scope="shared", align=128, offset_factor=1)
        for i0_outer_outer_outer_outer in range(0, 16, annotation = {"loop_type":"lazy_blockIdx.x"}):
            for i1_outer_outer_outer_outer in range(0, 4, annotation = {"loop_type":"lazy_blockIdx.x"}):
                for i0_outer_outer_outer_inner in range(0, 8, annotation = {"loop_type":"lazy_vthread"}):
                    for i1_outer_outer_outer_inner in range(0, 1, annotation = {"loop_type":"lazy_vthread"}):
                        for i0_outer_outer_inner in range(0, 2, annotation = {"loop_type":"lazy_threadIdx.x"}):
                            for i1_outer_outer_inner in range(0, 8, annotation = {"loop_type":"lazy_threadIdx.x"}):
                                for i2_outer_outer in range(0, 64):
                                    for ax0_ax1_fused_outer in range(0, 16, annotation = {"loop_type":"lazy_cooperative_fetch"}):
                                        for ax0_ax1_fused_inner in range(0, 1, annotation = {"loop_type":"lazy_vectorize"}):
                                            with tir.block([512, 512], "") as [v0, v1]:
                                                tir.bind(v0, ((((i0_outer_outer_outer_outer*32) + (i0_outer_outer_outer_inner*4)) + (i0_outer_outer_inner*2)) + tir.floordiv((ax0_ax1_fused_outer + ax0_ax1_fused_inner), 8)))
                                                tir.bind(v1, ((i2_outer_outer*8) + tir.floormod((ax0_ax1_fused_outer + ax0_ax1_fused_inner), 8)))
                                                tir.reads([A[v0:(v0 + 1), v1:(v1 + 1)]])
                                                tir.writes([A_shared[v0:(v0 + 1), v1:(v1 + 1)]])
                                                A_shared[v0, v1] = A[v0, v1]
                                    for ax0_ax1_fused_outer_1 in range(0, 32, annotation = {"loop_type":"lazy_cooperative_fetch"}):
                                        for ax0_ax1_fused_inner_1 in range(0, 4, annotation = {"loop_type":"lazy_vectorize"}):
                                            with tir.block([512, 512], "") as [v0_1, v1_1]:
                                                tir.bind(v0_1, ((i2_outer_outer*8) + tir.floordiv(((ax0_ax1_fused_outer_1*4) + ax0_ax1_fused_inner_1), 16)))
                                                tir.bind(v1_1, (((i1_outer_outer_outer_outer*128) + (i1_outer_outer_inner*16)) + tir.floormod(((ax0_ax1_fused_outer_1*4) + ax0_ax1_fused_inner_1), 16)))
                                                tir.reads([B[v0_1:(v0_1 + 1), v1_1:(v1_1 + 1)]])
                                                tir.writes([B_shared[v0_1:(v0_1 + 1), v1_1:(v1_1 + 1)]])
                                                B_shared[v0_1, v1_1] = B[v0_1, v1_1]
                                    for i2_outer_inner in range(0, 8):
                                        for i0_outer_inner in range(0, 1):
                                            for i1_outer_inner in range(0, 1):
                                                for i2_inner in range(0, 1):
                                                    for i0_inner in range(0, 2):
                                                        for i1_inner in range(0, 16):
                                                            with tir.block([512, 512, tir.reduce_axis(0, 512)], "C") as [vi, vj, vk]:
                                                                tir.bind(vi, ((((i0_outer_outer_outer_outer*32) + (i0_outer_outer_outer_inner*4)) + (i0_outer_outer_inner*2)) + i0_inner))
                                                                tir.bind(vj, (((i1_outer_outer_outer_outer*128) + (i1_outer_outer_inner*16)) + i1_inner))
                                                                tir.bind(vk, ((i2_outer_outer*8) + i2_outer_inner))
                                                                tir.reads([C_local[vi:(vi + 1), vj:(vj + 1)], A_shared[vi:(vi + 1), vk:(vk + 1)], B_shared[vk:(vk + 1), vj:(vj + 1)]])
                                                                tir.writes([C_local[vi:(vi + 1), vj:(vj + 1)]])
                                                                reducer.step(C_local[vi, vj], (A_shared[vi, vk]*B_shared[vk, vj]))
                                for ax0 in range(0, 2):
                                    for ax1 in range(0, 16):
                                        with tir.block([512, 512], "") as [v0_2, v1_2]:
                                            tir.bind(v0_2, ((((i0_outer_outer_outer_outer*32) + (i0_outer_outer_outer_inner*4)) + (i0_outer_outer_inner*2)) + ax0))
                                            tir.bind(v1_2, (((i1_outer_outer_outer_outer*128) + (i1_outer_outer_inner*16)) + ax1))
                                            tir.reads([C_local[v0_2:(v0_2 + 1), v1_2:(v1_2 + 1)]])
                                            tir.writes([C[v0_2:(v0_2 + 1), v1_2:(v1_2 + 1)]])
                                            C[v0_2, v1_2] = C_local[v0_2, v1_2]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unused-variable


def test_meta_schedule_sketch_cuda_matmul():
    func = te.create_func(te_workload.matmul(512, 512, 512))
    support = _get_support(func=func, task_name="matmul")
    expected = [_matmul_sketch_0]
    possible_decisions = [
        [
            [16, 8, 2, 1, 2],
            [4, 1, 8, 1, 16],
            [64, 8, 1],
            [32, 4],
            [16, 1],
        ],
    ]
    assert len(support) == 1
    _fix_sampling_tile_size(
        sch=support[0],
        possible_decisions=possible_decisions,
        expected=expected,
    )


if __name__ == "__main__":
    test_meta_schedule_sketch_cuda_matmul()
