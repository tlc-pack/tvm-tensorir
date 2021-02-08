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
            tile_binds=["blockIdx.x", "vthread", "threadIdx.x"],
        ),
    ],
    postprocs=[
        ms.postproc.rewrite_cooperative_fetch(),
        ms.postproc.rewrite_unbound_blocks(),
        ms.postproc.verify_gpu_code(),
    ],
)


def _fix_sampling_tile_size(
    sch: ms.Schedule,
    possible_decisions: List[List[List[int]]],
    expected: List[tir.PrimFunc],
):
    insts = [
        inst
        for inst in sch.trace.insts
        if isinstance(inst.inst_attrs, ms.instruction.SamplePerfectTileAttrs)
    ]
    for decisions in possible_decisions:
        if len(insts) != len(decisions):
            continue
        new_decisions = {
            k: v
            for k, v in sch.trace.decisions.items()  # pylint: disable=unnecessary-comprehension
        }
        for inst, decision in zip(insts, decisions):
            new_decisions[inst] = decision
        trace = ms.Trace(sch.trace.insts, new_decisions)
        new_sch = ms.Schedule(sch.orig_func)
        trace.apply(new_sch)
        results = [tvm.ir.structural_equal(new_sch.sch.func, i) for i in expected]
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
        for inst in sch.trace.insts:
            if inst in sch.trace.decisions:
                print(sch.trace.decisions[inst], ",")


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unused-variable
# fmt: off

@tvm.script.tir
def _matmul_sketch_0(var_A: ty.handle, var_B: ty.handle, var_C: ty.handle) -> None:
    A = tir.match_buffer(var_A, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(var_B, [512, 512], elem_offset=0, align=128, offset_factor=1)
    C = tir.match_buffer(var_C, [512, 512], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.buffer_allocate([512, 512], elem_offset=0, scope="local", align=128, offset_factor=1)
        B_shared = tir.buffer_allocate([512, 512], elem_offset=0, scope="shared", align=128, offset_factor=1)
        A_shared = tir.buffer_allocate([512, 512], elem_offset=0, scope="shared", align=128, offset_factor=1)
        for i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused in range(0, 32, annotation = {"loop_type":"blockIdx.x"}):
            for i0_outer_outer_outer_inner_i1_outer_outer_outer_inner_fused in range(0, 1, annotation = {"loop_type":"vthread"}):
                for i0_outer_outer_inner_i1_outer_outer_inner_fused in range(0, 4, annotation = {"loop_type":"threadIdx.x"}):
                    for i2_outer_outer in range(0, 8):
                        for ax0_ax1_fused_outer in range(0, 2048, annotation = {"loop_type":"lazy_cooperative_fetch"}):
                            for ax0_ax1_fused_inner in range(0, 4, annotation = {"loop_type":"vectorize"}):
                                with tir.block([512, 512], "A_shared") as [v0, v1]:
                                    tir.bind(v0, ((tir.floordiv(i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused, 8)*128) + tir.floordiv(((ax0_ax1_fused_outer*4) + ax0_ax1_fused_inner), 64)))
                                    tir.bind(v1, ((i2_outer_outer*64) + tir.floormod(((ax0_ax1_fused_outer*4) + ax0_ax1_fused_inner), 64)))
                                    tir.reads([A[v0:(v0 + 1), v1:(v1 + 1)]])
                                    tir.writes([A_shared[v0:(v0 + 1), v1:(v1 + 1)]])
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_outer_1 in range(0, 4096, annotation = {"loop_type":"lazy_cooperative_fetch"}):
                            for ax0_ax1_fused_inner_1 in range(0, 1):
                                with tir.block([512, 512], "B_shared") as [v0_1, v1_1]:
                                    tir.bind(v0_1, ((i2_outer_outer*64) + tir.floordiv(ax0_ax1_fused_outer_1, 64)))
                                    tir.bind(v1_1, ((tir.floormod(i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused, 8)*64) + tir.floormod(ax0_ax1_fused_outer_1, 64)))
                                    tir.reads([B[v0_1:(v0_1 + 1), v1_1:(v1_1 + 1)]])
                                    tir.writes([B_shared[v0_1:(v0_1 + 1), v1_1:(v1_1 + 1)]])
                                    B_shared[v0_1, v1_1] = B[v0_1, v1_1]
                        for i2_outer_inner, i0_outer_inner, i1_outer_inner, i2_inner, i0_inner, i1_inner in tir.grid(16, 2, 4, 4, 16, 16):
                            with tir.block([512, 512, tir.reduce_axis(0, 512)], "C") as [i, j, k]:
                                tir.bind(i, ((((tir.floordiv(i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused, 8)*128) + (i0_outer_outer_inner_i1_outer_outer_inner_fused*32)) + (i0_outer_inner*16)) + i0_inner))
                                tir.bind(j, (((tir.floormod(i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused, 8)*64) + (i1_outer_inner*16)) + i1_inner))
                                tir.bind(k, (((i2_outer_outer*64) + (i2_outer_inner*4)) + i2_inner))
                                tir.reads([C_local[i:(i + 1), j:(j + 1)], A_shared[i:(i + 1), k:(k + 1)], B_shared[k:(k + 1), j:(j + 1)]])
                                tir.writes([C_local[i:(i + 1), j:(j + 1)]])
                                with tir.init():
                                    C_local[i, j] = tir.float32(0)
                                C_local[i, j] = (C_local[i, j] + (A_shared[i, k]*B_shared[k, j]))
                    for ax0, ax1 in tir.grid(32, 64):
                        with tir.block([512, 512], "C_local") as [v0_2, v1_2]:
                            tir.bind(v0_2, (((tir.floordiv(i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused, 8)*128) + (i0_outer_outer_inner_i1_outer_outer_inner_fused*32)) + ax0))
                            tir.bind(v1_2, ((tir.floormod(i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused, 8)*64) + ax1))
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
            [4, 1, 4, 2, 16],
            [8, 1, 1, 4, 16],
            [8, 16, 4],
            [4096, 1],
            [2048, 4],
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
