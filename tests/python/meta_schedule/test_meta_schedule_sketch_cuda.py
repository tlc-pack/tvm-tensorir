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
        ms.rule.multi_level_tiling(
            structure="SSSRRSRS",
            must_cache_read=True,
            cache_read_scope="shared",
            can_cache_write=True,
            must_cache_write=True,
            cache_write_scope="local",
            consumer_inline_strict=False,
            fusion_levels=[3],
            vector_load_max_len=4,
            tile_binds=["blockIdx.x", "vthread", "threadIdx.x"],
        ),
        ms.rule.inline_pure_spatial(strict_mode=False),
    ],
    postprocs=[
        ms.postproc.rewrite_cooperative_fetch(),
        ms.postproc.rewrite_unbound_blocks(),
        ms.postproc.rewrite_parallel_vectorize_unroll(),
        ms.postproc.rewrite_reduction_block(),
        ms.postproc.disallow_dynamic_loops(),
        ms.postproc.verify_gpu_code(),
    ],
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
    return SPACE.get_support(
        task=ms.SearchTask(
            workload=func,
            task_name=task_name,
            target=TARGET,
            target_host="llvm",
        )
    )


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
def _matmul_sketch_0(var_A: ty.handle, var_B: ty.handle, var_C: ty.handle) -> None:
    A = tir.match_buffer(var_A, [512, 512], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(var_B, [512, 512], elem_offset=0, align=128, offset_factor=1)
    C = tir.match_buffer(var_C, [512, 512], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.alloc_buffer([512, 512], elem_offset=0, scope="local", align=128, offset_factor=1)
        B_shared = tir.alloc_buffer([512, 512], elem_offset=0, scope="shared", align=128, offset_factor=1)
        A_shared = tir.alloc_buffer([512, 512], elem_offset=0, scope="shared", align=128, offset_factor=1)
        for i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused in tir.thread_binding(0, 32, thread="blockIdx.x"):
            for i0_outer_outer_outer_inner_i1_outer_outer_outer_inner_fused in tir.thread_binding(0, 1, thread="vthread"):
                for i0_outer_outer_inner_i1_outer_outer_inner_fused in tir.thread_binding(0, 4, thread="threadIdx.x"):
                    for i2_outer_outer in range(0, 8):
                        for ax0_ax1_fused_outer in range(0, 2048, annotations={"loop_type": "lazy_cooperative_fetch"}):
                            for ax0_ax1_fused_inner in tir.vectorized(0, 4):
                                with tir.block([512, 512], "A_shared") as [v0, v1]:
                                    tir.bind(v0, ((tir.floordiv(i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused, 8)*128) + tir.floordiv(((ax0_ax1_fused_outer*4) + ax0_ax1_fused_inner), 64)))
                                    tir.bind(v1, ((i2_outer_outer*64) + tir.floormod(((ax0_ax1_fused_outer*4) + ax0_ax1_fused_inner), 64)))
                                    tir.reads([A[v0:(v0 + 1), v1:(v1 + 1)]])
                                    tir.writes([A_shared[v0:(v0 + 1), v1:(v1 + 1)]])
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_outer_1 in range(0, 4096, annotations={"loop_type": "lazy_cooperative_fetch"}):
                            for ax0_ax1_fused_inner_1 in tir.vectorized(0, 1):
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
    func = te.create_prim_func(te_workload.matmul(512, 512, 512))
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
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unused-variable,unexpected-keyword-arg,misplaced-comparison-constant
# fmt: off


@tvm.script.tir
def _conv2d_nchw_bias_bn_relu_sketch_0(
    var_X: ty.handle,
    var_W: ty.handle,
    var_B: ty.handle,
    var_bn_scale: ty.handle,
    var_bn_offset: ty.handle,
    var_compute: ty.handle,
) -> None:
    B = tir.match_buffer(var_B, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    bn_offset = tir.match_buffer(var_bn_offset, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    W = tir.match_buffer(var_W, [512, 512, 3, 3], elem_offset=0, align=128, offset_factor=1)
    compute = tir.match_buffer(var_compute, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    X = tir.match_buffer(var_X, [1, 512, 56, 56], elem_offset=0, align=128, offset_factor=1)
    bn_scale = tir.match_buffer(var_bn_scale, [512, 1, 1], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        compute_local = tir.alloc_buffer([1, 512, 56, 56], elem_offset=0, scope="local", align=128, offset_factor=1)
        W_shared = tir.alloc_buffer([512, 512, 3, 3], elem_offset=0, scope="shared", align=128, offset_factor=1)
        pad_temp_shared = tir.alloc_buffer([1, 512, 58, 58], elem_offset=0, scope="shared", align=128, offset_factor=1)
        for i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused_i2_outer_outer_outer_outer_fused_i3_outer_outer_outer_outer_fused in (tir.thread_binding(0, 4, thread="blockIdx.x")):
            for i0_outer_outer_outer_inner_i1_outer_outer_outer_inner_fused_i2_outer_outer_outer_inner_fused_i3_outer_outer_outer_inner_fused in (tir.thread_binding(0, 16, thread="vthread")):
                for i0_outer_outer_inner_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused_i3_outer_outer_inner_fused in (tir.thread_binding(0, 224, thread="threadIdx.x")):
                    for i4_outer_outer, i5_outer_outer, i6_outer_outer in tir.grid(1, 1, 1):
                        for ax0_ax1_fused_ax2_fused_ax3_fused_outer in range(0, 430592, annotations={"loop_type": "lazy_cooperative_fetch"}):
                            for ax0_ax1_fused_ax2_fused_ax3_fused_inner in tir.vectorized(0, 4):
                                with tir.block([1, 512, 58, 58], "pad_temp_shared") as [v0, v1, v2, v3]:
                                    tir.bind(v0, 0)
                                    tir.bind(v1, tir.floordiv(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner), 3364))
                                    tir.bind(v2, tir.floormod(tir.floordiv(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner), 58), 58))
                                    tir.bind(v3, tir.floormod(((ax0_ax1_fused_ax2_fused_ax3_fused_outer * 4) + ax0_ax1_fused_ax2_fused_ax3_fused_inner), 58))
                                    tir.reads([X[v0 : (v0 + 1), v1 : (v1 + 1), (v2 - 1) : ((v2 - 1) + 1), (v3 - 1) : ((v3 - 1) + 1)]])
                                    tir.writes([pad_temp_shared[v0 : (v0 + 1), v1 : (v1 + 1), v2 : (v2 + 1), v3 : (v3 + 1)]])
                                    pad_temp_shared[v0, v1, v2, v3] = tir.if_then_else(
                                        ((((v2 >= 1) and (v2 < 57)) and (v3 >= 1)) and (v3 < 57)),
                                        X[v0, v1, (v2 - 1), (v3 - 1)],
                                        tir.float32(0),
                                        dtype="float32",
                                    )
                        for ax0_ax1_fused_ax2_fused_ax3_fused_outer_1 in range(0, 589824, annotations={"loop_type": "lazy_cooperative_fetch"}):
                            for ax0_ax1_fused_ax2_fused_ax3_fused_inner_1 in tir.vectorized(0, 1):
                                with tir.block([512, 512, 3, 3], "W_shared") as [v0_1, v1_1, v2_1, v3_1]:
                                    tir.bind(v0_1, ((i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused_i2_outer_outer_outer_outer_fused_i3_outer_outer_outer_outer_fused * 128) + tir.floordiv(ax0_ax1_fused_ax2_fused_ax3_fused_outer_1, 4608)))
                                    tir.bind(v1_1, tir.floormod(tir.floordiv(ax0_ax1_fused_ax2_fused_ax3_fused_outer_1, 9), 512))
                                    tir.bind(v2_1, tir.floormod(tir.floordiv(ax0_ax1_fused_ax2_fused_ax3_fused_outer_1, 3), 3))
                                    tir.bind(v3_1, tir.floormod(ax0_ax1_fused_ax2_fused_ax3_fused_outer_1, 3))
                                    tir.reads([W[v0_1 : (v0_1 + 1), v1_1 : (v1_1 + 1), v2_1 : (v2_1 + 1), v3_1 : (v3_1 + 1)]])
                                    tir.writes([W_shared[v0_1 : (v0_1 + 1), v1_1 : (v1_1 + 1), v2_1 : (v2_1 + 1), v3_1 : (v3_1 + 1)]])
                                    W_shared[v0_1, v1_1, v2_1, v3_1] = W[v0_1, v1_1, v2_1, v3_1]
                        for i4_outer_inner, i5_outer_inner, i6_outer_inner, i0_outer_inner, i1_outer_inner, i2_outer_inner, i3_outer_inner, i4_inner, i5_inner, i6_inner, i0_inner, i1_inner, i2_inner, i3_inner in tir.grid(64, 1, 1, 1, 2, 2, 1, 8, 3, 3, 1, 1, 14, 2):
                            with tir.block([1, 512, 56, 56, tir.reduce_axis(0, 512), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "compute") as [nn, ff, yy, xx, rc, ry, rx]:
                                tir.bind(nn, 0)
                                tir.bind(ff, ((((i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused_i2_outer_outer_outer_outer_fused_i3_outer_outer_outer_outer_fused * 128) + (tir.floordiv(i0_outer_outer_outer_inner_i1_outer_outer_outer_inner_fused_i2_outer_outer_outer_inner_fused_i3_outer_outer_outer_inner_fused, 8) * 64)) + (tir.floordiv(i0_outer_outer_inner_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused_i3_outer_outer_inner_fused, 7) * 2)) + i1_outer_inner))
                                tir.bind(yy, (((tir.floormod(tir.floordiv(i0_outer_outer_outer_inner_i1_outer_outer_outer_inner_fused_i2_outer_outer_outer_inner_fused_i3_outer_outer_outer_inner_fused, 4), 2) * 28) + (i2_outer_inner * 14)) + i2_inner))
                                tir.bind(xx, (((tir.floormod(i0_outer_outer_outer_inner_i1_outer_outer_outer_inner_fused_i2_outer_outer_outer_inner_fused_i3_outer_outer_outer_inner_fused, 4) * 14) + (tir.floormod(i0_outer_outer_inner_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused_i3_outer_outer_inner_fused, 7) * 2)) + i3_inner))
                                tir.bind(rc, ((i4_outer_inner * 8) + i4_inner))
                                tir.bind(ry, i5_inner)
                                tir.bind(rx, i6_inner)
                                tir.reads([compute_local[nn : (nn + 1), ff : (ff + 1), yy : (yy + 1), xx : (xx + 1)], pad_temp_shared[nn : (nn + 1), rc : (rc + 1), (yy + ry) : ((yy + ry) + 1), (xx + rx) : ((xx + rx) + 1)], W_shared[ff : (ff + 1), rc : (rc + 1), ry : (ry + 1), rx : (rx + 1)]])
                                tir.writes([compute_local[nn : (nn + 1), ff : (ff + 1), yy : (yy + 1), xx : (xx + 1)]])
                                with tir.init():
                                    compute_local[nn, ff, yy, xx] = tir.float32(0)
                                compute_local[nn, ff, yy, xx] = compute_local[nn, ff, yy, xx] + (
                                    pad_temp_shared[nn, rc, (yy + ry), (xx + rx)]
                                    * W_shared[ff, rc, ry, rx]
                                )
                    for ax0, ax1, ax2, ax3 in tir.grid(1, 2, 28, 2):
                        with tir.block([1, 512, 56, 56], "compute_1") as [i0, i1, i2, i3]:
                            tir.bind(i0, 0)
                            tir.bind(i1, ((((i0_outer_outer_outer_outer_i1_outer_outer_outer_outer_fused_i2_outer_outer_outer_outer_fused_i3_outer_outer_outer_outer_fused * 128) + (tir.floordiv(i0_outer_outer_outer_inner_i1_outer_outer_outer_inner_fused_i2_outer_outer_outer_inner_fused_i3_outer_outer_outer_inner_fused, 8) * 64)) + (tir.floordiv(i0_outer_outer_inner_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused_i3_outer_outer_inner_fused, 7) * 2)) + ax1))
                            tir.bind(i2, ((tir.floordiv(tir.floormod(i0_outer_outer_outer_inner_i1_outer_outer_outer_inner_fused_i2_outer_outer_outer_inner_fused_i3_outer_outer_outer_inner_fused, 8), 4) * 28) + ax2))
                            tir.bind(i3, (((tir.floormod(i0_outer_outer_outer_inner_i1_outer_outer_outer_inner_fused_i2_outer_outer_outer_inner_fused_i3_outer_outer_outer_inner_fused, 4) * 14) + (tir.floormod(i0_outer_outer_inner_i1_outer_outer_inner_fused_i2_outer_outer_inner_fused_i3_outer_outer_inner_fused, 7) * 2)) + ax3))
                            tir.reads([compute_local[i0 : (i0 + 1), i1 : (i1 + 1), i2 : (i2 + 1), i3 : (i3 + 1)], B[i1 : (i1 + 1), 0:1, 0:1], bn_scale[i1 : (i1 + 1), 0:1, 0:1], bn_offset[i1 : (i1 + 1), 0:1, 0:1]])
                            tir.writes([compute[i0 : (i0 + 1), i1 : (i1 + 1), i2 : (i2 + 1), i3 : (i3 + 1)]])
                            compute[i0, i1, i2, i3] = tir.max((((compute_local[i0, i1, i2, i3] + B[i1, 0, 0]) * bn_scale[i1, 0, 0]) + bn_offset[i1, 0, 0]), tir.float32(0))


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unused-variable,unexpected-keyword-arg,misplaced-comparison-constant


def test_meta_schedule_sketch_cuda_conv2d_nchw_bias_bn_relu():  # pylint: disable=invalid-name
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
    expected = [_conv2d_nchw_bias_bn_relu_sketch_0]
    assert len(support) == 1
    possible_decisions = [
        [
            [1, 1, 1, 1, 1],
            [4, 2, 32, 2, 1],
            [1, 2, 1, 2, 14],
            [1, 4, 7, 1, 2],
            [1, 64, 8],
            [1, 1, 3],
            [1, 1, 3],
            [589824, 1],
            [430592, 4],
        ],
    ]
    assert len(support) == 1
    _fix_sampling_tile_size(
        sch=support[0],
        func=func,
        possible_decisions=possible_decisions,
        expected=expected,
    )


if __name__ == "__main__":
    test_meta_schedule_sketch_cuda_matmul()
    test_meta_schedule_sketch_cuda_conv2d_nchw_bias_bn_relu()
