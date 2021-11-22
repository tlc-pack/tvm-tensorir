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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import tvm
from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing.schedule_rule import (
    auto_inline,
    auto_inline_after_tiling,
)
from tvm.meta_schedule.tune_context import TuneContext
from tvm.script import tir as T
from tvm.target import Target

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm.script.ir_module
class Conv2DBiasBnReLU:
    @T.prim_func
    def main(var_X: T.handle, var_W: T.handle, var_B: T.handle, var_bn_scale: T.handle, var_bn_offset: T.handle, var_compute: T.handle) -> None:
        X = T.match_buffer(var_X, [1, 512, 56, 56], dtype="float32")
        W = T.match_buffer(var_W, [512, 512, 3, 3], dtype="float32")
        B = T.match_buffer(var_B, [512, 1, 1], dtype="float32")
        bn_scale = T.match_buffer(var_bn_scale, [512, 1, 1], dtype="float32")
        bn_offset = T.match_buffer(var_bn_offset, [512, 1, 1], dtype="float32")
        compute = T.match_buffer(var_compute, [1, 512, 56, 56], dtype="float32")
        pad_temp = T.alloc_buffer([1, 512, 58, 58], dtype="float32")
        compute_1 = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        bias_add = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        bn_mul = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        bn_add = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 512, 58, 58):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads([X[i0_1, i1_1, i2_1 - 1, i3_1 - 1]])
                T.writes([pad_temp[i0_1, i1_1, i2_1, i3_1]])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(i2_1 >= 1 and i2_1 < 57 and i3_1 >= 1 and i3_1 < 57, X[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 512, 56, 56, 512, 3, 3):
            with T.block("compute"):
                nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads([compute_1[nn, ff, yy, xx], pad_temp[nn, rc, yy + ry, xx + rx], W[ff, rc, ry, rx]])
                T.writes([compute_1[nn, ff, yy, xx]])
                with T.init():
                    compute_1[nn, ff, yy, xx] = T.float32(0)
                compute_1[nn, ff, yy, xx] = compute_1[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * W[ff, rc, ry, rx]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("bias_add"):
                i, j, k, l = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads([compute_1[i, j, k, l], B[j, 0, 0]])
                T.writes([bias_add[i, j, k, l]])
                bias_add[i, j, k, l] = compute_1[i, j, k, l] + B[j, 0, 0]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("bn_mul"):
                i, j, k, l = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads([bias_add[i, j, k, l], bn_scale[j, 0, 0]])
                T.writes([bn_mul[i, j, k, l]])
                bn_mul[i, j, k, l] = bias_add[i, j, k, l] * bn_scale[j, 0, 0]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("bn_add"):
                i, j, k, l = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads([bn_mul[i, j, k, l], bn_offset[j, 0, 0]])
                T.writes([bn_add[i, j, k, l]])
                bn_add[i, j, k, l] = bn_mul[i, j, k, l] + bn_offset[j, 0, 0]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("compute_1"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads([bn_add[i0_2, i1_2, i2_2, i3_2]])
                T.writes([compute[i0_2, i1_2, i2_2, i3_2]])
                compute[i0_2, i1_2, i2_2, i3_2] = T.max(bn_add[i0_2, i1_2, i2_2, i3_2], T.float32(0))


@tvm.script.ir_module
class Conv2DBiasBnReLUInlined:
    @T.prim_func
    def main(var_X: T.handle, var_W: T.handle, var_B: T.handle, var_bn_scale: T.handle, var_bn_offset: T.handle, var_compute: T.handle) -> None:
        X = T.match_buffer(var_X, [1, 512, 56, 56], dtype="float32")
        W = T.match_buffer(var_W, [512, 512, 3, 3], dtype="float32")
        B = T.match_buffer(var_B, [512, 1, 1], dtype="float32")
        bn_scale = T.match_buffer(var_bn_scale, [512, 1, 1], dtype="float32")
        bn_offset = T.match_buffer(var_bn_offset, [512, 1, 1], dtype="float32")
        compute = T.match_buffer(var_compute, [1, 512, 56, 56], dtype="float32")
        pad_temp = T.alloc_buffer([1, 512, 58, 58], dtype="float32")
        compute_1 = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        for i0, i1, i2, i3 in T.grid(1, 512, 58, 58):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads([X[i0_1, i1_1, i2_1 - 1, i3_1 - 1]])
                T.writes([pad_temp[i0_1, i1_1, i2_1, i3_1]])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(i2_1 >= 1 and i2_1 < 57 and i3_1 >= 1 and i3_1 < 57, X[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(1, 512, 56, 56, 512, 3, 3):
            with T.block("compute"):
                nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads([compute_1[nn, ff, yy, xx], pad_temp[nn, rc, yy + ry, xx + rx], W[ff, rc, ry, rx]])
                T.writes([compute_1[nn, ff, yy, xx]])
                with T.init():
                    compute_1[nn, ff, yy, xx] = T.float32(0)
                compute_1[nn, ff, yy, xx] = compute_1[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * W[ff, rc, ry, rx]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("compute_1"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads([compute_1[i0_2, i1_2, i2_2, i3_2], B[i1_2, 0, 0], bn_scale[i1_2, 0, 0], bn_offset[i1_2, 0, 0]])
                T.writes([compute[i0_2, i1_2, i2_2, i3_2]])
                compute[i0_2, i1_2, i2_2, i3_2] = T.max((compute_1[i0_2, i1_2, i2_2, i3_2] + B[i1_2, 0, 0]) * bn_scale[i1_2, 0, 0] + bn_offset[i1_2, 0, 0], T.float32(0))


@tvm.script.ir_module
class NeedsInlinePaddingAndEpilogue:
    @T.prim_func
    def main(var_X: T.handle, var_W: T.handle, var_B: T.handle, var_bn_scale: T.handle, var_bn_offset: T.handle, var_compute: T.handle) -> None:
        X = T.match_buffer(var_X, [1, 512, 56, 56], dtype="float32")
        W = T.match_buffer(var_W, [512, 512, 3, 3], dtype="float32")
        B = T.match_buffer(var_B, [512, 1, 1], dtype="float32")
        bn_scale = T.match_buffer(var_bn_scale, [512, 1, 1], dtype="float32")
        bn_offset = T.match_buffer(var_bn_offset, [512, 1, 1], dtype="float32")
        compute = T.match_buffer(var_compute, [1, 512, 56, 56], dtype="float32")
        pad_temp = T.alloc_buffer([1, 512, 58, 58], dtype="float32")
        compute_1 = T.alloc_buffer([1, 512, 56, 56], dtype="float32")
        compute_local = T.alloc_buffer([1, 512, 56, 56], dtype="float32", scope="local")
        pad_temp_shared = T.alloc_buffer([1, 512, 58, 58], dtype="float32", scope="shared")
        W_shared = T.alloc_buffer([512, 512, 3, 3], dtype="float32", scope="shared")
        for i0, i1, i2, i3 in T.grid(1, 512, 58, 58):
            with T.block("pad_temp"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads([X[i0_1, i1_1, i2_1 - 1, i3_1 - 1]])
                T.writes([pad_temp[i0_1, i1_1, i2_1, i3_1]])
                pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(i2_1 >= 1 and i2_1 < 57 and i3_1 >= 1 and i3_1 < 57, X[i0_1, i1_1, i2_1 - 1, i3_1 - 1], T.float32(0), dtype="float32")
        for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(0, 224, thread="blockIdx.x"):
            for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(0, 2, thread="vthread.x"):
                for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(0, 8, thread="threadIdx.x"):
                    for i4_0, i5_0, i6_0 in T.grid(1, 3, 1):
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(0, 40960, annotations={"meta_schedule.cooperative_fetch":1}):
                            for ax0_ax1_ax2_ax3_fused_1 in T.vectorized(0, 3):
                                with T.block("pad_temp_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(512, (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) // 30 // 8 % 512)
                                    v2 = T.axis.spatial(58, i0_0_i1_0_i2_0_i3_0_fused % 14 // 2 * 8 + i5_0 + (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) // 30 % 8)
                                    v3 = T.axis.spatial(58, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) % 30)
                                    T.reads([pad_temp[v0, v1, v2, v3]])
                                    T.writes([pad_temp_shared[v0, v1, v2, v3]])
                                    T.block_attr({"meta_schedule.cache_type":0})
                                    pad_temp_shared[v0, v1, v2, v3] = pad_temp[v0, v1, v2, v3]
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(0, 12288, annotations={"meta_schedule.cooperative_fetch":1}):
                            for ax0_ax1_ax2_ax3_fused_1 in T.vectorized(0, 4):
                                with T.block("W_shared"):
                                    v0 = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) // 1536)
                                    v1 = T.axis.spatial(512, (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) // 3 % 512)
                                    v2 = T.axis.spatial(3, i5_0)
                                    v3 = T.axis.spatial(3, (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) % 3)
                                    T.reads([W[v0, v1, v2, v3]])
                                    T.writes([W_shared[v0, v1, v2, v3]])
                                    T.block_attr({"meta_schedule.cache_type":0})
                                    W_shared[v0, v1, v2, v3] = W[v0, v1, v2, v3]
                        for i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3, i4_2, i5_2, i6_2, i0_4, i1_4, i2_4, i3_4 in T.grid(32, 1, 1, 1, 1, 1, 1, 16, 1, 3, 1, 8, 2, 28):
                            with T.block("compute"):
                                nn = T.axis.spatial(1, 0)
                                ff = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + i1_4)
                                yy = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused // 2 % 7 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + i2_4)
                                xx = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + i3_4)
                                rc = T.axis.reduce(512, i4_1 * 16 + i4_2)
                                ry, rx = T.axis.remap("RR", [i5_0, i6_2])
                                T.reads([compute_local[nn, ff, yy, xx], pad_temp_shared[nn, rc, yy + ry, xx + rx], W_shared[ff, rc, ry, rx]])
                                T.writes([compute_local[nn, ff, yy, xx]])
                                with T.init():
                                    compute_local[nn, ff, yy, xx] = T.float32(0)
                                compute_local[nn, ff, yy, xx] = compute_local[nn, ff, yy, xx] + pad_temp_shared[nn, rc, yy + ry, xx + rx] * W_shared[ff, rc, ry, rx]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 8, 2, 28):
                        with T.block("compute_local"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + ax1)
                            v2 = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 14 // 2 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + ax2)
                            v3 = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + ax3)
                            T.block_attr({"meta_schedule.cache_type":1})
                            T.reads([compute_local[v0, v1, v2, v3]])
                            T.writes([compute_1[v0, v1, v2, v3]])
                            compute_1[v0, v1, v2, v3] = compute_local[v0, v1, v2, v3]
        for i0, i1, i2, i3 in T.grid(1, 512, 56, 56):
            with T.block("compute_1"):
                i0_2, i1_2, i2_2, i3_2 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads([compute_1[i0_2, i1_2, i2_2, i3_2], B[i1_2, 0, 0], bn_scale[i1_2, 0, 0], bn_offset[i1_2, 0, 0]])
                T.writes([compute[i0_2, i1_2, i2_2, i3_2]])
                compute[i0_2, i1_2, i2_2, i3_2] = T.max((compute_1[i0_2, i1_2, i2_2, i3_2] + B[i1_2, 0, 0]) * bn_scale[i1_2, 0, 0] + bn_offset[i1_2, 0, 0], T.float32(0))


@tvm.script.ir_module
class PaddingAndEpilogueInlined:
    @T.prim_func
    def main(var_X: T.handle, var_W: T.handle, var_B: T.handle, var_bn_scale: T.handle, var_bn_offset: T.handle, var_compute: T.handle) -> None:
        X = T.match_buffer(var_X, [1, 512, 56, 56], dtype="float32")
        W = T.match_buffer(var_W, [512, 512, 3, 3], dtype="float32")
        B = T.match_buffer(var_B, [512, 1, 1], dtype="float32")
        bn_scale = T.match_buffer(var_bn_scale, [512, 1, 1], dtype="float32")
        bn_offset = T.match_buffer(var_bn_offset, [512, 1, 1], dtype="float32")
        compute = T.match_buffer(var_compute, [1, 512, 56, 56], dtype="float32")
        compute_local = T.alloc_buffer([1, 512, 56, 56], dtype="float32", scope="local")
        pad_temp_shared = T.alloc_buffer([1, 512, 58, 58], dtype="float32", scope="shared")
        W_shared = T.alloc_buffer([512, 512, 3, 3], dtype="float32", scope="shared")
        for i0_0_i1_0_i2_0_i3_0_fused in T.thread_binding(0, 224, thread="blockIdx.x"):
            for i0_1_i1_1_i2_1_i3_1_fused in T.thread_binding(0, 2, thread="vthread.x"):
                for i0_2_i1_2_i2_2_i3_2_fused in T.thread_binding(0, 8, thread="threadIdx.x"):
                    for i4_0, i5_0, i6_0 in T.grid(1, 3, 1):
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(0, 40960, annotations={"meta_schedule.cooperative_fetch":1}):
                            for ax0_ax1_ax2_ax3_fused_1 in T.vectorized(0, 3):
                                with T.block("pad_temp_shared"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(512, (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) // 30 // 8 % 512)
                                    v2 = T.axis.spatial(58, i0_0_i1_0_i2_0_i3_0_fused % 14 // 2 * 8 + i5_0 + (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) // 30 % 8)
                                    v3 = T.axis.spatial(58, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + (ax0_ax1_ax2_ax3_fused_0 * 3 + ax0_ax1_ax2_ax3_fused_1) % 30)
                                    T.reads([X[v0, v1, v2 - 1, v3 - 1]])
                                    T.writes([pad_temp_shared[v0, v1, v2, v3]])
                                    T.block_attr({"meta_schedule.cache_type":0})
                                    pad_temp_shared[v0, v1, v2, v3] = T.if_then_else(v2 >= 1 and v2 < 57 and v3 >= 1 and v3 < 57, X[v0, v1, v2 - 1, v3 - 1], T.float32(0), dtype="float32")
                        for ax0_ax1_ax2_ax3_fused_0 in T.serial(0, 12288, annotations={"meta_schedule.cooperative_fetch":1}):
                            for ax0_ax1_ax2_ax3_fused_1 in T.vectorized(0, 4):
                                with T.block("W_shared"):
                                    v0 = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) // 1536)
                                    v1 = T.axis.spatial(512, (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) // 3 % 512)
                                    v2 = T.axis.spatial(3, i5_0)
                                    v3 = T.axis.spatial(3, (ax0_ax1_ax2_ax3_fused_0 * 4 + ax0_ax1_ax2_ax3_fused_1) % 3)
                                    T.reads([W[v0, v1, v2, v3]])
                                    T.writes([W_shared[v0, v1, v2, v3]])
                                    T.block_attr({"meta_schedule.cache_type":0})
                                    W_shared[v0, v1, v2, v3] = W[v0, v1, v2, v3]
                        for i4_1, i5_1, i6_1, i0_3, i1_3, i2_3, i3_3, i4_2, i5_2, i6_2, i0_4, i1_4, i2_4, i3_4 in T.grid(32, 1, 1, 1, 1, 1, 1, 16, 1, 3, 1, 8, 2, 28):
                            with T.block("compute"):
                                nn = T.axis.spatial(1, 0)
                                ff = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + i1_4)
                                yy = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused // 2 % 7 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + i2_4)
                                xx = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + i3_4)
                                rc = T.axis.reduce(512, i4_1 * 16 + i4_2)
                                ry, rx = T.axis.remap("RR", [i5_0, i6_2])
                                T.reads([compute_local[nn, ff, yy, xx], pad_temp_shared[nn, rc, yy + ry, xx + rx], W_shared[ff, rc, ry, rx]])
                                T.writes([compute_local[nn, ff, yy, xx]])
                                with T.init():
                                    compute_local[nn, ff, yy, xx] = T.float32(0)
                                compute_local[nn, ff, yy, xx] = compute_local[nn, ff, yy, xx] + pad_temp_shared[nn, rc, yy + ry, xx + rx] * W_shared[ff, rc, ry, rx]
                    for ax0, ax1, ax2, ax3 in T.grid(1, 8, 2, 28):
                        with T.block("compute_local"):
                            v0 = T.axis.spatial(1, ax0)
                            v1 = T.axis.spatial(512, i0_0_i1_0_i2_0_i3_0_fused // 14 * 32 + i0_2_i1_2_i2_2_i3_2_fused // 2 * 8 + ax1)
                            v2 = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 14 // 2 * 8 + i0_1_i1_1_i2_1_i3_1_fused * 4 + i0_2_i1_2_i2_2_i3_2_fused % 2 * 2 + ax2)
                            v3 = T.axis.spatial(56, i0_0_i1_0_i2_0_i3_0_fused % 2 * 28 + ax3)
                            T.reads([compute_local[v0, v1, v2, v3], B[v1, 0, 0], bn_scale[v1, 0, 0], bn_offset[v1, 0, 0]])
                            T.writes([compute[v0, v1, v2, v3]])
                            T.block_attr({"meta_schedule.cache_type":1})
                            compute[v0, v1, v2, v3] = T.max((compute_local[v0, v1, v2, v3] + B[v1, 0, 0]) * bn_scale[v1, 0, 0] + bn_offset[v1, 0, 0], T.float32(0))

# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def _create_context(mod, target, rule):
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    ctx.space_generator.initialize_with_tune_context(ctx)
    for sch_rule in ctx.sch_rules:
        sch_rule.initialize_with_tune_context(ctx)
    return ctx


def test_inline_consumer_chain():
    mod = Conv2DBiasBnReLU
    target = Target("llvm")
    ctx = _create_context(
        mod=mod,
        target=target,
        rule=auto_inline(target=target),
    )
    (space,) = ctx.space_generator.generate_design_space(mod=mod)
    tvm.ir.assert_structural_equal(lhs=space.mod, rhs=Conv2DBiasBnReLUInlined)


def test_inline_into_cache():
    mod = NeedsInlinePaddingAndEpilogue
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        mod=NeedsInlinePaddingAndEpilogue,
        target=target,
        rule=auto_inline_after_tiling(target=target),
    )
    (space,) = ctx.space_generator.generate_design_space(mod=mod)
    tvm.ir.assert_structural_equal(lhs=space.mod, rhs=PaddingAndEpilogueInlined)


if __name__ == "__main__":
    test_inline_consumer_chain()
    test_inline_into_cache()
