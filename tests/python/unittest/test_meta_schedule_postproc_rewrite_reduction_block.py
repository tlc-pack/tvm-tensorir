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
from tvm import tir
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.postproc import RewriteReductionBlock
from tvm.script import tir as T
from tvm.target import Target


def _target() -> Target:
    return Target("cuda", host="llvm")


def _create_context(mod, target) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        postprocs=[
            RewriteReductionBlock(),
        ],
        task_name="test",
    )
    for rule in ctx.postprocs:
        rule.initialize_with_tune_context(ctx)
    return ctx


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm.script.ir_module
class Before0:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle) -> None:
        A = T.match_buffer(var_A, [512, 512], dtype="float32")
        B = T.match_buffer(var_B, [512, 512], dtype="float32")
        C = T.match_buffer(var_C, [512, 512], dtype="float32")
        C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(0, 16, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(0, 16, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(0, 8, thread="threadIdx.x"):
                    for i2_0 in T.serial(0, 1):
                        for ax0_ax1_fused_0 in T.serial(0, 32768):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                with T.block("A_shared"):
                                    v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) // 512)
                                    v1 = T.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) % 512)
                                    T.reads([A[v0, v1]])
                                    T.writes([A_shared[v0, v1]])
                                    T.block_attr({"meta_schedule.cooperative_fetch":1})
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(0, 1024):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(0, 2):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) // 32)
                                        v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) % 32)
                                        T.reads([B[v0, v1]])
                                        T.writes([B_shared[v0, v1]])
                                        T.block_attr({"meta_schedule.cooperative_fetch":2})
                                        B_shared[v0, v1] = B[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(16, 2, 2, 32, 16, 2):
                            with T.block("C"):
                                i = T.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3 * 16 + i0_4)
                                j = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3 * 2 + i1_4)
                                k = T.axis.reduce(512, i2_1 * 32 + i2_2)
                                T.reads([C_local[i, j], A_shared[i, k], B_shared[k, j]])
                                T.writes([C_local[i, j]])
                                with T.init():
                                    C_local[i, j] = T.float32(0)
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(32, 4):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(512, i0_1_i1_1_fused * 32 + ax0)
                            v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + ax1)
                            T.reads([C_local[v0, v1]])
                            T.writes([C[v0, v1]])
                            C[v0, v1] = C_local[v0, v1]


@tvm.script.ir_module
class After0:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle) -> None:
        A = T.match_buffer(var_A, [512, 512], dtype="float32")
        B = T.match_buffer(var_B, [512, 512], dtype="float32")
        C = T.match_buffer(var_C, [512, 512], dtype="float32")
        C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(0, 16, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(0, 16, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(0, 8, thread="threadIdx.x"):
                    for i2_0 in T.serial(0, 1):
                        for ax0_ax1_fused_0 in T.serial(0, 32768):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                with T.block("A_shared"):
                                    v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) // 512)
                                    v1 = T.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) % 512)
                                    T.reads([A[v0, v1]])
                                    T.writes([A_shared[v0, v1]])
                                    T.block_attr({"meta_schedule.cooperative_fetch":1})
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(0, 1024):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(0, 2):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) // 32)
                                        v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) % 32)
                                        T.reads([B[v0, v1]])
                                        T.writes([B_shared[v0, v1]])
                                        T.block_attr({"meta_schedule.cooperative_fetch":2})
                                        B_shared[v0, v1] = B[v0, v1]
                        for i0_3_init, i1_3_init, i0_4_init, i1_4_init in T.grid(2, 2, 16, 2):
                            with T.block("C_init"):
                                i = T.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3_init * 16 + i0_4_init)
                                j = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3_init * 2 + i1_4_init)
                                T.reads([])
                                T.writes([C_local[i, j]])
                                C_local[i, j] = T.float32(0)
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(16, 2, 2, 32, 16, 2):
                            with T.block("C_update"):
                                i = T.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3 * 16 + i0_4)
                                j = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3 * 2 + i1_4)
                                k = T.axis.reduce(512, i2_1 * 32 + i2_2)
                                T.reads([C_local[i, j], A_shared[i, k], B_shared[k, j]])
                                T.writes([C_local[i, j]])
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(32, 4):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(512, i0_1_i1_1_fused * 32 + ax0)
                            v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + ax1)
                            T.reads([C_local[v0, v1]])
                            T.writes([C[v0, v1]])
                            C[v0, v1] = C_local[v0, v1]


@tvm.script.ir_module
class Before1:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle) -> None:
        A = T.match_buffer(var_A, [512, 512], dtype="float16")
        B = T.match_buffer(var_B, [512, 512], dtype="float16")
        C = T.match_buffer(var_C, [512, 512], dtype="float32")
        # body
        with T.block("root"):
            T.reads([])
            T.writes([])
            T.block_attr({"meta_schedule.tensor_core_enabled":"1"})
            C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
            C_local_wmma_accumulator = T.alloc_buffer([512, 512], dtype="float32", scope="wmma.accumulator")
            A_shared = T.alloc_buffer([512, 512], dtype="float16", scope="shared")
            B_shared = T.alloc_buffer([512, 512], dtype="float16", scope="shared")
            A_shared_wmma_matrix_a = T.alloc_buffer([512, 512], dtype="float16", scope="wmma.matrix_a")
            B_shared_wmma_matrix_b = T.alloc_buffer([512, 512], dtype="float16", scope="wmma.matrix_b")
            for i0_0_0_i1_0_0_fused in T.thread_binding(0, 1, thread="blockIdx.x"):
                for i0_0_1_i1_0_1_fused in T.thread_binding(0, 4, thread="blockIdx.y"):
                    for i0_0_2_i1_0_2_fused in T.thread_binding(0, 8, thread="threadIdx.y"):
                        for i2_0_0 in T.serial(0, 4):
                            for ax0_ax1_fused_0 in T.serial(0, 128):
                                for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(0, 32, thread="threadIdx.x"):
                                        with T.block("A_shared"):
                                            v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2) // 128)
                                            v1 = T.axis.spatial(512, i2_0_0 * 128 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2) % 128)
                                            T.reads([A[v0, v1]])
                                            T.writes([A_shared[v0, v1]])
                                            T.block_attr({"meta_schedule.cooperative_fetch":1})
                                            A_shared[v0, v1] = A[v0, v1]
                            for ax0_ax1_fused_0 in T.serial(0, 32):
                                for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(0, 32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(0, 4):
                                            with T.block("B_shared"):
                                                v0 = T.axis.spatial(512, i2_0_0 * 128 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 256)
                                                v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 256)
                                                T.reads([B[v0, v1]])
                                                T.writes([B_shared[v0, v1]])
                                                T.block_attr({"meta_schedule.cooperative_fetch":4})
                                                B_shared[v0, v1] = B[v0, v1]
                            for i2_0_1, i0_0_3, i1_0_3, i2_0_2 in T.grid(8, 1, 1, 1):
                                for ax0, ax1 in T.grid(256, 16):
                                    with T.block("A_shared_wmma.matrix_a"):
                                        v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + ax0)
                                        v1 = T.axis.spatial(512, i2_0_0 * 128 + i2_0_1 * 16 + ax1)
                                        T.reads([A_shared[v0, v1]])
                                        T.writes([A_shared_wmma_matrix_a[v0, v1]])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_a"})
                                        A_shared_wmma_matrix_a[v0, v1] = A_shared[v0, v1]
                                for ax0, ax1 in T.grid(16, 32):
                                    with T.block("B_shared_wmma.matrix_b"):
                                        v0 = T.axis.spatial(512, i2_0_0 * 128 + i2_0_1 * 16 + ax0)
                                        v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + i0_0_2_i1_0_2_fused * 32 + ax1)
                                        T.reads([B_shared[v0, v1]])
                                        T.writes([B_shared_wmma_matrix_b[v0, v1]])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_b"})
                                        B_shared_wmma_matrix_b[v0, v1] = B_shared[v0, v1]
                                for i0_0_4, i1_0_4 in T.grid(16, 2):
                                    with T.block("blockized_C"):
                                        io = T.axis.spatial(32, i0_0_1_i1_0_1_fused // 2 * 16 + i0_0_4)
                                        jo = T.axis.spatial(32, i0_0_1_i1_0_1_fused % 2 * 16 + i0_0_2_i1_0_2_fused * 2 + i1_0_4)
                                        ko = T.axis.reduce(32, i2_0_0 * 8 + i2_0_1)
                                        T.reads([C_local_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], A_shared_wmma_matrix_a[io * 16 : io * 16 + 16, ko * 16 : ko * 16 + 16], B_shared_wmma_matrix_b[ko * 16 : ko * 16 + 16, jo * 16 : jo * 16 + 16]])
                                        T.writes([C_local_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_fill"})
                                        with T.init():
                                            for i0_1, i1_1 in T.grid(16, 16):
                                                with T.block("C_init"):
                                                    i_init = T.axis.spatial(512, io * 16 + i0_1)
                                                    j_init = T.axis.spatial(512, jo * 16 + i1_1)
                                                    T.reads([])
                                                    T.writes([C_local_wmma_accumulator[i_init, j_init]])
                                                    C_local_wmma_accumulator[i_init, j_init] = T.float32(0)
                                        for i0_1, i1_1, i2_1 in T.grid(16, 16, 16):
                                            with T.block("C"):
                                                i = T.axis.spatial(512, io * 16 + i0_1)
                                                j = T.axis.spatial(512, jo * 16 + i1_1)
                                                k = T.axis.reduce(512, ko * 16 + i2_1)
                                                T.reads([C_local_wmma_accumulator[i, j], A_shared_wmma_matrix_a[i, k], B_shared_wmma_matrix_b[k, j]])
                                                T.writes([C_local_wmma_accumulator[i, j]])
                                                T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync"})
                                                C_local_wmma_accumulator[i, j] = C_local_wmma_accumulator[i, j] + T.cast(A_shared_wmma_matrix_a[i, k], "float32") * T.cast(B_shared_wmma_matrix_b[k, j], "float32")
                        for ax0, ax1 in T.grid(256, 32):
                            with T.block("C_local_wmma.accumulator"):
                                v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + ax0)
                                v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + i0_0_2_i1_0_2_fused * 32 + ax1)
                                T.reads([C_local_wmma_accumulator[v0, v1]])
                                T.writes([C_local[v0, v1]])
                                T.block_attr({"meta_schedule.auto_tensorize":"wmma_store"})
                                C_local[v0, v1] = C_local_wmma_accumulator[v0, v1]
                        for ax0, ax1 in T.grid(256, 32):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + ax0)
                                v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + i0_0_2_i1_0_2_fused * 32 + ax1)
                                T.reads([C_local[v0, v1]])
                                T.writes([C[v0, v1]])
                                C[v0, v1] = C_local[v0, v1]


@tvm.script.ir_module
class After1:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle) -> None:
        A = T.match_buffer(var_A, [512, 512], dtype="float16")
        B = T.match_buffer(var_B, [512, 512], dtype="float16")
        C = T.match_buffer(var_C, [512, 512], dtype="float32")
        # body
        with T.block("root"):
            T.reads([])
            T.writes([])
            T.block_attr({"meta_schedule.tensor_core_enabled":"1"})
            C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
            C_local_wmma_accumulator = T.alloc_buffer([512, 512], dtype="float32", scope="wmma.accumulator")
            A_shared = T.alloc_buffer([512, 512], dtype="float16", scope="shared")
            B_shared = T.alloc_buffer([512, 512], dtype="float16", scope="shared")
            A_shared_wmma_matrix_a = T.alloc_buffer([512, 512], dtype="float16", scope="wmma.matrix_a")
            B_shared_wmma_matrix_b = T.alloc_buffer([512, 512], dtype="float16", scope="wmma.matrix_b")
            for i0_0_0_i1_0_0_fused in T.thread_binding(0, 1, thread="blockIdx.x"):
                for i0_0_1_i1_0_1_fused in T.thread_binding(0, 4, thread="blockIdx.y"):
                    for i0_0_2_i1_0_2_fused in T.thread_binding(0, 8, thread="threadIdx.y"):
                        for i0_0_4_init, i1_0_4_init in T.grid(16, 2):
                            with T.block("blockized_C_init"):
                                io = T.axis.spatial(32, i0_0_1_i1_0_1_fused // 2 * 16 + i0_0_4_init)
                                jo = T.axis.spatial(32, i0_0_1_i1_0_1_fused % 2 * 16 + i0_0_2_i1_0_2_fused * 2 + i1_0_4_init)
                                T.reads([])
                                T.writes([C_local_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                                for i0_1, i1_1 in T.grid(16, 16):
                                    with T.block("C_init"):
                                        i_init = T.axis.spatial(512, io * 16 + i0_1)
                                        j_init = T.axis.spatial(512, jo * 16 + i1_1)
                                        T.reads([])
                                        T.writes([C_local_wmma_accumulator[i_init, j_init]])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_fill"})
                                        C_local_wmma_accumulator[i_init, j_init] = T.float32(0)
                        for i2_0_0 in T.serial(0, 4):
                            for ax0_ax1_fused_0 in T.serial(0, 128):
                                for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(0, 32, thread="threadIdx.x"):
                                        with T.block("A_shared"):
                                            v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2) // 128)
                                            v1 = T.axis.spatial(512, i2_0_0 * 128 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2) % 128)
                                            T.reads([A[v0, v1]])
                                            T.writes([A_shared[v0, v1]])
                                            T.block_attr({"meta_schedule.cooperative_fetch":1})
                                            A_shared[v0, v1] = A[v0, v1]
                            for ax0_ax1_fused_0 in T.serial(0, 32):
                                for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(0, 32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(0, 4):
                                            with T.block("B_shared"):
                                                v0 = T.axis.spatial(512, i2_0_0 * 128 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 256)
                                                v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 256)
                                                T.reads([B[v0, v1]])
                                                T.writes([B_shared[v0, v1]])
                                                T.block_attr({"meta_schedule.cooperative_fetch":4})
                                                B_shared[v0, v1] = B[v0, v1]
                            for i2_0_1, i0_0_3, i1_0_3, i2_0_2 in T.grid(8, 1, 1, 1):
                                for ax0, ax1 in T.grid(256, 16):
                                    with T.block("A_shared_wmma.matrix_a"):
                                        v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + ax0)
                                        v1 = T.axis.spatial(512, i2_0_0 * 128 + i2_0_1 * 16 + ax1)
                                        T.reads([A_shared[v0, v1]])
                                        T.writes([A_shared_wmma_matrix_a[v0, v1]])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_a"})
                                        A_shared_wmma_matrix_a[v0, v1] = A_shared[v0, v1]
                                for ax0, ax1 in T.grid(16, 32):
                                    with T.block("B_shared_wmma.matrix_b"):
                                        v0 = T.axis.spatial(512, i2_0_0 * 128 + i2_0_1 * 16 + ax0)
                                        v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + i0_0_2_i1_0_2_fused * 32 + ax1)
                                        T.reads([B_shared[v0, v1]])
                                        T.writes([B_shared_wmma_matrix_b[v0, v1]])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_b"})
                                        B_shared_wmma_matrix_b[v0, v1] = B_shared[v0, v1]
                                for i0_0_4, i1_0_4 in T.grid(16, 2):
                                    with T.block("blockized_C_update"):
                                        io = T.axis.spatial(32, i0_0_1_i1_0_1_fused // 2 * 16 + i0_0_4)
                                        jo = T.axis.spatial(32, i0_0_1_i1_0_1_fused % 2 * 16 + i0_0_2_i1_0_2_fused * 2 + i1_0_4)
                                        ko = T.axis.reduce(32, i2_0_0 * 8 + i2_0_1)
                                        T.reads([C_local_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], A_shared_wmma_matrix_a[io * 16 : io * 16 + 16, ko * 16 : ko * 16 + 16], B_shared_wmma_matrix_b[ko * 16 : ko * 16 + 16, jo * 16 : jo * 16 + 16]])
                                        T.writes([C_local_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                                        for i0_1, i1_1, i2_1 in T.grid(16, 16, 16):
                                            with T.block("C"):
                                                i = T.axis.spatial(512, io * 16 + i0_1)
                                                j = T.axis.spatial(512, jo * 16 + i1_1)
                                                k = T.axis.reduce(512, ko * 16 + i2_1)
                                                T.reads([C_local_wmma_accumulator[i, j], A_shared_wmma_matrix_a[i, k], B_shared_wmma_matrix_b[k, j]])
                                                T.writes([C_local_wmma_accumulator[i, j]])
                                                T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync"})
                                                C_local_wmma_accumulator[i, j] = C_local_wmma_accumulator[i, j] + T.cast(A_shared_wmma_matrix_a[i, k], "float32") * T.cast(B_shared_wmma_matrix_b[k, j], "float32")
                        for ax0, ax1 in T.grid(256, 32):
                            with T.block("C_local_wmma.accumulator"):
                                v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + ax0)
                                v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + i0_0_2_i1_0_2_fused * 32 + ax1)
                                T.reads([C_local_wmma_accumulator[v0, v1]])
                                T.writes([C_local[v0, v1]])
                                T.block_attr({"meta_schedule.auto_tensorize":"wmma_store"})
                                C_local[v0, v1] = C_local_wmma_accumulator[v0, v1]
                        for ax0, ax1 in T.grid(256, 32):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + ax0)
                                v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + i0_0_2_i1_0_2_fused * 32 + ax1)
                                T.reads([C_local[v0, v1]])
                                T.writes([C[v0, v1]])
                                C[v0, v1] = C_local[v0, v1]


# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def test_rewrite_reduction_block():
    mod = Before0
    target = _target()
    ctx = _create_context(mod, target)
    sch = tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod, After0)


def test_rewrite_reduction_block_tensor_core():
    mod = Before1
    target = _target()
    ctx = _create_context(mod, target)
    sch = tir.Schedule(mod, debug_mask="all")
    sch.enter_postproc()
    assert ctx.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod, After1)


if __name__ == "__main__":
    test_rewrite_reduction_block()
    test_rewrite_reduction_block_tensor_core()
