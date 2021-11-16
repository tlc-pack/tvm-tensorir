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

import pytest

import tvm
from tvm import tir
from tvm.script import tir as T
import numpy as np
from tvm.contrib import nvcc
from tvm import meta_schedule as ms
import os
import tir_tensor_intrin
TASK = "transpose"
USE_MANUAL_CODE = False


def _measure_transpose(original):
    mod = tvm.IRModule.from_expr(original)
    f = tvm.build(mod["main"], target="cuda")
    ctx = tvm.cuda(0)
    time_f = f.time_evaluator(f.entry_name, ctx, 1000)
    print(f.imported_modules[0].get_source())
    a_np = np.random.uniform(size=(2048, 2048)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((2048, 2048), dtype="float32"), ctx)
    f(a,b)
    print(time_f(a,b).mean)
    b_np = a_np.T
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)

def _check_packed_matmul_tensorcore(test_func, standard_func):
    mod = tvm.IRModule.from_expr(test_func)
    f = tvm.build(mod["main"], target="cuda")
    ctx = tvm.cuda(0)
    a_np = np.random.uniform(size=(16, 16, 16, 16)).astype("float16")
    b_np = np.random.uniform(size=(16, 16, 16, 16)).astype("float16")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c1 = tvm.nd.array(np.zeros((16, 16, 16, 16), dtype="float32"), ctx)
    f(a,b,c1)
    time_f = f.time_evaluator(f.entry_name, ctx, 1000)
    print(time_f(a,b,c1).mean) #5.074027e-06

    cpu_ctx = tvm.cpu(0)
    mod = tvm.IRModule.from_expr(standard_func)
    f = tvm.build(mod["main"], target="llvm")
    a_cpu = tvm.nd.array(a_np, cpu_ctx)
    b_cpu = tvm.nd.array(b_np, cpu_ctx)
    c2 = tvm.nd.array(np.zeros((16, 16, 16, 16), dtype="float32"), cpu_ctx)
    f(a_cpu,b_cpu,c2)
    np.testing.assert_allclose(c1.asnumpy(), c2.asnumpy(), rtol=1e-4, atol=1e-4)

def _check_nonpacked_matmul_tensorcore(test_func):
    mod = tvm.IRModule.from_expr(test_func)
    f = tvm.build(mod["main"], target="cuda")
    print(f.imported_modules[0].get_source())
    ctx = tvm.cuda(0)
    a_np = np.random.uniform(size=(1024, 1024)).astype("float16")
    b_np = np.random.uniform(size=(1024, 1024)).astype("float16")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c1 = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), ctx)
    f(a,b,c1)
    time_f = f.time_evaluator(f.entry_name, ctx, 10)
    print(time_f(a,b,c1).mean * 1e3)

    c_numpy = a_np.dot(b_np)
    np.testing.assert_allclose(c1.asnumpy(), c_numpy, rtol=1e-3)


def _check(original):
    mod = tvm.IRModule.from_expr(original)
    mod=tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
    mod= tvm.tir.transform.CompactBufferAllocation()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(mod['main'].script())
    mod = tvm.tir.transform.LowerAutoCopy()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(mod['main'].script())


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code

# @tir.prim_func
# def transpose(a: tir.handle, b: tir.handle) -> None:
#     A = tir.match_buffer(a, [32, 64])
#     B = tir.match_buffer(b, [64, 32])
#     with tir.block([32, 64], "transpose") as [vi, vj]:
#         B[vj, vi] = A[vi, vj]
#
#
# @tir.prim_func
# def A0_func(a: tir.handle, b: tir.handle) -> None:
#     B = tir.match_buffer(b, [128, 64], elem_offset=0, align=128, offset_factor=1)
#     A = tir.match_buffer(a, [64, 128], elem_offset=0, align=128, offset_factor=1)
#     # body
#     with tir.block([], "root"):
#         tir.reads([])
#         tir.writes([])
#         A_shared = tir.alloc_buffer([64, 128], elem_offset=0, scope="shared", align=128, offset_factor=1)
#         B_local = tir.alloc_buffer([128, 64], elem_offset=0, scope="local", align=128, offset_factor=1)
#         B_shared = tir.alloc_buffer([128, 64], elem_offset=0, scope="shared", align=128, offset_factor=1)
#         for bx in tir.thread_binding(0,2,thread="blockIdx.x"):
#             for tx in tir.thread_binding(0,2, thread="threadIdx.x"):
#                 with tir.block([2, 2],"A_shared") as [v0, v1]:
#                     tir.bind(v0, bx)
#                     tir.bind(v1, tx)
#                     tir.block_attr({"auto_copy":1,"vector_bytes":16})
#                     for ax0, ax1 in tir.grid(32, 64):
#                         A_shared[v0*32+ax0, v1*64+ax1] = A[v0*32+ax0, v1*64+ax1]
#                 for i0, i1 in tir.grid(32, 64):
#                     with tir.block([64, 128], "transpose") as [vi, vj]:
#                         tir.bind(vi, bx*32+i0)
#                         tir.bind(vj, tx*64+i1)
#                         B_local[vj, vi] = A_shared[vi, vj]
#                 for i0, i1 in tir.grid(64, 32):
#                     with tir.block([64, 128], "transpose") as [vi, vj]:
#                         tir.bind(vi, tx*64+i0)
#                         tir.bind(vj, bx*32+i1)
#                         B_shared[vi, vj] = B_local[vi, vj]
#                 with tir.block([2, 2],"A_shared") as [v0, v1]:
#                     tir.bind(v0, tx)
#                     tir.bind(v1, bx)
#                     tir.block_attr({"auto_copy":1,"vector_bytes":16})
#                     for ax0, ax1 in tir.grid(64, 32):
#                         B[v0*64+ax0, v1*32+ax1] = B_shared[v0*64+ax0, v1*32+ax1]
#
# @tir.prim_func
# def A1_func1(a: tir.handle, b: tir.handle) -> None:
#     B = tir.match_buffer(b, [128, 64], elem_offset=0, align=128, offset_factor=1)
#     A = tir.match_buffer(a, [64, 128], elem_offset=0, align=128, offset_factor=1)
#     # body
#     with tir.block([], "root"):
#         tir.reads([])
#         tir.writes([])
#         A_shared = tir.alloc_buffer([128, 64], elem_offset=0, scope="shared", align=128, offset_factor=1)
#         for bx in tir.thread_binding(0,2,thread="blockIdx.x"):
#             for tx in tir.thread_binding(0,2, thread="threadIdx.x"):
#                 with tir.block([2, 2],"A_shared") as [v0, v1]:
#                     tir.bind(v0, bx)
#                     tir.bind(v1, tx)
#                     tir.block_attr({"auto_copy":1,"vector_bytes":4})
#                     for ax0, ax1 in tir.grid(32, 64):
#                         A_shared[v1*64+ax1, v0*32+ax0] = A[v0*32+ax0, v1*64+ax1]
#                 with tir.block([2, 2],"A_shared") as [v0, v1]:
#                     tir.bind(v0, tx)
#                     tir.bind(v1, bx)
#                     tir.block_attr({"auto_copy":1,"vector_bytes":4})
#                     for ax0, ax1 in tir.grid(64, 32):
#                         B[v0*64+ax0, v1*32+ax1] = A_shared[v0*64+ax0, v1*32+ax1]
#
# @tir.prim_func
# def A1_func2(a: tir.handle, b: tir.handle) -> None:
#     B = tir.match_buffer(b, [128, 64], elem_offset=0, align=128, offset_factor=1)
#     A = tir.match_buffer(a, [64, 128], elem_offset=0, align=128, offset_factor=1)
#     # body
#     with tir.block([], "root"):
#         tir.reads([])
#         tir.writes([])
#         A_shared = tir.alloc_buffer([64, 128], elem_offset=0, scope="shared", align=128, offset_factor=1)
#         for bx in tir.thread_binding(0,4,thread="blockIdx.x"):
#             for tx in tir.thread_binding(0,2, thread="threadIdx.x"):
#                 with tir.block([4, 2],"A_shared") as [v0, v1]:
#                     tir.bind(v0, bx)
#                     tir.bind(v1, tx)
#                     tir.block_attr({"auto_copy":1,"vector_bytes":16})
#                     for ax0, ax1 in tir.grid(16, 64):
#                         A_shared[v0*16+ax0, v1*64+ax1] = A[v0*16+ax0, v1*64+ax1]
#                 with tir.block([4, 2],"A_shared") as [v0, v1]:
#                     tir.bind(v0, bx)
#                     tir.bind(v1, tx)
#                     tir.block_attr({"auto_copy":1,"vector_bytes":16})
#                     for ax0, ax1 in tir.grid(16, 64):
#                         B[v1*64+ax1, v0*16+ax0] = A_shared[v0*16+ax0, v1*64+ax1]
#
# @tir.prim_func
# def realworld_transpose(a: tir.handle, b: tir.handle)->None:
#     A = tir.match_buffer(a, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
#     B = tir.match_buffer(b, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
#     A_shared = tir.alloc_buffer([2048, 2048], elem_offset=0, scope="shared", align=128, offset_factor=1)
#     for by in tir.thread_binding(0, 128,thread="blockIdx.y"):
#         for bx in tir.thread_binding(0, 64,thread="blockIdx.x"):
#             for tir in tir.thread_binding(0, 4,thread="threadIdx.y"):
#                 for tx in tir.thread_binding(0, 32, thread="threadIdx.x"):
#                     with tir.block([64, 128],"A_shared") as [v0, v1]:
#                         tir.bind(v0, bx)
#                         tir.bind(v1, by)
#                         tir.block_attr({"auto_copy":1,"vector_bytes":8})
#                         for ax0, ax1 in tir.grid(32, 16):
#                             A_shared[v1*16+ax1, v0*32+ax0] = A[v0*32+ax0, v1*16+ax1]
#                     with tir.block([64, 128],"B") as [v0, v1]:
#                         tir.bind(v0, bx)
#                         tir.bind(v1, by)
#                         tir.block_attr({"auto_copy":1,"vector_bytes":4})
#                         for ax0, ax1 in tir.grid(16, 32):
#                             B[v1*16+ax0, v0*32+ax1] = A_shared[v1*16+ax0, v0*32+ax1]
#
# #have to elimlinate all loops with extent=1
# @tir.prim_func
# def A3_func(var_A: tir.handle, var_B: tir.handle, var_C: tir.handle) -> None:
#     C = tir.match_buffer(var_C, [16, 16, 16, 16], elem_offset=0, align=128, offset_factor=1)
#     B = tir.match_buffer(var_B, [16, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1)
#     A = tir.match_buffer(var_A, [16, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1)
#     # body
#     with tir.block([], "root"):
#         tir.reads([])
#         tir.writes([])
#         A_shared = tir.alloc_buffer([16, 16, 16, 16], dtype="float16", elem_offset=0, scope="shared", align=128, offset_factor=1)
#         B_shared = tir.alloc_buffer([16, 16, 16, 16], dtype="float16", elem_offset=0, scope="shared", align=128, offset_factor=1)
#         A_shared_wmma_matrix_a = tir.alloc_buffer([16, 16, 16, 16], dtype="float16", elem_offset=0, scope="wmma.matrix_a", align=128, offset_factor=1)
#         B_shared_wmma_matrix_b = tir.alloc_buffer([16, 16, 16, 16], dtype="float16", elem_offset=0, scope="wmma.matrix_b", align=128, offset_factor=1)
#         C_wmma_accumulator = tir.alloc_buffer([16, 16, 16, 16], elem_offset=0, scope="wmma.accumulator", align=128, offset_factor=1)
#         C_shared = tir.alloc_buffer([16, 16, 16, 16], elem_offset=0, scope="shared", align=128,
#                                     offset_factor=1)
#         for i0_0_i1_0_fused in tir.thread_binding(0, 4, thread = "blockIdx.x"):
#             for i0_1_i1_1_fused in tir.thread_binding(0, 16, thread = "blockIdx.y"):
#                 for i0_2_i1_2_fused in tir.thread_binding(0, 4, thread = "threadIdx.y"):
#                     for tx in tir.thread_binding(0, 32, thread = "threadIdx.x"):
#                         with tir.block([16, 16, 1, 1], "blockized_C_init") as [io_init, jo_init, iio_init, jio_init]:
#                             tir.bind(io_init, ((tir.floordiv(i0_1_i1_1_fused, 2)*2) + tir.floordiv(i0_2_i1_2_fused, 2)))
#                             tir.bind(jo_init, (((i0_0_i1_0_fused*4) + (tir.floormod(i0_1_i1_1_fused, 2)*2)) + tir.floormod(i0_2_i1_2_fused, 2)))
#                             tir.bind(iio_init, 0)
#                             tir.bind(jio_init, 0)
#                             tir.reads([])
#                             tir.writes([C_wmma_accumulator[io_init, jo_init, 0:16, 0:16]])
#                             tir.evaluate(tir.tvm_fill_fragment(C_wmma_accumulator.data, 16, 16, 16, tir.floordiv(tir.get_elem_offset(C_wmma_accumulator[io_init, jo_init, 0, 0], dtype="int32"), 256), tir.float32(0), dtype="handle"))
#                         for i2_0 in tir.serial(0, 16):
#                             with tir.block([16,8],"B_shared") as [v0,v1]:
#                                 tir.bind(v0, i2_0)
#                                 tir.bind(v1, ((i0_0_i1_0_fused*2) + (tir.floormod(i0_1_i1_1_fused, 2))))
#                                 tir.block_attr({"auto_copy":1,"vector_bytes":8})
#                                 for ax1, ax2, ax3 in tir.grid(2, 16, 16):
#                                     B_shared[v0, v1*2+ax1, ax2, ax3] = B[v0, v1*2+ax1, ax2, ax3]
#                             with tir.block([8,16],"A_shared") as [v0,v1]:
#                                 tir.bind(v0, tir.floordiv(i0_1_i1_1_fused, 2))
#                                 tir.bind(v1, i2_0)
#                                 tir.block_attr({"auto_copy":1,"vector_bytes":8})
#                                 for ax0, ax2, ax3 in tir.grid(2, 16, 16):
#                                     A_shared[v0*2+ax0, v1, ax2, ax3] = A[v0*2+ax0, v1, ax2, ax3]
#                             for i2_1, i0_3, i1_3, i2_2 in tir.grid(1, 1, 1, 1):
#                                 with tir.block([16,16],"B_shared_wmma.matrix_b") as[v0,v1]:
#                                     tir.bind(v0, i2_0)
#                                     tir.bind(v1, (((i0_0_i1_0_fused*4) + (tir.floormod(i0_1_i1_1_fused, 2)*2)) + tir.floormod(i0_2_i1_2_fused, 2)))
#                                     tir.block_attr({"auto_copy":1,"vector_bytes":4,"layout":"row_major"})
#                                     for  ax2, ax3 in tir.grid( 16, 16):
#                                         B_shared_wmma_matrix_b[v0, v1, ax2, ax3] = B_shared[v0, v1, ax2, ax3]
#
#                                 with tir.block([16,16],"A_shared_wmma.matrix_b") as[v0,v1]:
#                                     tir.bind(v0, ((tir.floordiv(i0_1_i1_1_fused, 2)*2) + tir.floordiv(i0_2_i1_2_fused, 2)))
#                                     tir.bind(v1, i2_0)
#                                     tir.block_attr({"auto_copy":1,"vector_bytes":4})
#                                     for  ax2, ax3 in tir.grid( 16, 16):
#                                         A_shared_wmma_matrix_a[v0, v1, ax2, ax3] = A_shared[v0, v1, ax2, ax3]
#                                 for i0_4, i1_4 in tir.grid(1, 1):
#                                     with tir.block([16, 16, tir.reduce_axis(0, 16), 1, 1, tir.reduce_axis(0, 1)], "blockized_C_update") as [io, jo, ko, iio, jio, kio]:
#                                         tir.bind(io, ((tir.floordiv(i0_1_i1_1_fused, 2)*2) + tir.floordiv(i0_2_i1_2_fused, 2)))
#                                         tir.bind(jo, (((i0_0_i1_0_fused*4) + (tir.floormod(i0_1_i1_1_fused, 2)*2)) + tir.floormod(i0_2_i1_2_fused, 2)))
#                                         tir.bind(ko, i2_0)
#                                         tir.bind(iio, 0)
#                                         tir.bind(jio, 0)
#                                         tir.bind(kio, 0)
#                                         tir.reads([C_wmma_accumulator[io, jo, 0:16, 0:16], A_shared_wmma_matrix_a[io, ko, 0:16, 0:16], B_shared_wmma_matrix_b[ko, jo, 0:16, 0:16]])
#                                         tir.writes([C_wmma_accumulator[io, jo, 0:16, 0:16]])
#                                         tir.evaluate(tir.tvm_mma_sync(C_wmma_accumulator.data, tir.floordiv(tir.get_elem_offset(C_wmma_accumulator[io, jo, 0, 0], dtype="int32"), 256), A_shared_wmma_matrix_a.data, tir.floordiv(tir.get_elem_offset(A_shared_wmma_matrix_a[io, ko, 0, 0], dtype="int32"), 256), B_shared_wmma_matrix_b.data, tir.floordiv(tir.get_elem_offset(B_shared_wmma_matrix_b[ko, jo, 0, 0], dtype="int32"), 256), C_wmma_accumulator.data, tir.floordiv(tir.get_elem_offset(C_wmma_accumulator[io, jo, 0, 0], dtype="int32"), 256), dtype="handle"))
#                         with tir.block([16,16],"C_shared") as [v0, v1]:
#                             tir.bind(v0, ((tir.floordiv(i0_1_i1_1_fused, 2)*2) + tir.floordiv(i0_2_i1_2_fused, 2)))
#                             tir.bind(v1, (((i0_0_i1_0_fused*4) + (tir.floormod(i0_1_i1_1_fused, 2)*2)) + tir.floormod(i0_2_i1_2_fused, 2)))
#                             tir.block_attr({"auto_copy":1,"vector_bytes":4})
#                             for  ax2, ax3 in tir.grid( 16, 16):
#                                 C[v0, v1, ax2, ax3] = C_wmma_accumulator[v0, v1, ax2, ax3]
#                         # with tir.block([16,16],"C") as [v0, v1]:
#                         #     tir.bind(v0, ((tir.floordiv(i0_1_i1_1_fused, 2)*2) + tir.floordiv(i0_2_i1_2_fused, 2)))
#                         #     tir.bind(v1, (((i0_0_i1_0_fused*4) + (tir.floormod(i0_1_i1_1_fused, 2)*2)) + tir.floormod(i0_2_i1_2_fused, 2)))
#                         #     tir.block_attr({"auto_copy":1,"vector_bytes":8})
#                         #     for ax2, ax3 in tir.grid(16, 16):
#                         #         C[v0, v1, ax2, ax3] = C_shared[v0, v1, ax2, ax3]
#
#
#
# @tir.prim_func
# def matmul_fp16_packed(var_A: tir.handle, var_B: tir.handle, var_C: tir.handle) -> None:
#     A = tir.match_buffer(
#         var_A, [16, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
#     )
#     B = tir.match_buffer(
#         var_B, [16, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1
#     )
#     C = tir.match_buffer(var_C, [16, 16, 16, 16], elem_offset=0, align=128, offset_factor=1)
#     # body
#     with tir.block([], "root"):
#         tir.reads([])
#         tir.writes([])
#         for i0, i1, i2, i3, i4, i5 in tir.grid(16, 16, 16, 16, 16, 16):
#             with tir.block([16, 16, tir.reduce_axis(0, 16), 16, 16, tir.reduce_axis(0, 16)], "C") as [
#                 io,
#                 jo,
#                 ko,
#                 ii,
#                 ji,
#                 ki,
#             ]:
#                 tir.bind(io, i0)
#                 tir.bind(jo, i1)
#                 tir.bind(ko, i2)
#                 tir.bind(ii, i3)
#                 tir.bind(ji, i4)
#                 tir.bind(ki, i5)
#                 tir.reads([C[io, jo, ii, ji], A[io, ko, ii, ki], B[ko, jo, ki, ji]])
#                 tir.writes([C[io, jo, ii, ji]])
#                 with tir.init():
#                     C[io, jo, ii, ji] = tir.float32(0)
#                 C[io, jo, ii, ji] = C[io, jo, ii, ji] + (
#                         tir.cast(A[io, ko, ii, ki], "float32") * tir.cast(B[ko, jo, ki, ji], "float32")
#                 )

@T.prim_func
def nonpacked_matmul(var_A: T.handle, var_B: T.handle, var_C: T.handle) -> None:
    A = T.match_buffer(var_A, [1024, 1024], dtype="float16")
    B = T.match_buffer(var_B, [1024, 1024], dtype="float16")
    C = T.match_buffer(var_C, [1024, 1024], dtype="float32")
    # body
    # with T.block("root")
    A_wmma_matrix_a = T.alloc_buffer([1024, 1024], dtype="float16", scope="wmma.matrix_a")
    B_wmma_matrix_b = T.alloc_buffer([1024, 1024], dtype="float16", scope="wmma.matrix_b")
    A_shared = T.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
    B_shared = T.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
    C_wmma_accumulator = T.alloc_buffer([1024, 1024], dtype="float32", scope="wmma.accumulator")
    for i0_0_0_i1_0_0_fused in T.thread_binding(0, 8, thread="blockIdx.x"):
        for i0_0_1_i1_0_1_fused in T.thread_binding(0, 8, thread="blockIdx.y"):
            for i0_0_2_i1_0_2_fused in T.thread_binding(0, 8, thread="threadIdx.y"):
                for i0_0_4_init, i1_0_4_init in T.grid(4, 2):
                    with T.block([64, 64], "blockized_C_init") as [io, jo]:
                        T.bind(io, i0_0_0_i1_0_0_fused * 8 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 4 + i0_0_4_init)
                        T.bind(jo, i0_0_1_i1_0_1_fused * 8 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 2 + i1_0_4_init)
                        T.reads([])
                        T.writes([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                        with T.block([1, 1], "blockized_C_init") as [i_inito, j_inito]:
                            T.bind(i_inito, 0)
                            T.bind(j_inito, 0)
                            T.reads([])
                            T.writes([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                            C_1 = T.match_buffer(C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                            T.evaluate(T.tvm_fill_fragment(C_1.data, 16, 16, 16, T.floordiv(C_1.elem_offset, 256) + T.floordiv(T.floormod(C_1.elem_offset, 256), 16), T.float32(0), dtype="handle"))
                for i2_0_0 in T.serial(0, 32):
                    with T.block([], "A_shared"):
                        T.reads([A[i0_0_0_i1_0_0_fused * 128 : i0_0_0_i1_0_0_fused * 128 + 128, i2_0_0 * 32 : i2_0_0 * 32 + 32]])
                        T.writes([A_shared[i0_0_0_i1_0_0_fused * 128 : i0_0_0_i1_0_0_fused * 128 + 128, i2_0_0 * 32 : i2_0_0 * 32 + 32]])
                        T.block_attr({"auto_copy":1,"local_stage":True})
                        for ax0, ax1 in T.grid(128, 32):
                            A_shared[i0_0_0_i1_0_0_fused * 128 + ax0, i2_0_0 * 32 + ax1] = A[i0_0_0_i1_0_0_fused * 128 + ax0, i2_0_0 * 32 + ax1]
                    with T.block([], "B_shared"):
                        T.reads([B[i2_0_0 * 32 : i2_0_0 * 32 + 32, i0_0_1_i1_0_1_fused * 128 : i0_0_1_i1_0_1_fused * 128 + 128]])
                        T.writes([B_shared[i2_0_0 * 32 : i2_0_0 * 32 + 32, i0_0_1_i1_0_1_fused * 128 : i0_0_1_i1_0_1_fused * 128 + 128]])
                        T.block_attr({"auto_copy":1,"local_stage":True})
                        for ax0, ax1 in T.grid(32, 128):
                            B_shared[i2_0_0 * 32 + ax0, i0_0_1_i1_0_1_fused * 128 + ax1] = B[i2_0_0 * 32 + ax0, i0_0_1_i1_0_1_fused * 128 + ax1]
                    for i2_0_1 in T.serial(0, 2):
                        with T.block([], "A_wmma.matrix_a"):
                            T.reads([A_shared[i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 : i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 + 64, i2_0_0 * 32 + i2_0_1 * 16 : i2_0_0 * 32 + i2_0_1 * 16 + 16]])
                            T.writes([A_wmma_matrix_a[i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 : i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 + 64, i2_0_0 * 32 + i2_0_1 * 16 : i2_0_0 * 32 + i2_0_1 * 16 + 16]])
                            T.block_attr({"auto_copy":1})
                            for ax0, ax1 in T.grid(64, 16):
                                A_wmma_matrix_a[i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 + ax0, i2_0_0 * 32 + i2_0_1 * 16 + ax1] = A_shared[i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 + ax0, i2_0_0 * 32 + i2_0_1 * 16 + ax1]
                        with T.block([], "B_wmma.matrix_b"):
                            T.reads([B_shared[i2_0_0 * 32 + i2_0_1 * 16 : i2_0_0 * 32 + i2_0_1 * 16 + 16, i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 : i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 + 32]])
                            T.writes([B_wmma_matrix_b[i2_0_0 * 32 + i2_0_1 * 16 : i2_0_0 * 32 + i2_0_1 * 16 + 16, i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 : i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 + 32]])
                            T.block_attr({"auto_copy":1})
                            for ax0, ax1 in T.grid(16, 32):
                                B_wmma_matrix_b[i2_0_0 * 32 + i2_0_1 * 16 + ax0, i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 + ax1] = B_shared[i2_0_0 * 32 + i2_0_1 * 16 + ax0, i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 + ax1]
                        for i0_0_3, i1_0_3, i2_0_2, i0_0_4, i1_0_4 in T.grid(1, 1, 1, 4, 2):
                            with T.block([64, 64, T.reduce_axis(0, 64)], "blockized_C_update") as [io, jo, ko]:
                                T.bind(io, i0_0_0_i1_0_0_fused * 8 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 4 + i0_0_4)
                                T.bind(jo, i0_0_1_i1_0_1_fused * 8 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 2 + i1_0_4)
                                T.bind(ko, i2_0_0 * 2 + i2_0_1)
                                T.reads([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], A_wmma_matrix_a[io * 16 : io * 16 + 16, ko * 16 : ko * 16 + 16], B_wmma_matrix_b[ko * 16 : ko * 16 + 16, jo * 16 : jo * 16 + 16]])
                                T.writes([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                                with T.block([1, 1, T.reduce_axis(0, 1)], "blockized_C") as [io_1, jo_1, ko_1]:
                                    T.bind(io_1, 0)
                                    T.bind(jo_1, 0)
                                    T.bind(ko_1, 0)
                                    T.reads([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], A_wmma_matrix_a[io * 16 : io * 16 + 16, ko * 16 : ko * 16 + 16], B_wmma_matrix_b[ko * 16 : ko * 16 + 16, jo * 16 : jo * 16 + 16]])
                                    T.writes([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                                    A_1 = T.match_buffer(A_wmma_matrix_a[io * 16 : io * 16 + 16, ko * 16 : ko * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_a", offset_factor=16)
                                    B_1 = T.match_buffer(B_wmma_matrix_b[ko * 16 : ko * 16 + 16, jo * 16 : jo * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_b", offset_factor=16)
                                    C_2 = T.match_buffer(C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                                    T.evaluate(T.tvm_mma_sync(C_2.data, T.floordiv(C_2.elem_offset, 256) + T.floordiv(T.floormod(C_2.elem_offset, 256), 16), A_1.data, T.floordiv(A_1.elem_offset, 256) + T.floordiv(T.floormod(A_1.elem_offset, 256), 16), B_1.data, T.floordiv(B_1.elem_offset, 256) + T.floordiv(T.floormod(B_1.elem_offset, 256), 16), C_2.data, T.floordiv(C_2.elem_offset, 256) + T.floordiv(T.floormod(C_2.elem_offset, 256), 16), dtype="handle"))
                with T.block([], "C_wmma.accumulator"):
                    T.reads([C_wmma_accumulator[i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 : i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 + 64, i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 : i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 + 32]])
                    T.writes([C[i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 : i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 + 64, i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 : i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 + 32]])
                    T.block_attr({"auto_copy":1})
                    for ax0, ax1 in T.grid(64, 32):
                        C[i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 + ax0, i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 + ax1] = C_wmma_accumulator[i0_0_0_i1_0_0_fused * 128 + T.floordiv(i0_0_2_i1_0_2_fused, 4) * 64 + ax0, i0_0_1_i1_0_1_fused * 128 + T.floormod(i0_0_2_i1_0_2_fused, 4) * 32 + ax1]

# print(tvm.lower(nonpacked_matmul, None))
# _check(A3_func)
# _check(realworld_transpose)
_check(nonpacked_matmul)
# _check_nonpacked_matmul_tensorcore(nonpacked_matmul)
# _check_packed_matmul_tensorcore(A3_func, matmul_fp16_packed)
# _measure_transpose(realworld_transpose)
# _measure(realworld_transpose)