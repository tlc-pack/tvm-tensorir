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
from tvm.script import ty
import numpy as np

def _measure(original):
    mod = tvm.IRModule.from_expr(original)
    f = tvm.build(mod["main"], target="cuda")
    ctx = tvm.cuda(0)
    time_f = f.time_evaluator(f.entry_name, ctx, 1000)
    print(f.imported_modules[0].get_source())
    a_np = np.random.uniform(size=(2048, 2048)).astype("float32")
    ctx = tvm.cuda(0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((2048, 2048), dtype="float32"), ctx)
    f(a,b)
    print(time_f(a,b).mean)
    b_np = a_np.T
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)

def _check(original):
    mod = tvm.IRModule.from_expr(original)
    mod=tvm.tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
    mod= tvm.tir.transform.CompactBufferAllocation()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(tvm.script.asscript(mod["main"]))
    mod = tvm.tir.transform.LowerAutoCopy()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    print(tvm.script.asscript(mod["main"]))


@tvm.script.tir
def transpose(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [32, 64])
    B = tir.match_buffer(b, [64, 32])
    with tir.block([32, 64], "transpose") as [vi, vj]:
        B[vj, vi] = A[vi, vj]


@tvm.script.tir
def A0_func(a: ty.handle, b: ty.handle) -> None:
    B = tir.match_buffer(b, [128, 64], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [64, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        A_shared = tir.alloc_buffer([64, 128], elem_offset=0, scope="shared", align=128, offset_factor=1)
        B_local = tir.alloc_buffer([128, 64], elem_offset=0, scope="local", align=128, offset_factor=1)
        B_shared = tir.alloc_buffer([128, 64], elem_offset=0, scope="shared", align=128, offset_factor=1)
        for bx in tir.thread_binding(0,2,thread="blockIdx.x"):
            for tx in tir.thread_binding(0,2, thread="threadIdx.x"):
                with tir.block([2, 2],"A_shared") as [v0, v1]:
                    tir.bind(v0, bx)
                    tir.bind(v1, tx)
                    tir.block_attr({"auto_copy":1,"vector_bytes":16})
                    for ax0, ax1 in tir.grid(32, 64):
                        A_shared[v0*32+ax0, v1*64+ax1] = A[v0*32+ax0, v1*64+ax1]
                for i0, i1 in tir.grid(32, 64):
                    with tir.block([64, 128], "transpose") as [vi, vj]:
                        tir.bind(vi, bx*32+i0)
                        tir.bind(vj, tx*64+i1)
                        B_local[vj, vi] = A_shared[vi, vj]
                for i0, i1 in tir.grid(64, 32):
                    with tir.block([64, 128], "transpose") as [vi, vj]:
                        tir.bind(vi, tx*64+i0)
                        tir.bind(vj, bx*32+i1)
                        B_shared[vi, vj] = B_local[vi, vj]
                with tir.block([2, 2],"A_shared") as [v0, v1]:
                    tir.bind(v0, tx)
                    tir.bind(v1, bx)
                    tir.block_attr({"auto_copy":1,"vector_bytes":16})
                    for ax0, ax1 in tir.grid(64, 32):
                        B[v0*64+ax0, v1*32+ax1] = B_shared[v0*64+ax0, v1*32+ax1]

@tvm.script.tir
def A1_func1(a: ty.handle, b: ty.handle) -> None:
    B = tir.match_buffer(b, [128, 64], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [64, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        A_shared = tir.alloc_buffer([128, 64], elem_offset=0, scope="shared", align=128, offset_factor=1)
        for bx in tir.thread_binding(0,2,thread="blockIdx.x"):
            for tx in tir.thread_binding(0,2, thread="threadIdx.x"):
                with tir.block([2, 2],"A_shared") as [v0, v1]:
                    tir.bind(v0, bx)
                    tir.bind(v1, tx)
                    tir.block_attr({"auto_copy":1,"vector_bytes":4})
                    for ax0, ax1 in tir.grid(32, 64):
                        A_shared[v1*64+ax1, v0*32+ax0] = A[v0*32+ax0, v1*64+ax1]
                with tir.block([2, 2],"A_shared") as [v0, v1]:
                    tir.bind(v0, tx)
                    tir.bind(v1, bx)
                    tir.block_attr({"auto_copy":1,"vector_bytes":4})
                    for ax0, ax1 in tir.grid(64, 32):
                        B[v0*64+ax0, v1*32+ax1] = A_shared[v0*64+ax0, v1*32+ax1]

@tvm.script.tir
def A1_func2(a: ty.handle, b: ty.handle) -> None:
    B = tir.match_buffer(b, [128, 64], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [64, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        A_shared = tir.alloc_buffer([64, 128], elem_offset=0, scope="shared", align=128, offset_factor=1)
        for bx in tir.thread_binding(0,4,thread="blockIdx.x"):
            for tx in tir.thread_binding(0,2, thread="threadIdx.x"):
                with tir.block([4, 2],"A_shared") as [v0, v1]:
                    tir.bind(v0, bx)
                    tir.bind(v1, tx)
                    tir.block_attr({"auto_copy":1,"vector_bytes":16})
                    for ax0, ax1 in tir.grid(16, 64):
                        A_shared[v0*16+ax0, v1*64+ax1] = A[v0*16+ax0, v1*64+ax1]
                with tir.block([4, 2],"A_shared") as [v0, v1]:
                    tir.bind(v0, bx)
                    tir.bind(v1, tx)
                    tir.block_attr({"auto_copy":1,"vector_bytes":16})
                    for ax0, ax1 in tir.grid(16, 64):
                        B[v1*64+ax1, v0*16+ax0] = A_shared[v0*16+ax0, v1*64+ax1]

@tvm.script.tir
def realworld_transpose(a: ty.handle, b: ty.handle)->None:
    A = tir.match_buffer(a, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    A_shared = tir.alloc_buffer([2048, 2048], elem_offset=0, scope="shared", align=128, offset_factor=1)
    for by in tir.thread_binding(0, 128,thread="blockIdx.y"):
        for bx in tir.thread_binding(0, 64,thread="blockIdx.x"):
            for ty in tir.thread_binding(0, 4,thread="threadIdx.y"):
                for tx in tir.thread_binding(0, 32, thread="threadIdx.x"):
                    with tir.block([64, 128],"A_shared") as [v0, v1]:
                        tir.bind(v0, bx)
                        tir.bind(v1, by)
                        tir.block_attr({"auto_copy":1,"vector_bytes":4})
                        for ax0, ax1 in tir.grid(32, 16):
                            A_shared[v1*16+ax1, v0*32+ax0] = A[v0*32+ax0, v1*16+ax1]
                    with tir.block([64, 128],"B") as [v0, v1]:
                        tir.bind(v0, bx)
                        tir.bind(v1, by)
                        tir.block_attr({"auto_copy":1,"vector_bytes":4})
                        for ax0, ax1 in tir.grid(16, 32):
                            B[v1*16+ax0, v0*32+ax1] = A_shared[v1*16+ax0, v0*32+ax1]
@tvm.script.tir
def A2_func(var_A: ty.handle, var_B: ty.handle, var_C: ty.handle) -> None:
    A = tir.match_buffer(var_A, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(var_B, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    C = tir.match_buffer(var_C, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        tir.block_attr({"auto_unroll_explicit":"64"})
        C_local = tir.alloc_buffer([1024, 1024], elem_offset=0, scope="local", align=128, offset_factor=1)
        B_shared = tir.alloc_buffer([1024, 1024], elem_offset=0, scope="shared", align=128, offset_factor=1)
        A_shared = tir.alloc_buffer([1024, 1024], elem_offset=0, scope="shared", align=128, offset_factor=1)
        for i0_0_i1_0_fused in tir.thread_binding(0, 256, thread = "blockIdx.x"):
            for i0_1_i1_1_fused in tir.thread_binding(0, 1, thread = "vthread"):
                for i0_2_i1_2_fused in tir.thread_binding(0, 64, thread = "threadIdx.x"):
                    for i2_0 in tir.serial(0, 64):
                        for ax0_ax1_fused_0 in tir.serial(0, 512, annotation = {"loop_type":"lazy_cooperative_fetch"}):
                            for ax0_ax1_fused_1 in tir.vectorized(0, 2):
                                with tir.block([1024, 1024], "A_shared") as [v0, v1]:
                                    tir.bind(v0, ((tir.floordiv(i0_0_i1_0_fused, 16)*64) + tir.floordiv(((ax0_ax1_fused_0*2) + ax0_ax1_fused_1), 16)))
                                    tir.bind(v1, ((i2_0*16) + tir.floormod(((ax0_ax1_fused_0*2) + ax0_ax1_fused_1), 16)))
                                    tir.reads([A[v0, v1]])
                                    tir.writes([A_shared[v0, v1]])
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0_1 in tir.serial(0, 256, annotation = {"loop_type":"lazy_cooperative_fetch"}):
                            for ax0_ax1_fused_1_1 in tir.vectorized(0, 4):
                                with tir.block([1024, 1024], "B_shared") as [v0_1, v1_1]:
                                    tir.bind(v0_1, ((i2_0*16) + tir.floordiv(((ax0_ax1_fused_0_1*4) + ax0_ax1_fused_1_1), 64)))
                                    tir.bind(v1_1, ((tir.floormod(i0_0_i1_0_fused, 16)*64) + tir.floormod(((ax0_ax1_fused_0_1*4) + ax0_ax1_fused_1_1), 64)))
                                    tir.reads([B[v0_1, v1_1]])
                                    tir.writes([B_shared[v0_1, v1_1]])
                                    B_shared[v0_1, v1_1] = B[v0_1, v1_1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in tir.grid(4, 4, 2, 4, 4, 2):
                            with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "C") as [i, j, k]:
                                tir.bind(i, ((((tir.floordiv(i0_0_i1_0_fused, 16)*64) + (tir.floordiv(i0_2_i1_2_fused, 16)*16)) + (i0_3*4)) + i0_4))
                                tir.bind(j, ((((tir.floormod(i0_0_i1_0_fused, 16)*64) + (tir.floormod(i0_2_i1_2_fused, 16)*4)) + (i1_3*2)) + i1_4))
                                tir.bind(k, (((i2_0*16) + (i2_1*4)) + i2_2))
                                tir.reads([C_local[i, j], A_shared[i, k], B_shared[k, j]])
                                tir.writes([C_local[i, j]])
                                with tir.init():
                                    C_local[i, j] = tir.float32(0)
                                C_local[i, j] = (C_local[i, j] + (A_shared[i, k]*B_shared[k, j]))
                    for ax0, ax1 in tir.grid(16, 4):
                        with tir.block([1024, 1024], "C_local") as [v0_2, v1_2]:
                            tir.bind(v0_2, (((tir.floordiv(i0_0_i1_0_fused, 16)*64) + (tir.floordiv(i0_2_i1_2_fused, 16)*16)) + ax0))
                            tir.bind(v1_2, (((tir.floormod(i0_0_i1_0_fused, 16)*64) + (tir.floormod(i0_2_i1_2_fused, 16)*4)) + ax1))
                            tir.reads([C_local[v0_2, v1_2]])
                            tir.writes([C[v0_2, v1_2]])
                            C[v0_2, v1_2] = C_local[v0_2, v1_2]

_check(realworld_transpose)
# _measure(realworld_transpose)