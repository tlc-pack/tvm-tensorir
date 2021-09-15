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
    f = tvm.build(mod["main"], target="cuda")
    print(f.imported_modules[0].get_source())
    a_np = np.random.uniform(size=(64, 128)).astype("float32")
    ctx = tvm.cuda(0)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((128, 64), dtype="float32"), ctx)
    f(a,b)
    b_np = a_np.T
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)

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
                    tir.block_attr({"auto_copy":1,"vector_bytes":16})
                    for ax0, ax1 in tir.grid(32, 64):
                        A_shared[v1*64+ax1, v0*32+ax0] = A[v0*32+ax0, v1*64+ax1]
                with tir.block([2, 2],"A_shared") as [v0, v1]:
                    tir.bind(v0, tx)
                    tir.bind(v1, bx)
                    tir.block_attr({"auto_copy":1,"vector_bytes":16})
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
        for bx in tir.thread_binding(0,2,thread="blockIdx.x"):
            for tx in tir.thread_binding(0,2, thread="threadIdx.x"):
                with tir.block([2, 2],"A_shared") as [v0, v1]:
                    tir.bind(v0, bx)
                    tir.bind(v1, tx)
                    tir.block_attr({"auto_copy":1,"vector_bytes":16})
                    for ax0, ax1 in tir.grid(32, 64):
                        A_shared[v0*32+ax0, v1*64+ax1] = A[v0*32+ax0, v1*64+ax1]
                with tir.block([2, 2],"A_shared") as [v0, v1]:
                    tir.bind(v0, bx)
                    tir.bind(v1, tx)
                    tir.block_attr({"auto_copy":1,"vector_bytes":16})
                    for ax0, ax1 in tir.grid(32, 64):
                        B[v1*64+ax1, v0*32+ax0] = A_shared[v0*32+ax0, v1*64+ax1]

# sch = tir.Schedule(func)
# block_transpose=sch.get_block("transpose")
# block_local=sch.cache_write(block_transpose,0,"local")
# print(tvm.script.asscript(sch.mod["main"]))

_check(A1_func2)