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
# pylint: disable=missing-function-docstring,missing-module-docstring
import sys

import pytest

import tvm
from tvm import tir
from tvm.script import tir as T
from tvm.tir.schedule.testing import verify_trace_roundtrip
import numpy as np


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks,not-callable

@T.prim_func
def cuda_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = T.match_buffer(a, [2048, 2048], "float32")
    B = T.match_buffer(b, [2048, 2048], "float32")
    C = T.match_buffer(c, [2048, 2048], "float32")
    for by in T.thread_binding(0, 32, thread = "blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread = "blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread = "vthread.y"):
                for vx in T.thread_binding(0, 2, thread = "vthread.x"):
                    for ty in T.thread_binding(0, 8, thread = "threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread = "threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                for k1 in T.unroll(0, 8):
                                    for _, i, j in T.grid(1, 4, 4):
                                        with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            T.bind(vk, k0 * 8 + k1)
                                            T.reads([C[vi, vj], A[vi, vk], B[vk, vj]])
                                            T.writes([C[vi, vj]])
                                            with T.init():
                                                C[vi, vj] = 0.0
                                            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def cuda_matmul_read_at_a(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [2048, 2048], dtype="float32")
    B = T.match_buffer(b, [2048, 2048], dtype="float32")
    C = T.match_buffer(c, [2048, 2048], dtype="float32")
    A_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    for by in T.thread_binding(0, 32, thread="blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread="blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread="vthread.y"):
                for vx in T.thread_binding(0, 2, thread="vthread.x"):
                    for ty in T.thread_binding(0, 8, thread="threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread="threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                with T.block([], "A_shared"):
                                    T.reads([A[by * 64 : by * 64 + 64, k0 * 8 : k0 * 8 + 8]])
                                    T.writes([A_shared[by * 64 : by * 64 + 64, k0 * 8 : k0 * 8 + 8]])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(64, 8):
                                        A_shared[by * 64 + ax0, k0 * 8 + ax1] = A[by * 64 + ax0, k0 * 8 + ax1]
                                for k1 in T.unroll(0, 8):
                                    for v_, i, j in T.grid(1, 4, 4):
                                        with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            T.bind(vk, k0 * 8 + k1)
                                            T.reads([C[vi, vj], A_shared[vi, vk], B[vk, vj]])
                                            T.writes([C[vi, vj]])
                                            with T.init():
                                                C[vi, vj] = T.float32(0)
                                            C[vi, vj] = C[vi, vj] + A_shared[vi, vk] * B[vk, vj]


@T.prim_func
def cuda_matmul_read_at_ab(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [2048, 2048], dtype="float32")
    B = T.match_buffer(b, [2048, 2048], dtype="float32")
    C = T.match_buffer(c, [2048, 2048], dtype="float32")
    A_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    for by in T.thread_binding(0, 32, thread="blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread="blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread="vthread.y"):
                for vx in T.thread_binding(0, 2, thread="vthread.x"):
                    for ty in T.thread_binding(0, 8, thread="threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread="threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                with T.block([], "A_shared"):
                                    T.reads([A[by * 64 : by * 64 + 64, k0 * 8 : k0 * 8 + 8]])
                                    T.writes([A_shared[by * 64 : by * 64 + 64, k0 * 8 : k0 * 8 + 8]])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(64, 8):
                                        A_shared[by * 64 + ax0, k0 * 8 + ax1] = A[by * 64 + ax0, k0 * 8 + ax1]
                                with T.block([], "B_shared"):
                                    T.reads([B[k0 * 8 : k0 * 8 + 8, bx * 64 : bx * 64 + 64]])
                                    T.writes([B_shared[k0 * 8 : k0 * 8 + 8, bx * 64 : bx * 64 + 64]])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(8, 64):
                                        B_shared[k0 * 8 + ax0, bx * 64 + ax1] = B[k0 * 8 + ax0, bx * 64 + ax1]
                                for k1 in T.unroll(0, 8):
                                    for v_, i, j in T.grid(1, 4, 4):
                                        with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            T.bind(vk, k0 * 8 + k1)
                                            T.reads([C[vi, vj], A_shared[vi, vk], B_shared[vk, vj]])
                                            T.writes([C[vi, vj]])
                                            with T.init():
                                                C[vi, vj] = T.float32(0)
                                            C[vi, vj] = C[vi, vj] + A_shared[vi, vk] * B_shared[vk, vj]

@T.prim_func
def cuda_matmul_write_at_c(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [2048, 2048], dtype="float32")
    B = T.match_buffer(b, [2048, 2048], dtype="float32")
    C = T.match_buffer(c, [2048, 2048], dtype="float32")
    A_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    B_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    C_shared = T.alloc_buffer([2048, 2048], dtype="float32", scope="shared")
    for by in T.thread_binding(0, 32, thread="blockIdx.y"):
        for bx in T.thread_binding(0, 32, thread="blockIdx.x"):
            for vy in T.thread_binding(0, 2, thread="vthread.y"):
                for vx in T.thread_binding(0, 2, thread="vthread.x"):
                    for ty in T.thread_binding(0, 8, thread="threadIdx.y"):
                        for tx in T.thread_binding(0, 8, thread="threadIdx.x"):
                            for k0 in T.serial(0, 256):
                                with T.block([], "A_shared"):
                                    T.reads([A[by * 64 : by * 64 + 64, k0 * 8 : k0 * 8 + 8]])
                                    T.writes([A_shared[by * 64 : by * 64 + 64, k0 * 8 : k0 * 8 + 8]])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(64, 8):
                                        A_shared[by * 64 + ax0, k0 * 8 + ax1] = A[by * 64 + ax0, k0 * 8 + ax1]
                                with T.block([], "B_shared"):
                                    T.reads([B[k0 * 8 : k0 * 8 + 8, bx * 64 : bx * 64 + 64]])
                                    T.writes([B_shared[k0 * 8 : k0 * 8 + 8, bx * 64 : bx * 64 + 64]])
                                    T.block_attr({"auto_copy":1})
                                    for ax0, ax1 in T.grid(8, 64):
                                        B_shared[k0 * 8 + ax0, bx * 64 + ax1] = B[k0 * 8 + ax0, bx * 64 + ax1]
                                for k1 in T.serial(0, 8):
                                    for v_, i, j in T.grid(1, 4, 4):
                                        with T.block([2048, 2048, T.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
                                            T.bind(vi, by * 64 + vy * 32 + ty * 4 + i)
                                            T.bind(vj, bx * 64 + vx * 32 + tx * 4 + j)
                                            T.bind(vk, k0 * 8 + k1)
                                            T.reads([C_shared[vi, vj], A_shared[vi, vk], B_shared[vk, vj]])
                                            T.writes([C_shared[vi, vj]])
                                            with T.init():
                                                C_shared[vi, vj] = T.float32(0)
                                            C_shared[vi, vj] = C_shared[vi, vj] + A_shared[vi, vk] * B_shared[vk, vj]
                            with T.block([], "C_shared"):
                                T.reads([C_shared[by * 64 : by * 64 + 64, bx * 64 : bx * 64 + 64]])
                                T.writes([C[by * 64 : by * 64 + 64, bx * 64 : bx * 64 + 64]])
                                T.block_attr({"auto_copy":1})
                                for ax0, ax1 in T.grid(64, 64):
                                    C[by * 64 + ax0, bx * 64 + ax1] = C_shared[by * 64 + ax0, bx * 64 + ax1]


@T.prim_func
def simple(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (32,))
    B = T.match_buffer(b, (32,))
    for tx in T.thread_binding(0, 4, thread="threadIdx.x"):
        for i in T.serial(0, 8):
            with T.block(
                    [
                        32,
                    ],
                    "B",
            ) as vi:
                T.bind(vi, tx * 8 + i)
                B[vi] = A[vi] + 1.0


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg,too-many-nested-blocks,not-callable
# fmt: on

def test_simple():
    sch = tir.Schedule(simple, debug_mask="all")
    block = sch.get_block("B")
    _, i = sch.get_loops(block)
    sch.write_at(i, block, 0, "shared")
    print(sch.mod['main'].script())

def test_read_at_global_to_shared_a():
    sch = tir.Schedule(cuda_matmul, debug_mask="all")
    block = sch.get_block("C")
    # pylint: disable=invalid-name
    _by, _bx, _vy, _vx, _ty, _tx, k0, _k1, _, _i, _j = sch.get_loops(block)
    # pylint: enable=invalid-name
    sch.read_at(k0, block, 1, "shared")
    tvm.ir.assert_structural_equal(sch.mod["main"], cuda_matmul_read_at_a)
    verify_trace_roundtrip(sch, cuda_matmul)


def test_read_at_global_to_shared_ab():
    sch = tir.Schedule(cuda_matmul_read_at_a, debug_mask="all")
    block = sch.get_block("C")
    # pylint: disable=invalid-name
    _by, _bx, _vy, _vx, _ty, _tx, k0, _k1, _, _i, _j = sch.get_loops(block)
    # pylint: enable=invalid-name
    sch.read_at(k0, block, 2, "shared")
    tvm.ir.assert_structural_equal(sch.mod["main"], cuda_matmul_read_at_ab)
    verify_trace_roundtrip(sch, cuda_matmul_read_at_a)


def test_read_at_local_to_shared_c():
    sch = tir.Schedule(cuda_matmul_read_at_ab, debug_mask="all")
    block = sch.get_block("C")
    # pylint: disable=invalid-name
    _by, _bx, _vy, _vx, _ty, tx, _k0, _k1, _, _i, _j = sch.get_loops(block)
    # pylint: enable=invalid-name
    sch.write_at(tx, block, 0, "shared")
    mod = sch.mod["main"]
    print(tvm.lower(mod, None, simple_mode=True))

    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(2048, 2048)).astype("float32")
    b_np = np.random.uniform(size=(2048, 2048)).astype("float32")
    c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((2048, 2048), dtype="float32"), dev)
    f = tvm.build(mod, target="cuda", name="dense")
    # print(f.imported_modules[0].get_source())
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)
    # tvm.ir.assert_structural_equal(sch.mod["main"], cuda_matmul_write_at_c)
    # verify_trace_roundtrip(sch, cuda_matmul_read_at_ab)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
