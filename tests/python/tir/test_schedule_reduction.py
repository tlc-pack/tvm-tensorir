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
# pylint: disable=missing-function-docstring,missing-module-docstring
import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import ty

import util

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg

@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
def matmul_decompose0(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([128, 128], "init") as [vi, vj]:
        C[vi, vj] = 0.0

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
def matmul_decompose1(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        if vk == 0:
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
def matmul_blockized(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 128):
            with tir.block([128, 1, tir.reduce_axis(0, 1)], "blockized_update") as [vi, vjo, vko]:
                tir.bind(vi, i0)
                tir.bind(vjo, 0)
                tir.bind(vko, 0)
                with tir.init():
                    for i1 in range(0, 128):
                        with tir.block([128], "update_init") as [vj_init]:
                            tir.bind(vj_init, i1)
                            C[vi, vj_init] = tir.float32(0)
                for i1, i2 in tir.grid(128, 128):
                    with tir.block([128, tir.reduce_axis(0, 128)], "update") as [vj, vk]:
                        tir.bind(vj, i1)
                        tir.bind(vk, i2)
                        C[vi, vj] = C[vi, vj] + (A[vi, vk] * B[vj, vk])


@tvm.script.tir
def matmul_scale(a: ty.handle, b: ty.handle, e: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    E = tir.match_buffer(e, [128, 128])

    C = tir.buffer_allocate([128, 128])
    D = tir.buffer_allocate([128, 128])
    with tir.block([128, 128], "D") as [vi, vj]:
        D[vi, vj] = A[vi, vj] * 2.0

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + D[vi, vk] * B[vj, vk]

    with tir.block([128, 128], "E") as [vi, vj]:
        E[vi, vj] = C[vi, vj] + 1.0


@tvm.script.tir
def matmul_scale_inline(a: ty.handle, b: ty.handle, e: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    E = tir.match_buffer(e, [128, 128])

    C = tir.buffer_allocate([128, 128])
    with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + (A[vi, vk] * 2.0) * B[vj, vk]

    with tir.block([128, 128], "E") as [vi, vj]:
        E[vi, vj] = C[vi, vj] + 1.0


@tvm.script.tir
def matmul_rfactor(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    C_rf = tir.buffer_allocate([4, 128, 128])

    for i2_inner_inner, i0, i1, i2_outer, i2_inner_outer in tir.grid(4, 128, 128, 4, 8):
        with tir.block([4, 128, 128, tir.reduce_axis(0, 4), tir.reduce_axis(0, 8)], "update_rf") as [vi2_inner_inner, vi, vj, vi2_outer, vi2_inner_outer]:
            tir.bind(vi2_inner_inner, i2_inner_inner)
            tir.bind(vi, i0)
            tir.bind(vj, i1)
            tir.bind(vi2_outer, i2_outer)
            tir.bind(vi2_inner_outer, i2_inner_outer)
            with tir.init():
                C_rf[vi2_inner_inner, vi, vj] = tir.float32(0)
            C_rf[vi2_inner_inner, vi, vj] = (C_rf[vi2_inner_inner, vi, vj] + (A[vi, (((vi2_outer*32) + (vi2_inner_outer*4)) + vi2_inner_inner)]*B[vj, (((vi2_outer*32) + (vi2_inner_outer*4)) + vi2_inner_inner)]))

    for i0_1, i1_1, i2_inner_inner_1 in tir.grid(128, 128, 4):
        with tir.block([128, 128, tir.reduce_axis(0, 4)], "update") as [vi_1, vj_1, vi2_inner_inner_1]:
            tir.bind(vi_1, i0_1)
            tir.bind(vj_1, i1_1)
            tir.bind(vi2_inner_inner_1, i2_inner_inner_1)
            with tir.init():
                C[vi_1, vj_1] = tir.float32(0)
            C[vi_1, vj_1] = (C[vi_1, vj_1] + C_rf[vi2_inner_inner_1, vi_1, vj_1])


@tvm.script.tir
def square_sum(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 256, 256])
    C = tir.match_buffer(c, [16])

    with tir.block([16, tir.reduce_axis(0, 256), tir.reduce_axis(0, 256)], "C") as [b, i, j]:
        with tir.init():
            C[b] = tir.float32(0)
        C[b] = C[b] + A[b, i, j] * A[b, i, j]


@tvm.script.tir
def square_sum_rfactor(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 256, 256])
    C = tir.match_buffer(c, [16])
    C_rf = tir.buffer_allocate([16, 256])

    for i2, i0, i1 in tir.grid(256, 16, 256):
        with tir.block([256, 16, tir.reduce_axis(0, 256)], "C_rf") as [vi2, b, i]:
            tir.bind(vi2, i2)
            tir.bind(b, i0)
            tir.bind(i, i1)
            with tir.init():
                C_rf[b, vi2] = tir.float32(0)
            C_rf[b, vi2] = (C_rf[b, vi2] + (A[b, i, vi2]*A[b, i, vi2]))

    for i0_1, i2_1 in tir.grid(16, 256):
        with tir.block([16, tir.reduce_axis(0, 256)], "C") as [b_1, vi2_1]:
            tir.bind(b_1, i0_1)
            tir.bind(vi2_1, i2_1)
            with tir.init():
                C[b_1] = tir.float32(0)
            C[b_1] = (C[b_1] + C_rf[b_1, vi2_1])


@tvm.script.tir
def square_sum_square_root(a: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 256, 256])
    D = tir.match_buffer(d, [16])
    C = tir.buffer_allocate([16])

    with tir.block([16, tir.reduce_axis(0, 256), tir.reduce_axis(0, 256)], "C") as [b, i, j]:
        with tir.init():
            C[b] = tir.float32(0)
        C[b] = C[b] + A[b, i, j] * A[b, i, j]

    with tir.block([16], "D") as [b]:
        D[b] = tir.sqrt(C[b], dtype="float32")


@tvm.script.tir
def square_sum_square_root_rfactor(a: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 256, 256])
    D = tir.match_buffer(d, [16])
    C = tir.buffer_allocate([16])
    C_rf = tir.buffer_allocate([1, 16])

    for i1_i2_fused_inner, i0, i1_i2_fused_outer in tir.grid(1, 16, 65536):
        with tir.block([1, 16, tir.reduce_axis(0, 256), tir.reduce_axis(0, 256)], "C_rf") as [vi1_i2_fused_inner, b, i, j]:
            tir.bind(vi1_i2_fused_inner, i1_i2_fused_inner)
            tir.bind(b, i0)
            tir.bind(i, tir.floordiv(i1_i2_fused_outer, 256))
            tir.bind(j, tir.floormod(i1_i2_fused_outer, 256))
            with tir.init():
                C_rf[vi1_i2_fused_inner, b] = tir.float32(0)
            C_rf[vi1_i2_fused_inner, b] = (C_rf[vi1_i2_fused_inner, b] + (A[b, i, j]*A[b, i, j]))

    for i0_1, i1_i2_fused_inner_1 in tir.grid(16, 1):
        with tir.block([16, tir.reduce_axis(0, 1)], "C") as [b_1, vi1_i2_fused_inner_1]:
            tir.bind(b_1, i0_1)
            tir.bind(vi1_i2_fused_inner_1, i1_i2_fused_inner_1)
            with tir.init():
                C[b_1] = tir.float32(0)
            C[b_1] = (C[b_1] + C_rf[vi1_i2_fused_inner_1, b_1])

    for i0_2 in tir.serial(0, 16):
        with tir.block([16], "D") as [b_2]:
            tir.bind(b_2, i0_2)
            D[b_2] = tir.sqrt(C[b_2], dtype="float32")


@tvm.script.tir
def rowsum_allreduce(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16, 16), "float32")
    B = tir.match_buffer(b, (16,), "float32")

    with tir.block([16, tir.reduce_axis(0, 16), tir.reduce_axis(0, 16)], "B") as [vii, vi, vj]:
        with tir.init():
            B[vii] = tir.float32(0)
        B[vii] = B[vii] + A[vii, vi, vj]


# fmt: on
# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg


# pylint: disable=invalid-name


def test_reduction_roundtrip():
    func_rt = tvm.script.from_source(tvm.script.asscript(matmul))
    tvm.ir.assert_structural_equal(matmul, func_rt)


def test_reduction_decompose():
    # Test 1
    s = tir.Schedule(matmul, debug_mode=True)
    C = s.get_block("update")
    i, _, _ = s.get_axes(C)
    s.decompose_reduction(C, i)
    tvm.ir.assert_structural_equal(matmul_decompose0, s.mod["main"])
    # Test 2
    s = tir.Schedule(matmul, debug_mode=True)
    C = s.get_block("update")
    s.decompose_reduction(C, loop=None)
    tvm.ir.assert_structural_equal(matmul_decompose1, s.mod["main"])


def test_reduction_merge():
    s = tir.Schedule(matmul_decompose0, debug_mode=True)
    init = s.get_block("init")
    update = s.get_block("update")
    s.merge_reduction(init, update)
    tvm.ir.assert_structural_equal(matmul, s.mod["main"])


def test_reduction_blockize():
    s = tir.Schedule(matmul, debug_mode=True)
    C = s.get_block("update")
    _, j, _ = s.get_axes(C)
    s.blockize(j)
    tvm.ir.assert_structural_equal(matmul_blockized, s.mod["main"])


def test_reduction_compute_inline():
    s = tir.Schedule(matmul_scale, debug_mode=True)
    D = s.get_block("D")
    s.compute_inline(D)
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_scale_inline)


def test_reduction_rfactor():
    # Test 1
    func = util.matmul_stmt()
    s = tir.Schedule(func, debug_mode=True)
    C = s.get_block("update")
    i, j, k = s.get_axes(C)
    _, ki = s.split(k, factor=32)
    _, kii = s.split(ki, factor=4)
    _ = s.rfactor(kii, factor=0)
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_rfactor)

    f = tvm.build(s.mod["main"], target="llvm")
    a_np = np.random.uniform(size=(128, 128)).astype("float32")
    b_np = np.random.uniform(size=(128, 128)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((128, 128), dtype="float32"))
    f(a, b, c)
    c_np = np.matmul(a_np, b_np.T)
    tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-4, atol=1e-4)

    # Test 2
    s = tir.Schedule(square_sum, debug_mode=True)
    C = s.get_block("C")
    b, i, j = s.get_axes(C)
    _ = s.rfactor(j, 1)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_rfactor)

    f = tvm.build(s.mod["main"], target="llvm")
    a_np = np.random.uniform(size=(16, 256, 256)).astype("float32")
    a = tvm.nd.array(a_np)
    c = tvm.nd.array(np.zeros((16,), dtype="float32"))
    f(a, c)
    c_np = np.sum(a_np * a_np, axis=(1, 2))
    tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-4, atol=1e-4)

    # Test 3
    s = tir.Schedule(square_sum_square_root, debug_mode=True)
    C = s.get_block("C")
    b, i, j = s.get_axes(C)
    fuse = s.fuse(i, j)
    _, fi = s.split(fuse, factor=1)
    _ = s.rfactor(fi, 0)
    tvm.ir.assert_structural_equal(s.mod["main"], square_sum_square_root_rfactor)

    f = tvm.build(s.mod["main"], target="llvm")
    a_np = np.random.uniform(size=(16, 256, 256)).astype("float32")
    a = tvm.nd.array(a_np)
    c = tvm.nd.array(np.zeros((16,), dtype="float32"))
    f(a, c)
    c_np = np.sqrt(np.sum(a_np * a_np, axis=(1, 2)))
    tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-4, atol=1e-4)


@pytest.mark.skip("Needs GPU")
def test_reduction_allreduce():
    ctx = tvm.gpu(0)
    # Test 1
    s = tir.Schedule(rowsum_allreduce, debug_mode=True)
    thread_x = tir.thread_axis((0, 16), "threadIdx.x")
    thread_y = tir.thread_axis((0, 16), "threadIdx.y")

    B_block = s.get_block("B")
    _, ax_i, ax_j = s.get_axes(B_block)
    s.bind(ax_j, thread_x)
    s.bind(ax_i, thread_y)

    f = tvm.build(s.mod["main"], target="cuda")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((16,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1, 2))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)

    # Test 2
    s = tir.Schedule(rowsum_allreduce, debug_mode=True)
    thread_x = tir.thread_axis((0, 16), "threadIdx.x")

    B_block = s.get_block("B")
    _, ax_i, ax_j = s.get_axes(B_block)
    s.bind(ax_j, thread_x)

    f = tvm.build(s.mod["main"], target="cuda")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((16,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1, 2))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)

    # Test 3
    s = tir.Schedule(rowsum_allreduce, debug_mode=True)
    thread_x = tir.thread_axis((0, 16), "threadIdx.x")

    B_block = s.get_block("B")
    _, ax_i, ax_j = s.get_axes(B_block)
    s.bind(ax_i, thread_x)

    f = tvm.build(s.mod["main"], target="cuda")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((16,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1, 2))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)

    # Test 4
    s = tir.Schedule(rowsum_allreduce, debug_mode=True)
    thread_x = tir.thread_axis((0, 16), "threadIdx.x")

    B_block = s.get_block("B")
    _, ax_i, ax_j = s.get_axes(B_block)

    B_rf = s.rfactor(ax_i, 0)
    ax_i_rf, ax_ii_rf, ax_j_rf = s.get_axes(B_rf)
    s.reorder(ax_ii_rf, ax_i_rf, ax_j_rf)
    ax_i_rf_o, _ = s.split(ax_i_rf, factor=4)

    s.bind(ax_ii_rf, tir.thread_axis("blockIdx.x"))
    s.bind(ax_i_rf_o, tir.thread_axis("threadIdx.x"))

    _ = s.cache_write(B_rf, 0, "local")
    s.compute_inline(B_rf)

    s.reverse_compute_at(B_block, ax_i_rf_o)

    f = tvm.build(s.mod["main"], target="cuda")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((16,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1, 2))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_reduction_roundtrip()
    test_reduction_decompose()
    test_reduction_merge()
    test_reduction_blockize()
    test_reduction_compute_inline()
    test_reduction_rfactor()
    test_reduction_allreduce()
