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

import numpy as np
import pytest
import tvm
import util
from tvm import tir
from tvm.script import ty


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_reduction_roundtrip():
    func_rt = tvm.script.from_source(tvm.script.asscript(matmul))
    tvm.ir.assert_structural_equal(matmul, func_rt)


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


def test_reduction_decompose():
    # Test 1
    s = tir.create_schedule(matmul)
    C = s.get_block("update")
    i, j, k = s.get_axes(C)
    s.decompose_reduction(C, i)
    tvm.ir.assert_structural_equal(matmul_decompose0, s.func)

    # Test 2
    s = tir.create_schedule(matmul)
    C = s.get_block("update")
    s.decompose_reduction(C)
    tvm.ir.assert_structural_equal(matmul_decompose1, s.func)


def test_reduction_merge():
    s = tir.create_schedule(matmul_decompose0)
    init = s.get_block("init")
    update = s.get_block("update")
    s.merge_reduction(init, update)
    tvm.ir.assert_structural_equal(matmul, s.func)


@tvm.script.tir
def matmul_blockzied(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
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


def test_reduction_blockize():
    s = tir.create_schedule(matmul)
    C = s.get_block("update")
    i, j, k = s.get_axes(C)
    s.blockize(j)
    tvm.ir.assert_structural_equal(matmul_blockzied, s.func)


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


def test_reduction_compute_inline():
    s = tir.create_schedule(matmul_scale)
    D = s.get_block("D")
    s.compute_inline(D)
    tvm.ir.assert_structural_equal(s.func, matmul_scale_inline)


@tvm.script.tir
def matmul_rfactor(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    B = tir.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    C_rf = tir.buffer_allocate([4, 128, 128], elem_offset=0, align=128, offset_factor=1)
    for i2_inner_inner in range(0, 4):
        for i0 in range(0, 128):
            for i1 in range(0, 128):
                for i2_outer in range(0, 4):
                    for i2_inner_outer in range(0, 8):
                        with tir.block([128, 128, tir.reduce_axis(0, 32), 4], "update") as [
                            vi,
                            vj,
                            vk,
                            vi2_inner_inner,
                        ]:
                            tir.bind(vi, i0)
                            tir.bind(vj, i1)
                            tir.bind(vk, ((i2_outer * 8) + i2_inner_outer))
                            tir.bind(vi2_inner_inner, i2_inner_inner)
                            with tir.init():
                                C_rf[vi2_inner_inner, vi, vj] = 0.0
                            C_rf[vi2_inner_inner, vi, vj] = (
                                C_rf[vi2_inner_inner, vi, vj]
                                + A[vi, ((vk * 4) + vi2_inner_inner)]
                                * B[vj, ((vk * 4) + vi2_inner_inner)]
                            )

    with tir.block([128, 128, tir.reduce_axis(0, 4)], "update") as [vi, vj, vi2_inner_inner]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + C_rf[vi2_inner_inner, vi, vj]


def test_reduction_rfactor():
    func = util.matmul_stmt()

    s = tir.create_schedule(func)
    C = s.get_block("update")
    i, j, k = s.get_axes(C)
    ko, ki = s.split(k, 32)
    kio, kii = s.split(ki, 4)
    wb = s.rfactor(kii, 0)
    tvm.ir.assert_structural_equal(s.func, matmul_rfactor)


@tvm.script.tir
def rowsum_allreduce(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16, 16), "float32")
    B = tir.match_buffer(b, (16,), "float32")

    with tir.block([16, tir.reduce_axis(0, 16), tir.reduce_axis(0, 16)], "B") as [vii, vi, vj]:
        with tir.init():
            B[vii] = tir.float32(0)
        B[vii] = B[vii] + A[vii, vi, vj]


@pytest.mark.skip("Needs GPU")
def test_reduction_allreduce():
    ctx = tvm.gpu(0)
    # Test 1
    s = tir.create_schedule(rowsum_allreduce)
    thread_x = tir.thread_axis((0, 16), "threadIdx.x")
    thread_y = tir.thread_axis((0, 16), "threadIdx.y")

    B_block = s.get_block("B")
    ax_ii, ax_i, ax_j = s.get_axes(B_block)
    s.bind(ax_j, thread_x)
    s.bind(ax_i, thread_y)

    f = tvm.build(s.func, target="cuda")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((16,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1, 2))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)

    # Test 2
    s = tir.create_schedule(rowsum_allreduce)
    thread_x = tir.thread_axis((0, 16), "threadIdx.x")

    B_block = s.get_block("B")
    ax_ii, ax_i, ax_j = s.get_axes(B_block)
    s.bind(ax_j, thread_x)

    f = tvm.build(s.func, target="cuda")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((16,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1, 2))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)

    # Test 3
    s = tir.create_schedule(rowsum_allreduce)
    thread_x = tir.thread_axis((0, 16), "threadIdx.x")

    B_block = s.get_block("B")
    ax_ii, ax_i, ax_j = s.get_axes(B_block)
    s.bind(ax_i, thread_x)

    f = tvm.build(s.func, target="cuda")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((16,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1, 2))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)

    # Test 4
    s = tir.create_schedule(rowsum_allreduce)
    thread_x = tir.thread_axis((0, 16), "threadIdx.x")

    B_block = s.get_block("B")
    ax_ii, ax_i, ax_j = s.get_axes(B_block)

    B_rf = s.rfactor(ax_i, 0)
    ax_i_rf, ax_ii_rf, ax_j_rf = s.get_axes(B_rf)
    s.reorder(ax_ii_rf, ax_i_rf, ax_j_rf)
    ax_i_rf_o, ax_i_rf_i = s.split(ax_i_rf, factor=4)

    s.bind(ax_ii_rf, tir.thread_axis('blockIdx.x'))
    s.bind(ax_i_rf_o, tir.thread_axis('threadIdx.x'))

    B_rf_local = s.cache_write(B_rf, 0, 'local')
    s.compute_inline(B_rf)

    s.reverse_compute_at(B_block, ax_i_rf_o)

    f = tvm.build(s.func, target="cuda")
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
