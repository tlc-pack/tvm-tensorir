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

    C = tir.alloc_buffer([128, 128])
    D = tir.alloc_buffer([128, 128])
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

    C = tir.alloc_buffer([128, 128])
    with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + (A[vi, vk] * 2.0) * B[vj, vk]

    with tir.block([128, 128], "E") as [vi, vj]:
        E[vi, vj] = C[vi, vj] + 1.0


@tvm.script.tir
def rowsum_allreduce(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16, 16), "float32")
    B = tir.match_buffer(b, (16,), "float32")

    with tir.block([16, tir.reduce_axis(0, 16), tir.reduce_axis(0, 16)], "B") as [vii, vi, vj]:
        with tir.init():
            B[vii] = tir.float32(0)
        B[vii] = B[vii] + A[vii, vi, vj]


@tvm.script.tir
def rowsum_blockized(a: ty.handle, b: ty.handle) -> None:
    B = tir.match_buffer(b, [32, 4])
    A = tir.match_buffer(a, [32, 4, 128])
    for i0, i2_0 in tir.grid(32, 16):
        with tir.block([32, tir.reduce_axis(0, 16)], "blockized_B") as [io, ko]:
            tir.bind(io, i0)
            tir.bind(ko, i2_0)
            with tir.init():
                for i1 in tir.serial(0, 4):
                    with tir.block([4], "B_init") as [ii_init]:
                        tir.bind(ii_init, i1)
                        B[io, ii_init] = 0.0
            for i1_1, i2_1 in tir.grid(4, 8):
                with tir.block([4, tir.reduce_axis(0, 128)], "B") as [ii, k]:
                    tir.bind(ii, i1_1)
                    tir.bind(k, ko * 8 + i2_1)
                    B[io, ii] = B[io, ii] + A[io, ii, k]


@tvm.script.tir
def func(a: ty.handle, b: ty.handle) -> None:
    B = tir.match_buffer(b, [32, 4])
    A = tir.match_buffer(a, [32, 4, 128])
    for i0_init in tir.serial(0, 32):
        with tir.block([32], "blockized_B_init") as [io_init]:
            tir.bind(io_init, i0_init)
            for i1 in tir.serial(0, 4):
                with tir.block([4], "B_init") as [ii_init]:
                    tir.bind(ii_init, i1)
                    B[io_init, ii_init] = 0.0
    for i0, i2_0 in tir.grid(32, 16):
        with tir.block([32, tir.reduce_axis(0, 16)], "blockized_B_update") as [io, ko]:
            tir.bind(io, i0)
            tir.bind(ko, i2_0)
            for i1_1, i2_1 in tir.grid(4, 8):
                with tir.block([4, tir.reduce_axis(0, 128)], "B") as [ii, k]:
                    tir.bind(ii, i1_1)
                    tir.bind(k, ko * 8 + i2_1)
                    B[io, ii] = B[io, ii] + A[io, ii, k]


@tvm.script.tir
def non_multiple_allreduce(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (256, 250), "float32")
    B = tir.match_buffer(b, (256,), "float32")

    with tir.block([256, tir.reduce_axis(0, 250)], "B") as [i, k]:
        with tir.init():
            B[i] = tir.float32(0)
        B[i] = B[i] + A[i, k]


# fmt: on
# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,unexpected-keyword-arg


# pylint: disable=invalid-name


def test_reduction_roundtrip():
    func_rt = tvm.script.from_source(tvm.script.asscript(matmul))
    tvm.ir.assert_structural_equal(matmul, func_rt)


def test_reduction_decompose1():
    s = tir.Schedule(matmul, debug_mode=True)
    C = s.get_block("update")
    i, _, _ = s.get_loops(C)
    s.decompose_reduction(C, i)
    tvm.ir.assert_structural_equal(matmul_decompose0, s.mod["main"])


def test_reduction_decompose2():
    s = tir.Schedule(matmul, debug_mode=True)
    C = s.get_block("update")
    s.decompose_reduction(C, loop=None)
    tvm.ir.assert_structural_equal(matmul_decompose1, s.mod["main"])


def test_reduction_decompose3():
    s = tir.Schedule(rowsum_blockized, debug_mode=True)
    blockized_B = s.get_block("blockized_B")
    io, ko = s.get_loops(blockized_B)
    s.decompose_reduction(blockized_B, ko)


def test_reduction_merge():
    s = tir.Schedule(matmul_decompose0, debug_mode=True)
    init = s.get_block("init")
    update = s.get_block("update")
    s.merge_reduction(init, update)
    tvm.ir.assert_structural_equal(matmul, s.mod["main"])


def test_reduction_blockize():
    s = tir.Schedule(matmul, debug_mode=True)
    C = s.get_block("update")
    _, j, _ = s.get_loops(C)
    s.blockize(j)
    tvm.ir.assert_structural_equal(matmul_blockized, s.mod["main"])


def test_reduction_compute_inline():
    s = tir.Schedule(matmul_scale, debug_mode=True)
    D = s.get_block("D")
    s.compute_inline(D)
    tvm.ir.assert_structural_equal(s.mod["main"], matmul_scale_inline)


def test_reduction_allreduce_1():
    ctx = tvm.cuda(0)
    s = tir.Schedule(rowsum_allreduce, debug_mode=True)
    B_block = s.get_block("B")
    _, ax_i, ax_j = s.get_loops(B_block)
    s.bind(ax_j, "threadIdx.x")
    s.bind(ax_i, "threadIdx.y")

    f = tvm.build(s.mod["main"], target="cuda")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((16,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1, 2))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)


def test_reduction_allreduce_2():
    ctx = tvm.cuda(0)
    s = tir.Schedule(rowsum_allreduce, debug_mode=True)
    B_block = s.get_block("B")
    _, _, ax_j = s.get_loops(B_block)
    s.bind(ax_j, "threadIdx.x")

    f = tvm.build(s.mod["main"], target="cuda")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((16,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1, 2))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)


def test_reduction_allreduce_3():
    ctx = tvm.cuda(0)
    s = tir.Schedule(rowsum_allreduce, debug_mode=True)
    B_block = s.get_block("B")
    _, ax_i, _ = s.get_loops(B_block)
    s.bind(ax_i, "threadIdx.x")

    f = tvm.build(s.mod["main"], target="cuda")
    a_np = np.random.uniform(size=(16, 16, 16)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((16,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1, 2))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)


def test_reduction_allreduce_4():
    ctx = tvm.cuda(0)
    s = tir.Schedule(rowsum_allreduce, debug_mode=True)
    B_block = s.get_block("B")
    _, ax_i, _ = s.get_loops(B_block)
    B_rf = s.rfactor(ax_i, 0)
    ax_ii_rf, ax_i_rf, ax_j_rf = s.get_loops(B_rf)
    ax_i_rf_o, _ = s.split(ax_i_rf, factors=[None, 4])
    s.bind(ax_ii_rf, "blockIdx.x")
    s.bind(ax_i_rf_o, "threadIdx.x")
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


def test_reduction_allreduce_5():
    ctx = tvm.cuda(0)
    s = tir.Schedule(non_multiple_allreduce, debug_mode=True)
    B = s.get_block("B")
    _, k = s.get_loops(B)
    _, ki = s.split(k, factors=[None, 32])
    s.bind(ki, "threadIdx.x")

    f = tvm.build(s.mod["main"], target="cuda")
    a_np = np.random.uniform(size=(256, 250)).astype("float32")
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(np.zeros((256,), dtype="float32"), ctx)
    f(a, b)
    b_np = np.sum(a_np, axis=(1,))
    np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_reduction_roundtrip()
    test_reduction_decompose1()
    test_reduction_decompose2()
    test_reduction_decompose3()
    test_reduction_merge()
    test_reduction_blockize()
    test_reduction_compute_inline()
    test_reduction_allreduce_1()
    test_reduction_allreduce_2()
    test_reduction_allreduce_3()
    test_reduction_allreduce_4()
    test_reduction_allreduce_5()
