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
import sys

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import tir
from tvm.script import ty

# pylint: disable=no-member,invalid-name,unused-variable


@tvm.script.tir
def element_wise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))

    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0


@tvm.script.tir
def element_wise_parallelized(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    for i0 in tir.parallel(0, 128):
        for i1 in tir.serial(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i0)
                tir.bind(vj, i1)
                B[vi, vj] = A[vi, vj] * 2.0


@tvm.script.tir
def element_wise_i_bound(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    for i0 in tir.thread_binding(0, 128, thread="threadIdx.x"):
        for i1 in tir.serial(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i0)
                tir.bind(vj, i1)
                B[vi, vj] = A[vi, vj] * 2.0


@tvm.script.tir
def element_wise_compute_at_split(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))
    for i in tir.serial(0, 128):
        for j0 in tir.serial(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j0)
                B[vi, vj] = A[vi, vj] * 2.0
        for j1o, j1i in tir.grid(32, 4):
            with tir.block([128, 128], "C") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j1o * 4 + j1i)
                C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def element_wise_compute_at_split_vectorized(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))
    for i in tir.serial(0, 128):
        for j0 in tir.serial(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j0)
                B[vi, vj] = A[vi, vj] * 2.0
        for j1o in tir.serial(0, 32):
            for j1i in tir.vectorized(0, 4):
                with tir.block([128, 128], "C") as [vi, vj]:
                    tir.bind(vi, i)
                    tir.bind(vj, j1o * 4 + j1i)
                    C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def element_wise_compute_at_split_j0_j1o_bound(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))
    for i in tir.serial(0, 128):
        for j0 in tir.thread_binding(0, 128, thread="threadIdx.x"):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j0)
                B[vi, vj] = A[vi, vj] * 2.0
        for j1o in tir.thread_binding(0, 32, thread="threadIdx.x"):
            for j1i in tir.serial(0, 4):
                with tir.block([128, 128], "C") as [vi, vj]:
                    tir.bind(vi, i)
                    tir.bind(vj, j1o * 4 + j1i)
                    C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
def rowsum(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))

    with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
        with tir.init():
            B[vi] = 0.0
        B[vi] = B[vi] + A[vi, vk]


@tvm.script.tir
def rowsum_unrolled(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))
    for i0 in tir.unroll(0, 128):
        for i1 in tir.serial(0, 128):
            with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
                tir.bind(vi, i0)
                tir.bind(vk, i1)
                with tir.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]


@tvm.script.tir
def rowsum_not_quasi_affine(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))

    for i, k in tir.grid(128, 16):
        with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
            tir.bind(vi, i)
            tir.bind(vk, tir.floordiv(k * k, 2))
            with tir.init():
                B[vi] = 0.0
            B[vi] = B[vi] + A[vi, vk]


@tvm.script.tir
def rowsum_not_compact_data_flow(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))

    with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
        with tir.init():
            B[vk] = 0.0
        B[vk] = B[vk] + A[vi, vk]


@tvm.script.tir
def rowsum_cross_thread_reduction(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128,))
    for i0 in tir.serial(0, 128):
        for i1 in tir.thread_binding(0, 128, thread="threadIdx.x"):
            with tir.block([128, tir.reduce_axis(0, 128)], "B") as [vi, vk]:
                tir.bind(vi, i0)
                tir.bind(vk, i1)
                with tir.init():
                    B[vi] = 0.0
                B[vi] = B[vi] + A[vi, vk]


# pylint: enable=no-member,invalid-name,unused-variable


def test_parallel():
    s = tir.Schedule(element_wise, debug_mode=True)
    i, _ = s.get_loops(s.get_block("B"))
    s.parallel(i)
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_parallelized)


def test_parallel_reduction_block_iter():
    s = tir.Schedule(matmul, debug_mode=True)
    _, _, k = s.get_loops(s.get_block("C"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.parallel(k)


def test_parallel_not_quasi_affine():
    s = tir.Schedule(rowsum_not_quasi_affine, debug_mode=True)
    i, _ = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.parallel(i)


def test_parallel_not_compact_data_flow():
    s = tir.Schedule(rowsum_not_compact_data_flow, debug_mode=True)
    i, _ = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.parallel(i)


def test_vectorize():
    s = tir.Schedule(element_wise_compute_at_split, debug_mode=True)
    _, _, j1i = s.get_loops(s.get_block("C"))
    s.vectorize(j1i)
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_compute_at_split_vectorized)


def test_unroll():
    s = tir.Schedule(rowsum, debug_mode=True)
    i, _ = s.get_loops(s.get_block("B"))
    s.unroll(i)
    tvm.ir.assert_structural_equal(s.mod["main"], rowsum_unrolled)


def test_bind1():
    s = tir.Schedule(element_wise, debug_mode=True)
    i, _ = s.get_loops(s.get_block("B"))
    s.bind(i, "threadIdx.x")
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_i_bound)


def test_bind2():
    s = tir.Schedule(element_wise_compute_at_split, debug_mode=True)
    _, j0 = s.get_loops(s.get_block("B"))
    _, j1o, _ = s.get_loops(s.get_block("C"))
    s.bind(j0, "threadIdx.x")
    s.bind(j1o, "threadIdx.x")
    tvm.ir.assert_structural_equal(s.mod["main"], element_wise_compute_at_split_j0_j1o_bound)


def test_bind_cross_thread_reduction():
    s = tir.Schedule(rowsum, debug_mode=True)
    _, k = s.get_loops(s.get_block("B"))
    s.bind(k, "threadIdx.x")
    tvm.ir.assert_structural_equal(s.mod["main"], rowsum_cross_thread_reduction)


def test_bind_not_cross_thread_reduction():
    s = tir.Schedule(rowsum, debug_mode=True)
    _, k = s.get_loops(s.get_block("B"))
    with pytest.raises(tvm.tir.ScheduleError):
        s.bind(k, "blockIdx.x")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
