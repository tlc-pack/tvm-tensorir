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
import pytest
import tvm
from tvm import tir
from tvm.script import ty

# pylint: disable=no-member,invalid-name,unused-variable


@tvm.script.tir
def elementwise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
        B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_with_seq(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    C = tir.alloc_buffer((128, 128, 128))
    for i, j in tir.grid(128, 128):
        for k in tir.serial(0, 128):
            with tir.block([128, 128, 128], "C") as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.reads([A[vi, vj, vk]])
                tir.writes([C[vi, vj, vk]])
                C[vi, vj, vk] = A[vi, vj, vk] * 2.0
        for k in tir.serial(0, 128):
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.reads([C[vi, vj, vk]])
                tir.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = C[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_with_anno(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for i, j in tir.grid(128, 128):
        for k in tir.serial(0, 128, annotations={"useless_annotation": True}):
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.reads([A[vi, vj, vk]])
                tir.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0

@tvm.script.tir
def elementwise_with_starting_point(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for i, j in tir.grid(128, 128):
        for k in tir.serial(10, 128):
            with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.reads([A[vi, vj, vk]])
                tir.writes([B[vi, vj, vk]])
                B[vi, vj, vk] = A[vi, vj, vk] * 2.0

@tvm.script.tir
def elementwise_fused(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128, 128))
    B = tir.match_buffer(b, (128, 128, 128))
    for fused in tir.serial(0, 2097152):
        with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
            tir.bind(vi, tir.floordiv(fused, 16384))
            tir.bind(vj, tir.floormod(tir.floordiv(fused, 128), 128))
            tir.bind(vk, tir.floormod(fused, 128))
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_split_case0(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128, 128])
    B = tir.match_buffer(b, [128, 128, 128])
    for i1, i2, i3, j1, j2, k1, k2 in tir.grid(2, 1, 64, 4, 32, 16, 8):
        with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
            tir.bind(vi, ((i1 * 64) + i3))
            tir.bind(vj, ((j1 * 32) + j2))
            tir.bind(vk, ((k1 * 8) + k2))
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0


@tvm.script.tir
def elementwise_split_case1(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128, 128])
    B = tir.match_buffer(b, [128, 128, 128])
    for i1, i2, i3, j1, j2, j3, k1, k2, k3 in tir.grid(2, 1, 64, 2, 1, 64, 2, 1, 64):
        with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
            tir.bind(vi, i1 * 64 + i3)
            tir.bind(vj, j1 * 64 + j3)
            tir.bind(vk, k1 * 64 + k3)
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0

@tvm.script.tir
def elementwise_split_with_predicate(a: ty.handle, b: ty.handle) -> None:
    B = tir.match_buffer(b, [128, 128, 128])
    A = tir.match_buffer(a, [128, 128, 128])
    for i0, i1, i2, j0, j1, k0, k1 in tir.grid(43, 1, 3, 1, 129, 129, 1):
        with tir.block([128, 128, 128], "B") as [vi, vj, vk]:
            tir.where(((((((i0 + i1)*3) + i2) < 128) and (((j0*129) + j1) < 128)) and ((k0 + k1) < 128)))
            tir.bind(vi, ((i0*3) + i2))
            tir.bind(vj, ((j0*127) + j1))
            tir.bind(vk, ((k0*2) + k1))
            tir.reads([A[vi, vj, vk]])
            tir.writes([B[vi, vj, vk]])
            B[vi, vj, vk] = A[vi, vj, vk] * 2.0

# pylint: enable=no-member,invalid-name,unused-variable


def test_fuse():
    sch = tir.Schedule(elementwise, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.fuse(i, j, k)
    tvm.ir.assert_structural_equal(elementwise_fused, sch.mod["main"])


def test_split():
    sch = tir.Schedule(elementwise, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[2, 1, 64])
    sch.split(j, factor=32)
    sch.split(k, nparts=16)
    tvm.ir.assert_structural_equal(elementwise_split_case0, sch.mod["main"])


def test_split_with_none_factor():
    sch = tir.Schedule(elementwise, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[None, 1, 64])
    sch.split(j, factors=[2, None, 64])
    sch.split(k, factors=[2, 1, None])
    tvm.ir.assert_structural_equal(elementwise_split_case1, sch.mod["main"])


# this test fails because of a bug in iter_affine_map.cc
def test_split_with_predicate():
    sch = tir.Schedule(elementwise, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    sch.split(i, factors=[None, 1, 3])
    sch.split(j, factor=129)
    sch.split(k, nparts=129)
    tvm.ir.assert_structural_equal(elementwise_split_with_predicate, sch.mod["main"])

def test_fuse_fail_not_only_child():
    sch = tir.Schedule(elementwise_with_seq, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)


def test_fuse_split_fail_with_annotation():
    sch = tir.Schedule(elementwise_with_anno, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(k, factor=10)

def test_fuse_split_fail_not_start_with_zero():
    sch = tir.Schedule(elementwise_with_anno, debug_mode=True)
    block_b = sch.get_block("B")
    i, j, k = sch.get_loops(block_b)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.fuse(j, k)
    with pytest.raises(tvm.tir.ScheduleError):
        sch.split(k, factor=10)


if __name__ == "__main__":
    test_fuse()
    test_split()
    test_split_with_none_factor()
    # test_split_with_predicate()
    test_fuse_fail_not_only_child()
    test_fuse_split_fail_with_annotation()
    test_fuse_split_fail_not_start_with_zero()
