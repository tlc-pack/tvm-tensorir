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
def simple1(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (32,))
    B = tir.match_buffer(b, (32,))
    B_shared = tir.alloc_buffer((32,), scope="shared")
    for tx in tir.thread_binding(0, 4, thread="threadIdx.x"):
        for i in tir.serial(0, 8):
            with tir.block(
                [
                    32,
                ],
                "B_shared",
            ) as vi:
                tir.bind(vi, tx * 8 + i)
                B_shared[vi] = A[vi] + 1.0
    for j in tir.serial(0, 32):
        with tir.block(
            [
                32,
            ],
            "B",
        ) as vi:
            tir.bind(vi, j)
            B[vi] = B_shared[vi]


@tvm.script.tir
def after_promotion(a: ty.handle, b: ty.handle) -> None:
    B = tir.match_buffer(b, [32], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [32], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        B_shared = tir.alloc_buffer(
            [4, 8], elem_offset=0, scope="shared", align=128, offset_factor=1
        )
        for tx in tir.thread_binding(0, 4, thread="threadIdx.x"):
            for i in tir.serial(0, 8):
                with tir.block([32], "B_shared") as [vi]:
                    tir.bind(vi, ((tx * 8) + i))
                    tir.reads([A[vi]])
                    tir.writes(
                        [B_shared[tir.floormod(tir.floordiv(vi, 8), 4), tir.floormod(vi, 8)]]
                    )
                    B_shared[tir.floormod(tir.floordiv(vi, 8), 4), tir.floormod(vi, 8)] = A[
                        vi
                    ] + tir.float32(1)
        for j in tir.serial(0, 32):
            with tir.block([32], "B") as [vi_1]:
                tir.bind(vi_1, j)
                tir.reads([B_shared[tir.floormod(tir.floordiv(vi_1, 8), 4), tir.floormod(vi_1, 8)]])
                tir.writes([B[vi_1]])
                B[vi_1] = B_shared[tir.floormod(tir.floordiv(vi_1, 8), 4), tir.floormod(vi_1, 8)]


@tvm.script.tir
def simple2(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024))
    B = tir.match_buffer(b, (1024, 1024))
    B_shared = tir.alloc_buffer((1024, 1024), scope="shared")
    for bx in tir.thread_binding(0, 8, thread="blockIdx.x"):
        for by in tir.thread_binding(0, 8, thread="blockIdx.y"):
            for ty in tir.thread_binding(0, 8, thread="threadIdx.y"):
                for ax0_0 in tir.serial(0, 4):
                    for ax1_0 in tir.serial(0, 2):
                        for i, j in tir.grid(16, 16):
                            with tir.block([1024, 1024], "B_shared") as [v0, v1]:
                                tir.bind(v0, (bx * 8 + tir.floordiv(ty, 4) * 4 + ax0_0) * 16 + i)
                                tir.bind(v1, (by * 8 + tir.floormod(ty, 4) * 2 + ax1_0) * 16 + j)
                                B[v0, v1] = A[v0, v1]
    # for i, j in tir.grid(1024, 1024):
    #     with tir.block([1024, 1024], "B") as [vi, vj]:
    #         B[vi, vj] = B_shared[vi, vj]

    write_at(B_shared, j)


@tvm.script.tir
def complex(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 32))
    A_shared = tir.alloc_buffer((32, 16))

    for i in tir.serial(0, 2):
        for ty in tir.thread_binding(0, 4, thread="threadIdx.y"):
            for tx in tir.thread_binding(0, 32, thread="threadIdx.x"):
                for vec in tir.vectorized(0, 2):
                    with tir.block([32, 16], "transpose") as [v0, v1]:
                        tir.bind(
                            v0,
                            (
                                tir.floormod(
                                    tir.floordiv(((((((i * 4) + ty) * 32) + tx) * 2) + vec), 16), 32
                                )
                            ),
                        )
                        tir.bind(v1, (tir.floormod(((((((i * 4) + ty) * 32) + tx) * 2) + vec), 16)))
                        A_shared[v1, v0] = A[v0, v1]


@tvm.script.tir
def simple_with_offset(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (64,))
    B = tir.match_buffer(b, (64,))
    B_shared = tir.alloc_buffer((32,), scope="shared")
    for tx in tir.thread_binding(0, 4, thread="threadIdx.x"):
        for i in tir.serial(0, 8):
            with tir.block(
                [
                    32,
                ],
                "B_shared",
            ) as vi:
                tir.bind(vi, tx * 8 + i)
                B_shared[vi + 32] = A[vi] + 1.0
    for j in tir.serial(0, 32):
        with tir.block(
            [
                32,
            ],
            "B",
        ) as vi:
            tir.bind(vi, j)
            B[vi] = B_shared[vi + 32]


def test_simple():
    sch = tir.Schedule(simple1, debug_mode=True)
    block_b_shared = sch.get_block("B_shared")
    sch.promote_rank(block_b_shared, 0)
    print(tvm.script.asscript(sch.mod["main"]))


def test_simple2():
    sch = tir.Schedule(simple2, debug_mode=True)
    block_b_shared = sch.get_block("B_shared")
    sch.promote_rank(block_b_shared, 0)
    print(tvm.script.asscript(sch.mod["main"]))


def test_complex_pattern():
    sch = tir.Schedule(complex, debug_mode=True)
    block_transpose = sch.get_block("transpose")
    sch.promote_rank(block_transpose, 0)
    print(tvm.script.asscript(sch.mod["main"]))


def test_simple_with_offset():
    sch = tir.Schedule(simple_with_offset, debug_mode=True)
    block_b_shared = sch.get_block("B_shared")
    sch.promote_rank(block_b_shared, 0)
    print(tvm.script.asscript(sch.mod["main"]))


if __name__ == "__main__":
    test_simple()
    # test_simple2()
    # test_complex_pattern()
    # test_simple_with_offset()
