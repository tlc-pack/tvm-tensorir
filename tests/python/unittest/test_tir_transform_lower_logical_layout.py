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


def _check(original, transformed):
    mod = tvm.IRModule.from_expr(original)
    mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tvm.tir.transform.LowerLogicalLayout()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed)


def _check_fail(original):
    mod = tvm.IRModule.from_expr(original)
    with pytest.raises(tvm.TVMError):
        mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
        mod = tvm.tir.transform.LowerLogicalLayout()(mod)


def lower_warp_logical_layout(i, j):
    return tir.floormod(i, 16) * 4 + tir.floormod(j, 4), tir.floordiv(i, 16), tir.floordiv(j, 4)


def lower_non_bijective_layout(i, j):
    return (i + j,)


tir.LogicalLayout.register('warp.logical', lower_warp_logical_layout)
tir.LogicalLayout.register('warp.non_bijective', lower_non_bijective_layout)


@tvm.script.tir
def cache_read(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [32, 32], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [32, 32], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        A_warp_logical = tir.alloc_buffer([32, 32], scope="warp.logical")
        for ax0, ax1 in tir.grid(32, 32):
            with tir.block([32, 32], "A_warp.logical") as [v0, v1]:
                tir.bind(v0, ax0)
                tir.bind(v1, ax1)
                tir.reads([A[v0, v1]])
                tir.writes([A_warp_logical[v0, v1]])
                A_warp_logical[v0, v1] = A[v0, v1]
        for i0, i1 in tir.grid(32, 32):
            with tir.block([32, 32], "C") as [i, j]:
                tir.bind(i, i0)
                tir.bind(j, i1)
                tir.reads([A_warp_logical[i, j]])
                tir.writes([C[i, j]])
                C[i, j] = (A_warp_logical[i, j] + tir.float32(1))


@tvm.script.tir
def transformed_cache_read(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [32, 32], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [32, 32], elem_offset=0, align=128, offset_factor=1)
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        A_warp_logical = tir.alloc_buffer([64, 2, 8], scope="warp")

        for v0 in tir.thread_binding(0, 64, thread = "threadIdx.x"):
            for v1, v2 in tir.grid(2, 8):
                with tir.block([], "A_warp.logical"):
                    tir.reads([A[((v1*16) + tir.floordiv(v0, 4)), ((v2*4) + tir.floormod(v0, 4))]])
                    tir.writes([A_warp_logical[v0, v1, v2]])
                    A_warp_logical[v0, v1, v2] = A[((v1*16) + tir.floordiv(v0, 4)), ((v2*4) + tir.floormod(v0, 4))]
        for i0, i1 in tir.grid(32, 32):
            with tir.block([], "C"):
                tir.reads([A_warp_logical[((tir.floormod(i0, 16)*4) + tir.floormod(i1, 4)), tir.floordiv(i0, 16), tir.floordiv(i1, 4)]])
                tir.writes([C[i0, i1]])
                C[i0, i1] = (A_warp_logical[((tir.floormod(i0, 16)*4) + tir.floormod(i1, 4)), tir.floordiv(i0, 16), tir.floordiv(i1, 4)] + tir.float32(1))


@tvm.script.tir
def cache_read_extra_dim(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [4, 32, 32], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [4, 32, 32], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        A_warp_logical = tir.alloc_buffer([4, 32, 32], scope="warp.logical")
        for ax0, ax1, ax2 in tir.grid(4, 32, 32):
            with tir.block([4, 32, 32], "A_warp.logical") as [v0, v1, v2]:
                tir.bind(v0, ax0)
                tir.bind(v1, ax1)
                tir.bind(v2, ax2)
                tir.reads([A[v0, v1, v2]])
                tir.writes([A_warp_logical[v0, v1, v2]])
                A_warp_logical[v0, v1, v2] = A[v0, v1, v2]
        for i0, i1, i2 in tir.grid(4, 32, 32):
            with tir.block([4, 32, 32], "C") as [i, j, k]:
                tir.bind(i, i0)
                tir.bind(j, i1)
                tir.bind(k, i2)
                tir.reads([A_warp_logical[i, j, k]])
                tir.writes([C[i, j, k]])
                C[i, j, k] = (A_warp_logical[i, j, k] + tir.float32(1))


@tvm.script.tir
def transformed_cache_read_extra_dim(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [4, 32, 32], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [4, 32, 32], elem_offset=0, align=128, offset_factor=1)
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        A_warp_logical = tir.alloc_buffer([4, 64, 2, 8], scope="warp")

        for ax0 in tir.grid(4):
            for v0 in tir.thread_binding(0, 64, thread = "threadIdx.x"):
                for v1, v2 in tir.grid(2, 8):
                    with tir.block([], "A_warp.logical"):
                        tir.reads([A[ax0, ((v1*16) + tir.floordiv(v0, 4)), ((v2*4) + tir.floormod(v0, 4))]])
                        tir.writes([A_warp_logical[ax0, v0, v1, v2]])
                        A_warp_logical[ax0, v0, v1, v2] = A[ax0, ((v1*16) + tir.floordiv(v0, 4)), ((v2*4) + tir.floormod(v0, 4))]
        for i0, i1, i2 in tir.grid(4, 32, 32):
            with tir.block([], "C"):
                tir.reads([A_warp_logical[i0, ((tir.floormod(i1, 16)*4) + tir.floormod(i2, 4)), tir.floordiv(i1, 16), tir.floordiv(i2, 4)]])
                tir.writes([C[i0, i1, i2]])
                C[i0, i1, i2] = (A_warp_logical[i0, ((tir.floormod(i1, 16)*4) + tir.floormod(i2, 4)), tir.floordiv(i1, 16), tir.floordiv(i2, 4)] + tir.float32(1))


@tvm.script.tir
def cache_read_non_bijective(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [32, 32], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [32, 32], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        A_warp_logical = tir.alloc_buffer([32, 32], scope="warp.non_bijective")
        for ax0, ax1 in tir.grid(32, 32):
            with tir.block([32, 32], "A_warp.logical") as [v0, v1]:
                tir.bind(v0, ax0)
                tir.bind(v1, ax1)
                tir.reads([A[v0, v1]])
                tir.writes([A_warp_logical[v0, v1]])
                A_warp_logical[v0, v1] = A[v0, v1]
        for i0, i1 in tir.grid(32, 32):
            with tir.block([32, 32], "C") as [i, j]:
                tir.bind(i, i0)
                tir.bind(j, i1)
                tir.reads([A_warp_logical[i, j]])
                tir.writes([C[i, j]])
                C[i, j] = (A_warp_logical[i, j] + tir.float32(1))


@tvm.script.tir
def cache_write(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [32, 32], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [32, 32], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        C_warp_logical = tir.alloc_buffer([32, 32], scope="warp.logical")
        for ax0, ax1 in tir.grid(32, 32):
            with tir.block([32, 32], "A_warp.logical") as [v0, v1]:
                tir.bind(v0, ax0)
                tir.bind(v1, ax1)
                tir.reads([A[v0, v1]])
                tir.writes([C_warp_logical[v0, v1]])
                C_warp_logical[v0, v1] = A[v0, v1] + tir.float32(1)
        for i0, i1 in tir.grid(32, 32):
            with tir.block([32, 32], "C") as [i, j]:
                tir.bind(i, i0)
                tir.bind(j, i1)
                tir.reads([C_warp_logical[i, j]])
                tir.writes([C[i, j]])
                C[i, j] = C_warp_logical[i, j]


@tvm.script.tir
def transformed_cache_write(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [32, 32], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [32, 32], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        C_warp_logical = tir.alloc_buffer([64, 2, 8], scope="warp.logical")
        for v0 in tir.thread_binding(0, 64, thread = "threadIdx.x"):
            for v1, v2 in tir.grid(2, 8):
                with tir.block([], "C_warp.logical"):
                    tir.reads([A[((v1*16) + tir.floordiv(v0, 4)), ((v2*4) + tir.floormod(v0, 4))]])
                    tir.writes([C_warp_logical[v0, v1, v2]])
                    C_warp_logical[v0, v1, v2] = A[((v1*16) + tir.floordiv(v0, 4)), ((v2*4) + tir.floormod(v0, 4))] + tir.float32(1)
        for i, j in tir.grid(32, 32):
            with tir.block([], "C"):
                tir.reads([C_warp_logical[tir.floormod(i, 16)* 4 + tir.floormod(j, 4), tir.floordiv(i, 16), tir.floordiv(j, 4)]])
                tir.writes([C[i, j]])
                C[i, j] = C_warp_logical[tir.floormod(i, 16)* 4 + tir.floormod(j, 4), tir.floordiv(i, 16), tir.floordiv(j, 4)]


def test_cache_read():
    _check(cache_read, transformed_cache_read)


def test_cache_read_extra_dim():
    _check(cache_read_extra_dim, transformed_cache_read_extra_dim)


def test_non_bijective():
    _check_fail(cache_read_non_bijective)


def test_cache_write():
    _check(cache_write, transformed_cache_write)


if __name__ == "__main__":
    test_cache_read()
    test_cache_read_extra_dim()
    test_non_bijective()
    test_cache_write()
