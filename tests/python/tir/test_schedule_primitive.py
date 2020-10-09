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

import tvm
import util
from tvm import tir
from tvm.script import ty


@tvm.script.tir
def fused_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.buffer_allocate((128, 128))

    for i in range(0, 16384):
        with tir.block([128, 128], "B") as [vi, vj]:
            tir.bind(vi, i // 128)
            tir.bind(vj, i % 128)
            B[vi, vj] = A[vi, vj] * 2.0

    for j in range(0, 16384):
        with tir.block([128, 128], "C") as [vi, vj]:
            tir.bind(vi, j // 128)
            tir.bind(vj, j % 128)
            C[vi, vj] = B[vi, vj] + 1.0


def test_fuse():
    func = util.element_wise_stmt()

    # schedule
    s = tir.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(B)
    s.fuse(outer, inner)
    outer, inner = s.get_axes(C)
    s.fuse(outer, inner)

    mod = tvm.script.create_module({"fused_element_wise": fused_element_wise})
    fused_func = mod["fused_element_wise"]
    tvm.ir.assert_structural_equal(fused_func, s.func)
    assert s.validate_sref()


@tvm.script.tir
def split_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.buffer_allocate((128, 128))

    for io, ii, j in tir.grid(8, 16, 128):
        with tir.block([128, 128], "B") as [vi, vj]:
            tir.bind(vi, io * 16 + ii)
            tir.bind(vj, j)
            B[vi, vj] = A[vi, vj] * 2.0

    for i, jo, ji in tir.grid(128, 10, 13):
        with tir.block([128, 128], "C") as [vi, vj]:
            tir.where(jo * 13 + ji < 128)
            tir.bind(vi, i)
            tir.bind(vj, jo * 13 + ji)
            C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def split_fuse_element_wise(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.buffer_allocate((128, 128), "float32")
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "B") as [vi, vj]:
            tir.bind(vi, ((tir.floordiv(i, 16) * 16) + tir.floormod(i, 16)))
            tir.bind(vj, j)
            B[vi, vj] = (A[vi, vj] * tir.float32(2))
    for i, j in tir.grid(128, 130):
        with tir.block([128, 128], "C") as [vi, vj]:
            tir.where((((tir.floordiv(j, 13) * 13) + tir.floormod(j, 13)) < 128))
            tir.bind(vi, i)
            tir.bind(vj, ((tir.floordiv(j, 13) * 13) + tir.floormod(j, 13)))
            C[vi, vj] = (B[vi, vj] + tir.float32(1))


def test_split_fuse():
    func = util.element_wise_stmt()

    # schedule
    s = tir.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(B)
    s.split(outer, factor=16)
    outer, inner = s.get_axes(C)
    s.split(inner, nparts=10)

    mod = tvm.script.create_module({"split_element_wise": split_element_wise})
    split_func = mod["split_element_wise"]

    tvm.ir.assert_structural_equal(split_func, s.func)
    assert s.validate_sref()


@tvm.script.tir
def compute_at_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    B = tir.buffer_allocate((128, 128))
    for i in range(0, 128):
        for j in range(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                B[vi, vj] = A[vi, vj] * 2.0
        for j in range(0, 128):
            with tir.block([128, 128], "C") as [vi, vj]:
                C[vi, vj] = B[vi, vj] + 1.0


def test_compute_at():
    func = util.element_wise_stmt()

    # schedule
    s = tir.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(C)
    s.compute_at(B, outer)

    mod = tvm.script.create_module({"compute_at_element_wise": compute_at_element_wise})
    split_func = mod["compute_at_element_wise"]

    tvm.ir.assert_structural_equal(split_func, s.func)
    assert s.validate_sref()


@tvm.script.tir
def reverse_compute_at_element_wise(a: ty.handle, c: ty.handle) -> None:
    # function attr dict
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.buffer_allocate([128, 128], elem_offset=0, align=128, offset_factor=1)

    # body
    for i0_outer in range(0, 8):
        for i1_outer in range(0, 8):
            for i0_inner in range(0, 16):
                for i1_inner in range(0, 16):
                    with tir.block([128, 128], "B") as [vi, vj]:
                        tir.bind(vi, ((i0_outer*16) + i0_inner))
                        tir.bind(vj, ((i1_outer*16) + i1_inner))
                        B[vi, vj] = (A[vi, vj]*tir.float32(2))
                for ax1 in range(0, 16):
                    with tir.block([128, 128], "C") as [vi, vj]:
                        tir.bind(vi, ((i0_outer*16) + i0_inner))
                        tir.bind(vj, ((i1_outer*16) + ax1))
                        C[vi, vj] = (B[vi, vj] + tir.float32(1))


def test_reverse_compute_at():
    func = util.element_wise_stmt()

    # schedule
    s = tir.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    i, j = s.get_axes(B)
    i1, i2 = s.split(i, 16)
    j1, j2 = s.split(j, 16)
    s.reorder(i1, j1, i2, j2)
    s.reverse_compute_at(C, i2)

    tvm.ir.assert_structural_equal(reverse_compute_at_element_wise, s.func)
    assert s.validate_sref()


@tvm.script.tir
def predicate_fuse(b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (16, 16), "float32")
    B = tir.match_buffer(b, (16, 16), "float32")
    for i in range(0, 256):
        with tir.block([16, 16], "update") as [vi, vj]:
            tir.where((((tir.floormod(tir.floordiv(i, 4), 4) * 4) + tir.floormod(i, 4)) < 16))
            tir.bind(vi, tir.floordiv(tir.floordiv(i, 4), 4))
            tir.bind(vj, ((tir.floormod(tir.floordiv(i, 4), 4) * 4) + tir.floormod(i, 4)))
            C[vi, vj] = (B[vi, vj] + tir.float32(1))


def test_fuse_loop_sref():
    func = util.predicate_stmt()

    # schedule
    s = tir.create_schedule(func)
    update = s.get_block("update")
    i, j, k = s.get_axes(update)
    ij = s.fuse(i, j)
    s.fuse(ij, k)

    mod = tvm.script.create_module({"predicate_fuse": predicate_fuse})
    predicate_fuse_func = mod["predicate_fuse"]

    tvm.ir.assert_structural_equal(s.func, predicate_fuse_func)
    assert s.validate_sref()


@tvm.script.tir
def matmul_reorder(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.match_buffer(b, (128, 128), "float32")

    for i0, j0 in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)
    for k, i in tir.grid(128, 16384):
        with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
            tir.bind(vi, tir.floordiv(i, 128))
            tir.bind(vj, tir.floormod(i, 128))
            tir.bind(vk, k)
            C[vi, vj] = (C[vi, vj] + (A[vi, vk] * B[vj, vk]))


def test_reorder_normal():
    func = util.matmul_stmt()
    # schedule
    s = tir.create_schedule(func)
    update = s.get_block("update")
    i, j, k = s.get_axes(update)
    s.reorder(k, i)
    s.reorder(i, j)
    s.decompose_reduction(update, k)
    s.fuse(i, j)
    mod = tvm.script.create_module({"matmul_reorder": matmul_reorder})
    matmul_reorder_func = mod["matmul_reorder"]
    tvm.ir.assert_structural_equal(s.func, matmul_reorder_func)
    assert s.validate_sref()


@tvm.script.tir
def inline_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = A[vi, vj] * 2.0 + 1.0


def test_compute_inline():
    func = util.element_wise_stmt()

    # schedule
    s = tir.create_schedule(func)
    B = s.get_block("B")
    s.compute_inline(B)

    inlined_func = inline_element_wise

    tvm.ir.assert_structural_equal(inlined_func, s.func)
    assert s.validate_sref()


@tvm.script.tir
def element_wise_reverse_inline(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")

    with tir.block([], "root") as []:
        tir.reads([A[0:128, 0:128]])
        tir.writes([C[0:128, 0:128]])
        with tir.block([128, 128], "B") as [vi, vj]:
            C[vi, vj] = (A[vi, vj] * 2.0) + 1.0


def test_reverse_compute_inline():
    func = util.element_wise_stmt()

    # schedule
    s = tir.create_schedule(func)
    C = s.get_block("C")
    s.reverse_compute_inline(C)
    tvm.ir.assert_structural_equal(element_wise_reverse_inline, s.func)
    assert s.validate_sref()


@tvm.script.tir
def compute_at_case(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")

    B = tir.buffer_allocate((128, 128))
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "B0") as [vi, vj]:
            A[vi, vj] = 2.0
        for k in range(0, 128):
            with tir.block([128, 128], "B1") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                B[vi, vj] = A[vi, vj] * 2.0
            with tir.block([128, 128], "C") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                C[vi, vj] = B[vi, vj] * 2.0


def test_compute_at_fail():
    mod = tvm.script.create_module({"compute_at_case": compute_at_case})
    func = mod["compute_at_case"]
    s = tir.create_schedule(func)
    B1 = s.get_block("B1")
    C = s.get_block("C")
    i, j, k = s.get_axes(C)
    try:
        s.compute_at(C, j)
        assert False
    except:
        pass

    try:
        s.compute_at(B1, i)
        assert False
    except:
        pass


@tvm.script.tir
def matmul_reduction(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.match_buffer(b, (128, 128), "float32")

    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)
        for k in range(0, 128):
            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = (C[vi, vj] + (A[vi, vk] * B[vj, vk]))


def test_reduction():
    func = util.matmul_stmt()

    # schedule
    s = tir.create_schedule(func)
    update = s.get_block("update")
    i, j, k = s.get_axes(update)
    init = s.decompose_reduction(update, i)
    i, j_i = s.get_axes(init)
    s.split(j_i, 4)
    s.merge_reduction(init, update)
    s.decompose_reduction(update, k)

    mod = tvm.script.create_module({"matmul_reduction": matmul_reduction})
    matmul_reduction_func = mod["matmul_reduction"]

    tvm.ir.assert_structural_equal(s.func, matmul_reduction_func)
    assert s.validate_sref()


@tvm.script.tir
def cache_read(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.buffer_allocate((128, 128), "float32")
    AA = tir.buffer_allocate((128, 128), "float32", scope="local")
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "AA") as [vi, vj]:
            AA[vi, vj] = A[vi, vj]
    for i in range(0, 128):
        for j in range(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                B[vi, vj] = (AA[vi, vj] * tir.float32(2))
        for j in range(0, 128):
            with tir.block([128, 128], "C") as [vi, vj]:
                C[vi, vj] = (B[vi, vj] + tir.float32(1))


def test_cache_read():
    func = util.element_wise_stmt()
    buffer_a = func.buffer_map[func.params[0]]

    # schedule
    s = tir.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(C)
    s.compute_at(B, outer)
    AA = s.cache_read(buffer_a, 'local')

    mod = tvm.script.create_module({"cache_read": cache_read})
    cached_func = mod["cache_read"]

    tvm.ir.assert_structural_equal(cached_func, s.func)
    assert s.validate_sref()


@tvm.script.tir
def cache_write(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.buffer_allocate((128, 128), "float32")
    CC = tir.buffer_allocate((128, 128), "float32", scope="local")
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "B") as [vi, vj]:
            B[vi, vj] = (A[vi, vj] * tir.float32(2))
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "CC") as [vi, vj]:
            CC[vi, vj] = (B[vi, vj] + tir.float32(1))
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = CC[vi, vj]


def test_cache_write():
    func = util.element_wise_stmt()
    buffer_c = func.buffer_map[func.params[1]]

    # schedule
    s = tir.create_schedule(func)
    C = s.get_block(buffer_c)
    CC = s.cache_write(buffer_c, 'local')

    mod = tvm.script.create_module({"cache_write": cache_write})
    cached_func = mod["cache_write"]

    tvm.ir.assert_structural_equal(cached_func, s.func)
    assert s.validate_sref()


@tvm.script.tir
def blockize(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.buffer_allocate((128, 128), "float32")
    for i, j in tir.grid(8, 8):
        with tir.block([128, 128], "blockized_B") as [vi, vj]:
            tir.bind(vi, i * 16)
            tir.bind(vj, j * 16)
            for ii, jj in tir.grid(16, 16):
                with tir.block([128, 128], "B") as [vii, vjj]:
                    tir.bind(vii, vi + ii)
                    tir.bind(vjj, vj + jj)
                    B[vii, vjj] = (A[vii, vjj] * tir.float32(2))
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = (B[vi, vj] + tir.float32(1))


def test_blockize():
    func = util.element_wise_stmt()

    # schedule
    s = tir.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    x, y = s.get_axes(B)
    xo, xi = s.split(x, 16)
    yo, yi = s.split(y, 16)
    s.reorder(xo, yo, xi, yi)
    s.blockize(xi)

    mod = tvm.script.create_module({"blockize": blockize})
    blockized_func = mod["blockize"]

    tvm.ir.assert_structural_equal(blockized_func, s.func)
    assert s.validate_sref()


@tvm.script.tir
def test_func_cache_rw(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    B = tir.buffer_allocate((128, 128), "float32")
    D = tir.buffer_allocate((128, 128), "float32")

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "A") as [vi, vj, vk]:
        A[vi, vj] = A[vi, vj] + B[vi, vk] * C[vj, vk]

    with tir.block([128, 128], "D") as [vi, vj]:
        D[vi, vj] = A[vi, vj]


@tvm.script.tir
def test_func_cache_read(a: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    B = tir.buffer_allocate([128, 128], elem_offset=0, align=128, offset_factor=1)
    D = tir.buffer_allocate([128, 128], elem_offset=0, align=128, offset_factor=1)
    A_local = tir.buffer_allocate([128, 128], elem_offset=0, scope="local", align=128, offset_factor=1)
    with tir.block([128, 128, tir.reduce_axis(0, 128)], "A") as [vi, vj, vk]:
        A[vi, vj] = (A[vi, vj] + (B[vi, vk]*C[vj, vk]))
    with tir.block([128, 128], "") as [v0, v1]:
        A_local[v0, v1] = A[v0, v1]
    with tir.block([128, 128], "D") as [vi_1, vj_1]:
        D[vi_1, vj_1] = A_local[vi_1, vj_1]


@tvm.script.tir
def test_func_cache_write(a: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    B = tir.buffer_allocate([128, 128], elem_offset=0, align=128, offset_factor=1)
    D = tir.buffer_allocate([128, 128], elem_offset=0, align=128, offset_factor=1)
    A_local = tir.buffer_allocate([128, 128], elem_offset=0, scope="local", align=128, offset_factor=1)
    with tir.block([128, 128, tir.reduce_axis(0, 128)], "A") as [vi, vj, vk]:
        A_local[vi, vj] = (A_local[vi, vj] + (B[vi, vk]*C[vj, vk]))
    with tir.block([128, 128], "") as [v0, v1]:
        A[v0, v1] = A_local[v0, v1]
    with tir.block([128, 128], "D") as [vi_1, vj_1]:
        D[vi_1, vj_1] = A[vi_1, vj_1]


def test_cache_read_write():
    func = test_func_cache_rw

    # schedule cache read
    s = tir.create_schedule(func)
    blockA = tir.schedule.get_stmt(s.get_block("A"))
    A = blockA.writes[0].buffer
    s.cache_read(A, "local")

    tvm.ir.assert_structural_equal(test_func_cache_read, s.func)
    assert s.validate_sref()

    # schedule cache write
    s = tir.create_schedule(func)
    blockA = tir.schedule.get_stmt(s.get_block("A"))
    A = blockA.writes[0].buffer
    s.cache_write(A, "local")

    tvm.ir.assert_structural_equal(test_func_cache_write, s.func)
    assert s.validate_sref()


if __name__ == "__main__":
    test_fuse()
    test_split_fuse()
    test_fuse_loop_sref()
    test_reorder_normal()
    test_compute_at()
    test_reverse_compute_at()
    test_compute_inline()
    test_reverse_compute_inline()
    test_compute_at_fail()
    test_reduction()
    test_cache_read()
    test_cache_write()
    test_cache_read_write()
    test_blockize()
