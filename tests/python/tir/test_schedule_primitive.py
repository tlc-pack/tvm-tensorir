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
from tvm.tir.ir_pass import AssertEqual


@tvm.tir.hybrid.script
def fused_element_wise(a, c):
    A = buffer_bind(a, (128, 128))
    C = buffer_bind(c, (128, 128))

    with block({}, A[0: 128, 0: 128], C[0: 128, 0: 128], name="root"):
        B = buffer_allocate((128, 128))

        for i in range(0, 128 * 128):
            with block({vi(0, 128): i // 128, vj(0, 128): i % 128},
                       reads=A[vi: vi + 1, vj: vj + 1], writes=B[vi: vi + 1, vj: vj + 1], name="B"):
                B[vi, vj] = A[vi, vj] * 2

        for j in range(0, 128 * 128):
            with block({vi(0, 128): j // 128, vj(0, 128): j % 128},
                       reads=B[vi: vi + 1, vj: vj + 1], writes=C[vi: vi + 1, vj: vj + 1], name="C"):
                C[vi, vj] = B[vi, vj] + 1


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

    mod = tvm.tir.hybrid.create_module([fused_element_wise])
    fused_func = mod["fused_element_wise"]
    assert AssertEqual(fused_func, s.func)
    assert s.validate_sref()


@tvm.tir.hybrid.script
def split_element_wise(a, c):
    A = buffer_bind(a, (128, 128))
    C = buffer_bind(c, (128, 128))

    with block({}, A[0: 128, 0: 128], C[0: 128, 0: 128], name="root"):
        B = buffer_allocate((128, 128))

        for io in range(0, 8):
            for ii in range(0, 16):
                for j in range(0, 128):
                    with block({vi(0, 128): io * 16 + ii, vj(0, 128): j},
                               reads=A[vi: vi + 1, vj: vj + 1], writes=B[vi: vi + 1, vj: vj + 1],
                               name="B"):
                        B[vi, vj] = A[vi, vj] * 2

        for i in range(0, 128):
            for jo in range(0, 10):
                for ji in range(0, 13):
                    with block({vi(0, 128): i, vj(0, 128): jo * 13 + ji},
                               reads=B[vi: vi + 1, vj: vj + 1], writes=C[vi: vi + 1, vj: vj + 1],
                               predicate=jo * 13 + ji < 128, name="C"):
                        C[vi, vj] = B[vi, vj] + 1


@tvm.tir.hybrid.script
def split_fuse_element_wise(a, c):
    C = buffer_bind(c, (128, 128), "float32")
    A = buffer_bind(a, (128, 128), "float32")
    with block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128]], name="root"):
        B = buffer_allocate((128, 128), "float32", "")
        for i in range(0, 128):
            for j in range(0, 128):
                with block({vi(0, 128): ((floordiv(i, 16) * 16) + floormod(i, 16)), vj(0, 128): j},
                           writes=[B[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[A[vi:(vi + 1), vj:(vj + 1)]], name="B"):
                    B[vi, vj] = (A[vi, vj] * float32(2))
        for i in range(0, 128):
            for j in range(0, 130):
                with block({vi(0, 128): i, vj(0, 128): ((floordiv(j, 13) * 13) + floormod(j, 13))},
                           writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[B[vi:(vi + 1), vj:(vj + 1)]],
                           predicate=(((floordiv(j, 13) * 13) + floormod(j, 13)) < 128), name="C"):
                    C[vi, vj] = (B[vi, vj] + float32(1))


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

    mod = tvm.tir.hybrid.create_module([split_element_wise])
    split_func = mod["split_element_wise"]

    assert AssertEqual(split_func, s.func)
    assert s.validate_sref()


@tvm.tir.hybrid.script
def compute_at_element_wise(a, c):
    A = buffer_bind(a, (128, 128))
    C = buffer_bind(c, (128, 128))

    with block({}, A[0: 128, 0: 128], C[0: 128, 0: 128], name="root"):
        B = buffer_allocate((128, 128))
        for i in range(0, 128):
            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j},
                           reads=A[vi: vi + 1, vj: vj + 1], writes=B[vi: vi + 1, vj: vj + 1],
                           name="B"):
                    B[vi, vj] = A[vi, vj] * 2
            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j},
                           reads=B[vi: vi + 1, vj: vj + 1], writes=C[vi: vi + 1, vj: vj + 1],
                           name="C"):
                    C[vi, vj] = B[vi, vj] + 1


def test_compute_at():
    func = util.element_wise_stmt()

    # schedule
    s = tir.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(C)
    s.compute_at(B, outer)

    mod = tvm.tir.hybrid.create_module([compute_at_element_wise])
    split_func = mod["compute_at_element_wise"]

    assert AssertEqual(split_func, s.func)
    assert s.validate_sref()


@tvm.tir.hybrid.script
def predicate_fuse(b, c):
    C = buffer_bind(c, (16, 16), "float32")
    B = buffer_bind(b, (16, 16), "float32")
    with block({}, writes=[], reads=[], name="root"):
        for i in range(0, 256):
            with block({vi(0, 16): floordiv(floordiv(i, 4), 4),
                        vj(0, 16): ((floormod(floordiv(i, 4), 4) * 4) + floormod(i, 4))},
                       writes=[C[vi:(vi + 1), vj:(vj + 1)]], reads=[B[vi:(vi + 1), vj:(vj + 1)]],
                       predicate=(((floormod(floordiv(i, 4), 4) * 4) + floormod(i, 4)) < 16),
                       name="update"):
                C[vi, vj] = (B[vi, vj] + float32(1))


def test_fuse_loop_sref():
    func = util.predicate_stmt()

    # schedule
    s = tir.create_schedule(func)
    update = s.get_block("update")
    i, j, k = s.get_axes(update)
    ij = s.fuse(i, j)
    s.fuse(ij, k)

    mod = tvm.tir.hybrid.create_module([predicate_fuse])
    predicate_fuse_func = mod["predicate_fuse"]

    assert AssertEqual(s.func, predicate_fuse_func)
    assert s.validate_sref()


@tvm.tir.hybrid.script
def matmul_reorder(a, b, c):
    C = buffer_bind(c, (128, 128), "float32")
    A = buffer_bind(a, (128, 128), "float32")
    B = buffer_bind(b, (128, 128), "float32")
    with block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128], B[0:128, 0:128]], name="root"):
        for i0 in range(0, 128):
            for j0 in range(0, 128):
                with block({vi(0, 128): i0, vj(0, 128): j0}, writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[], name="init"):
                    C[vi, vj] = float32(0)
        for k in range(0, 128):
            for i in range(0, 16384):
                with block({vi(0, 128): floordiv(i, 128), vj(0, 128): floormod(i, 128),
                            vk(0, 128, iter_type="reduce"): k},
                           writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)],
                                  B[vj:(vj + 1), vk:(vk + 1)]], name="update"):
                    C[vi, vj] = (C[vi, vj] + (A[vi, vk] * B[vj, vk]))


def test_reorder_normal():
    func = util.matmul_stmt()
    # schedule
    s = tir.create_schedule(func)
    update = s.get_block("update")
    i, j, k = s.get_axes(update)
    s.reorder(k, i)
    s.reorder(i, j)
    s.fuse(i, j)
    mod = tvm.tir.hybrid.create_module([matmul_reorder])
    matmul_reorder_func = mod["matmul_reorder"]

    assert AssertEqual(s.func, matmul_reorder_func)
    assert s.validate_sref()


@tvm.tir.hybrid.script
def matmul_reorder(a, b, c):
    C = buffer_bind(c, (128, 128), "float32")
    A = buffer_bind(a, (128, 128), "float32")
    B = buffer_bind(b, (128, 128), "float32")
    with block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128], B[0:128, 0:128]], name="root"):
        for i0 in range(0, 128):
            for j0 in range(0, 128):
                with block({vi(0, 128): i0, vj(0, 128): j0}, writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[], name="init"):
                    C[vi, vj] = float32(0)
        for k in range(0, 128):
            for i in range(0, 16384):
                with block({vi(0, 128): floordiv(i, 128), vj(0, 128): floormod(i, 128),
                            vk(0, 128, iter_type="reduce"): k},
                           writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[C[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)],
                                  B[vj:(vj + 1), vk:(vk + 1)]], name="update"):
                    C[vi, vj] = (C[vi, vj] + (A[vi, vk] * B[vj, vk]))


@tvm.tir.hybrid.script
def compute_at_case(a, c):
    A = buffer_bind(a, (128, 128), "float32")
    C = buffer_bind(c, (128, 128), "float32")

    with block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128]], name="root"):
        B = buffer_allocate((128, 128))
        for i in range(0, 128):
            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j},
                           reads=[], writes=A[vi: vi + 1, vj: vj + 1], name="B0"):
                    A[vi, vj] = 2
                for k in range(0, 128):
                    with block({vi(0, 128): i, vj(0, 128): j},
                               reads=A[vi: vi + 1, vj: vj + 1], writes=B[vi: vi + 1, vj: vj + 1],
                               name="B1"):
                        B[vi, vj] = A[vi, vj] * 2
                    with block({vi(0, 128): i, vj(0, 128): j},
                               reads=B[vi: vi + 1, vj: vj + 1], writes=C[vi: vi + 1, vj: vj + 1],
                               name="C"):
                        C[vi, vj] = B[vi, vj] * 2


def test_compute_at_fail():
    mod = tvm.tir.hybrid.create_module([compute_at_case])
    func = mod["compute_at_case"]
    s = tir.create_schedule(func)
    B1 = s.get_block("B1")
    C = s.get_block("C")
    i, j, k = s.get_axes(C)
    try:
        s.compute_at(C, j)
        assert False
    except tvm._ffi.base.TVMError as e:
        assert str(e).split(':')[-1].strip() == "Can not compute_at an output block"

    try:
        s.compute_at(B1, i)
        assert False
    except tvm._ffi.base.TVMError as e:
        assert str(e).split(':')[-1].strip() == "Cannot satisfy dependency"

@tvm.tir.hybrid.script
def cache_read(a, c):
    C = buffer_bind(c, (128, 128), "float32")
    A = buffer_bind(a, (128, 128), "float32")
    with block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128]], name="root"):
        B = buffer_allocate((128, 128), "float32", "")
        AA = buffer_allocate((128, 128), "float32", "local")
        for i in range(0, 128):
            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j},
                           writes=[AA[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[A[vi:(vi + 1), vj:(vj + 1)]]):
                    AA[vi, vj] = A[vi, vj]
        for i in range(0, 128):
            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j},
                           writes=[B[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[AA[vi:(vi + 1), vj:(vj + 1)]], name="B"):
                    B[vi, vj] = (AA[vi, vj] * float32(2))
        for i in range(0, 128):
            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j},
                           writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[B[vi:(vi + 1), vj:(vj + 1)]], name="C"):
                    C[vi, vj] = (B[vi, vj] + float32(1))


def test_cache_read():
    func = util.element_wise_stmt()
    buffer_a = func.buffer_map[func.params[0]]

    # schedule
    s = tir.create_schedule(func)
    AA = s.cache_read(buffer_a, 'local')

    mod = tvm.tir.hybrid.create_module([cache_read])
    cached_func = mod["cache_read"]

    assert AssertEqual(cached_func, s.func)
    assert s.validate_sref()


@tvm.tir.hybrid.script
def cache_write(a, c):
    C = buffer_bind(c, (128, 128), "float32")
    A = buffer_bind(a, (128, 128), "float32")
    with block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128]], name="root"):
        B = buffer_allocate((128, 128), "float32", "")
        CC = buffer_allocate((128, 128), "float32", "local")
        for i in range(0, 128):
            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j},
                           writes=[B[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[A[vi:(vi + 1), vj:(vj + 1)]], name="B"):
                    B[vi, vj] = (A[vi, vj] * float32(2))
        for i in range(0, 128):
            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j},
                           writes=[CC[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[B[vi:(vi + 1), vj:(vj + 1)]], name="C"):
                    CC[vi, vj] = (B[vi, vj] + float32(1))
        for i in range(0, 128):
            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j},
                           writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[CC[vi:(vi + 1), vj:(vj + 1)]]):
                    C[vi, vj] = CC[vi, vj]


def test_cache_write():
    func = util.element_wise_stmt()
    buffer_c = func.buffer_map[func.params[1]]

    # schedule
    s = tir.create_schedule(func)
    C = s.get_block(buffer_c)
    CC = s.cache_write(buffer_c, 'local')

    mod = tvm.tir.hybrid.create_module([cache_write])
    cached_func = mod["cache_write"]

    assert AssertEqual(cached_func, s.func)
    assert s.validate_sref()


if __name__ == "__main__":
    test_fuse()
    test_split_fuse()
    test_fuse_loop_sref()
    test_reorder_normal()
    test_compute_at()
    test_compute_at_fail()
    test_cache_read()
    test_cache_write()
