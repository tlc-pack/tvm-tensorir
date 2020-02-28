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
from tvm.ir_pass import Equal, AssertEqual


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
    assert Equal(fused_func, s.func)


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

    assert Equal(split_func, s.func)

    io, ii, j = s.get_axes(B)
    s.fuse(io, ii)
    i, jo, ji = s.get_axes(C)
    s.fuse(jo, ji)

    mod = tvm.tir.hybrid.create_module([split_fuse_element_wise])
    split_fuse_func = mod["split_fuse_element_wise"]

    assert AssertEqual(split_fuse_func, s.func)


@tvm.tir.hybrid.script
def predicate_fuse(b, c):
    C = buffer_bind(c, (16, 16), "float32")
    B = buffer_bind(b, (16, 16), "float32")
    with block({}, writes=[], reads=[], name="root"):
        for i in range(0, 256):
            with block({vi(0, 16):floordiv(floordiv(i, 4), 4), vj(0, 16):((floormod(floordiv(i, 4), 4)*3) + floormod(i, 4))}, writes=[C[vi:(vi + 1), vj:(vj + 1)]], reads=[B[vi:(vi + 1), vj:(vj + 1)]], predicate=(((floormod(floordiv(i, 4), 4)*4) + floormod(i, 4)) < 16), name="update"):
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


if __name__ == "__main__":
    test_fuse()
    test_split_fuse()
    test_fuse_loop_sref()
