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
from tvm import te
import util
from tvm.ir_pass import Equal

@tvm.hybrid_te.script
def fused_element_wise(a, c):
    A = buffer_bind(a, (16, 16), name="A")
    C = buffer_bind(c, (16, 16), name="C")

    with block({}, A[0: 16, 0: 16], C[0: 16, 0: 16], name="root"):
        B = buffer_allocate((16, 16), name="B")

        for i in range(0, 256):
            with block({vi(0, 16) : i // 16, vj(0, 16) : i % 16},
                    reads=A[vi: vi + 1, vj: vj + 1], writes=B[vi: vi + 1, vj: vj + 1], name="B"):
                B[vi, vj] = A[vi, vj] * 2

        for j in range(0, 256):
            with block({vi(0, 16) : j // 16, vj(0, 16) : j % 16},
                    reads=B[vi: vi + 1, vj: vj + 1], writes=C[vi: vi + 1, vj: vj + 1], name="C"):
                C[vi, vj] = B[vi, vj] + 1


def test_fuse():
    m, n = 16, 16
    func, tensors, tensor_map, _ = util.element_wise_stmt(m, n)

    # schedule
    s = te.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(B)
    s.fuse(outer, inner)
    outer, inner = s.get_axes(C)
    s.fuse(outer, inner)

    a = tvm.var("a")
    c = tvm.var("c")
    fused_func = fused_element_wise(a, c)
    assert Equal(fused_func, s.func)


@tvm.hybrid_te.script
def split_element_wise(a, c):
    A = buffer_bind(a, (16, 16), name="A")
    C = buffer_bind(c, (16, 16), name="C")

    with block({}, A[0: 16, 0: 16], C[0: 16, 0: 16], name="root"):
        B = buffer_allocate((16, 16), name="B")

        for io in range(0, 4):
            for ii in range(0, 4):
                for j in range(0, 16):
                    with block({vi(0, 16) : io * 4 + ii, vj(0, 16) : j},
                            reads=A[vi: vi + 1, vj: vj + 1], writes=B[vi: vi + 1, vj: vj + 1],
                            name="B"):
                        B[vi, vj] = A[vi, vj] * 2

        for i in range(0, 16):
            for jo in range(0, 5):
                for ji in range(0, 4):
                    with block({vi(0, 16) : i, vj(0, 16) : jo * 3 + ji},
                            reads=B[vi: vi + 1, vj: vj + 1], writes=C[vi: vi + 1, vj: vj + 1],
                            predicate=jo * 4 + ji < 16, name="C"):
                        C[vi, vj] = B[vi, vj] + 1


def test_split():
    m, n = 16, 16
    func, tensors, tensor_map, _ = util.element_wise_stmt(m, n)

    # schedule
    s = te.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(B)
    s.split(outer, factor=4)
    outer, inner = s.get_axes(C)
    s.split(inner, nparts=5)

    a = tvm.var("a")
    c = tvm.var("c")
    # TODO: It will work once Bohan adds support for block predicate
    # split_func = split_element_wise(a, c)
    # assert Equal(split_func, s.func)


if __name__ == "__main__":
    test_fuse()
    test_split()