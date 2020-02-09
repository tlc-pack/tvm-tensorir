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
from tvm import tir
import util
from tvm.ir_pass import Equal, AssertStructEqual

@tvm.tir.hybrid.script
def fused_element_wise(a, c):
    A = buffer_bind(a, (128, 128))
    C = buffer_bind(c, (128, 128))

    with block({}, A[0: 128, 0: 128], C[0: 128, 0: 128], name="root"):
        B = buffer_allocate((128, 128))

        for i in range(0, 128 * 128):
            with block({vi(0, 128) : i // 128, vj(0, 128) : i % 128},
                    reads=A[vi: vi + 1, vj: vj + 1], writes=B[vi: vi + 1, vj: vj + 1], name="B"):
                B[vi, vj] = A[vi, vj] * 2

        for j in range(0, 128 * 128):
            with block({vi(0, 128) : j // 128, vj(0, 128) : j % 128},
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
                    with block({vi(0, 128) : io * 16 + ii, vj(0, 128) : j},
                            reads=A[vi: vi + 1, vj: vj + 1], writes=B[vi: vi + 1, vj: vj + 1],
                            name="B"):
                        B[vi, vj] = A[vi, vj] * 2

        for i in range(0, 128):
            for jo in range(0, 10):
                for ji in range(0, 13):
                    with block({vi(0, 128) : i, vj(0, 128) : jo * 13 + ji},
                            reads=B[vi: vi + 1, vj: vj + 1], writes=C[vi: vi + 1, vj: vj + 1],
                            predicate=jo * 13 + ji < 128, name="C"):
                        C[vi, vj] = B[vi, vj] + 1


def test_split():
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

    print(tvm.tir.hybrid.ashybrid(split_func))
    print(tvm.tir.hybrid.ashybrid(s.func))
    assert Equal(split_func, s.func)


if __name__ == "__main__":
    test_fuse()
    test_split()
