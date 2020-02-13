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
from tvm import ir_pass
from tvm import ir_builder
import util


def test_no_allocate():
    func = util.matmul_stmt()
    func = ir_pass.BufferFlatten(func)
    stmt0 = ir_pass.CanonicalSimplify(func.body)

    def no_allocate_after_stmt():
        ib = tvm.ir_builder.create()
        A = ib.pointer("float32", 'A')
        B = ib.pointer("float32", 'B')
        C = ib.pointer("float32", 'C')
        with ib.for_range(0, 128, "i") as i:
            with ib.for_range(0, 128, "j") as j:
                C[i*128 + j] = 0.
                with ib.for_range(0, 128, "k") as k:
                    C[i*128 + j] = C[i*128 + j] + A[i*128 + k] * B[j*128 + k]
        return ib.get()

    stmt1 = no_allocate_after_stmt()
    assert ir_pass.AssertStructEqual(stmt0, stmt1)


def test_global_allocate():
    func = util.element_wise_stmt()
    func = ir_pass.BufferFlatten(func)
    stmt0 = ir_pass.CanonicalSimplify(func.body)

    def no_allocate_after_stmt():
        ib = tvm.ir_builder.create()
        A = ib.pointer("float32", 'A')
        C = ib.pointer("float32", 'C')
        B = ib.allocate("float32", 128*128, name='B', scope="global")
        with ib.for_range(0, 128, "i") as i:
            with ib.for_range(0, 128, "j") as j:
                B[i*128 + j] = A[i*128 + j] * 2.0

        with ib.for_range(0, 128, "i") as i:
            with ib.for_range(0, 128, "j") as j:
                C[i*128 + j] = B[i*128 + j] + 1.0
        return ib.get()

    stmt1 = no_allocate_after_stmt()
    assert ir_pass.AssertStructEqual(stmt0, stmt1)


@tvm.hybrid_tir.script
def comput_at_element_wise(a, c):
    A = buffer_bind(a, (128, 128), "float32", name="A")
    C = buffer_bind(c, (128, 128), "float32", name="C")

    with block({}, A[0: 128, 0: 128], C[0: 128, 0: 128], name="root"):
        B = buffer_allocate((128, 128), "float32", name="B")

        for i in range(0, 128):
            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j}, A[vi: vi + 1, vj: vj + 1],
                           B[vi: vi + 1, vj: vj + 1],
                           name="B"):
                    B[vi, vj] = A[vi, vj] * 2

            for j in range(0, 128):
                with block({vi(0, 128): i, vj(0, 128): j}, B[vi: vi + 1, vj: vj + 1],
                           C[vi: vi + 1, vj: vj + 1],
                           name="C"):
                    C[vi, vj] = B[vi, vj] + 1


def test_local_allocate():
    a, c = tvm.var('a'), tvm.var('c')
    func = comput_at_element_wise(a, c)
    func = ir_pass.BufferFlatten(func)
    stmt0 = ir_pass.CanonicalSimplify(func.body)

    def no_allocate_after_stmt():
        ib = tvm.ir_builder.create()
        A = ib.pointer("float32", 'A')
        C = ib.pointer("float32", 'C')
        B = ib.allocate("float32", 128, name='B', scope="global")
        with ib.for_range(0, 128, "i") as i:
            with ib.for_range(0, 128, "j") as j:
                B[j] = A[i*128 + j] * 2.0

            with ib.for_range(0, 128, "j") as j:
                C[i*128 + j] = B[j] + 1.0
        return ib.get()

    stmt1 = no_allocate_after_stmt()
    assert ir_pass.AssertStructEqual(stmt0, stmt1)


def test_shared_allocate():
    pass


if __name__ == "__main__":
    test_no_allocate()
    test_global_allocate()
    test_local_allocate()
    test_shared_allocate()
