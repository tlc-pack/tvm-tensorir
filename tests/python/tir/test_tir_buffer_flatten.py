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
import tvm
from tvm import tir
from tvm.script import ty

import util

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name


@tvm.script.tir
def compute_at_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32", name="A")
    C = tir.match_buffer(c, (128, 128), "float32", name="C")
    B = tir.alloc_buffer((128, 128), "float32", name="B")
    for i in range(0, 128):
        for j in range(0, 128):
            with tir.block([128, 128], "B") as [vi, vj]:
                B[vi, vj] = A[vi, vj] * 2.0

        for j in range(0, 128):
            with tir.block([128, 128], "C") as [vi, vj]:
                C[vi, vj] = B[vi, vj] + 1.0


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name
# fmt: on
# pylint: disable=invalid-name


def test_no_allocate():
    mod = tvm.script.create_module({"func": util.matmul_stmt_original()})
    trans = tvm.transform.Sequential(
        [
            tvm.tir.transform.AllreduceTransform(),
            tvm.tir.transform.BufferFlatten(),
            tvm.tir.transform.Simplify(),
        ]
    )
    mod = trans(mod)

    def no_allocate_after_stmt():
        ib = tvm.tir.ir_builder.create()
        A = ib.pointer("float32", "A")
        B = ib.pointer("float32", "B")
        C = ib.pointer("float32", "C")
        with ib.for_range(0, 128, "i") as i:
            with ib.for_range(0, 128, "j") as j:
                C[i * 128 + j] = 0.0
                with ib.for_range(0, 128, "k") as k:
                    C[i * 128 + j] = C[i * 128 + j] + A[i * 128 + k] * B[j * 128 + k]
        return ib.get()

    stmt = no_allocate_after_stmt()
    tvm.ir.assert_structural_equal(mod["func"].body, stmt, map_free_vars=True)


def test_global_allocate():
    mod = tvm.script.create_module({"func": util.element_wise_stmt()})
    trans = tvm.transform.Sequential(
        [
            tvm.tir.transform.AllreduceTransform(),
            tvm.tir.transform.BufferFlatten(),
            tvm.tir.transform.Simplify(),
        ]
    )
    mod = trans(mod)

    def no_allocate_after_stmt():
        ib = tvm.tir.ir_builder.create()
        A = ib.pointer("float32", "A")
        C = ib.pointer("float32", "C")
        B = ib.allocate("float32", 128 * 128, name="B", scope="global")
        with ib.for_range(0, 128, "i") as i:
            with ib.for_range(0, 128, "j") as j:
                B[i * 128 + j] = A[i * 128 + j] * 2.0

        with ib.for_range(0, 128, "i") as i:
            with ib.for_range(0, 128, "j") as j:
                C[i * 128 + j] = B[i * 128 + j] + 1.0
        return ib.get()

    stmt = no_allocate_after_stmt()
    tvm.ir.assert_structural_equal(mod["func"].body, stmt, map_free_vars=True)


def test_local_allocate():
    mod = tvm.script.create_module({"func": compute_at_element_wise})
    trans = tvm.transform.Sequential(
        [
            tvm.tir.transform.AllreduceTransform(),
            tvm.tir.transform.BufferFlatten(),
            tvm.tir.transform.Simplify(),
        ]
    )
    mod = trans(mod)

    def no_allocate_after_stmt():
        ib = tvm.tir.ir_builder.create()
        A = ib.pointer("float32", "A")
        C = ib.pointer("float32", "C")
        with ib.for_range(0, 128, "i") as i:
            B = ib.allocate("float32", 128, name="B", scope="global")
            with ib.for_range(0, 128, "j") as j:
                B[j] = A[i * 128 + j] * 2.0

            with ib.for_range(0, 128, "j") as j:
                C[i * 128 + j] = B[j] + 1.0
        return ib.get()

    stmt = no_allocate_after_stmt()
    tvm.ir.assert_structural_equal(mod["func"].body, stmt, map_free_vars=True)


def test_shared_allocate():
    pass


if __name__ == "__main__":
    test_no_allocate()
    test_global_allocate()
    test_local_allocate()
    test_shared_allocate()
