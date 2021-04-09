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
from tvm.script import ty


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.FlattenBuffer()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


@tvm.script.tir
def narrowed_elementwise_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with tir.block([]):
            tir.reads(A[i, 0:16])
            tir.writes(C[i, 0:16])
            B = tir.alloc_buffer([1, 16], "float32")
            for j in range(0, 16):
                with tir.block() as []:
                    tir.reads(A[i, j])
                    tir.writes(B[0, j])
                    B[0, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with tir.block() as []:
                    tir.reads(B[0, j])
                    tir.writes(C[i, j])
                    C[i, j] = B[0, j] * 2.0


@tvm.script.tir
def flattened_elementwise_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")
    for i in tir.serial(0, 16):
        B_new = tir.allocate([16], "float32", "global")
        for j in tir.serial(0, 16):
            B_new[j] = (tir.load("float32", A.data, ((i * 16) + j)) + tir.float32(1))
        for j in tir.serial(0, 16):
            C.data[((i * 16) + j)] = (tir.load("float32", B_new, j) * tir.float32(2))


def test_elementwise():
    _check(narrowed_elementwise_func, flattened_elementwise_func)


if __name__ == "__main__":
    test_elementwise()
