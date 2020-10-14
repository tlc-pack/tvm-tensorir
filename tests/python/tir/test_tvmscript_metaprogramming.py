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


@tvm.script.tir
def matmul(a: ty.int32, b: ty.int32, c: ty.int32) -> None:
    m = tir.var("int32")
    tir.func_attr({"free_vars": [m]})
    A = tir.match_buffer(a, [m, m])
    B = tir.match_buffer(b, [m, m])
    C = tir.match_buffer(c, [m, m])
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    with tir.block([m, m, tir.reduce_axis(0, m)], "update") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vj, vk])


@tvm.script.tir
def matmul_128(a: ty.int32, b: ty.int32, c: ty.int32) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vj, vk])


def test_tensor_dimension_invariant_code_matmul():
    m = matmul.attrs["free_vars"][0]
    func1 = matmul.bind({m: 128})  # syntax 1
    tvm.ir.assert_structural_equal(func1, matmul_128)

    func2 = matmul.bind(128)  # syntax 2
    tvm.ir.assert_structural_equal(func2, matmul_128)


@tvm.script.tir
def element_wise(a: ty.handle, c: ty.handle) -> None:
    m = tir.var("int32")
    n = tir.var("int32")
    tir.func_attr({"free_vars": [m, n]})
    A = tir.match_buffer(a, (m, n), "float32")
    C = tir.match_buffer(c, (m, n), "float32")

    B = tir.buffer_allocate((m, n), "float32")

    with tir.block([m, n], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0

    with tir.block([m, n], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def element_wise_128_n(a: ty.handle, c: ty.handle) -> None:
    n = tir.var("int32")
    tir.func_attr({"free_vars": [n]})
    A = tir.match_buffer(a, (128, n), "float32")
    C = tir.match_buffer(c, (128, n), "float32")
    B = tir.buffer_allocate((128, n), "float32")

    with tir.block([128, n], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0

    with tir.block([128, n], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def element_wise_128_64(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 64), "float32")
    C = tir.match_buffer(c, (128, 64), "float32")
    B = tir.buffer_allocate((128, 64), "float32")

    with tir.block([128, 64], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0

    with tir.block([128, 64], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


def test_tensor_dimension_invariant_code_elemwise():
    m, n = element_wise.attrs["free_vars"]
    func1 = element_wise.bind({m: 128, n: 64})  # syntax 1
    tvm.ir.assert_structural_equal(func1, element_wise_128_64)

    func2 = element_wise.bind([128, 64])  # syntax 2
    tvm.ir.assert_structural_equal(func2, element_wise_128_64)

    func3 = element_wise.bind({m: 128})  # syntax 3: partially bind
    tvm.ir.assert_structural_equal(func3, element_wise_128_n)


if __name__ == '__main__':
    test_tensor_dimension_invariant_code_matmul()
    test_tensor_dimension_invariant_code_elemwise()
