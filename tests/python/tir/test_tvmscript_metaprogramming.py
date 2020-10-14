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


def test_tensor_dimension_invariant_code():
    m = matmul.attrs["free_vars"][0]
    func1 = matmul.bind({m: 128})  # syntax 1
    tvm.ir.assert_structural_equal(func1, matmul_128)

    func2 = matmul.bind(128)  # syntax 2
    tvm.ir.assert_structural_equal(func2, matmul_128)


if __name__ == '__main__':
    test_tensor_dimension_invariant_code()
