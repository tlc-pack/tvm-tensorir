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
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_reduction_roundtrip():
    func_rt = tvm.script.from_source(tvm.script.asscript(matmul))
    tvm.ir.assert_structural_equal(matmul, func_rt)


@tvm.script.tir
def matmul_decompose0(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([128, 128], "init") as [vi, vj]:
        C[vi, vj] = 0.0

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_reduction_decompose():
    s = tir.create_schedule(matmul)
    C = s.get_block("update")
    i, j, k = s.get_axes(C)
    s.decompose_reduction(C, i)
    tvm.ir.assert_structural_equal(matmul_decompose0, s.func)


def test_reduction_merge():
    s = tir.create_schedule(matmul_decompose0)
    init = s.get_block("init")
    update = s.get_block("update")
    s.merge_reduction(init, update)
    tvm.ir.assert_structural_equal(matmul, s.func)


def test_reduction_blockize():
    s = tir.create_schedule(matmul)
    C = s.get_block("update")
    i, j, k = s.get_axes(C)
    s.blockize(j)
    print(tvm.script.asscript(s.func))


if __name__ == "__main__":
    test_reduction_roundtrip()
    test_reduction_decompose()
    test_reduction_merge()
    test_reduction_blockize()
