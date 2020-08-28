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
from tvm.hybrid import ty


@tvm.hybrid.script
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.buffer_bind(a, [128, 128])
    B = tir.buffer_bind(b, [128, 128])
    C = tir.buffer_bind(c, [128, 128])
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vj, vk])


@tvm.hybrid.script
def matmul_original(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.buffer_bind(a, [128, 128])
    B = tir.buffer_bind(b, [128, 128])
    C = tir.buffer_bind(c, [128, 128])

    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)

        for k in range(0, 128):
            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.hybrid.script
def element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.buffer_bind(a, (128, 128), "float32")
    C = tir.buffer_bind(c, (128, 128), "float32")
    B = tir.buffer_allocate((128, 128), "float32")

    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0

    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.hybrid.script
def predicate(b: ty.handle, c: ty.handle) -> None:
    B = tir.buffer_bind(b, (16, 16), "float32")
    C = tir.buffer_bind(c, (16, 16), "float32")

    for i, jo, ji in tir.grid(16, 4, 4):
        with tir.block([16, 16], "update") as [vi, vj]:
            tir.where(jo * 4 + ji < 16)
            tir.bind(vi, i)
            tir.bind(vj, jo * 4 + ji)
            C[vi, vj] = B[vi, vj] + 1.0


def matmul_stmt():
    mod = tvm.hybrid.create_module({"matmul": matmul})
    return mod["matmul"]


def matmul_stmt_original():
    mod = tvm.hybrid.create_module({"matmul_original": matmul_original})
    return mod["matmul_original"]


def element_wise_stmt():
    mod = tvm.hybrid.create_module({"element_wise": element_wise})
    return mod["element_wise"]


def predicate_stmt():
    mod = tvm.hybrid.create_module({"predicate": predicate})
    return mod["predicate"]
