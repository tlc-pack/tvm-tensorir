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


@tvm.script.tir
def matmul_original(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)

        for k in range(0, 128):
            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
def element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    B = tir.buffer_allocate((128, 128), "float32")

    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0

    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def predicate(b: ty.handle, c: ty.handle) -> None:
    B = tir.match_buffer(b, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")

    for i, jo, ji in tir.grid(16, 4, 4):
        with tir.block([16, 16], "update") as [vi, vj]:
            tir.bind(vi, i)
            tir.bind(vj, jo * 4 + ji)
            C[vi, vj] = B[vi, vj] + 1.0


# @tvm.script.tir
# def block_in_opaque_block(a: ty.handle, b: ty.handle) -> None:
#     # TODO
#     A = tir.match_buffer(a, (128, 128), "float32")
#     B = tir.match_buffer(b, (128, 128), "float32")
#     with tir.block([128], "B") as vi:
#         B[vi, 0] = A[vi, 0]
#         if A[vi, 0] == 0.0:
#             with tir.block([], "C") as ():
#                 with tir.block([128], "D") as vj:
#                     B[vi, vj] = A[vi, vj] * 3.0
#         else:
#             with tir.block([], "E") as ():
#                 with tir.block([128], "F") as vj:
#                     B[vi, vj] = A[vi, vj] * 2.0


def matmul_stmt():
    mod = tvm.script.create_module({"matmul": matmul})
    return mod["matmul"]


def matmul_stmt_original():
    mod = tvm.script.create_module({"matmul_original": matmul_original})
    return mod["matmul_original"]


def element_wise_stmt():
    mod = tvm.script.create_module({"element_wise": element_wise})
    return mod["element_wise"]


def predicate_stmt():
    mod = tvm.script.create_module({"predicate": predicate})
    return mod["predicate"]
