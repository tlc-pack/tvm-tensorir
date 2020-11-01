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
"""A collection of TIR tensor intrinsics"""
# pylint: disable=missing-function-docstring
import tvm
from tvm import tir
from tvm.script import ty

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks
# fmt: off

@tvm.script.tir
def tensorcore_desc(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = tir.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = tir.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        for i, j, k in tir.grid(16, 16, 16):
            with tir.block([16, 16, tir.reduce_axis(0, 16)], "update") as [vii, vjj, vkk]:
                tir.bind(vii, vi * 16 + i)
                tir.bind(vjj, vj * 16 + j)
                tir.bind(vkk, vk * 16 + k)
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@tvm.script.tir
def tensorcore_impl(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = tir.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = tir.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        tir.reads([
            C[vi * 16 : vi * 16 + 16, vj * 16 : vj + 16],
            A[vi * 16 : vi * 16 + 16, vk * 16 : vk + 16],
            B[vj * 16 : vj * 16 + 16, vk * 16 : vk + 16],
        ])
        tir.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        tir.evaluate(
            tir.tvm_mma_sync(
                C.data,
                C.elem_offset // 256,
                A.data,
                A.elem_offset // 256,
                B.data,
                B.elem_offset // 256,
                C.data,
                C.elem_offset // 256,
                dtype="handle",
            )
        )


@tvm.script.tir
def dot_product_desc(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (4,))
    B = tir.match_buffer(b, (4,))
    C = tir.match_buffer(c, (1,))

    with tir.block([tir.reduce_axis(0, 4)], "root") as [v0]:
        tir.bind(v0, 0)
        for i in range(0, 4):
            with tir.block([tir.reduce_axis(0, 4)], "update") as [vi]:
                tir.bind(vi, v0 + i)
                C[0] = C[0] + A[vi] * B[vi]


@tvm.script.tir
def dot_product_impl(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (4,))
    B = tir.match_buffer(b, (4,))
    C = tir.match_buffer(c, (1,))

    with tir.block([tir.reduce_axis(0, 4)], "root") as [v0]:
        tir.bind(v0, 0)
        tir.evaluate(C.data + A.data + B.data)

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks
