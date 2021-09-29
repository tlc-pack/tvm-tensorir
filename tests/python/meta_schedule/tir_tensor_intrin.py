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
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                tir.bind(vkk, vk + k)
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
            C[vi : vi + 16, vj : vj + 16],
            A[vi : vi + 16, vk : vk + 16],
            B[vj : vj + 16, vk : vk + 16],
        ])
        tir.writes(C[vi : vi + 16, vj : vj + 16])
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
        tir.reads([C[0 : 1], A[v0 : v0 + 4], B[v0 : v0 + 4]])
        tir.writes([C[0 : 1]])
        tir.evaluate(tir.call_extern(  # pylint: disable=redundant-keyword-arg
            "vec4add",
            C.data, C.elem_offset,
            A.data, A.elem_offset,
            B.data, B.elem_offset,
            dtype="int32",
        ))

@tvm.script.tir
def wmma_sync_desc(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a")
    B = tir.match_buffer(b, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b")
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=1, scope="wmma.accumulator")

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        for i, j, k in tir.grid(16, 16, 16):
            with tir.block([16, 16, tir.reduce_axis(0, 16)], "update") as [vii, vjj, vkk]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                tir.bind(vkk, vk + k)
                C[vii, vjj] = C[vii, vjj] + tir.cast(A[vii, vkk], "float32") * tir.cast(B[vkk, vjj],
                                                                                        "float32")


@tvm.script.tir
def wmma_sync_impl(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")
    B = tir.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16,
                         scope="wmma.accumulator")

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        tir.reads([C[vi: vi+16, vj: vj+16], A[vi: vi+16, vk: vk+16], B[vk: vk+16, vj: vj+16]])
        tir.writes(C[vi: vi+16, vj: vj+16])
        tir.evaluate(tir.tvm_mma_sync(C.data, C.elem_offset // 256 + tir.floordiv(tir.floormod(C.elem_offset, 256), 16),
                                      A.data, A.elem_offset // 256 + tir.floordiv(tir.floormod(A.elem_offset, 256), 16),
                                      B.data, B.elem_offset // 256 + tir.floordiv(tir.floormod(B.elem_offset, 256), 16),
                                      C.data, C.elem_offset // 256 + tir.floordiv(tir.floormod(C.elem_offset, 256), 16),
                                      dtype="handle"))


@tvm.script.tir
def wmma_load_a_desc(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16,
                         scope="shared")
    C = tir.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16,
                         scope="wmma.matrix_a")

    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        for i, j in tir.grid(16, 16):
            with tir.block([16, 16], "load") as [vii, vjj]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                C[vii, vjj] = A[vii, vjj]


@tvm.script.tir
def wmma_load_a_impl(a: ty.handle, c: ty.handle) -> None:
    s1 = tir.var("int32")
    s0 = tir.var("int32")
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s1, s0])
    C = tir.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")

    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads(A[vi: vi+16, vj: vj+16])
        tir.writes(C[vi: vi+16, vj: vj+16])
        tir.evaluate(tir.tvm_load_matrix_sync(
            C.data, 16, 16, 16, C.elem_offset // 256 + tir.floordiv(tir.floormod(C.elem_offset, 256), 16), A.access_ptr("r"), s1, "row_major",
            dtype="handle"))


@tvm.script.tir
def wmma_load_b_desc(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared")
    C = tir.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        for i, j in tir.grid(16, 16):
            with tir.block([16, 16], "load") as [vii, vjj]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                C[vii, vjj] = A[vii, vjj]


@tvm.script.tir
def wmma_load_b_impl(a: ty.handle, c: ty.handle) -> None:
    s1 = tir.var("int32")
    s0 = tir.var("int32")
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s1, s0])
    C = tir.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")
    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads(A[vi: vi+16, vj: vj+16])
        tir.writes(C[vi: vi+16, vj: vj+16])
        tir.evaluate(tir.tvm_load_matrix_sync(
            C.data, 16, 16, 16, C.elem_offset // 256 + tir.floordiv(tir.floormod(C.elem_offset, 256), 16), A.access_ptr("r"), s1, "row_major",
            dtype="handle"))


@tvm.script.tir
def wmma_fill_desc(c: ty.handle) -> None:
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        for i, j in tir.grid(16, 16):
            with tir.block([16, 16], "init") as [vii, vjj]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                C[vii, vjj] = tir.float32(0)


@tvm.script.tir
def wmma_fill_impl(c: ty.handle) -> None:
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads([])
        tir.writes(C[vi : vi + 16, vj : vj + 16])
        tir.evaluate(tir.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // 256 + tir.floordiv(tir.floormod(C.elem_offset, 256), 16), tir.float32(0), dtype="handle"))


@tvm.script.tir
def wmma_store_desc(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global")
    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        for i, j in tir.grid(16, 16):
            with tir.block([16, 16], "store") as [vii, vjj]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                C[vii, vjj] = A[vii, vjj]


@tvm.script.tir
def wmma_store_impl(a: ty.handle, c: ty.handle) -> None:
    s1 = tir.var("int32")
    s0 = tir.var("int32")
    A = tir.match_buffer(a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global", strides=[s1, s0])
    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads(A[vi: vi + 16, vj: vj + 16])
        tir.writes(C[vi: vi+16, vj: vj+16])
        tir.evaluate(tir.tvm_store_matrix_sync(
            A.data, 16, 16, 16, A.elem_offset // 256 + tir.floordiv(tir.floormod(A.elem_offset, 256), 16), C.access_ptr("w"), s1, "row_major",
            dtype="handle"))


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks

TENSORCORE_WMMA = tir.TensorIntrin.register(
    "test.tensorcore.wmma",
    tensorcore_desc,
    tensorcore_impl,
)

NEON_DOT = tir.TensorIntrin.register(
    "test.neon.dot",
    dot_product_desc,
    dot_product_impl,
)

WMMA_SYNC = tir.TensorIntrin.register(
    "wmma_sync",
    wmma_sync_desc,
    wmma_sync_impl,
)

WMMA_LOAD_A = tir.TensorIntrin.register(
    "wmma_load_a",
    wmma_load_a_desc,
    wmma_load_a_impl,
)

WMMA_LOAD_B = tir.TensorIntrin.register(
    "wmma_load_b",
    wmma_load_b_desc,
    wmma_load_b_impl,
)

WMMA_FILL = tir.TensorIntrin.register(
    "wmma_fill",
    wmma_fill_desc,
    wmma_fill_impl,
)

WMMA_FILL = tir.TensorIntrin.register(
    "wmma_store",
    wmma_store_desc,
    wmma_store_impl,
)
