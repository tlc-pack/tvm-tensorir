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
""" Test multi-level tiling """
# pylint: disable=missing-function-docstring

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import ty

# pylint: disable=invalid-name,no-member


@tvm.script.tir
def batched_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 128, 128])
    B = tir.match_buffer(b, [16, 128, 128])
    C = tir.match_buffer(c, [16, 128, 128])

    with tir.block([16, 128, 128]) as [vn, vi, vj]:
        C[vn, vi, vj] = tir.float32(0)

    with tir.block([16, 128, 128, tir.reduce_axis(0, 128)], "update") as [
        vn,
        vi,
        vj,
        vk,
    ]:
        C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


@tvm.script.tir
def desc_tensorcore(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = tir.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = tir.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        for i, j, k in tir.grid(16, 16, 16):
            with tir.block([16, 16, tir.reduce_axis(0, 16)], "update") as [
                vii,
                vjj,
                vkk,
            ]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                tir.bind(vkk, vk + k)
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@tvm.script.tir
def impl_tensorcore(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = tir.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = tir.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        tir.reads(
            [
                C[vi : vi + 16, vj : vj + 16],
                A[vi : vi + 16, vk : vk + 16],
                B[vj : vj + 16, vk : vk + 16],
            ]
        )
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
def tensorized_batch_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    C = tir.match_buffer(c, [16, 128, 128])
    B = tir.match_buffer(b, [16, 128, 128])
    A = tir.match_buffer(a, [16, 128, 128])

    with tir.block([16, 128, 128]) as [vn, vi, vj]:
        C[vn, vi, vj] = tir.float32(0)
    # body
    for n in range(0, 16):
        for i, j, k in tir.grid(8, 8, 8):
            with tir.block([16, 16, 16, tir.reduce_axis(0, 16)], "update") as [
                vn,
                vi,
                vj,
                vk,
            ]:
                tir.bind(vn, n)
                tir.bind(vi, i * 16)
                tir.bind(vj, j * 16)
                tir.bind(vk, k * 16)
                tir.reads(
                    [
                        C[vn : vn + 1, vi : vi + 16, vj : vj + 16],
                        A[vn : vn + 1, vi : vi + 16, vk : vk + 16],
                        B[vn : vn + 1, vj : vj + 16, vk : vk + 16],
                    ]
                )
                tir.writes(C[vn : vn + 1, vi : vi + 16, vj : vj + 16])
                tir.evaluate(
                    tir.tvm_mma_sync(
                        C.data,
                        tir.floordiv(
                            tir.get_elem_offset(C[vn, vi, vj], dtype="int32"), 256
                        ),
                        A.data,
                        tir.floordiv(
                            tir.get_elem_offset(A[vn, vi, vk], dtype="int32"), 256
                        ),
                        B.data,
                        tir.floordiv(
                            tir.get_elem_offset(B[vn, vj, vk], dtype="int32"), 256
                        ),
                        C.data,
                        tir.floordiv(
                            tir.get_elem_offset(C[vn, vi, vj], dtype="int32"), 256
                        ),
                        dtype="handle",
                    )
                )


@tvm.script.tir
def desc_dot_product(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
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
def impl_dot_product(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (4,))
    B = tir.match_buffer(b, (4,))
    C = tir.match_buffer(c, (1,))

    with tir.block([tir.reduce_axis(0, 4)], "root") as [v0]:
        tir.bind(v0, 0)
        tir.evaluate(C.data + A.data + B.data)


@tvm.script.tir
def dot_product_batch_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 128, 128])
    B = tir.match_buffer(b, [16, 128, 128])
    C = tir.match_buffer(c, [16, 128, 128])
    with tir.block([16, 128, 128]) as [vn, vi, vj]:
        C[vn, vi, vj] = tir.float32(0)
    for i0_1 in range(0, 16):
        for i1_1 in range(0, 128):
            for i2_1 in range(0, 128):
                for i3_outer in range(0, 32):
                    with tir.block([16, 128, 128, tir.reduce_axis(0, 4)], "root") as [
                        vn_1,
                        vi_1,
                        vj_1,
                        v0,
                    ]:
                        tir.bind(vn_1, i0_1)
                        tir.bind(vi_1, i1_1)
                        tir.bind(vj_1, i2_1)
                        tir.bind(v0, (i3_outer * 4))
                        tir.reads([])
                        tir.writes([])
                        tir.evaluate(((C.data + A.data) + B.data))


# pylint: enable=invalid-name,no-member


def test_auto_tensorize_tensorcore():
    sch = ms.Schedule(batched_matmul)
    block = sch.get_block(name="update")
    assert ms.analysis.can_tensorize_rewrite(sch.sch, sch.evaluate(block), desc_tensorcore)
    ms.analysis.do_tensorize_rewrite(sch, block, desc_tensorcore)

    i, j, k = sch.sch.get_axes(sch.evaluate(block))[-3:]
    tensor_intrin = tvm.tir.TensorIntrin(desc_tensorcore, impl_tensorcore)
    sch.sch.tensorize(i, tensor_intrin)

    tvm.ir.assert_structural_equal(tensorized_batch_matmul, sch.sch.func)


def test_auto_tensorize_dot_product():
    sch = ms.Schedule(batched_matmul)
    block = sch.get_block(name="update")
    assert ms.analysis.can_tensorize_rewrite(sch.sch, sch.evaluate(block), desc_dot_product)
    ms.analysis.do_tensorize_rewrite(sch, block, desc_dot_product)

    k = sch.sch.get_axes(sch.evaluate(block))[-1]
    tensor_intrin = tvm.tir.TensorIntrin(desc_dot_product, impl_dot_product)
    sch.sch.tensorize(k, tensor_intrin)

    tvm.ir.assert_structural_equal(dot_product_batch_matmul, sch.sch.func)


if __name__ == "__main__":
    test_auto_tensorize_tensorcore()
    test_auto_tensorize_dot_product()
