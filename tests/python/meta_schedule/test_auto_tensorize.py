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
    reducer = tir.comm_reducer(lambda x, y: (x + y), tir.float32(0))

    with tir.block([16, 128, 128, tir.reduce_axis(0, 128)], "update") as [
        vn,
        vi,
        vj,
        vk,
    ]:
        reducer.step(C[vn, vi, vj], (A[vn, vi, vk] * B[vn, vj, vk]))


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
def tensorcore_impl(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
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
def tensorcore_blockized(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: (x + y), tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 16):
            for i1_outer in range(0, 8):
                for i2_outer in range(0, 8):
                    for i3_outer in range(0, 8):
                        for i1_inner in range(0, 16):
                            for i2_inner in range(0, 16):
                                for i3_inner in range(0, 16):
                                    with tir.block(
                                        [16, 128, 128, tir.reduce_axis(0, 128)], "update"
                                    ) as [vn, vi, vj, vk]:
                                        tir.bind(vn, i0)
                                        tir.bind(vi, ((i1_outer * 16) + i1_inner))
                                        tir.bind(vj, ((i2_outer * 16) + i2_inner))
                                        tir.bind(vk, ((i3_outer * 16) + i3_inner))
                                        tir.reads(
                                            [
                                                C[vn : (vn + 1), vi : (vi + 1), vj : (vj + 1)],
                                                A[vn : (vn + 1), vi : (vi + 1), vk : (vk + 1)],
                                                B[vn : (vn + 1), vj : (vj + 1), vk : (vk + 1)],
                                            ]
                                        )
                                        tir.writes([C[vn : (vn + 1), vi : (vi + 1), vj : (vj + 1)]])
                                        reducer.step(C[vn, vi, vj], (A[vn, vi, vk] * B[vn, vj, vk]))


@tvm.script.tir
def tensorcore_tensorized(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 16):
            for i1_outer in range(0, 8):
                for i2_outer in range(0, 8):
                    for i3_outer in range(0, 8):
                        with tir.block(
                            [16, 128, 128, tir.reduce_axis(0, 128)], "blockized_update"
                        ) as [vn, vi, vj, vk]:
                            tir.bind(vn, i0)
                            tir.bind(vi, (i1_outer * 16))
                            tir.bind(vj, (i2_outer * 16))
                            tir.bind(vk, (i3_outer * 16))
                            tir.reads(
                                [
                                    C[vn : (vn + 1), vi : (vi + 16), vj : (vj + 16)],
                                    A[vn : (vn + 1), vi : (vi + 16), vk : (vk + 16)],
                                    B[vn : (vn + 1), vj : (vj + 16), vk : (vk + 16)],
                                ]
                            )
                            tir.writes([C[vn : (vn + 1), vi : (vi + 16), vj : (vj + 16)]])
                            for i1_inner_init in range(0, 16):
                                for i2_inner_init in range(0, 16):
                                    with tir.block([16, 128, 128], "update_init") as [
                                        vn_init,
                                        vi_init,
                                        vj_init,
                                    ]:
                                        tir.bind(vn_init, vn)
                                        tir.bind(vi_init, (vi + i1_inner_init))
                                        tir.bind(vj_init, (vj + i2_inner_init))
                                        tir.reads([])
                                        tir.writes(
                                            [
                                                C[
                                                    vn_init : (vn_init + 1),
                                                    vi_init : (vi_init + 1),
                                                    vj_init : (vj_init + 1),
                                                ]
                                            ]
                                        )
                                        C[vn_init, vi_init, vj_init] = tir.float32(0)
                            with tir.block([16, 16, 16, tir.reduce_axis(0, 16)], "root") as [
                                vn_1,
                                vi_1,
                                vj_1,
                                vk_1,
                            ]:
                                tir.bind(vn_1, vn)
                                tir.bind(vi_1, vi)
                                tir.bind(vj_1, vj)
                                tir.bind(vk_1, vk)
                                tir.reads(
                                    [
                                        C[
                                            vn_1 : (vn_1 + 1),
                                            vi_1 : (vi_1 + 16),
                                            vj_1 : (vj_1 + 16),
                                        ],
                                        A[
                                            vn_1 : (vn_1 + 1),
                                            vi_1 : (vi_1 + 16),
                                            vk_1 : (vk_1 + 16),
                                        ],
                                        B[
                                            vn_1 : (vn_1 + 1),
                                            vj_1 : (vj_1 + 16),
                                            vk_1 : (vk_1 + 16),
                                        ],
                                    ]
                                )
                                tir.writes(
                                    [C[vn_1 : (vn_1 + 1), vi_1 : (vi_1 + 16), vj_1 : (vj_1 + 16)]]
                                )
                                tir.evaluate(
                                    tir.tvm_mma_sync(
                                        C.data,
                                        tir.floordiv(
                                            tir.get_elem_offset(C[vn_1, vi_1, vj_1], dtype="int32"),
                                            256,
                                        ),
                                        A.data,
                                        tir.floordiv(
                                            tir.get_elem_offset(A[vn_1, vi_1, vk_1], dtype="int32"),
                                            256,
                                        ),
                                        B.data,
                                        tir.floordiv(
                                            tir.get_elem_offset(B[vn_1, vj_1, vk_1], dtype="int32"),
                                            256,
                                        ),
                                        C.data,
                                        tir.floordiv(
                                            tir.get_elem_offset(C[vn_1, vi_1, vj_1], dtype="int32"),
                                            256,
                                        ),
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


@tvm.script.tir
def dot_product_blockized(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: (x + y), tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 16):
            for i1 in range(0, 128):
                for i2 in range(0, 128):
                    for i3_outer in range(0, 32):
                        for i3_inner in range(0, 4):
                            with tir.block([16, 128, 128, tir.reduce_axis(0, 128)], "update") as [
                                vn,
                                vi,
                                vj,
                                vk,
                            ]:
                                tir.bind(vn, i0)
                                tir.bind(vi, i1)
                                tir.bind(vj, i2)
                                tir.bind(vk, ((i3_outer * 4) + i3_inner))
                                tir.reads(
                                    [
                                        C[vn : (vn + 1), vi : (vi + 1), vj : (vj + 1)],
                                        A[vn : (vn + 1), vi : (vi + 1), vk : (vk + 1)],
                                        B[vn : (vn + 1), vj : (vj + 1), vk : (vk + 1)],
                                    ]
                                )
                                tir.writes([C[vn : (vn + 1), vi : (vi + 1), vj : (vj + 1)]])
                                reducer.step(C[vn, vi, vj], (A[vn, vi, vk] * B[vn, vj, vk]))


@tvm.script.tir
def dot_product_tensorized(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 16):
            for i1 in range(0, 128):
                for i2 in range(0, 128):
                    for i3_outer in range(0, 32):
                        with tir.block(
                            [16, 128, 128, tir.reduce_axis(0, 128)], "blockized_update"
                        ) as [vn, vi, vj, vk]:
                            tir.bind(vn, i0)
                            tir.bind(vi, i1)
                            tir.bind(vj, i2)
                            tir.bind(vk, (i3_outer * 4))
                            tir.reads(
                                [
                                    C[vn : (vn + 1), vi : (vi + 1), vj : (vj + 1)],
                                    A[vn : (vn + 1), vi : (vi + 1), vk : (vk + 4)],
                                    B[vn : (vn + 1), vj : (vj + 1), vk : (vk + 4)],
                                ]
                            )
                            tir.writes([C[vn : (vn + 1), vi : (vi + 1), vj : (vj + 1)]])
                            with tir.block([16, 128, 128], "update_init") as [
                                vn_init,
                                vi_init,
                                vj_init,
                            ]:
                                tir.bind(vn_init, vn)
                                tir.bind(vi_init, vi)
                                tir.bind(vj_init, vj)
                                tir.reads([])
                                tir.writes(
                                    [
                                        C[
                                            vn_init : (vn_init + 1),
                                            vi_init : (vi_init + 1),
                                            vj_init : (vj_init + 1),
                                        ]
                                    ]
                                )
                                C[vn_init, vi_init, vj_init] = tir.float32(0)
                            with tir.block([16, 128, 128, tir.reduce_axis(0, 4)], "root") as [
                                vn_1,
                                vi_1,
                                vj_1,
                                v0,
                            ]:
                                tir.bind(vn_1, vn)
                                tir.bind(vi_1, vi)
                                tir.bind(vj_1, vj)
                                tir.bind(v0, vk)
                                tir.reads([])
                                tir.writes([])
                                tir.evaluate(((C.data + A.data) + B.data))


# pylint: enable=invalid-name,no-member


def test_auto_tensorize_tensorcore():
    sch = ms.Schedule(batched_matmul)
    block = sch.get_block(name="update")
    assert (
        ms.analysis.get_tensorize_loop_mapping(
            sch.sch,
            sch.evaluate(block),
            tensorcore_desc,
        )
        is not None
    )
    ms.analysis.do_tensorize_rewrite(sch, block, tensorcore_desc)
    tvm.ir.assert_structural_equal(tensorcore_blockized, sch.sch.func)
    # Blockize
    block = sch.evaluate(sch.get_block(name="update"))
    _, _, _, _, i, _, _ = sch.sch.get_axes(block)
    sch.sch.blockize(i)
    # Decompose reduction
    sch.sch.decompose_reduction(block, i)
    # Tensorize
    tensor_intrin = tvm.tir.TensorIntrin(tensorcore_desc, tensorcore_impl)
    sch.sch.tensorize(i, tensor_intrin)
    tvm.ir.assert_structural_equal(tensorcore_tensorized, sch.sch.func)


def test_auto_tensorize_dot_product():
    sch = ms.Schedule(batched_matmul)
    block = sch.get_block(name="update")
    assert (
        ms.analysis.get_tensorize_loop_mapping(
            sch.sch,
            sch.evaluate(block),
            dot_product_desc,
        )
        is not None
    )
    ms.analysis.do_tensorize_rewrite(sch, block, dot_product_desc)
    tvm.ir.assert_structural_equal(dot_product_blockized, sch.sch.func)

    # Blockize
    block = sch.evaluate(sch.get_block(name="update"))
    _, _, _, _, i = sch.sch.get_axes(block)
    sch.sch.blockize(i)
    # Decompose reduction
    sch.sch.decompose_reduction(block, i)
    # Tensorize
    tensor_intrin = tvm.tir.TensorIntrin(dot_product_desc, dot_product_impl)
    sch.sch.tensorize(i, tensor_intrin)
    tvm.ir.assert_structural_equal(dot_product_tensorized, sch.sch.func)


if __name__ == "__main__":
    test_auto_tensorize_tensorcore()
    test_auto_tensorize_dot_product()
