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
"""Test auto tensorization"""
# pylint: disable=missing-function-docstring

import tvm
from tir_tensor_intrin import (
    dot_product_desc,
    dot_product_impl,
    tensorcore_desc,
    tensorcore_impl,
)
from tir_workload import batch_matmul
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import ty


def _check_sketch(result, expected):
    assert len(result) == len(expected)
    for x in result:
        found = False
        for y in expected:
            if tvm.ir.structural_equal(x.sch.func, y):
                found = True
                break
        assert found


# pylint: disable=invalid-name,no-member


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


def test_auto_tensorize_rule_tensorcore():
    mma_sync = tvm.tir.TensorIntrin(tensorcore_desc, tensorcore_impl)
    task = ms.SearchTask(func=batch_matmul)
    space = ms.space.PostOrderApply(
        stages=[
            ms.rule.mark_tensorize(tensor_intrins=[mma_sync]),
        ]
    )
    postproc = ms.postproc.rewrite_tensorize(tensor_intrins=[mma_sync])
    schs = space.get_support(task=task)
    for sch in schs:
        postproc.apply(sch)
    _check_sketch(schs, [batch_matmul, tensorcore_tensorized])


def test_auto_tensorize_rule_dot_product():
    dot_prod = tvm.tir.TensorIntrin(dot_product_desc, dot_product_impl)
    task = ms.SearchTask(func=batch_matmul)
    space = ms.space.PostOrderApply(
        stages=[
            ms.rule.mark_tensorize(tensor_intrins=[dot_prod]),
        ]
    )
    postproc = ms.postproc.rewrite_tensorize(tensor_intrins=[dot_prod])
    schs = space.get_support(task=task)
    for sch in schs:
        postproc.apply(sch)
    _check_sketch(schs, [batch_matmul, dot_product_tensorized])


if __name__ == "__main__":
    test_auto_tensorize_rule_tensorcore()
    test_auto_tensorize_rule_dot_product()
