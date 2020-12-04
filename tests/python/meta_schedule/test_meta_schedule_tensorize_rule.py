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
    # assert len(result) == len(expected)
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
    C = tir.match_buffer(c, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 16):
            for i1_outer in range(0, 8):
                for i2_outer in range(0, 8):
                    for i3_outer in range(0, 8):
                        with tir.block([16, 8, 8, tir.reduce_axis(0, 8)], "blockized_update") as [vn, vio, vjo, vko]:
                            tir.bind(vn, i0)
                            tir.bind(vio, i1_outer)
                            tir.bind(vjo, i2_outer)
                            tir.bind(vko, i3_outer)
                            tir.reads([C[vn:(vn + 1), (vio*16):((vio*16) + 16), (vjo*16):((vjo*16) + 16)], A[vn:(vn + 1), (vio*16):((vio*16) + 16), (vko*16):((vko*16) + 16)], B[vn:(vn + 1), (vjo*16):((vjo*16) + 16), (vko*16):((vko*16) + 16)]])
                            tir.writes([C[vn:(vn + 1), (vio*16):((vio*16) + 16), (vjo*16):((vjo*16) + 16)]])
                            with tir.init():
                                for i1_inner in range(0, 16):
                                    for i2_inner in range(0, 16):
                                        C[vn, ((vio*16) + i1_inner), ((vjo*16) + i2_inner)] = tir.float32(0)
                            with tir.block([1, 1, tir.reduce_axis(0, 1)], "blockized_update") as [vio_1, vjo_1, vko_1]:
                                tir.bind(vio_1, 0)
                                tir.bind(vjo_1, 0)
                                tir.bind(vko_1, 0)
                                tir.reads([C[vn:(vn + 1), (vio*16):((vio*16) + 16), (vjo*16):((vjo*16) + 16)], A[vn:(vn + 1), (vio*16):((vio*16) + 16), (vko*16):((vko*16) + 16)], B[vn:(vn + 1), (vjo*16):((vjo*16) + 16), (vko*16):((vko*16) + 16)]])
                                tir.writes([C[vn:(vn + 1), (vio*16):((vio*16) + 16), (vjo*16):((vjo*16) + 16)]])
                                tir.evaluate(tir.tvm_mma_sync(C.data, tir.floordiv(tir.get_elem_offset(C[vn, (vio*16), (vjo*16)], dtype="int32"), 256), A.data, tir.floordiv(tir.get_elem_offset(A[vn, (vio*16), (vko*16)], dtype="int32"), 256), B.data, tir.floordiv(tir.get_elem_offset(B[vn, (vjo*16), (vko*16)], dtype="int32"), 256), C.data, tir.floordiv(tir.get_elem_offset(C[vn, (vio*16), (vjo*16)], dtype="int32"), 256), dtype="handle"))


@tvm.script.tir
def dot_product_tensorized(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [16, 128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        for i0 in range(0, 16):
            for i1 in range(0, 128):
                for i2 in range(0, 128):
                    for i3_outer in range(0, 32):
                        with tir.block([16, 128, 128, tir.reduce_axis(0, 32)], "blockized_update") as [vn, vi, vj, vko]:
                            tir.bind(vn, i0)
                            tir.bind(vi, i1)
                            tir.bind(vj, i2)
                            tir.bind(vko, i3_outer)
                            tir.reads([C[vn:(vn + 1), vi:(vi + 1), vj:(vj + 1)], A[vn:(vn + 1), vi:(vi + 1), (vko*4):((vko*4) + 4)], B[vn:(vn + 1), vj:(vj + 1), (vko*4):((vko*4) + 4)]])
                            tir.writes([C[vn:(vn + 1), vi:(vi + 1), vj:(vj + 1)]])
                            with tir.init():
                                C[vn, vi, vj] = tir.float32(0)
                            with tir.block([tir.reduce_axis(0, 1)], "blockized_update") as [vko_1]:
                                tir.bind(vko_1, 0)
                                tir.reads([C[vn:(vn + 1), vi:(vi + 1), vj:(vj + 1)], A[vn:(vn + 1), vi:(vi + 1), (vko*4):((vko*4) + 4)], B[vn:(vn + 1), vj:(vj + 1), (vko*4):((vko*4) + 4)]])
                                tir.writes([C[vn:(vn + 1), vi:(vi + 1), vj:(vj + 1)]])
                                tir.evaluate(tir.call_extern("vec4add", C.data, 0, A.data, 0, B.data, 0, dtype="int32"))


# pylint: enable=invalid-name,no-member


def test_auto_tensorize_rule_tensorcore():
    mma_sync = tvm.tir.TensorIntrin(tensorcore_desc, tensorcore_impl)
    task = ms.SearchTask(workload=batch_matmul)
    space = ms.space.PostOrderApply(
        stages=[
            ms.rule.mark_tensorize(tensor_intrins=[mma_sync]),
        ]
    )
    postproc = ms.postproc.rewrite_tensorize(tensor_intrins=[mma_sync])
    schs = space.get_support(task=task)
    for sch in schs:
        postproc.apply(task, sch)
    _check_sketch(schs, [batch_matmul, tensorcore_tensorized])


def test_auto_tensorize_rule_dot_product():
    dot_prod = tvm.tir.TensorIntrin(dot_product_desc, dot_product_impl)
    task = ms.SearchTask(workload=batch_matmul)
    space = ms.space.PostOrderApply(
        stages=[
            ms.rule.mark_tensorize(tensor_intrins=[dot_prod]),
        ]
    )
    postproc = ms.postproc.rewrite_tensorize(tensor_intrins=[dot_prod])
    schs = space.get_support(task=task)
    for sch in schs:
        postproc.apply(task, sch)
    _check_sketch(schs, [batch_matmul, dot_product_tensorized])


if __name__ == "__main__":
    test_auto_tensorize_rule_tensorcore()
    test_auto_tensorize_rule_dot_product()
