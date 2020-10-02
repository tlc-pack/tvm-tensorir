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
import util
from tvm import tir

import numpy as np
from tvm.hybrid import ty


@tvm.hybrid.script
def desc_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
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


@tvm.hybrid.script
def intrin_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
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
                C[vii, vjj] = C[vii, vjj] + B[vjj, vkk] * A[vii, vkk]


def test_tensorize_gemm():
    func = util.matmul_stmt()
    # schedule
    s = tir.create_schedule(func)
    update = s.get_block("update")
    i, j, k = s.get_axes(update)
    io, ii = s.split(i, 16)
    jo, ji = s.split(j, 16)
    ko, ki = s.split(k, 16)
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction(update, ko)

    tensor_intrin = tvm.tir.TensorIntrin(desc_func, intrin_func)

    s.tensorize(ii, tensor_intrin)

    func = tvm.build(s.func)

    a_np = np.random.uniform(size=(128, 128)).astype("float32")
    b_np = np.random.uniform(size=(128, 128)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((128, 128)).astype("float32"))
    func(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np.dot(a_np, b_np.transpose()), rtol=1e-6)


@tvm.hybrid.script
def lower_intrin_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = tir.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = tir.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        tir.reads([C[vi:vi + 16, vj:vj + 16], A[vi:vi + 16, vk:vk + 16], B[vj:vj + 16, vk:vk + 16]])
        tir.writes(C[vi:vi + 16, vj:vj + 16])
        tir.evaluate(tir.tvm_mma_sync(C.data, C.elem_offset // 256,
                                      A.data, A.elem_offset // 256,
                                      B.data, B.elem_offset // 256,
                                      C.data, C.elem_offset // 256,
                                      dtype="handle"))


@tvm.hybrid.script
def tensorized_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    for i_outer, j_outer in tir.grid(8, 8):
        for i_inner_init, j_inner_init in tir.grid(16, 16):
            with tir.block([128, 128], "init") as [vi_init, vj_init]:
                tir.bind(vi_init, ((i_outer * 16) + i_inner_init))
                tir.bind(vj_init, ((j_outer * 16) + j_inner_init))
                C[vi_init, vj_init] = tir.float32(0)
        for k_outer in tir.grid(8):
            with tir.block([16, 16, tir.reduce_axis(0, 16)], "update") as [vi, vj, vk]:
                tir.bind(vi, i_outer * 16)
                tir.bind(vj, j_outer * 16)
                tir.bind(vk, k_outer * 16)
                tir.reads([C[vi:vi + 16, vj:vj + 16], A[vi:vi + 16, vk:vk + 16], B[vj:vj + 16, vk:vk + 16]])
                tir.writes(C[vi:vi + 16, vj:vj + 16])
                tir.evaluate(
                    tir.tvm_mma_sync(C.data, tir.floordiv(tir.get_elem_offset(C[vi, vj], dtype="int32"), 256),
                                     A.data, tir.floordiv(tir.get_elem_offset(A[vi, vk], dtype="int32"), 256),
                                     B.data, tir.floordiv(tir.get_elem_offset(B[vj, vk], dtype="int32"), 256),
                                     C.data, tir.floordiv(tir.get_elem_offset(C[vi, vj], dtype="int32"), 256),
                                     dtype="handle"))

def test_tensorize_buffer_bind():
    func = util.matmul_stmt()
    # schedule
    s = tir.create_schedule(func)
    update = s.get_block("update")
    i, j, k = s.get_axes(update)
    io, ii = s.split(i, 16)
    jo, ji = s.split(j, 16)
    ko, ki = s.split(k, 16)
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction(update, ko)

    tensor_intrin = tvm.tir.TensorIntrin(desc_func, lower_intrin_func)
    s.tensorize(ii, tensor_intrin)

    tvm.ir.assert_structural_equal(tensorized_func, s.func)


@tvm.hybrid.script
def batch_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 128, 128])
    B = tir.match_buffer(b, [16, 128, 128])
    C = tir.match_buffer(c, [16, 128, 128])

    with tir.block([16, 128, 128]) as [vn, vi, vj]:
        C[vn, vi, vj] = tir.float32(0)

    with tir.block([16, 128, 128, tir.reduce_axis(0, 128)], "update") as [vn, vi, vj, vk]:
        C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


@tvm.hybrid.script
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
            with tir.block([16, 16, 16, tir.reduce_axis(0, 16)], "update") as [vn, vi, vj, vk]:
                tir.bind(vn, n)
                tir.bind(vi, i * 16)
                tir.bind(vj, j * 16)
                tir.bind(vk, k * 16)
                tir.reads([C[vn:vn + 1, vi:vi + 16, vj:vj + 16], A[vn:vn + 1, vi:vi + 16, vk:vk + 16],
                           B[vn:vn + 1, vj:vj + 16, vk:vk + 16]])
                tir.writes(C[vn:vn + 1, vi:vi + 16, vj:vj + 16])
                tir.evaluate(
                    tir.tvm_mma_sync(C.data, tir.floordiv(tir.get_elem_offset(C[vn, vi, vj], dtype="int32"), 256),
                                     A.data, tir.floordiv(tir.get_elem_offset(A[vn, vi, vk], dtype="int32"), 256),
                                     B.data, tir.floordiv(tir.get_elem_offset(B[vn, vj, vk], dtype="int32"), 256),
                                     C.data, tir.floordiv(tir.get_elem_offset(C[vn, vi, vj], dtype="int32"), 256),
                                     dtype="handle"))


def test_high_dim_tensorize():
    s = tir.create_schedule(batch_matmul)
    update = s.get_block("update")
    n, i, j, k = s.get_axes(update)
    io, ii = s.split(i, 16)
    jo, ji = s.split(j, 16)
    ko, ki = s.split(k, 16)
    s.reorder(io, jo, ko, ii, ji, ki)

    tensor_intrin = tvm.tir.TensorIntrin(desc_func, lower_intrin_func)
    s.tensorize(ii, tensor_intrin)

    tvm.ir.assert_structural_equal(tensorized_batch_matmul, s.func)


if __name__ == "__main__":
    test_tensorize_gemm()
    test_tensorize_buffer_bind()
    test_high_dim_tensorize()
