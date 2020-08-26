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
    A = tir.buffer_bind(a, (16, 16), align=128, offset_factor=1)
    B = tir.buffer_bind(b, (16, 16), align=128, offset_factor=1)
    C = tir.buffer_bind(c, (16, 16), align=128, offset_factor=1)

    with tir.block(16, 16, tir.reduce_axis(0, 16)) as [vi, vj, vk]:
        tir.block_name("root")
        for i, j, k in tir.grid(16, 16, 16):
            with tir.block(16, 16, tir.reduce_axis(0, 16)) as [vii, vjj, vkk]:
                tir.block_name("update")
                tir.block_bind(vii, vi + i)
                tir.block_bind(vjj, vj + j)
                tir.block_bind(vkk, vk + k)
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@tvm.hybrid.script
def intrin_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.buffer_bind(a, (16, 16), align=128, offset_factor=1)
    B = tir.buffer_bind(b, (16, 16), align=128, offset_factor=1)
    C = tir.buffer_bind(c, (16, 16), align=128, offset_factor=1)

    with tir.block(16, 16, tir.reduce_axis(0, 16)) as [vi, vj, vk]:
        tir.block_name("root")
        for i, j, k in tir.grid(16, 16, 16):
            with tir.block(16, 16, tir.reduce_axis(0, 16)) as [vii, vjj, vkk]:
                tir.block_name("update")
                tir.block_bind(vii, vi + i)
                tir.block_bind(vjj, vj + j)
                tir.block_bind(vkk, vk + k)
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

    mod = tvm.hybrid.create_module({"desc_func": desc_func, "intrin_func": intrin_func})

    tensor_intrin = tvm.tir.TensorIntrin(mod["desc_func"], mod["intrin_func"])
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
    A = tir.buffer_bind(a, (16, 16), align=128, offset_factor=1)
    B = tir.buffer_bind(b, (16, 16), align=128, offset_factor=1)
    C = tir.buffer_bind(c, (16, 16), align=128, offset_factor=1)

    with tir.block(16, 16, tir.reduce_axis(0, 16)) as [vi, vj, vk]:
        tir.block_name("root")
        tir.block_reads([C[vi:vi + 16, vj:vj + 16], A[vi:vi + 16, vk:vk + 16], B[vj:vj + 16, vk:vk + 16]])
        tir.block_writes(C[vi:vi + 16, vj:vj + 16])
        tir.evaluate(tir.tvm_mma_sync(C.data, C.elem_offset // 256,
                                      A.data, A.elem_offset // 256,
                                      B.data, B.elem_offset // 256,
                                      C.data, C.elem_offset // 256,
                                      dtype="handle"))


@tvm.hybrid.script
def tensorized_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    C = tir.buffer_bind(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.buffer_bind(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.buffer_bind(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block():
        tir.block_name("root")
        for i_outer, j_outer in tir.grid(8, 8):
            for i_inner_init, j_inner_init in tir.grid(16, 16):
                with tir.block(128, 128) as [vi_init, vj_init]:
                    tir.block_name("update_init")
                    tir.block_bind(vi_init, ((i_outer * 16) + i_inner_init))
                    tir.block_bind(vj_init, ((j_outer * 16) + j_inner_init))
                    C[vi_init, vj_init] = tir.float32(0)
            for k_outer in tir.grid(8):
                with tir.block(16, 16, tir.reduce_axis(0, 16)) as [vi, vj, vk]:
                    tir.block_name("root")
                    tir.block_bind(vi, i_outer * 16)
                    tir.block_bind(vj, j_outer * 16)
                    tir.block_bind(vk, k_outer * 16)
                    tir.block_reads([C[vi:vi + 16, vj:vj + 16], A[vi:vi + 16, vk:vk + 16], B[vj:vj + 16, vk:vk + 16]])
                    tir.block_writes(C[vi:vi + 16, vj:vj + 16])
                    tir.evaluate(
                        tir.tvm_mma_sync(C.data, tir.floordiv(((vi * 128) + vj), 256), A.data,
                                         tir.floordiv(((vi * 128) + vk), 256), B.data,
                                         tir.floordiv(((vj * 128) + vk), 256), C.data,
                                         tir.floordiv(((vi * 128) + vj), 256), dtype="handle"))


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

    mod = tvm.hybrid.create_module(
        {"desc_func": desc_func, "intrin_func": lower_intrin_func})

    tensor_intrin = tvm.tir.TensorIntrin(mod["desc_func"], mod["intrin_func"])
    s.tensorize(ii, tensor_intrin)

    mod = tvm.hybrid.create_module({"tensorized_func": tensorized_func})
    new_func = mod["tensorized_func"]

    tvm.ir.assert_structural_equal(new_func, s.func)


if __name__ == "__main__":
    test_tensorize_gemm()
    test_tensorize_buffer_bind()
