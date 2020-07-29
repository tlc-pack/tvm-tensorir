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
    A = buffer_bind(a, (16, 16), align=128, offset_factor=1)
    B = buffer_bind(b, (16, 16), align=128, offset_factor=1)
    C = buffer_bind(c, (16, 16), align=128, offset_factor=1)

    with tir.block({vi(0, 16): 0, vj(0, 16): 0, vk(0, 16, iter_type="reduce"): 0},
                   [C[vi:vi + 16, vj:vj + 16], A[vi:vi + 16, vk:vk + 16],
                    B[vj:vj + 16, vk:vk + 16]],
                   C[vi:vi + 16, vj:vj + 16], name="root"):
        for i in tir.grid(0, 16):
            for j in tir.grid(0, 16):
                for k in tir.grid(0, 16):
                    with tir.block({vii(0, 16): vi + i, vjj(0, 16): vj + j,
                                    vkk(0, 16, iter_type="reduce"): vk + k},
                                   reads=[C[vii, vjj], A[vii, vkk], B[vjj, vkk]],
                                   writes=[C[vii, vjj]],
                                   name="update"):
                        C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@tvm.hybrid.script
def intrin_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = buffer_bind(a, (16, 16), align=128, offset_factor=1)
    B = buffer_bind(b, (16, 16), align=128, offset_factor=1)
    C = buffer_bind(c, (16, 16), align=128, offset_factor=1)

    with tir.block({vi(0, 16): 0, vj(0, 16): 0, vk(0, 16, iter_type="reduce"): 0},
                   [C[vi:vi + 16, vj:vj + 16], A[vi:vi + 16, vk:vk + 16],
                    B[vj:vj + 16, vk:vk + 16]],
                   C[vi:vi + 16, vj:vj + 16], name="root"):
        for i in tir.grid(0, 16):
            for j in tir.grid(0, 16):
                for k in tir.grid(0, 16):
                    with tir.block({vii(0, 16): vi + i, vjj(0, 16): vj + j,
                                    vkk(0, 16, iter_type="reduce"): vk + k},
                                   reads=[C[vii, vjj], A[vii, vkk], B[vjj, vkk]],
                                   writes=[C[vii, vjj]],
                                   name="update"):
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

    intrinsic = tvm.tir.Intrinsic(mod["desc_func"], mod["intrin_func"])
    s.tensorize(ii, intrinsic)

    print(tvm.hybrid.ashybrid(s.func))
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
    A = buffer_bind(a, (16, 16), align=128, offset_factor=1)
    B = buffer_bind(b, (16, 16), align=128, offset_factor=1)
    C = buffer_bind(c, (16, 16), align=128, offset_factor=1)

    with tir.block({vi(0, 16): 0, vj(0, 16): 0, vk(0, 16, iter_type="reduce"): 0},
                   [C[vi:vi + 16, vj:vj + 16], A[vi:vi + 16, vk:vk + 16],
                    B[vj:vj + 16, vk:vk + 16]],
                   C[vi:vi + 16, vj:vj + 16], name="root"):
        for i in tir.grid(0, 16):
            for j in tir.grid(0, 16):
                for k in tir.grid(0, 16):
                    with tir.block({vii(0, 16): vi + i, vjj(0, 16): vj + j,
                                    vkk(0, 16, iter_type="reduce"): vk + k},
                                   reads=[C[vii, vjj], A[vii, vkk], B[vjj, vkk]],
                                   writes=[C[vii, vjj]],
                                   name="update"):
                        tir.evaluate(tir.tvm_mma_sync(C.data, C.elem_offset // 256,
                                                      A.data, A.elem_offset // 256,
                                                      B.data, B.elem_offset // 256,
                                                      C.data, C.elem_offset // 256,
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

    mod = tvm.hybrid.create_module(
        {"desc_func": desc_func, "intrin_func": lower_intrin_func})

    intrinsic = tvm.tir.Intrinsic(mod["desc_func"], mod["intrin_func"])
    s.tensorize(ii, intrinsic)

    print(tvm.hybrid.ashybrid(s.func))


if __name__ == "__main__":
    test_tensorize_gemm()
    test_tensorize_buffer_bind()
