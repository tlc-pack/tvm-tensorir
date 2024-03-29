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
# pylint: disable=invalid-name, too-many-locals, too-many-statements, unused-argument
"""Test workload for lowering and build"""
import tvm
from tvm import tir
from tvm.script import ty
import tvm.testing
import numpy as np

from tvm.ir import register_intrin_lowering, register_op_attr


@tvm.script.tir
def tvm_mfma_sync(d: ty.handle, index_d: ty.int32, a: ty.handle, index_a: ty.int32, b: ty.handle, index_b: ty.int32, c:ty.handle, index_c: ty.int32) -> None:
    tx = tir.env_thread("threadIdx.x")
    tir.launch_thread(tx, 64)
    num_warp_i = tir.var('int32')
    num_warp_j = tir.var('int32')
    wmma_A1 = tir.match_buffer(a, [64, num_warp_i, 16], dtype='float16')
    wmma_B1 = tir.match_buffer(b, [64, num_warp_j, 16], dtype='float16')
    wmma_C1 = tir.match_buffer(c, [64, num_warp_i, num_warp_j, 16], dtype='float32')
    wmma_D1 = tir.match_buffer(d, [64, num_warp_i, num_warp_j, 16], dtype='float32')

    with tir.block([], 'mfma_sync'):
        tir.reads([wmma_A1[0:64,0:num_warp_i,0:16], wmma_B1[0:64,0:num_warp_j,0:16], wmma_C1[0:64,0:num_warp_i,0:num_warp_j,0:16]])
        tir.writes([wmma_D1[0:64,0:num_warp_i,0:num_warp_j,0:16]])
        wmma_D1[tx, index_a, index_b, tir.ramp(0, 1, 4)] = tir.call_llvm_pure_intrin(tir.llvm_lookup_intrinsic_id('llvm.amdgcn.mfma.f32.16x16x16f16'), 6,
        wmma_B1[tx, index_b, tir.ramp(0, 1, 4)], wmma_A1[tx, index_a, tir.ramp(0, 1, 4)], wmma_C1[tx, index_a, index_b, tir.ramp(0, 1, 4)], 0, 0, 0, dtype="float32x4")

register_op_attr("tir.tvm_mfma_sync", "FLowerLogicalIntrin", tvm_mfma_sync);

def lower_mfma_16x16x16_matrix_a(i, j):
    return tir.floordiv(j, 4) * 16 + tir.floormod(i, 16), tir.floordiv(i, 16), tir.floormod(j, 4)

def lower_mfma_16x16x16_matrix_c(i, j):
    return tir.floordiv(tir.floormod(j, 16), 4) * 16 + tir.floormod(i, 16), tir.floordiv(i, 16), tir.floordiv(j, 16), tir.floormod(j, 4)

tir.LogicalLayout.register('warp.mfma_16x16x16_matrix_a', lower_mfma_16x16x16_matrix_a)
tir.LogicalLayout.register('warp.mfma_16x16x16_matrix_c', lower_mfma_16x16x16_matrix_c)


@tvm.script.tir
def tensorcore_gemm(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # match buffer
    A = tir.match_buffer(a, [1024, 1024], "float16")
    B = tir.match_buffer(b, [1024, 1024], "float16")
    C = tir.match_buffer(c, [1024, 1024], "float32")

    # body
    for blockIdx_x in tir.thread_binding(0, 16, "blockIdx.x"):
        for blockIdx_y in tir.thread_binding(0, 8, "blockIdx.y"):
            with tir.block([16, 8]) as [bx, by]:
                tir.bind(bx, blockIdx_x)
                tir.bind(by, blockIdx_y)
                shared_A = tir.alloc_buffer([1024, 1024], "float16", scope="shared")
                shared_B = tir.alloc_buffer([1024, 1024], "float16", scope="shared")
                wmma_A = tir.alloc_buffer([1024, 1024], "float16", scope="warp.mfma_16x16x16_matrix_a")
                wmma_B = tir.alloc_buffer([1024, 1024], "float16", scope="warp.mfma_16x16x16_matrix_a")
                wmma_C = tir.alloc_buffer([1024, 1024], "float32", scope="warp.mfma_16x16x16_matrix_c")
                for ty in tir.thread_binding(0, 2, "threadIdx.y"):
                    for tz in tir.thread_binding(0, 2, "threadIdx.z"):
                        for i, j in tir.grid(2, 4):
                            with tir.block([64, 64]) as [vi, vj]:
                                tir.bind(vi, bx * 4 + ty * 2 + i)
                                tir.bind(vj, by * 8 + tz * 4 + j)
                                tir.reads([])
                                tir.writes(wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                                C0 = tir.match_buffer(
                                    wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                                    (16, 16),
                                    "float32",
                                    strides=[16 * 4, 1],
                                    scope="warp.mfma_16x16x16_matrix_c",
                                    offset_factor=1,
                                )
                                for i1, j1 in tir.grid(16, 16):
                                    with tir.block([16, 16]) as [vii, vjj]:
                                        tir.bind(vii, i1)
                                        tir.bind(vjj, j1)
                                        tir.reads([])
                                        tir.writes(C0[vii : vii + 1, vjj : vjj + 1])

                                        C0[vii, vjj] = tir.float32(0)

                        for ko in range(0, 32):
                            # copy data from global to shared
                            for tx in tir.thread_binding(0, 64, "threadIdx.x"):
                                for i0, j0 in tir.grid(1, 2):
                                    for j1 in tir.vectorized(0, 4):
                                        with tir.block([1024, 1024]) as [vi, vj]:
                                            tir.bind(vi, bx * 64 + tx + i0)
                                            tir.bind(vj, ko * 32 + tz * 16 + ty * 8 + j0 * 4 + j1)
                                            shared_A[vi, vj] = A[vi, vj]

                                for i0, j0 in tir.grid(1, 4):
                                    for j1 in tir.vectorized(0, 4):
                                        with tir.block([1024, 1024]) as [vi, vj]:
                                            tir.bind(vi, by * 128 + ty * 64 + tx + i0)
                                            tir.bind(vj, ko * 32 + tz * 16 + j0 * 4 + j1)
                                            shared_B[vi, vj] = B[vi, vj]

                            for ki in range(0, 2):
                                for i in range(0, 2):
                                    with tir.block([64, 64]) as [vi, vk]:
                                        tir.bind(vi, bx * 4 + ty * 2 + i)
                                        tir.bind(vk, ko * 2 + ki)
                                        tir.reads(
                                            shared_A[
                                                vi * 16 : vi * 16 + 16,
                                                vk * 16 : vk * 16 + 16,
                                            ]
                                        )
                                        tir.writes(
                                            wmma_A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16]
                                        )
                                        s0 = tir.var("int32")
                                        s1 = tir.var("int32")
                                        A0 = tir.match_buffer(
                                            shared_A[
                                                vi * 16 : vi * 16 + 16,
                                                vk * 16 : vk * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=[s0, s1],
                                            scope="shared",
                                            offset_factor=1,
                                        )
                                        wmma_A0 = tir.match_buffer(
                                            wmma_A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="warp.mfma_16x16x16_matrix_a",
                                            offset_factor=1,
                                        )

                                        for i1, j1 in tir.grid(16, 16):
                                            with tir.block([16, 16]) as [vii, vjj]:
                                                tir.bind(vii, i1)
                                                tir.bind(vjj, j1)
                                                tir.reads([A0[vii, vjj]])
                                                tir.writes(wmma_A0[vii, vjj])
                                                wmma_A0[vii, vjj] = A0[vii, vjj]

                                for j in range(0, 4):
                                    with tir.block([64, 64]) as [vj, vk]:
                                        tir.bind(vj, by * 8 + tz * 4 + j)
                                        tir.bind(vk, ko * 2 + ki)
                                        tir.reads(
                                            shared_B[
                                                vj * 16 : vj * 16 + 16,
                                                vk * 16 : vk * 16 + 16,
                                            ]
                                        )
                                        tir.writes(
                                            wmma_B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16]
                                        )
                                        s0 = tir.var("int32")
                                        s1 = tir.var("int32")
                                        B0 = tir.match_buffer(
                                            shared_B[
                                                vj * 16 : vj * 16 + 16,
                                                vk * 16 : vk * 16 + 16,
                                            ],
                                            (16, 16),
                                            "float16",
                                            strides=[s0, s1],
                                            scope="shared",
                                            offset_factor=1,
                                        )
                                        wmma_B0 = tir.match_buffer(
                                            wmma_B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="warp.mfma_16x16x16_matrix_a",
                                            offset_factor=1,
                                        )

                                        for i1, j1 in tir.grid(16, 16):
                                            with tir.block([16, 16]) as [vii, vjj]:
                                                tir.bind(vii, i1)
                                                tir.bind(vjj, j1)
                                                tir.reads([B0[vii, vjj]])
                                                tir.writes([wmma_B0[vii, vjj]])
                                                wmma_B0[vii, vjj] = B0[vii, vjj]

                                for i, j in tir.grid(2, 4):
                                    with tir.block([64, 64, tir.reduce_axis(0, 64)]) as [
                                        vi,
                                        vj,
                                        vk,
                                    ]:
                                        tir.bind(vi, bx * 4 + ty * 2 + i)
                                        tir.bind(vj, by * 8 + tz * 4 + j)
                                        tir.bind(vk, ko * 2 + ki)
                                        tir.reads(
                                            [
                                                wmma_A[
                                                    vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16
                                                ],
                                                wmma_B[
                                                    vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16
                                                ],
                                                wmma_C[
                                                    vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16
                                                ],
                                            ]
                                        )
                                        tir.writes(
                                            wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16]
                                        )
                                        wmma_A1 = tir.match_buffer(
                                            wmma_A[vi * 16 : vi * 16 + 16, vk * 16 : vk * 16 + 16],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="warp.mfma_16x16x16_matrix_a",
                                            offset_factor=1,
                                        )
                                        wmma_B1 = tir.match_buffer(
                                            wmma_B[vj * 16 : vj * 16 + 16, vk * 16 : vk * 16 + 16],
                                            (16, 16),
                                            "float16",
                                            strides=[16, 1],
                                            scope="warp.mfma_16x16x16_matrix_a",
                                            offset_factor=1,
                                        )
                                        wmma_C1 = tir.match_buffer(
                                            wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                                            (16, 16),
                                            "float32",
                                            strides=[16 * 4, 1],
                                            scope="warp.mfma_16x16x16_matrix_c",
                                            offset_factor=1,
                                        )

                                        tir.evaluate(
                                            tir.tvm_mfma_sync(
                                                wmma_C1.data,
                                                i * 4 + j,
                                                wmma_A1.data,
                                                i,
                                                wmma_B1.data,
                                                j,
                                                wmma_C1.data,
                                                i * 4 + j,
                                                dtype="handle",
                                            )
                                        )
                        for i, j in tir.grid(2, 4):
                            with tir.block([64, 64]) as [vi, vj]:
                                tir.bind(vi, bx * 4 + ty * 2 + i)
                                tir.bind(vj, by * 8 + tz * 4 + j)
                                tir.reads(wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                                tir.writes(C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                                s0 = tir.var("int32")
                                s1 = tir.var("int32")
                                wmma_C2 = tir.match_buffer(
                                    wmma_C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                                    (16, 16),
                                    "float32",
                                    strides=[16 * 4, 1],
                                    scope="warp.mfma_16x16x16_matrix_c",
                                    offset_factor=1,
                                )
                                C1 = tir.match_buffer(
                                    C[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16],
                                    (16, 16),
                                    "float32",
                                    strides=[s0, s1],
                                    offset_factor=1,
                                )

                                for tx in tir.thread_binding(0, 64, "threadIdx.x"):
                                    for r in range(0, 4):
                                        with tir.block([16, 16]) as [vii, vjj]:
                                            tir.bind(vii, tir.floormod(tx, 16))
                                            tir.bind(vjj, tir.floordiv(tx, 16) * 4 + r)
                                            tir.reads([wmma_C2[vii, vjj]])
                                            tir.writes([C1[vii, vjj]])
                                            C1[vii, vjj] = wmma_C2[vii, vjj]


def test_gemm_tensorcore():
    dev = tvm.device("rocm", 0)
    print(tvm.script.asscript(tvm.lower(tensorcore_gemm, simple_mode=True)))
    f = tvm.build(tensorcore_gemm, target="rocm", name="dense")
    a_np = np.random.uniform(size=(1024, 1024)).astype("float16")

    b_np = np.random.uniform(size=(1024, 1024)).astype("float16")

    c_np = np.dot(a_np.astype("float32"), b_np.T.astype("float32"))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)
    f(a, b, c)
    cc = c.numpy()
    np.set_printoptions(threshold=np.inf)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=100)
    t = evaluator(a, b, c).mean
    num_flops = 2 * 1024 * 1024 * 1024
    gflops = num_flops / (t * 1e3) / 1e6
    print("gemm with tensor core: %f ms" % (t * 1e3))
    print("GFLOPS: %f" % gflops)


if __name__ == "__main__":
    test_gemm_tensorcore()
