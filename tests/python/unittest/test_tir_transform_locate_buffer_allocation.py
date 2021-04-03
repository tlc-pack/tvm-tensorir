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
from tvm import tir
from tvm.script import ty


@tvm.script.tir
def original_func() -> None:
    A = tir.alloc_buffer((128, 128), "float32")
    with tir.block([128, 128]) as [i, j]:
        A[i, j] = tir.float32(0)
    with tir.block([32, 32, tir.reduce_axis(0, 32)]) as [i, j, k]:
        B = tir.alloc_buffer((128, 128), "float32")
        C = tir.alloc_buffer((128, 128), "float32")
        D = tir.alloc_buffer((128, 128), "float32")
        with tir.init():
            for ii, jj in tir.grid(4, 4):
                B[i * 4 + ii, j * 4 + jj] = A[i * 4 + ii, j * 4 + jj]
        for ii, jj in tir.grid(4, 4):
            for kk in range(0, 4):
                B[i * 4 + ii, j * 4 + jj] += C[i * 4 + ii, k * 4 + kk]
            for kk in range(0, 4):
                B[i * 4 + ii, j * 4 + jj] += D[j * 4 + jj, k * 4 + kk] * C[i * 4 + ii, k * 4 + kk]


@tvm.script.tir
def transformed_func() -> None:
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        A = tir.alloc_buffer([128, 128])
        for i0, i1 in tir.grid(128, 128):
            with tir.block([128, 128], "") as [i, j]:
                tir.bind(i, i0)
                tir.bind(j, i1)
                tir.reads([])
                tir.writes([A[i, j]])
                A[i, j] = tir.float32(0)
        for i0_1, i1_1, i2 in tir.grid(32, 32, 32):
            with tir.block([32, 32, tir.reduce_axis(0, 32)], "") as [i_1, j_1, k]:
                tir.bind(i_1, i0_1)
                tir.bind(j_1, i1_1)
                tir.bind(k, i2)
                tir.reads([A[(i_1 * 4) : ((i_1 * 4) + 4), (j_1 * 4) : ((j_1 * 4) + 4)]])
                tir.writes([])
                B = tir.alloc_buffer([128, 128])
                with tir.init():
                    for ii, jj in tir.grid(4, 4):
                        B[((i_1 * 4) + ii), ((j_1 * 4) + jj)] = A[
                            ((i_1 * 4) + ii), ((j_1 * 4) + jj)
                        ]
                for ii_1, jj_1 in tir.grid(4, 4):
                    with tir.block([], ""):
                        tir.reads([])
                        tir.writes([])
                        C = tir.alloc_buffer([128, 128])
                        for kk in tir.serial(0, 4):
                            B[((i_1 * 4) + ii_1), ((j_1 * 4) + jj_1)] = (
                                B[((i_1 * 4) + ii_1), ((j_1 * 4) + jj_1)]
                                + C[((i_1 * 4) + ii_1), ((k * 4) + kk)]
                            )
                        for kk_1 in tir.serial(0, 4):
                            with tir.block([], "") as []:
                                tir.reads([])
                                tir.writes([])
                                D = tir.alloc_buffer([128, 128])
                                B[((i_1 * 4) + ii_1), ((j_1 * 4) + jj_1)] = B[
                                    ((i_1 * 4) + ii_1), ((j_1 * 4) + jj_1)
                                ] + (
                                    D[((j_1 * 4) + jj_1), ((k * 4) + kk_1)]
                                    * C[((i_1 * 4) + ii_1), ((k * 4) + kk_1)]
                                )


def test_locate_buffer_allocation():
    func = original_func
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.LocateBufferAllocation()(mod)

    # Blocked by parsing opaque block
    # tvm.ir.assert_structural_equal(mod["main"], transformed_func)


if __name__ == "__main__":
    test_locate_buffer_allocation()
