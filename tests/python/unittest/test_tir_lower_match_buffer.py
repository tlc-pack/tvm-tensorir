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


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.LowerMatchBuffer()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed)


@tvm.script.tir
def buffer_load_store(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16, 16))
    C = tir.match_buffer(c, (16, 16))
    for i, j, k in tir.grid(4, 16, 8):
        with tir.block([]):
            tir.reads(C[i * 4 : i * 4 + 4, k * 2 : k * 2 + 2])
            tir.writes(A[i * 4 : i * 4 + 4, j, k * 2 : k * 2 + 2])
            sub_A = tir.match_buffer_region(A[i * 4 : i * 4 + 4, j, k * 2 : k * 2 + 2])
            sub_C = tir.match_buffer_region(C[i * 4 : i * 4 + 4, k * 2 : k * 2 + 2])
            for ii, jj in tir.grid(4, 2):
                sub_A[ii, jj] += sub_C[ii, jj]


@tvm.script.tir
def transformed_buffer_load_store(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16, 16))
    C = tir.match_buffer(c, (16, 16))
    for i, j, k in tir.grid(4, 16, 8):
        with tir.block([]):
            tir.reads(C[i * 4 : i * 4 + 4, k * 2 : k * 2 + 2])
            tir.writes(A[i * 4 : i * 4 + 4, j, k * 2 : k * 2 + 2])
            for ii, jj in tir.grid(4, 2):
                A[i * 4 + ii, j, k * 2 + jj] += C[i * 4 + ii, k * 2 + jj]


@tvm.ir.register_op_attr("tir.test_intrin", "")
def test_intrin(data, elem_offset, stride_0, stride_1, shape_0, shape_1):
    return 0


@tvm.script.tir
def opaque_access(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (32, 64, 128))
    B = tir.match_buffer(b, (64, 64, 64))
    for i, j, k in tir.grid(2, 64, 8):
        with tir.block([]):
            tir.reads([])
            tir.writes(A[i * 16 : i * 16 + 16, j, k * 16 : k * 16 + 16])
            sub_A = tir.match_buffer_region(A[i * 16 : i * 16 + 16, j, k * 16 : k * 16 + 16])
            tir.evaluate(
                tir.test_intrin(
                    sub_A.data,
                    sub_A.elem_offset,
                    sub_A.strides[0],
                    sub_A.strides[1],
                    sub_A.shape[0],
                    sub_A.shape[1],
                    dtype="handle",
                )
            )
    for i, j, k in tir.grid(64, 2, 8):
        with tir.block([]):
            tir.reads([])
            tir.writes(B[i, j * 32 : j * 32 + 32, k * 8 : k * 8 + 8])
            sub_B = tir.match_buffer_region(B[i, j * 32 : j * 32 + 32, k * 8 : k * 8 + 8])
            tir.evaluate(
                tir.test_intrin(
                    sub_B.data,
                    sub_B.elem_offset,
                    sub_B.strides[0],
                    sub_B.strides[1],
                    sub_B.shape[0],
                    sub_B.shape[1],
                    dtype="handle",
                )
            )


@tvm.script.tir
def transformed_opaque_access(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (32, 64, 128))
    B = tir.match_buffer(b, (64, 64, 64))
    for i, j, k in tir.grid(2, 64, 8):
        with tir.block([]):
            tir.reads([])
            tir.writes(A[i * 16 : i * 16 + 16, j, k * 16 : k * 16 + 16])
            tir.evaluate(
                tir.test_intrin(
                    A.data,
                    i * 131072 + j * 128 + k * 16,
                    8192,
                    1,
                    16,
                    16,
                    dtype="handle",
                )
            )
    for i, j, k in tir.grid(64, 2, 8):
        with tir.block([]):
            tir.reads([])
            tir.writes(B[i, j * 32 : j * 32 + 32, k * 8 : k * 8 + 8])
            tir.evaluate(
                tir.test_intrin(
                    B.data,
                    i * 4096 + j * 2048 + k * 8,
                    64,
                    1,
                    32,
                    8,
                    dtype="handle",
                )
            )


def test_buffer_load_store():
    _check(buffer_load_store, transformed_buffer_load_store)


def test_opaque_access():
    _check(opaque_access, transformed_opaque_access)


if __name__ == "__main__":
    test_buffer_load_store()
    test_opaque_access()
