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

import pytest

import tvm
import util
from tvm import tir
from tvm.hybrid import ty


@tvm.hybrid.script
def predicate_vectorize(b: ty.handle, c: ty.handle) -> None:
    C = tir.buffer_bind(c, (16, 16), "float32")
    B = tir.buffer_bind(b, (16, 16), "float32")
    with tir.block({}, writes=[], reads=[], name="root"):
        for i, jo in tir.grid(16, 4):
            for ji in range(0, 4, annotation={"loop_type": "vectorize"}):
                with tir.block({vi(0, 16): i, vj(0, 16): ((jo * 4) + ji)},
                               writes=C[vi, vj], reads=B[vi, vj],
                               predicate=(((jo * 4) + ji) < 16), name="update"):
                    C[vi, vj] = (B[vi, vj] + tir.float32(1))


@tvm.hybrid.script
def predicate_unroll(b: ty.handle, c: ty.handle) -> None:
    C = tir.buffer_bind(c, (16, 16), "float32")
    B = tir.buffer_bind(b, (16, 16), "float32")
    with tir.block({}, writes=[], reads=[], name="root"):
        for i in range(0, 16, annotation={}):
            for jo in range(0, 4, annotation={}):
                for ji in range(0, 4, annotation={"loop_type": "unroll"}):
                    with tir.block({vi(0, 16): i, vj(0, 16): ((jo * 4) + ji)},
                                   writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                                   reads=[B[vi:(vi + 1), vj:(vj + 1)]],
                                   predicate=(((jo * 4) + ji) < 16), name="update"):
                        C[vi, vj] = (B[vi, vj] + tir.float32(1))


def test_vectorize_normal():
    func = util.predicate_stmt()

    s = tir.create_schedule(func)
    B = s.get_block("update")
    i, jo, ji = s.get_axes(B)
    s.vectorize(ji)

    mod = tvm.hybrid.create_module(
        {"predicate_vectorize": predicate_vectorize})
    tvm.ir.assert_structural_equal(s.func, mod["predicate_vectorize"])


@tvm.hybrid.script
def element_wise_compute_at(a: ty.handle, c: ty.handle) -> None:
    C = tir.buffer_bind(c, (128, 128), "float32")
    A = tir.buffer_bind(a, (128, 128), "float32")
    with tir.block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128]], name="root"):
        B = tir.buffer_allocate((128, 128), "float32")
        for i, j in tir.grid(128, 128):
            with tir.block({vi(0, 128): i, vj(0, 128): j}, writes=B[vi, vj],reads=A[vi, vj], name="B"):
                B[vi, vj] = (A[vi, vj] * tir.float32(2))
            with tir.block({vi(0, 128): i, vj(0, 128): j}, writes=C[vi, vj],reads=B[vi, vj], name="C"):
                C[vi, vj] = (B[vi, vj] + tir.float32(1))


@tvm.hybrid.script
def element_wise_compute_at_vectorize(a: ty.handle, c: ty.handle) -> None:
    A = tir.buffer_bind(a, (128, 128), "float32")
    C = tir.buffer_bind(c, (128, 128), "float32")
    with tir.block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128]], name="root"):
        B = tir.buffer_allocate((128, 128), "float32")
        for i in range(0, 128, annotation={}):
            for j_outer in range(0, 32, annotation={}):
                for j_inner in range(0, 4, annotation={"loop_type": "vectorize"}):
                    with tir.block({vi(0, 128): i, vj(0, 128): ((j_outer * 4) + j_inner)}, writes=B[vi, vj], reads=A[vi, vj], name="B"):
                        B[vi, vj] = (A[vi, vj] * tir.float32(2))
                    with tir.block({vi(0, 128): i, vj(0, 128): ((j_outer * 4) + j_inner)}, writes=C[vi, vj], reads=B[vi, vj], name="C"):
                        C[vi, vj] = (B[vi, vj] + tir.float32(1))


def test_vectorize_complete():
    mod = tvm.hybrid.create_module(
        {"element_wise_compute_at": element_wise_compute_at})
    func = mod["element_wise_compute_at"]

    # schedule
    s = tir.create_schedule(func)
    C = s.get_block("C")
    outer, inner = s.get_axes(C)
    i_o, i_i = s.split(inner, 4)
    s.vectorize(i_i)

    mod = tvm.hybrid.create_module(
        {"element_wise_compute_at_vectorize": element_wise_compute_at_vectorize})
    tvm.ir.assert_structural_equal(
        s.func, mod["element_wise_compute_at_vectorize"])


def test_vectorize_fail_on_reduce_var():
    func = util.matmul_stmt()
    s = tir.create_schedule(func)
    update = s.get_block("update")
    _, _, k = s.get_axes(update)
    with pytest.raises(ValueError):
        s.vectorize(k)


def test_unroll_normal():
    func = util.predicate_stmt()

    s = tir.create_schedule(func)
    B = s.get_block("update")
    i, jo, ji = s.get_axes(B)
    s.unroll(ji)

    mod = tvm.hybrid.create_module({"predicate_unroll": predicate_unroll})
    tvm.ir.assert_structural_equal(s.func, mod["predicate_unroll"])


if __name__ == "__main__":
    test_vectorize_normal()
    test_vectorize_complete()
    test_vectorize_fail_on_reduce_var()
    test_unroll_normal()
