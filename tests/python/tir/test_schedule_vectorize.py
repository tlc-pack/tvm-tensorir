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
# pylint: disable=missing-function-docstring,missing-module-docstring
import pytest
import tvm
from tvm import tir
from tvm.script import ty

import util

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name


@tvm.script.tir
def predicate_vectorize(b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (16, 16), "float32")
    B = tir.match_buffer(b, (16, 16), "float32")
    for i, jo in tir.grid(16, 4):
        for ji in tir.vectorized(0, 4):
            with tir.block([16, 16], "update") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, (jo * 4) + ji)
                C[vi, vj] = (B[vi, vj] + tir.float32(1))


@tvm.script.tir
def predicate_unroll(b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (16, 16), "float32")
    B = tir.match_buffer(b, (16, 16), "float32")
    for i, jo in tir.grid(16, 4):
        for ji in tir.unroll(0, 4):
            with tir.block([16, 16], "update") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, (jo * 4) + ji)
                C[vi, vj] = (B[vi, vj] + tir.float32(1))


@tvm.script.tir
def element_wise_compute_at(a: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (128, 128), "float32")
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "B") as [vi, vj]:
            B[vi, vj] = (A[vi, vj] * tir.float32(2))
        with tir.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = (B[vi, vj] + tir.float32(1))


@tvm.script.tir
def element_wise_compute_at_vectorize(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    for i, j_outer in tir.grid(128, 32):
        for j_inner in tir.vectorized(0, 4):
            with tir.block([128, 128], "B") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, ((j_outer * 4) + j_inner))
                B[vi, vj] = (A[vi, vj] * tir.float32(2))
            with tir.block([128, 128], "C") as [vi, vj]:
                tir.bind(vi, i)
                tir.bind(vj, (j_outer * 4) + j_inner)
                C[vi, vj] = (B[vi, vj] + tir.float32(1))

# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name
# fmt: on
# pylint: disable=invalid-name


def test_vectorize_normal():
    func = util.predicate_stmt()
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("update")
    _, _, ji = s.get_loops(B)
    s.vectorize(ji)
    mod = tvm.script.create_module({"predicate_vectorize": predicate_vectorize})
    tvm.ir.assert_structural_equal(s.mod["main"], mod["predicate_vectorize"])


def test_vectorize_complete():
    mod = tvm.script.create_module({"element_wise_compute_at": element_wise_compute_at})
    func = mod["element_wise_compute_at"]
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    C = s.get_block("C")
    _, inner = s.get_loops(C)
    _, i_i = s.split(inner, factors=[None, 4])
    s.vectorize(i_i)
    mod = tvm.script.create_module(
        {"element_wise_compute_at_vectorize": element_wise_compute_at_vectorize}
    )
    tvm.ir.assert_structural_equal(s.mod["main"], mod["element_wise_compute_at_vectorize"])


def test_vectorize_fail_on_reduce_var():
    func = util.matmul_stmt()
    s = tir.Schedule(func, debug_mode=True)
    update = s.get_block("update")
    _, _, k = s.get_loops(update)
    with pytest.raises(tvm.tir.ScheduleError):
        s.vectorize(k)


def test_unroll_normal():
    func = util.predicate_stmt()
    s = tir.Schedule(func, debug_mode=True)
    B = s.get_block("update")
    _, _, ji = s.get_loops(B)
    s.unroll(ji)
    mod = tvm.script.create_module({"predicate_unroll": predicate_unroll})
    tvm.ir.assert_structural_equal(s.mod["main"], mod["predicate_unroll"])


if __name__ == "__main__":
    test_vectorize_normal()
    test_vectorize_complete()
    test_vectorize_fail_on_reduce_var()
    test_unroll_normal()
