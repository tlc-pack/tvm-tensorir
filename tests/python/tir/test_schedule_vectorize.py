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


@tvm.tir.hybrid.script
def predicate_vectorize(b, c):
    C = buffer_bind(c, (16, 16), "float32")
    B = buffer_bind(b, (16, 16), "float32")
    with block({}, writes=[], reads=[], name="root"):
        for i in range(0, 16, annotation={}):
            for jo in range(0, 4, annotation={}):
                for ji in range(0, 4, annotation={"loop_type": "vectorize"}):
                    with block({vi(0, 16): i, vj(0, 16): ((jo * 4) + ji)},
                               writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                               reads=[B[vi:(vi + 1), vj:(vj + 1)]],
                               predicate=(((jo * 4) + ji) < 16), name="update"):
                        C[vi, vj] = (B[vi, vj] + float32(1))


@tvm.tir.hybrid.script
def predicate_unroll(b, c):
    C = buffer_bind(c, (16, 16), "float32")
    B = buffer_bind(b, (16, 16), "float32")
    with block({}, writes=[], reads=[], name="root"):
        for i in range(0, 16, annotation={}):
            for jo in range(0, 4, annotation={}):
                for ji in range(0, 4, annotation={"loop_type": "unroll"}):
                    with block({vi(0, 16): i, vj(0, 16): ((jo * 4) + ji)},
                               writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                               reads=[B[vi:(vi + 1), vj:(vj + 1)]],
                               predicate=(((jo * 4) + ji) < 16), name="update"):
                        C[vi, vj] = (B[vi, vj] + float32(1))


def test_vectorize_normal():
    func = util.predicate_stmt()

    s = tir.create_schedule(func)
    B = s.get_block("update")
    i, jo, ji = s.get_axes(B)
    s.vectorize(ji)

    mod = tir.hybrid.create_module({"predicate_vectorize": predicate_vectorize})
    tvm.ir.assert_structural_equal(s.func, mod["predicate_vectorize"])


@tvm.tir.hybrid.script
def element_wise_compute_at(a, c):
    C = buffer_bind(c, (128, 128), "float32")
    A = buffer_bind(a, (128, 128), "float32")
    with block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128]], name="root"):
        B = buffer_allocate((128, 128), "float32", "")
        for i in range(0, 128, annotation={}):
            for j in range(0, 128, annotation={}):
                with block({vi(0, 128): i, vj(0, 128): j}, writes=[B[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[A[vi:(vi + 1), vj:(vj + 1)]], name="B"):
                    B[vi, vj] = (A[vi, vj] * float32(2))
                with block({vi(0, 128): i, vj(0, 128): j}, writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                           reads=[B[vi:(vi + 1), vj:(vj + 1)]], name="C"):
                    C[vi, vj] = (B[vi, vj] + float32(1))


@tvm.tir.hybrid.script
def element_wise_compute_at_vectorize(a, c):
    A = buffer_bind(a, (128, 128), "float32")
    C = buffer_bind(c, (128, 128), "float32")
    with block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128]], name="root"):
        B = buffer_allocate((128, 128), "float32", "")
        for i in range(0, 128, annotation={}):
            for j_outer in range(0, 32, annotation={}):
                for j_inner in range(0, 4, annotation={"loop_type": "vectorize"}):
                    with block({vi(0, 128): i, vj(0, 128): ((j_outer * 4) + j_inner)},
                               writes=[B[vi:(vi + 1), vj:(vj + 1)]],
                               reads=[A[vi:(vi + 1), vj:(vj + 1)]], name="B"):
                        B[vi, vj] = (A[vi, vj] * float32(2))
                    with block({vi(0, 128): i, vj(0, 128): ((j_outer * 4) + j_inner)},
                               writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                               reads=[B[vi:(vi + 1), vj:(vj + 1)]], name="C"):
                        C[vi, vj] = (B[vi, vj] + float32(1))


def test_vectorize_complete():
    mod = tvm.tir.hybrid.create_module({"element_wise_compute_at": element_wise_compute_at})
    func = mod["element_wise_compute_at"]

    # schedule
    s = tir.create_schedule(func)
    C = s.get_block("C")
    outer, inner = s.get_axes(C)
    i_o, i_i = s.split(inner, 4)
    s.vectorize(i_i)

    mod = tir.hybrid.create_module(
        {"element_wise_compute_at_vectorize": element_wise_compute_at_vectorize})
    tvm.ir.assert_structural_equal(s.func, mod["element_wise_compute_at_vectorize"])


def test_unroll_normal():
    func = util.predicate_stmt()

    s = tir.create_schedule(func)
    B = s.get_block("update")
    i, jo, ji = s.get_axes(B)
    s.unroll(ji)

    mod = tir.hybrid.create_module({"predicate_unroll": predicate_unroll})
    tvm.ir.assert_structural_equal(s.func, mod["predicate_unroll"])


if __name__ == "__main__":
    test_vectorize_normal()
    test_vectorize_complete()
    test_unroll_normal()
