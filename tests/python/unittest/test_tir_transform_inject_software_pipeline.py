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
import sys

import tvm
from tvm import tir, te
from tvm.script import tir as T


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.InjectSoftwarePipeline()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed, True)


@T.prim_func
def simple_compute(a: T.handle, c: T.handle):
    A = T.match_buffer(a, (16, 16), dtype="float32")
    C = T.match_buffer(c, (16, 16), dtype="float32")
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        for i in T.serial(0, 16, annotations={"pipeline_scope": 1}):
            with T.block():
                T.reads(A[tx, i])
                T.writes(C[tx, i])
                B = T.alloc_buffer((16, 1), dtype="float32", scope="shared")
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(B[tx, 0])
                    B[tx, 0] = A[tx, i] * T.float32(2)
                with T.block():
                    T.reads(B[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = B[tx, 0] + T.float32(1)


@T.prim_func
def transformed_simple_compute(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [16, 16], dtype="float32")
    C = T.match_buffer(c, [16, 16], dtype="float32")
    for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
        with T.block():
            T.reads([A[tx, 0 : 16]])
            T.writes([C[tx, 0 : 16]])
            B = T.alloc_buffer([2, 16, 1], dtype="float32", scope="shared")
            with T.block():
                T.reads([A[tx, 0]])
                T.writes([B[0, tx, 0]])
                T.block_attr({"pipeline_prologue_scope":True})
                B[0, tx, 0] = A[tx, 0] * T.float32(2)
            with T.block():
                T.reads([A[tx, 1 : 16], B[0 : 2, tx, 0]])
                T.writes([B[0 : 2, tx, 0], C[tx, 0 : 15]])
                T.block_attr({"pipeline_body_scope":True})
                for i in T.serial(0, 15):
                    with T.block():
                        T.reads([A[tx, i + 1]])
                        T.writes([B[(i + 1) % 2, tx, 0]])
                        B[(i + 1) % 2, tx, 0] = A[tx, i + 1] * T.float32(2)
                    with T.block():
                        T.reads([B[i % 2, tx, 0]])
                        T.writes([C[tx, i]])
                        C[tx, i] = B[i % 2, tx, 0] + T.float32(1)
            with T.block():
                T.reads([B[1, tx, 0]])
                T.writes([C[tx, 15]])
                T.block_attr({"pipeline_epilogue_scope":True})
                C[tx, 15] = B[1, tx, 0] + T.float32(1)


def test_simple_compute():
    _check(simple_compute, transformed_simple_compute)


if __name__=='__main__':
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
