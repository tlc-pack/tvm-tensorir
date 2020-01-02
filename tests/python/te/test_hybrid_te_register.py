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

import numpy as np
import tvm


def add(a: tvm.expr.Expr, b: tvm.expr.Expr) -> tvm.expr.Expr:
    return a + b


def mul(a: tvm.expr.Expr, b: tvm.expr.Expr = 1) -> tvm.expr.Expr:
    return a * b


@tvm.hybrid_te.script
def element_wise(a, c):
    A = buffer_bind(a, (16, 16), "float32", name="A")
    C = buffer_bind(c, (16, 16), "float32", name="C")

    with block([], [], A[0: 16, 0: 16], C[0: 16, 0: 16], name="root"):
        B = buffer_allocate((16, 16), "float32", name="B")

        for i in range(0, 16):
            for j in range(0, 16):
                with block([vi(0, 16), vj(0, 16)], [i, j], A[vi: vi + 1, vj: vj + 1], B[vi: vi + 1, vj: vj + 1],
                           name="B"):
                    B[vi, vj] = mul(A[vi, vj])
                    B[vi, vj] = mul(A[vi, vj], 2)

        for i in range(0, 16):
            for j in range(0, 16):
                with block([vi(0, 16), vj(0, 16)], [i, j], B[vi: vi + 1, vj: vj + 1], C[vi: vi + 1, vj: vj + 1],
                           name="C"):
                    C[vi, vj] = add(B[vi, vj], 1)


def test_element_wise(a, c):
    m, n = 16, 16
    func, tensors, tensor_map = element_wise(a, c)

    assert isinstance(func.body, tvm.stmt.TeBlock)
    assert isinstance(func.body.body, tvm.stmt.Block)
    assert isinstance(func.body.body.first, tvm.stmt.Loop)
    assert isinstance(func.body.body.first.body, tvm.stmt.Loop)
    assert isinstance(func.body.body.first.body.body, tvm.stmt.TeBlock)

    assert isinstance(func.body.body.rest, tvm.stmt.Loop)
    assert isinstance(func.body.body.rest.body, tvm.stmt.Loop)
    assert isinstance(func.body.body.rest.body.body, tvm.stmt.TeBlock)

    func = tvm.ir_pass.TeLower(func, tensor_map)
    print(func)
    lower_func = tvm.lower(func, tensors)

    func = tvm.build(lower_func)
    a_np = np.random.uniform(size=(m, n)).astype("float32")
    a = tvm.nd.array(a_np)
    c = tvm.nd.array(np.zeros((m, n)).astype("float32"))
    func(a, c)
    tvm.testing.assert_allclose(c.asnumpy(), a_np * 2 + 1, rtol=1e-6)


if __name__ == '__main__':
    tvm.hybrid_te.register(add)
    tvm.hybrid_te.register(mul)
    a = tvm.var("a")
    c = tvm.var("c")

    test_element_wise(a, c)
