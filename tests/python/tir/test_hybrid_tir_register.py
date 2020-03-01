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


def add(a, b):
    return a + b


def mul(a, b=1):
    return a * b


@tvm.tir.hybrid.script
def element_wise(a, c):
    A = buffer_bind(a, (16, 16), "float32")
    C = buffer_bind(c, (16, 16), "float32")

    with block({}, A[0: 16, 0: 16], C[0: 16, 0: 16], name="root"):
        B = buffer_allocate((16, 16), "float32")

        for i in range(0, 16):
            for j in range(0, 16):
                with block({vi(0, 16): i, vj(0, 16): j}, A[vi: vi + 1, vj: vj + 1],
                           B[vi: vi + 1, vj: vj + 1],
                           name="B"):
                    B[vi, vj] = mul(A[vi, vj], 2)

        for i in range(0, 16):
            for j in range(0, 16):
                with block({vi(0, 16): i, vj(0, 16): j}, B[vi: vi + 1, vj: vj + 1],
                           C[vi: vi + 1, vj: vj + 1],
                           name="C"):
                    C[vi, vj] = add(B[vi, vj], 1)


def test_element_wise():
    tvm.tir.hybrid.register(add)
    tvm.tir.hybrid.register(mul)
    mod = tvm.tir.hybrid.create_module([element_wise])
    func = mod["element_wise"]

    assert isinstance(func.body.block, tir.stmt.Block)
    assert isinstance(func.body.block.body, tir.stmt.SeqStmt)
    assert isinstance(func.body.block.body[0], tir.stmt.Loop)
    assert isinstance(func.body.block.body[0].body, tir.stmt.Loop)
    assert isinstance(func.body.block.body[0].body.body.block, tir.stmt.Block)

    assert isinstance(func.body.block.body[1], tir.stmt.Loop)
    assert isinstance(func.body.block.body[1].body, tir.stmt.Loop)
    assert isinstance(func.body.block.body[1].body.body.block, tir.stmt.Block)


if __name__ == '__main__':
    test_element_wise()
