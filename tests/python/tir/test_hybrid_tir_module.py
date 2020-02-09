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


@tvm.tir.hybrid.script
def matmul(a, b, c):
    A = buffer_bind(a, (16, 16), "float32", name="A")
    B = buffer_bind(b, (16, 16), "float32", name="B")
    C = buffer_bind(c, (16, 16), "float32", name="C")

    with block({}, reads=[A[0: 16, 0: 16], B[0: 16, 0: 16]], writes=C[0: 16, 0: 16], name="root"):
        for i in range(0, 16):
            for j in range(0, 16):
                with block({vi(0, 16): i, vj(0, 16): j}, reads=[], writes=C[vi: vi + 1, vj: vj + 1],
                           name="init"):
                    C[vi, vj] = float32(0)
                for k in range(0, 16):
                    with block({vi(0, 16): i, vj(0, 16): j, vk(0, 16, iter_type="reduce"): k},
                               reads=[C[vi: vi + 1, vj: vj + 1], A[vi: vi + 1, vk: vk + 1],
                                      B[vj: vj + 1, vk: vk + 1]],
                               writes=[C[vi: vi + 1, vj: vj + 1]], name="update"):
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.tir.hybrid.script
def element_wise(a, c):
    A = buffer_bind(a, (16, 16), "float32", name="A")
    C = buffer_bind(c, (16, 16), "float32", name="C")

    with block({}, A[0: 16, 0: 16], C[0: 16,  0: 16], name="root"):
        B = buffer_allocate((16, 16), "float32", name="B")

        for i in range(0, 16):
            for j in range(0, 16):
                with block({vi(0, 16): i, vj(0, 16): j}, A[vi: vi + 1, vj: vj + 1],
                           B[vi: vi + 1, vj: vj + 1],
                           name="B"):
                    B[vi, vj] = A[vi, vj] * 2

        for i in range(0, 16):
            for j in range(0, 16):
                with block({vi(0, 16): i, vj(0, 16): j}, B[vi: vi + 1, vj: vj + 1],
                           C[vi: vi + 1, vj: vj + 1],
                           name="C"):
                    C[vi, vj] = B[vi, vj] + 1


@tvm.tir.hybrid.script
def predicate(b, c):
    B = buffer_bind(b, (16, 16), "float32", name="B")
    C = buffer_bind(c, (16, 16), "float32", name="C")

    with block({}, reads=[], writes=[], name="root"):
        for i in range(0, 16):
            for jo in range(0, 4):
                for ji in range(0, 4):
                    with block({vi(0, 16): i, vj(0, 16): jo * 3 + ji},
                               reads=B[vi: vi + 1, vj: vj + 1], writes=C[vi: vi + 1, vj: vj + 1],
                               predicate=jo * 4 + ji < 16):
                        C[vi, vj] = B[vi, vj] + 1


def test_module_define():
    func1_var = tvm.tir.hybrid.global_var("matmul")
    func2_var = tvm.tir.hybrid.global_var("element_wise")
    func3_var = tvm.tir.hybrid.global_var("predicate")

    a = tvm.var("a")
    b = tvm.var("b")
    c = tvm.var("c")

    func1 = matmul(a, b, c)
    func2 = element_wise(a, c)
    func3 = predicate(b, c)

    mod = tvm.tir.hybrid.create_module({func1_var: func1, func2_var: func2, func3_var: func3})

    print(tvm.tir.hybrid.to_python(mod))


if __name__ == '__main__':
    test_module_define()
