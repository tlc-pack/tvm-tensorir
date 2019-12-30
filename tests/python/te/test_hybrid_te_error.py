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


def wrap_error(func, *args):
    try:
        res, _, _ = func(*args)
        print(res)
    except BaseException as e:
        print(e)
        return

    assert False


@tvm.hybrid_te.script
def buffer_bind_missing_args(a):
    A = buffer_bind((16, 16), "float32", name="A")


@tvm.hybrid_te.script
def range_missing_args(a):
    A = buffer_bind(a, (16, 16), "float32", name="A")

    for i in range(16):
        for j in range(0, 16):
            with block([vi(0, 16), vj(0, 16)], [i, j], reads=[], writes=C[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vj] = 0.0


@tvm.hybrid_te.script
def block_missing_args(a):
    A = buffer_bind(a, (16, 16), "float32", name="A")

    for i in range(0, 16):
        for j in range(0, 16):
            with block([vi(0, 16), vj(0, 16)], [i, j], writes=A[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vj] = 0.0


@tvm.hybrid_te.script
def undefined_buffer(a):
    A = buffer_bind(a, (16, 16), "float32", name="A")

    for i in range(0, 16):
        for j in range(0, 16):
            with block([vi(0, 16), vj(0, 16)], [i, j], reads=[], writes=C[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vj] = 0.0


@tvm.hybrid_te.script
def undefined_block_var(a):
    A = buffer_bind(a, (16, 16), "float32", name="A")

    for i in range(0, 16):
        for j in range(0, 16):
            with block([vi(0, 16), vj(0, 16)], [i, j], reads=[], writes=A[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vk] = 0.0


@tvm.hybrid_te.script
def unsupported_stmt(a):
    if a > 0:
        print("I love tvm")


@tvm.hybrid_te.script
def unsupported_function_call(a):
    A = buffer_bind(a, (16, 16), "float32", name="A")

    for i in const_range(0, 16):
        for j in range(0, 16):
            with block([vi(0, 16), vj(0, 16)], [i, j], reads=[], writes=A[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vk] = 0.0


@tvm.hybrid_te.script
def type_check(a):
    A = buffer_bind(a, (16, 16), "float32", name="A")

    for i in range(0, 16):
        for j in range(0, 16):
            B = buffer_allocate(A, (16, 16), "float32", name="A")
            with block([vi(0, 16), vj(0, 16)], [i, j], reads=[], writes=A[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vj] = 0.0


if __name__ == '__main__':
    a = tvm.var("a")
    b = tvm.var("b")
    c = tvm.var("c")

    wrap_error(buffer_bind_missing_args, a)
    wrap_error(range_missing_args, a)
    wrap_error(block_missing_args, a)
    wrap_error(undefined_buffer, a)
    wrap_error(undefined_block_var, a)
    wrap_error(unsupported_stmt, a)
    wrap_error(unsupported_function_call, a)
    wrap_error(type_check, a)
