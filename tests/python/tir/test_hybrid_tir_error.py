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
from tvm.tir.hybrid import ty


@tvm.tir.hybrid.script
def buffer_bind_missing_args(a: ty.handle) -> None:
    A = tir.buffer_bind((16, 16), "float32")


@tvm.tir.hybrid.script
def range_missing_args(a: ty.handle) -> None:
    A = tir.buffer_bind(a, (16, 16), "float32")

    for i in range(16):
        for j in range(0, 16):
            with tir.block({vi(0, 16): i, vj(0, 16): j}, reads=[], writes=C[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vj] = 0.0


@tvm.tir.hybrid.script
def block_missing_args(a: ty.handle) -> None:
    A = tir.buffer_bind(a, (16, 16), "float32")

    for i in tir.grid(0, 16):
        for j in tir.grid(0, 16):
            with tir.block({vi(0, 16): i, vj(0, 16): j}, writes=A[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vj] = 0.0


@tvm.tir.hybrid.script
def undefined_buffer(a: ty.handle) -> None:
    A = tir.buffer_bind(a, (16, 16), "float32")

    for i in tir.grid(0, 16):
        for j in tir.grid(0, 16):
            with tir.block({vi(0, 16): i, vj(0, 16): j}, reads=[], writes=C[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vj] = 0.0


@tvm.tir.hybrid.script
def undefined_block_var(a: ty.handle) -> None:
    A = tir.buffer_bind(a, (16, 16), "float32")

    for i in tir.grid(0, 16):
        for j in tir.grid(0, 16):
            with tir.block({vi(0, 16): i, vj(0, 16): j}, reads=[], writes=A[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vk] = 0.0


@tvm.tir.hybrid.script
def unsupported_stmt(a: ty.int32) -> None:
    if a > 0:
        print("I love tvm")


@tvm.tir.hybrid.script
def unsupported_function_call(a: ty.handle) -> None:
    A = tir.buffer_bind(a, (16, 16), "float32")

    for i in tir.const_range(0, 16):
        for j in tir.grid(0, 16):
            with tir.block({vi(0, 16): i, vj(0, 16): j}, reads=[], writes=A[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vk] = 0.0


@tvm.tir.hybrid.script
def type_check(a: ty.handle) -> None:
    A = tir.buffer_bind(a, (16, 16), "float32")

    for i in tir.range(0, 16):
        for j in tir.range(0, 16):
            with tir.block({vi(0, 16): i, vj(0, 16): j}, reads=[], writes=a[vi: vi + 1, vj: vj + 1], name="init"):
                A[vi, vj] = 0.0


def wrap_error(func, lineno):
    try:
        mod = tvm.tir.hybrid.create_module({"func": func})
        res = mod[func.__name__]
    except BaseException as e:
        print(e)
        msg = str(e).split('\n')[-1].split(':', maxsplit=1)[0].strip().split(' ')[-1].strip()
        assert int(msg) == lineno
        return

    assert False


if __name__ == '__main__':
    wrap_error(buffer_bind_missing_args, 25)
    wrap_error(range_missing_args, 32)
    wrap_error(block_missing_args, 44)
    wrap_error(undefined_buffer, 54)
    wrap_error(undefined_block_var, 65)
    wrap_error(unsupported_stmt, 71)
    wrap_error(unsupported_function_call, 78)
    wrap_error(type_check, 90)
