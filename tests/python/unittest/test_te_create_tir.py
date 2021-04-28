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
import tvm
from tvm.script import ty
from tvm import te, tir


def test_unique_name():
    A = te.placeholder((16, 16), name="A")
    B = te.compute((16, 16), lambda x, y: A[x, y] * 2, name="main")
    C = te.compute((16, 16), lambda x, y: B[x, y] + 1, name="main")
    func = te.create_prim_func(C)
    s = tir.Schedule(func, debug_mode=True)
    assert isinstance(s.get_sref(s.get_block("main")), tir.schedule.StmtSRef)
    assert isinstance(s.get_sref(s.get_block("main_1")), tir.schedule.StmtSRef)


def _check_workload(te_workload, tir_workload):
    func = te.create_prim_func(te_workload())
    tvm.ir.assert_structural_equal(func, tir_workload)
    # make sure that we can create schedule from the func
    s = tir.Schedule(func, debug_mode=True)
    assert s


def te_matmul():
    k = te.reduce_axis((0, 128), "k")
    A = te.placeholder((128, 128), name="A")
    B = te.placeholder((128, 128), name="B")
    C = te.compute((128, 128), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")
    return C


@tvm.script.tir
def tir_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    with tir.block([128, 128, tir.reduce_axis(0, 128)]) as [i, j, k]:
        with tir.init():
            C[i, j] = 0.0
        C[i, j] += A[i, k] * B[j, k]


def te_element_wise():
    A = te.placeholder((128, 128), name="A")
    B = te.compute((128, 128), lambda x, y: A[x, y] * 2, name="B")
    C = te.compute((128, 128), lambda x, y: B[x, y] + 1, name="C")
    return C


@tvm.script.tir
def tir_element_wise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    B = tir.alloc_buffer((128, 128))

    with tir.block([128, 128]) as [i, j]:
        B[i, j] = A[i, j] * 2.0
    with tir.block([128, 128]) as [i, j]:
        C[i, j] = B[i, j] + 1.0


def test_matmul():
    _check_workload(te_matmul, tir_matmul)


def test_element_wise():
    _check_workload(te_element_wise, tir_element_wise)


def te_conv2d():
    batch = 16
    in_channel = 16
    out_channel = 32
    size = 14
    kernel = 3

    A = te.placeholder((batch, in_channel, size, size), name="A")
    W = te.placeholder((in_channel, kernel, kernel, out_channel), name="W")
    Apad = te.compute(
        (batch, in_channel, size + 2, size + 2),
        lambda nn, cc, yy, xx: tvm.tir.if_then_else(
            tvm.tir.all(yy >= 1, yy - 1 < size, xx >= 1, xx - 1 < size),
            A[nn, cc, yy - 1, xx - 1],
            0.0,
        ),
        name="Apad",
    )
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel), name="ry")
    rx = te.reduce_axis((0, kernel), name="rx")
    B = te.compute(
        (batch, out_channel, size, size),
        lambda nn, ff, yy, xx: te.sum(
            Apad[nn, rc, yy + ry, xx + rx] * W[rc, ry, rx, ff], axis=[rc, ry, rx]
        ),
        name="B",
    )
    return B


@tvm.script.tir
def tir_conv2d(a: ty.handle, w: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16, 14, 14])
    W = tir.match_buffer(w, [16, 3, 3, 32])
    B = tir.match_buffer(b, [16, 32, 14, 14])
    Apad = tir.alloc_buffer([16, 16, 16, 16])
    
    with tir.block([16, 16, 16, 16], "Apad") as [nn, cc, yy, xx]:
        Apad[nn, cc, yy, xx] = tir.if_then_else(
            1 <= yy and yy < 15 and 1 <= xx and xx < 15,
            A[nn, cc, yy - 1, xx - 1],
            0.0,
            dtype="float32"
        )
    with tir.block(
        [16, 32, 14, 14, tir.reduce_axis(0, 16), tir.reduce_axis(0, 3), tir.reduce_axis(0, 3)], "B"
    ) as [nn, ff, yy, xx, rc, ry, rx]:
        with tir.init():
            B[nn, ff, yy, xx] = 0.0
        B[nn, ff, yy, xx] += Apad[nn, rc, yy + ry, xx + rx] * W[rc, ry, rx, ff]


def test_conv2d():
    _check_workload(te_conv2d, tir_conv2d)


if __name__ == "__main__":
    test_unique_name()
    test_matmul()
    test_element_wise()
    test_conv2d()
