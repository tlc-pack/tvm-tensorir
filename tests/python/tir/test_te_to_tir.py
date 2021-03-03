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
import numpy as np
import tvm
import tvm.testing
from tvm import te, tir
from tvm.topi.testing import conv2d_nhwc_python

import util

M = 128
K = 128
N = 128

# pylint: disable=invalid-name


def test_unique_name():
    A = te.placeholder((M, N), name="A")
    B = te.compute((M, N), lambda x, y: A[x, y] * 2, name="main")
    C = te.compute((M, N), lambda x, y: B[x, y] + 1, name="main")
    func = te.create_func([C])
    s = tir.Schedule(func, debug_mode=True)
    assert isinstance(s.get_sref(s.get_block("main")), tir.schedule.StmtSRef)
    assert isinstance(s.get_sref(s.get_block("main_1")), tir.schedule.StmtSRef)


def test_matmul():
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((N, K), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[y, k], axis=k), name="C")

    func = te.create_func([C])
    tvm.ir.assert_structural_equal(func, util.matmul)
    a_np = np.random.uniform(size=(M, K)).astype("float32")
    b_np = np.random.uniform(size=(K, N)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((M, N)).astype("float32"))
    func = tvm.build(func)
    func(a, b, c)
    tvm.testing.assert_allclose(
        c.asnumpy(), np.matmul(a.asnumpy(), b.asnumpy().transpose()), rtol=1e-5
    )


def test_element_wise():
    A = te.placeholder((M, N), name="A")
    B = te.compute((M, N), lambda x, y: A[x, y] * 2, name="B")
    C = te.compute((M, N), lambda x, y: B[x, y] + 1, name="C")

    func = te.create_func([C])
    tvm.ir.assert_structural_equal(func, util.element_wise)
    a_np = np.random.uniform(size=(M, N)).astype("float32")
    a = tvm.nd.array(a_np)
    c = tvm.nd.array(np.zeros((M, N)).astype("float32"))
    func = tvm.build(func)
    func(a, c)
    tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() * 2 + 1, rtol=1e-5)


def test_conv2d():
    batch_size = 16
    height = 5
    width = 5
    in_channels = 16
    out_channels = 16
    kernel_h = 3
    kernel_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    block_size = 16

    # Input feature map: (N, H, W, IC, n, ic)
    data_shape = (
        batch_size // block_size,
        height,
        width,
        in_channels // block_size,
        block_size,
        block_size,
    )
    # Kernel: (H, W, IC, OC, ic, oc)
    kernel_shape = (
        kernel_h,
        kernel_w,
        in_channels // block_size,
        out_channels // block_size,
        block_size,
        block_size,
    )
    # Output feature map: (N, H, W, OC, n, oc)
    output_shape = (
        batch_size // block_size,
        height,
        width,
        out_channels // block_size,
        block_size,
        block_size,
    )

    # Reduction axes
    kh = te.reduce_axis((0, kernel_h), name="kh")
    kw = te.reduce_axis((0, kernel_w), name="kw")
    ic = te.reduce_axis((0, in_channels // block_size), name="ic")
    ii = te.reduce_axis((0, block_size), name="ii")

    # Algorithm
    A = te.placeholder(data_shape, name="A", dtype="float16")
    W = te.placeholder(kernel_shape, name="W", dtype="float16")
    Apad = te.compute(
        (
            batch_size // block_size,
            height + 2 * pad_h,
            width + 2 * pad_w,
            in_channels // block_size,
            block_size,
            block_size,
        ),
        lambda n, h, w, i, nn, ii: tvm.tir.if_then_else(
            tvm.tir.all(h >= pad_h, h - pad_h < height, w >= pad_w, w - pad_w < width),
            A[n, h - pad_h, w - pad_w, i, nn, ii],
            tvm.tir.const(0.0, "float16"),
        ),
        name="Apad",
    )
    Conv = te.compute(
        output_shape,
        lambda n, h, w, o, nn, oo: te.sum(
            Apad[n, h * stride_h + kh, w * stride_w + kw, ic, nn, ii].astype("float32")
            * W[kh, kw, ic, o, ii, oo].astype("float32"),
            axis=[ic, kh, kw, ii],
        ),
        name="Conv",
    )

    func = te.create_func(Conv)
    func = tvm.build(func)

    a_np = np.random.uniform(size=data_shape).astype(A.dtype)
    w_np = np.random.uniform(size=kernel_shape).astype(W.dtype)
    a = tvm.nd.array(a_np)
    w = tvm.nd.array(w_np)
    c = tvm.nd.array(np.zeros(output_shape, dtype=Conv.dtype))
    func(a, w, c)
    a_np = a_np.transpose((0, 4, 1, 2, 3, 5)).reshape((batch_size, height, width, in_channels))
    w_np = w_np.transpose((0, 1, 2, 4, 3, 5)).reshape(
        (kernel_h, kernel_w, in_channels, out_channels)
    )
    c_np = (
        c.asnumpy().transpose((0, 4, 1, 2, 3, 5)).reshape((batch_size, height, width, out_channels))
    )
    c_std = conv2d_nhwc_python(
        a_np.astype("float16"), w_np.astype("float16"), (1, 1), (1, 1)
    ).astype("float32")
    np.testing.assert_allclose(c_np, c_std, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_unique_name()
    test_matmul()
    test_element_wise()
    test_conv2d()
