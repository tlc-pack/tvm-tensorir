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
"""Test Ansor-like sketch generation in subgraphs in meta schedule"""
# pylint: disable=missing-function-docstring
from typing import Tuple

from tvm import te, topi, tir


def matmul(n: int, m: int, k: int) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    a = te.placeholder((n, k), name="A")
    b = te.placeholder((k, m), name="B")
    k = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (n, m),
        lambda i, j: te.sum(a[i, k] * b[k, j], axis=[k]),
        name="C",
    )
    return [a, b, c]


def matmul_fp16(n: int, m: int, k: int) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    a = te.placeholder((n, k), name="A", dtype="float16")
    b = te.placeholder((k, m), name="B", dtype="float16")
    k = te.reduce_axis((0, k), name="k")

    def f_compute(i, j):
        v_a = tir.Cast(dtype="float32", value=a[i, k])
        v_b = tir.Cast(dtype="float32", value=b[k, j])
        return te.sum(v_a * v_b, axis=[k])

    c = te.compute((n, m), f_compute, name="C")
    return [a, b, c]


def matmul_relu(n: int, m: int, k: int) -> Tuple[te.Tensor, te.Tensor, te.Tensor, te.Tensor]:
    a = te.placeholder((n, k), name="A")
    b = te.placeholder((m, k), name="B")
    k = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (n, m),
        lambda i, j: te.sum(a[i, k] * b[k, j], axis=[k]),
        name="C",
    )
    d = topi.nn.relu(c)  # pylint: disable=invalid-name
    return [a, b, d]


def conv2d_nchw(  # pylint: disable=invalid-name
    n: int,
    h: int,
    w: int,
    ci: int,
    co: int,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((n, ci, h, w), name="X")
    w = te.placeholder((co, ci, kh, kw), name="W")
    y = topi.nn.conv2d_nchw(Input=x, Filter=w, stride=stride, padding=padding, dilation=dilation)
    return [x, w, y]


def conv2d_nchwc(  # pylint: disable=invalid-name
    n: int,
    h: int,
    w: int,
    ci: int,
    co: int,
    kh: int,
    kw: int,
    stride: int,
    in_type: str,
    out_type: str,
):
    PACK_C = 16  # pylint: disable=invalid-name
    assert ci % PACK_C == 0
    assert co % PACK_C == 0
    assert stride == 1
    X = te.placeholder(
        (n, ci // PACK_C, h, w, PACK_C),
        dtype=in_type,
        name="X",
    )
    W = te.placeholder(
        (co // PACK_C, ci // PACK_C, kh, kw, PACK_C, PACK_C),
        dtype=in_type,
        name="W",
    )

    rc = te.reduce_axis((0, ci), "rc")
    rh = te.reduce_axis((0, kh), "rh")
    rw = te.reduce_axis((0, kw), "rw")

    def f_compute(n, c0, h, w, c1):
        rc0 = rc // PACK_C
        rc1 = rc % PACK_C
        x = X[n, rc0, h + rh, w + rw, rc1].astype(out_type)
        w = W[c0, rc0, rh, rw, rc1, c1].astype(out_type)
        return te.sum(x * w, axis=(rc, rh, rw))

    Conv = te.compute(
        (
            n,
            co // PACK_C,
            h - kh + 1,
            w - kw + 1,
            PACK_C,
        ),
        f_compute,
        name="conv2d_nchwc",
    )

    return [X, W, Conv]


def conv2d_nchw_bias_bn_relu(  # pylint: disable=invalid-name
    n: int,
    h: int,
    w: int,
    ci: int,
    co: int,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor, te.Tensor, te.Tensor, te.Tensor]:
    oh = (h + 2 * padding - (kh - 1) * dilation - 1) // stride + 1  # pylint: disable=invalid-name
    ow = (w + 2 * padding - (kw - 1) * dilation - 1) // stride + 1  # pylint: disable=invalid-name
    x = te.placeholder((n, ci, h, w), name="X")
    w = te.placeholder((co, ci, kh, kw), name="W")
    b = te.placeholder((co, 1, 1), name="B")
    bn_scale = te.placeholder((co, 1, 1), name="bn_scale")
    bn_offset = te.placeholder((co, 1, 1), name="bn_offset")
    y = topi.nn.conv2d_nchw(Input=x, Filter=w, stride=stride, padding=padding, dilation=dilation)
    y = te.compute((n, co, oh, ow), lambda i, j, k, l: y[i, j, k, l] + b[j, 0, 0], name="bias_add")
    y = te.compute(
        (n, co, oh, ow), lambda i, j, k, l: y[i, j, k, l] * bn_scale[j, 0, 0], name="bn_mul"
    )
    y = te.compute(
        (n, co, oh, ow), lambda i, j, k, l: y[i, j, k, l] + bn_offset[j, 0, 0], name="bn_add"
    )
    y = topi.nn.relu(y)
    return [x, w, b, bn_scale, bn_offset, y]


def max_pool2d_nchw(  # pylint: disable=invalid-name
    n: int,
    h: int,
    w: int,
    ci: int,
    padding: int,
) -> Tuple[te.Tensor, te.Tensor]:  # pylint: disable=invalid-name
    x = te.placeholder((n, ci, h, w), name="X")
    y = topi.nn.pool2d(x, [2, 2], [1, 1], [1, 1], [padding, padding, padding, padding], "max")
    return [x, y]
