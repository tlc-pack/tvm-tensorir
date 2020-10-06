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
""" Test for meta schedule search rules """
# pylint: disable=missing-function-docstring
import pytest

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.hybrid import ty

# pylint: disable=invalid-name,no-member


@tvm.hybrid.script
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])


@tvm.hybrid.script
def conv2d_relu_plus_one(x: ty.handle, w: ty.handle, y: ty.handle) -> None:
    X = tir.match_buffer(x, (1, 512, 7, 7), "float32")
    W = tir.match_buffer(w, (512, 512, 3, 3), "float32")
    X_padded = tir.buffer_allocate((1, 512, 9, 9), "float32")
    Y_i = tir.buffer_allocate((1, 512, 7, 7), "float32")
    Y_j = tir.buffer_allocate((1, 512, 7, 7), "float32")
    Y = tir.match_buffer(y, [1, 512, 7, 7], "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([1, 512, 9, 9], "conv2d_pad_x") as [i_n, i_ci, i_h, i_w]:
        X_padded[
            i_n, i_ci, i_h, i_w
        ] = tir.if_then_else(  # pylint: disable=unexpected-keyword-arg
            # guard
            ((1 <= i_h < 8) and (1 <= i_w < 8)),
            # the value from input
            X[i_n, i_ci, i_h - 1, i_w - 1],
            # the value padded
            tir.float32(0),
            dtype="float32",
        )

    with tir.block(
        [
            1,  # i_n
            512,  # i_co
            7,  # i_h
            7,  # i_w
            tir.reduce_axis(0, 512),  # i_ci
            tir.reduce_axis(0, 3),  # i_kh
            tir.reduce_axis(0, 3),  # i_kw
        ],
        "conv2d_nchw",
    ) as [i_n, i_co, i_h, i_w, i_ci, i_kh, i_kw]:
        reducer.step(
            Y_i[i_n, i_co, i_h, i_w],
            X_padded[i_n, i_ci, i_h + i_kh, i_w + i_kw] * W[i_co, i_ci, i_kh, i_kw],
        )

    with tir.block([1, 512, 7, 7], "relu") as [i_n, i_co, i_h, i_w]:
        Y_j[i_n, i_co, i_h, i_w] = tir.max(Y_i[i_n, i_co, i_h, i_w], 0.0)

    with tir.block([1, 512, 7, 7], "plus_one") as [i_n, i_co, i_h, i_w]:
        Y[i_n, i_co, i_h, i_w] = Y_j[i_n, i_co, i_h, i_w] + 1.0


# pylint: enable=invalid-name,no-member


def test_meta_schedule_rule_always_inline():
    task = ms.SearchTask(func=conv2d_relu_plus_one)
    sch = ms.Schedule(func=conv2d_relu_plus_one)
    block = sch.get_block("relu")
    ms.search_rule.always_inline()(task, sch, block)
    with pytest.raises(ValueError):
        sch.get_block("relu")
    sch.get_block("conv2d_pad_x")
    sch.get_block("conv2d_nchw")
    sch.get_block("plus_one")


def test_meta_schedule_rule_add_cache_write():
    task = ms.SearchTask(func=matmul)
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("matmul")
    ms.search_rule.add_cache_write()(task, sch, block)
    sch.get_block("matmul")
    assert sch.evaluate(block).stmt.tag == ""


def test_meta_schedule_rule_multi_level_tiling():
    task = ms.SearchTask(func=matmul)
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("matmul")
    ms.search_rule.multi_level_tiling("SSRSRS")(task, sch, block)
    assert len(sch.get_axes(block)) == 10


if __name__ == "__main__":
    test_meta_schedule_rule_always_inline()
    test_meta_schedule_rule_add_cache_write()
    test_meta_schedule_rule_multi_level_tiling()
