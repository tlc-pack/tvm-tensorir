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
""" Test multi-level tiling """
# pylint: disable=missing-function-docstring
import pytest

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.hybrid import ty

TILING_FORMAT = "SSRSRS"
SPATIAL = 0
REDUCTION = 2

# pylint: disable=invalid-name,no-member


@tvm.hybrid.script
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "C") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])


@tvm.hybrid.script
def conv2d(x: ty.handle, w: ty.handle, y: ty.handle) -> None:
    X = tir.match_buffer(x, (1, 512, 7, 7), "float32")
    W = tir.match_buffer(w, (512, 512, 3, 3), "float32")
    Y = tir.match_buffer(y, [1, 512, 7, 7], "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    Pad = tir.buffer_allocate((1, 512, 9, 9), "float32")
    with tir.block([1, 512, 9, 9], "conv2d_pad_x") as [i_n, i_ci, i_h, i_w]:
        Pad[
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
            Y[i_n, i_co, i_h, i_w],
            Pad[i_n, i_ci, i_h + i_kh, i_w + i_kw] * W[i_co, i_ci, i_kh, i_kw],
        )


# pylint: enable=invalid-name,no-member


def _print_prim_func(prim_func):
    print(tvm.hybrid.ashybrid(prim_func))


@ms.register_rule("do_nothing")
def do_nothing(sch: ms.Schedule, _block: ms.BlockRV):
    return sch


@pytest.mark.skip(reason="needs RPC")
def test_matmul_schedule_fn():
    def schedule_matmul(sch):
        block = sch.get_block(name="C")
        i, j, k = sch.get_axes(block=block)
        i_tiles = sch.sample_tile_factor(n=4, loop=i, where=[1, 2, 4])
        j_tiles = sch.sample_tile_factor(n=4, loop=j, where=[1, 2, 4])
        k_tiles = sch.sample_tile_factor(n=2, loop=k, where=[1, 2, 4])
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
        k_0, k_1 = sch.split(loop=k, factors=k_tiles)
        sch.reorder(after_axes=[i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3])

    sch = ms.autotune(
        task=matmul,
        space=schedule_matmul,
        strategy="replay",
        runner="rpc://0.0.0.0:3012:local * 16",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        _print_prim_func(sch.sch.func)


@pytest.mark.skip(reason="needs RPC")
def test_matmul_post_order_apply():
    rule = ms.SearchRule.compose(
        name="composed",
        rules=[
            do_nothing,
            ms.search_rule.multi_level_tiling(tiling_structure="SSRSRS"),
        ],
    )
    sch = ms.autotune(
        task=matmul,
        space=ms.PostOrderApply(rule=rule),
        strategy="replay",
        runner="rpc://0.0.0.0:3012:local * 16",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        _print_prim_func(sch.sch.func)


@pytest.mark.skip(reason="needs RPC")
def test_conv2d_schedule_fn():
    def schedule_conv2d(sch):
        block = sch.get_block(name="conv2d_nchw")
        i_n, i_co, i_h, i_w, i_ci, i_kh, i_kw = sch.get_axes(block=block)

        factors = sch.sample_tile_factor(n=4, loop=i_n, where=[1, 2, 4])
        i_n_0, i_n_1, i_n_2, i_n_3 = sch.split(loop=i_n, factors=factors)

        factors = sch.sample_tile_factor(n=4, loop=i_co, where=[1, 2, 4])
        i_co_0, i_co_1, i_co_2, i_co_3 = sch.split(loop=i_co, factors=factors)

        factors = sch.sample_tile_factor(n=4, loop=i_h, where=[1, 2, 4])
        i_h_0, i_h_1, i_h_2, i_h_3 = sch.split(loop=i_h, factors=factors)

        factors = sch.sample_tile_factor(n=4, loop=i_w, where=[1, 2, 4])
        i_w_0, i_w_1, i_w_2, i_w_3 = sch.split(loop=i_w, factors=factors)

        factors = sch.sample_tile_factor(n=2, loop=i_ci, where=[1, 2, 4])
        i_ci_0, i_ci_1 = sch.split(loop=i_ci, factors=factors)

        factors = sch.sample_tile_factor(n=2, loop=i_kh, where=[1, 2, 4])
        i_kh_0, i_kh_1 = sch.split(loop=i_kh, factors=factors)

        factors = sch.sample_tile_factor(n=2, loop=i_kw, where=[1, 2, 4])
        i_kw_0, i_kw_1 = sch.split(loop=i_kw, factors=factors)
        sch.reorder(
            [i_n_0, i_co_0, i_h_0, i_w_0]  # S
            + [i_n_1, i_co_1, i_h_1, i_w_1]  # S
            + [i_ci_0, i_kh_0, i_kw_0]  # R
            + [i_n_2, i_co_2, i_h_2, i_w_2]  # S
            + [i_ci_1, i_kh_1, i_kw_1]  # R
            + [i_n_3, i_co_3, i_h_3, i_w_3],  # S
        )

    sch = ms.autotune(
        task=conv2d,
        space=schedule_conv2d,
        strategy="replay",
        runner="rpc://0.0.0.0:3012:local * 16",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        _print_prim_func(sch.sch.func)


@pytest.mark.skip(reason="needs RPC")
def test_conv2d_post_order_apply():
    rule = ms.SearchRule.compose(
        name="composed",
        rules=[
            do_nothing,
            ms.search_rule.multi_level_tiling(tiling_structure="SSRSRS"),
        ],
    )
    sch = ms.autotune(
        task=conv2d,
        space=ms.PostOrderApply(rule=rule),
        strategy=ms.Replay(batch_size=1, num_iterations=1),
        builder=ms.LocalBuilder(n_parallel=1),
        runner="rpc://0.0.0.0:3012:local * 16",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        _print_prim_func(sch.sch.func)


if __name__ == "__main__":
    # test_matmul_schedule_fn()
    # test_matmul_post_order_apply()
    test_conv2d_schedule_fn()
    # test_conv2d_post_order_apply()
