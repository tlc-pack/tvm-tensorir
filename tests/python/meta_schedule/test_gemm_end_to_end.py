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
            tir.reduce_axis(0, 3),
        ],  # i_kw
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


@ms.register_rule("multi_level_tiling")
def multi_level_tiling(sch: ms.Schedule, block: ms.BlockRV):
    spatial_indices = [i for i, c in enumerate(TILING_FORMAT) if c == "S"]
    reduce_indices = [i for i, c in enumerate(TILING_FORMAT) if c == "R"]
    order = [list() for _ in TILING_FORMAT]
    axes = sch.get_axes(block=block)
    iter_vars = ms.helpers.block_from_sref(sch.evaluate(block)).iter_vars
    assert len(axes) == len(iter_vars)
    for axis, iter_var in zip(axes, iter_vars):
        for iter_type, indices in [
            (SPATIAL, spatial_indices),
            (REDUCTION, reduce_indices),
        ]:
            if iter_var.iter_type == iter_type:
                tiles = sch.sample_tile_factor(
                    n=len(indices), loop=axis, where=[1, 2, 4]
                )
                splits = sch.split(loop=axis, factors=tiles)
                for i, split in zip(indices, splits):
                    order[i].append(split)
    sch.reorder(after_axes=sum(order, []))
    return sch


def do_multi_level_tiling(sch: ms.Schedule, block: ms.BlockRV):
    _print_prim_func(sch.sch.func)
    spatial_indices = [i for i, c in enumerate(TILING_FORMAT) if c == "S"]
    reduce_indices = [i for i, c in enumerate(TILING_FORMAT) if c == "R"]
    order = [list() for _ in TILING_FORMAT]
    axes = sch.get_axes(block=block)
    iter_vars = ms.helpers.block_from_sref(sch.evaluate(block)).iter_vars
    assert len(axes) == len(iter_vars)
    for axis, iter_var in zip(axes, iter_vars):
        for iter_type, indices in [
            (SPATIAL, spatial_indices),
            (REDUCTION, reduce_indices),
        ]:
            if iter_var.iter_type == iter_type:
                tiles = sch.sample_tile_factor(
                    n=len(indices), loop=axis, where=[1, 2, 4]
                )
                splits = sch.split(loop=axis, factors=tiles)
                for i, split in zip(indices, splits):
                    order[i].append(split)
    sch.reorder(after_axes=sum(order, []))
    _print_prim_func(sch.sch.func)
    return "Apply"


def test_matmul_tiling_rule():
    sch = ms.Schedule(matmul)
    block = sch.get_block(name="C")
    do_multi_level_tiling(sch, block)


def test_conv2d_tiling_rule():
    sch = ms.Schedule(conv2d)
    block = sch.get_block("conv2d_nchw")
    do_multi_level_tiling(sch, block)


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
            multi_level_tiling,
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


if __name__ == "__main__":
    test_matmul_tiling_rule()
    test_conv2d_tiling_rule()
    test_matmul_schedule_fn()
    test_matmul_post_order_apply()
