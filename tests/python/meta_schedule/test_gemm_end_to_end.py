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
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])


@tvm.hybrid.script
def matmul_relu(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    D = tir.match_buffer(d, (1024, 1024), "float32")
    C = tir.buffer_allocate((1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])
    with tir.block([1024, 1024], "relu") as [vi, vj]:
        D[vi, vj] = tir.max(C[vi, vj], 0.0)


@tvm.hybrid.script
def conv2d(x: ty.handle, w: ty.handle, y: ty.handle) -> None:
    X = tir.match_buffer(x, (1, 512, 7, 7), "float32")
    W = tir.match_buffer(w, (512, 512, 3, 3), "float32")
    X_padded = tir.buffer_allocate((1, 512, 9, 9), "float32")
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
            Y[i_n, i_co, i_h, i_w],
            X_padded[i_n, i_ci, i_h + i_kh, i_w + i_kw] * W[i_co, i_ci, i_kh, i_kw],
        )


@tvm.hybrid.script
def conv2d_relu(x: ty.handle, w: ty.handle, y: ty.handle) -> None:
    X = tir.match_buffer(x, (1, 512, 7, 7), "float32")
    W = tir.match_buffer(w, (512, 512, 3, 3), "float32")
    X_padded = tir.buffer_allocate((1, 512, 9, 9), "float32")
    Y_i = tir.buffer_allocate((1, 512, 7, 7), "float32")
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
        Y[i_n, i_co, i_h, i_w] = tir.max(Y_i[i_n, i_co, i_h, i_w], 0.0)


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


@ms.search_rule.register_rule("do_nothing")
def do_nothing(_task, sch: ms.Schedule, _block: ms.BlockRV, _info):
    return sch


@pytest.mark.skip(reason="needs RPC")
def test_matmul_schedule_fn():
    def schedule_matmul(sch):
        block = sch.get_block(name="matmul")
        i, j, k = sch.get_axes(block=block)
        i_tiles = sch.sample_perfect_tile(n_splits=4, loop=i)
        j_tiles = sch.sample_perfect_tile(n_splits=4, loop=j)
        k_tiles = sch.sample_perfect_tile(n_splits=2, loop=k)
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
        print(tvm.hybrid.ashybrid(sch.sch.func))


@pytest.mark.skip(reason="needs RPC")
def test_matmul_post_order_apply():
    rule = ms.search_rule.compose(
        name="composed",
        rules=[
            do_nothing,
            ms.search_rule.multi_level_tiling(tiling_structure="SSRSRS"),
        ],
    )
    sch = ms.autotune(
        task=matmul,
        space=ms.space.PostOrderApply(rule=rule),
        strategy="replay",
        runner="rpc://0.0.0.0:3012:local * 16",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.hybrid.ashybrid(sch.sch.func))


@pytest.mark.skip(reason="needs RPC")
def test_matmul_relu_schedule_fn():
    def schedule_matmul(sch):
        block = sch.get_block(name="matmul")
        i, j, k = sch.get_axes(block=block)
        i_tiles = sch.sample_perfect_tile(n_splits=4, loop=i)
        j_tiles = sch.sample_perfect_tile(n_splits=4, loop=j)
        k_tiles = sch.sample_perfect_tile(n_splits=2, loop=k)
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
        k_0, k_1 = sch.split(loop=k, factors=k_tiles)
        sch.reorder(after_axes=[i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3])

    sch = ms.autotune(
        task=matmul_relu,
        space=schedule_matmul,
        strategy="replay",
        runner="rpc://0.0.0.0:3012:local * 16",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.hybrid.ashybrid(sch.sch.func))


@pytest.mark.skip(reason="needs RPC")
def test_matmul_relu_post_order_apply():
    rule = ms.search_rule.compose(
        name="composed",
        rules=[
            do_nothing,
            ms.search_rule.multi_level_tiling_with_fusion(tiling_structure="SSRSRS"),
            ms.search_rule.multi_level_tiling(tiling_structure="SSRSRS"),
        ],
    )
    sch = ms.autotune(
        task=matmul_relu,
        space=ms.space.PostOrderApply(rule=rule),
        strategy="replay",
        runner="rpc://0.0.0.0:3012:local * 16",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.hybrid.ashybrid(sch.sch.func))


@pytest.mark.skip(reason="needs RPC")
def test_conv2d_schedule_fn():
    def schedule_conv2d(sch):
        block = sch.get_block(name="conv2d_nchw")
        i_n, i_co, i_h, i_w, i_ci, i_kh, i_kw = sch.get_axes(block=block)

        factors = sch.sample_perfect_tile(n_splits=4, loop=i_n)
        i_n_0, i_n_1, i_n_2, i_n_3 = sch.split(loop=i_n, factors=factors)

        factors = sch.sample_perfect_tile(n_splits=4, loop=i_co)
        i_co_0, i_co_1, i_co_2, i_co_3 = sch.split(loop=i_co, factors=factors)

        factors = sch.sample_perfect_tile(n_splits=4, loop=i_h)
        i_h_0, i_h_1, i_h_2, i_h_3 = sch.split(loop=i_h, factors=factors)

        factors = sch.sample_perfect_tile(n_splits=4, loop=i_w)
        i_w_0, i_w_1, i_w_2, i_w_3 = sch.split(loop=i_w, factors=factors)

        factors = sch.sample_perfect_tile(n_splits=2, loop=i_ci)
        i_ci_0, i_ci_1 = sch.split(loop=i_ci, factors=factors)

        factors = sch.sample_perfect_tile(n_splits=2, loop=i_kh)
        i_kh_0, i_kh_1 = sch.split(loop=i_kh, factors=factors)

        factors = sch.sample_perfect_tile(n_splits=2, loop=i_kw)
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
        print(tvm.hybrid.ashybrid(sch.sch.func))


@pytest.mark.skip(reason="needs RPC")
def test_conv2d_post_order_apply():
    rule = ms.search_rule.compose(
        name="composed",
        rules=[
            do_nothing,
            ms.search_rule.multi_level_tiling(tiling_structure="SSRSRS"),
        ],
    )
    sch = ms.autotune(
        task=conv2d,
        space=ms.space.PostOrderApply(rule=rule),
        strategy="replay",
        runner="rpc://0.0.0.0:3012:local * 16",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.hybrid.ashybrid(sch.sch.func))


@pytest.mark.skip(reason="needs RPC")
def test_conv2d_relu_plus_one_post_order_apply():
    rule = ms.search_rule.compose(
        name="composed",
        rules=[
            do_nothing,
            ms.search_rule.always_inline(),
            ms.search_rule.add_cache_write(),
            ms.search_rule.multi_level_tiling_with_fusion(tiling_structure="SSRSRS"),
            ms.search_rule.multi_level_tiling(tiling_structure="SSRSRS"),
        ],
    )
    sch = ms.autotune(
        task=conv2d_relu_plus_one,
        space=ms.space.PostOrderApply(rule=rule),
        strategy="replay",
        runner="rpc://0.0.0.0:3012:local * 16",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.hybrid.ashybrid(sch.sch.func))


@pytest.mark.skip(reason="needs RPC")
def test_matmul_evolutionary():
    task = ms.SearchTask(func=matmul)
    strategy = ms.strategy.Evolutionary(
        num_measure_trials=128,
        num_measure_per_batch=16,
        num_iters_in_genetic_algo=1,
        eps_greedy=0.02,
        use_measured_ratio=0.05,
        population=16,
        p_mutate=0.85,
        mutators=[ms.mutator.MutateTileSize(p=1.0)],
        cost_model=ms.RandomModel(),
    )
    space = ms.space.PostOrderApply(
        rule=ms.search_rule.multi_level_tiling(tiling_structure="SSRSRS")
    )
    # Test API:
    #   sample_init_population
    #   evolve_with_cost_model
    strategy.evolve_with_cost_model(
        task=task,
        inits=strategy.sample_init_population(
            space.get_support(task=task), num_samples=15
        ),
        num_samples=100,
    )
    # End-to-end integration test
    sch = ms.autotune(
        task=task,
        space=space,
        strategy=strategy,
        runner="rpc://0.0.0.0:3012:local * 16",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.hybrid.ashybrid(sch.sch.func))


if __name__ == "__main__":
    # test_matmul_schedule_fn()
    # test_matmul_post_order_apply()
    # test_matmul_relu_schedule_fn()
    # test_matmul_relu_post_order_apply()
    # test_conv2d_schedule_fn()
    # test_conv2d_post_order_apply()
    test_conv2d_relu_plus_one_post_order_apply()
    test_matmul_evolutionary()
