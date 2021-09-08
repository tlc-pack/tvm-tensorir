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
import logging

# pylint: disable=missing-function-docstring
import os

import pytest
import tvm
from tir_workload import matmul, matmul_relu
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import ty

logging.basicConfig()
logging.getLogger("meta_schedule").setLevel(logging.DEBUG)

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks


@tvm.script.tir
def conv2d(x: ty.handle, w: ty.handle, y: ty.handle) -> None:
    X = tir.match_buffer(x, (1, 512, 7, 7), "float32")
    W = tir.match_buffer(w, (512, 512, 3, 3), "float32")
    X_padded = tir.alloc_buffer((1, 512, 9, 9), "float32")
    Y = tir.match_buffer(y, [1, 512, 7, 7], "float32")
    with tir.block([1, 512, 9, 9], "conv2d_pad_x") as [i_n, i_ci, i_h, i_w]:
        X_padded[i_n, i_ci, i_h, i_w] = tir.if_then_else(  # pylint: disable=unexpected-keyword-arg
            # guard
            ((1 <= i_h) and (i_h < 8) and (1 <= i_w) and (i_w < 8)),
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
        with tir.init():
            Y[i_n, i_co, i_h, i_w] = 0.0
        Y[i_n, i_co, i_h, i_w] = (
            Y[i_n, i_co, i_h, i_w]
            + X_padded[i_n, i_ci, i_h + i_kh, i_w + i_kw] * W[i_co, i_ci, i_kh, i_kw]
        )


@tvm.script.tir
def conv2d_relu(x: ty.handle, w: ty.handle, y: ty.handle) -> None:
    X = tir.match_buffer(x, (1, 512, 7, 7), "float32")
    W = tir.match_buffer(w, (512, 512, 3, 3), "float32")
    X_padded = tir.alloc_buffer((1, 512, 9, 9), "float32")
    Y_i = tir.alloc_buffer((1, 512, 7, 7), "float32")
    Y = tir.match_buffer(y, [1, 512, 7, 7], "float32")
    with tir.block([1, 512, 9, 9], "conv2d_pad_x") as [i_n, i_ci, i_h, i_w]:
        X_padded[i_n, i_ci, i_h, i_w] = tir.if_then_else(  # pylint: disable=unexpected-keyword-arg
            # guard
            ((1 <= i_h) and (i_h < 8) and (1 <= i_w) and (i_w < 8)),
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
        with tir.init():
            Y_i[i_n, i_co, i_h, i_w] = 0.0
        Y_i[i_n, i_co, i_h, i_w] = (
            Y_i[i_n, i_co, i_h, i_w]
            + X_padded[i_n, i_ci, i_h + i_kh, i_w + i_kw] * W[i_co, i_ci, i_kh, i_kw]
        )

    with tir.block([1, 512, 7, 7], "relu") as [i_n, i_co, i_h, i_w]:
        Y[i_n, i_co, i_h, i_w] = tir.max(Y_i[i_n, i_co, i_h, i_w], 0.0)


@tvm.script.tir
def conv2d_relu_plus_one(x: ty.handle, w: ty.handle, y: ty.handle) -> None:
    X = tir.match_buffer(x, (1, 512, 7, 7), "float32")
    W = tir.match_buffer(w, (512, 512, 3, 3), "float32")
    X_padded = tir.alloc_buffer((1, 512, 9, 9), "float32")
    Y_i = tir.alloc_buffer((1, 512, 7, 7), "float32")
    Y_j = tir.alloc_buffer((1, 512, 7, 7), "float32")
    Y = tir.match_buffer(y, [1, 512, 7, 7], "float32")
    with tir.block([1, 512, 9, 9], "conv2d_pad_x") as [i_n, i_ci, i_h, i_w]:
        X_padded[i_n, i_ci, i_h, i_w] = tir.if_then_else(  # pylint: disable=unexpected-keyword-arg
            # guard
            ((1 <= i_h) and (i_h < 8) and (1 <= i_w) and (i_w < 8)),
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
        with tir.init():
            Y_i[i_n, i_co, i_h, i_w] = 0.0
        Y_i[i_n, i_co, i_h, i_w] = (
            Y_i[i_n, i_co, i_h, i_w]
            + X_padded[i_n, i_ci, i_h + i_kh, i_w + i_kw] * W[i_co, i_ci, i_kh, i_kw]
        )

    with tir.block([1, 512, 7, 7], "relu") as [i_n, i_co, i_h, i_w]:
        Y_j[i_n, i_co, i_h, i_w] = tir.max(Y_i[i_n, i_co, i_h, i_w], 0.0)

    with tir.block([1, 512, 7, 7], "plus_one") as [i_n, i_co, i_h, i_w]:
        Y[i_n, i_co, i_h, i_w] = Y_j[i_n, i_co, i_h, i_w] + 1.0


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


@pytest.mark.skip(reason="needs RPC")
def test_matmul_schedule_fn():
    os.environ["TVM_TRACKER_KEY"] = "local"

    def schedule_matmul(sch):
        block = sch.get_block(name="matmul")
        i, j, k = sch.get_loops(block=block)
        i_tiles = sch.sample_perfect_tile(i, n=4)
        j_tiles = sch.sample_perfect_tile(j, n=4)
        k_tiles = sch.sample_perfect_tile(k, n=2)
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
        k_0, k_1 = sch.split(loop=k, factors=k_tiles)
        sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)

    sch = ms.autotune(
        task=matmul,
        space=schedule_matmul,
        strategy="replay",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.mod))


@pytest.mark.skip(reason="needs RPC")
def test_matmul_shape_generic_schedule_fn():
    os.environ["TVM_TRACKER_KEY"] = "local"

    def schedule_matmul(sch):
        block = sch.get_block(name="matmul")
        i, j, k = sch.get_loops(block=block)
        tiles = sch.sample_shape_generic_tiles([i, j, k], ns=[4, 4, 2],
                                               target=tvm.target.Target('nvidia/nvidia-t4'))
        i_tiles, j_tiles, k_tiles = list(tiles)
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
        k_0, k_1 = sch.split(loop=k, factors=k_tiles)
        sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)

    sch = ms.autotune(
        task=matmul,
        space=schedule_matmul,
        strategy="replay",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.mod))


@pytest.mark.skip(reason="needs RPC")
def test_matmul_relu_schedule_fn():
    os.environ["TVM_TRACKER_KEY"] = "local"

    def schedule_matmul(sch: tir.Schedule):
        matmul_block = sch.get_block(name="matmul")
        i, j, k = sch.get_loops(block=matmul_block)
        i_tiles = sch.sample_perfect_tile(i, n=4)
        j_tiles = sch.sample_perfect_tile(j, n=4)
        k_tiles = sch.sample_perfect_tile(k, n=2)
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
        k_0, k_1 = sch.split(loop=k, factors=k_tiles)
        sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
        relu_block = sch.get_block(name="relu")
        sch.reverse_compute_at(relu_block, j_0)

    sch = ms.autotune(
        task=matmul_relu,
        space=schedule_matmul,
        strategy="replay",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.mod))


@pytest.mark.skip(reason="needs RPC")
def test_conv2d_schedule_fn():
    os.environ["TVM_TRACKER_KEY"] = "local"

    def schedule_conv2d(sch):
        block = sch.get_block(name="conv2d_nchw")
        i_n, i_co, i_h, i_w, i_ci, i_kh, i_kw = sch.get_loops(block=block)

        factors = sch.sample_perfect_tile(i_n, 4)
        i_n_0, i_n_1, i_n_2, i_n_3 = sch.split(loop=i_n, factors=factors)

        factors = sch.sample_perfect_tile(i_co, 4)
        i_co_0, i_co_1, i_co_2, i_co_3 = sch.split(loop=i_co, factors=factors)

        factors = sch.sample_perfect_tile(i_h, 4)
        i_h_0, i_h_1, i_h_2, i_h_3 = sch.split(loop=i_h, factors=factors)

        factors = sch.sample_perfect_tile(i_w, 4)
        i_w_0, i_w_1, i_w_2, i_w_3 = sch.split(loop=i_w, factors=factors)

        factors = sch.sample_perfect_tile(i_ci, 2)
        i_ci_0, i_ci_1 = sch.split(loop=i_ci, factors=factors)

        factors = sch.sample_perfect_tile(i_kh, 2)
        i_kh_0, i_kh_1 = sch.split(loop=i_kh, factors=factors)

        factors = sch.sample_perfect_tile(i_kw, 2)
        i_kw_0, i_kw_1 = sch.split(loop=i_kw, factors=factors)
        sch.reorder(
            # fmt: off
            i_n_0, i_co_0, i_h_0, i_w_0,  # S
            i_n_1, i_co_1, i_h_1, i_w_1,  # S
            i_ci_0, i_kh_0, i_kw_0,  # R
            i_n_2, i_co_2, i_h_2, i_w_2,  # S
            i_ci_1, i_kh_1, i_kw_1,  # R
            i_n_3, i_co_3, i_h_3, i_w_3,
            # S
            # fmt: on
        )

    sch = ms.autotune(
        task=conv2d,
        space=schedule_conv2d,
        strategy="replay",
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.mod))


@pytest.mark.skip(reason="needs RPC")
def test_matmul_evolutionary_step_by_step():
    os.environ["TVM_TRACKER_KEY"] = "test"
    task = ms.SearchTask(workload=matmul)
    measurer = ms.ProgramMeasurer()
    strategy = ms.strategy.Evolutionary(
        total_measures=128,
        num_measures_per_iter=16,
        population=128,
        init_measured_ratio=0.05,
        genetic_algo_iters=1,
        p_mutate=0.85,
        mutator_probs={
            ms.mutator.mutate_tile_size(): 0.95,
            ms.mutator.mutate_compute_location(): 0.05,
        },
        cost_model=ms.RandCostModel(),
        eps_greedy=0.07,
    )
    space = ms.space.PostOrderApply(
        stages=[
            ms.rule.inline_pure_spatial(strict_mode=True),
            ms.rule.multi_level_tiling(
                structure="SSRSRS",
                must_cache_read=False,
                cache_read_scope="global",
                can_cache_write=True,
                must_cache_write=False,
                cache_write_scope="global",
                consumer_inline_strict=True,
                fusion_levels=[1, 2],
            ),
            ms.rule.parallelize_vectorize_unroll(
                max_jobs_per_core=16,
                max_vectorize_extent=32,
                unroll_max_steps=[0, 16, 64, 512],
                unroll_explicit=True,
            ),
            ms.rule.random_compute_location(),
        ],
        postprocs=[
            ms.postproc.rewrite_parallel_vectorize_unroll(),
            ms.postproc.rewrite_reduction_block(),
        ],
    )
    support = space.get_support(task=task)
    # Test API:
    #   sample_init_population
    #   evolve_with_cost_model
    #   pick_with_eps_greedy
    #   measure_and_update_cost_model
    inits = strategy.sample_init_population(support=support, task=task, space=space)
    bests = strategy.evolve_with_cost_model(inits=inits, task=task, space=space)
    picks = strategy.pick_with_eps_greedy(inits=inits, bests=bests, task=task, space=space)
    strategy.measure_and_update_cost_model(
        task=task,
        picks=picks,
        measurer=measurer,
        verbose=1,
    )


@pytest.mark.skip(reason="needs RPC")
def test_matmul_evolutionary_end_to_end():
    os.environ["TVM_TRACKER_KEY"] = "test"
    sch = ms.autotune(
        task=ms.SearchTask(workload=matmul),
        space=ms.space.PostOrderApply(
            stages=[
                ms.rule.inline_pure_spatial(strict_mode=True),
                ms.rule.multi_level_tiling(
                    structure="SSRSRS",
                    must_cache_read=False,
                    cache_read_scope="global",
                    can_cache_write=True,
                    must_cache_write=False,
                    cache_write_scope="global",
                    consumer_inline_strict=True,
                    fusion_levels=[1, 2],
                ),
                ms.rule.parallelize_vectorize_unroll(
                    max_jobs_per_core=16,
                    max_vectorize_extent=32,
                    unroll_max_steps=[0, 16, 64, 512],
                    unroll_explicit=True,
                ),
                ms.rule.random_compute_location(),
            ],
            postprocs=[
                ms.postproc.rewrite_parallel_vectorize_unroll(),
                ms.postproc.rewrite_reduction_block(),
            ],
        ),
        strategy=ms.strategy.Evolutionary(
            total_measures=128,
            num_measures_per_iter=16,
            population=128,
            init_measured_ratio=0.05,
            genetic_algo_iters=1,
            p_mutate=0.85,
            mutator_probs={
                ms.mutator.mutate_tile_size(): 0.95,
                ms.mutator.mutate_compute_location(): 0.05,
            },
            cost_model=ms.RandCostModel(),
            eps_greedy=0.07,
        ),
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.mod))


@pytest.mark.skip(reason="needs RPC")
def test_matmul_evolutionary_xgb():
    os.environ["TVM_TRACKER_KEY"] = "test"
    sch = ms.autotune(
        task=ms.SearchTask(workload=matmul),
        space=ms.space.PostOrderApply(
            stages=[
                ms.rule.inline_pure_spatial(strict_mode=True),
                ms.rule.multi_level_tiling(
                    structure="SSRSRS",
                    must_cache_read=False,
                    cache_read_scope="global",
                    can_cache_write=True,
                    must_cache_write=False,
                    cache_write_scope="global",
                    consumer_inline_strict=True,
                    fusion_levels=[1, 2],
                ),
                ms.rule.parallelize_vectorize_unroll(
                    max_jobs_per_core=16,
                    max_vectorize_extent=32,
                    unroll_max_steps=[0, 16, 64, 512],
                    unroll_explicit=True,
                ),
                ms.rule.random_compute_location(),
            ],
            postprocs=[
                ms.postproc.rewrite_parallel_vectorize_unroll(),
                ms.postproc.rewrite_reduction_block(),
            ],
        ),
        strategy=ms.strategy.Evolutionary(
            total_measures=128,
            num_measures_per_iter=16,
            population=128,
            init_measured_ratio=0.05,
            genetic_algo_iters=1,
            p_mutate=0.85,
            mutator_probs={
                ms.mutator.mutate_tile_size(): 0.95,
                ms.mutator.mutate_compute_location(): 0.05,
            },
            cost_model=ms.XGBModel(
                num_warmup_samples=0,
            ),
            eps_greedy=0.07,
        ),
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.mod))


if __name__ == "__main__":
    # ScheduleFn + Replay
    test_matmul_schedule_fn()
    test_matmul_relu_schedule_fn()
    test_conv2d_schedule_fn()
    # PostOrderApply + Evo Search
    test_matmul_evolutionary_step_by_step()
    test_matmul_evolutionary_end_to_end()
    # PostOrderApply + Evo Search + XGBModel
    test_matmul_evolutionary_xgb()
