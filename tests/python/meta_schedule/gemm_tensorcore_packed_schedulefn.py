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
"""Integration test for CUDA with Tensor Core"""
# pylint: disable=missing-function-docstring
import logging
import os

import te_workload
import tvm
import tir_tensor_intrin  # pylint: disable=unused-import
from tvm import meta_schedule as ms
from tvm import te
from tvm import tir
from tvm.contrib import nvcc
import numpy as np

RPC_KEY = "rtx-3080"
TARGET = tvm.target.Target("nvidia/geforce-rtx-3080")
TARGET_HOST = tvm.target.Target("llvm")


def test_integration_matmul():
    os.environ["TVM_TRACKER_KEY"] = RPC_KEY
    os.environ["TVM_TRACKER_HOST"] = "172.16.2.241"
    os.environ["TVM_TRACKER_PORT"] = "4445"

    block_num = 16

    def matmul_schedule(sch):
        # Step 1. Rule-Auto-Tensorize
        # pylint: disable=invalid-name
        block = sch.get_block("C")
        i, j, k, i_tc, j_tc, k_tc = sch.get_loops(block)
        block_inner = sch.blockize(i_tc)
        block_outer, block_inner = block_inner, block
        del block
        # Step 2. Rule-Multi-Level-Tiling
        i0, i1, i2, i3, i4 = sch.split(i, factors=sch.sample_perfect_tile(i, 5))
        j0, j1, j2, j3, j4 = sch.split(j, factors=sch.sample_perfect_tile(j, 5))
        k0, k1, k2 = sch.split(k, factors=sch.sample_perfect_tile(k, 3))
        # pylint: enable=invalid-name
        sch.reorder(
            # fmt: off
            i0, j0,   # S => blockIdx.x
            i1, j1,   # S => blockIdx.y
            i2, j2,   # S => threadIdx.y
            # cache_write here
            k0,       # R
            # vectorized cooperative fetching here
            k1,       # R
            i3, j3,   # S
            k2,       # R
            i4, j4,
            # S
            # fmt: on
        )
        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(i2, j2)
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, k0, True)
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])
            fused_0, fused_1 = sch.split(fused, factors=[None, 4])
            sch.mark_loop(fused_0, "loop_type", "lazy_cooperative_fetch")
            sch.vectorize(fused_1)

        fetch_to_shared(block_outer, 1, 4)
        fetch_to_shared(block_outer, 2, 4)

        # Step 3. Postproc-Rewrite-Tensorize
        # Step 3.1. Cache read
        loop = sch.get_loops(block_outer)[-3]
        block_read_a = sch.cache_read(block_inner, 1, "wmma.matrix_a")
        block_read_b = sch.cache_read(block_inner, 2, "wmma.matrix_b")
        sch.compute_at(block_read_a, loop, True)
        sch.compute_at(block_read_b, loop, True)
        # Step 3.2. Cache write
        block_write_c = sch.cache_write(block_outer, 0, "wmma.accumulator")
        block_outer, block_write_c = block_write_c, block_outer
        sch.reverse_compute_at(block_write_c, thread_idy, True)
        # Step 3.3. Decompose
        loop = sch.get_loops(block_outer)[3]
        block_init_c = sch.decompose_reduction(block_outer, loop)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]
        # # Step 3.4. Tensorize
        loop = sch.get_loops(block_inner)[-3]
        sch.tensorize(loop, "wmma_sync")
        loop = sch.get_loops(block_read_a)[-2]
        sch.tensorize(loop, "wmma_load_a")
        loop = sch.get_loops(block_read_b)[-2]
        sch.tensorize(loop, "wmma_load_b")
        loop = sch.get_loops(block_init_c_inner)[-2]
        sch.tensorize(loop, "wmma_fill")
        loop = sch.get_loops(block_write_c)[-2]
        sch.tensorize(loop, "wmma_store")

    space = ms.space.ScheduleFn(
        matmul_schedule,
        postprocs=[
            ms.postproc.rewrite_cooperative_fetch_tensorcore(),
            ms.postproc.rewrite_parallel_vectorize_unroll(),
            ms.postproc.verify_gpu_code(),
        ],
    )
    schedule = ms.autotune(
        task=ms.SearchTask(
            workload=te_workload.matmul_fp16_packed.specialize(
                {
                    te_workload.matmul_fp16_packed.params[0]:
                    tir.decl_buffer((block_num, block_num, 16, 16))
                }
            ),
            task_name="matmul_tensorize",
            target=TARGET,
            target_host=TARGET_HOST
        ),
        space=space,
        strategy=ms.strategy.Evolutionary(
            total_measures=1500,
            num_measures_per_iter=64,
            population=2048,
            init_measured_ratio=0.2,
            genetic_algo_iters=10,
            p_mutate=0.85,
            mutator_probs={
                ms.mutator.mutate_tile_size(): 0.90,
                ms.mutator.mutate_auto_unroll(): 0.10,
            },
            cost_model=ms.XGBModel(),
            eps_greedy=0.25,
        )
        # strategy=ms.strategy.Replay(10)
    )

    ctx = tvm.gpu(0)
    if nvcc.have_tensorcore(ctx.compute_version):
        with tvm.transform.PassContext():
            func = tvm.build(schedule.state.mod["main"], [], "cuda")
            print(tvm.script.asscript(schedule.state.mod["main"]))
            print(func.imported_modules[0].get_source())
            for inst in schedule.trace.as_python():
                print(inst)
        a_np = np.random.uniform(size=(block_num, block_num, 16, 16)).astype("float16")
        b_np = np.random.uniform(size=(block_num, block_num, 16, 16)).astype("float16")
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros((block_num, block_num, 16, 16), dtype="float32"), ctx)
        evaluator = func.time_evaluator(func.entry_name, ctx, number = 3, repeat = 1, min_repeat_ms = 40)
        print("matmul with tensor core: %f ms" % (evaluator(a, b, c).mean * 1e3))

        c_np = c.asnumpy()
        a_non_packed = np.array(
            [
                [a_np[i // 16][j // 16][i % 16][j % 16] for j in range(block_num * 16)]
                for i in range(block_num * 16)
            ]
        )
        b_non_packed = np.array(
            [
                [b_np[i // 16][j // 16][i % 16][j % 16] for j in range(block_num * 16)]
                for i in range(block_num * 16)
            ]
        )
        c_non_packed = np.array(
            [
                [c_np[i // 16][j // 16][i % 16][j % 16] for j in range(block_num * 16)]
                for i in range(block_num * 16)
            ]
        )

        np.testing.assert_allclose(
            c_non_packed,
            np.matmul(a_non_packed.astype("float32"), b_non_packed.astype("float32")),
            rtol=1e-4,
            atol=1e-4,
        )


if __name__ == "__main__":
    test_integration_matmul()