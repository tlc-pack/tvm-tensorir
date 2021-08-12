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
"""Integration test for CUDA"""
import logging

# pylint: disable=missing-function-docstring
import os

import pytest
import te_workload
import tvm
from tvm import meta_schedule as ms
from tvm import te

logging.basicConfig()
logging.getLogger("meta_schedule").setLevel(logging.DEBUG)

RPC_KEY = "rtx-3070"
TARGET = tvm.target.Target("nvidia/geforce-rtx-3070")
TARGET_HOST = tvm.target.Target("llvm")
SPACE = ms.space.PostOrderApply(
    stages=[
        ms.rule.multi_level_tiling(
            structure="SSSRRSRS",
            must_cache_read=True,
            cache_read_scope="shared",
            can_cache_write=True,
            must_cache_write=True,
            cache_write_scope="local",
            consumer_inline_strict=False,
            fusion_levels=[3],
            vector_load_max_len=4,
            tile_binds=["blockIdx.x", "vthread", "threadIdx.x"],
        ),
        ms.rule.inline_pure_spatial(strict_mode=False),
        ms.rule.parallelize_vectorize_unroll(
            max_jobs_per_core=-1,  # disable parallelize
            max_vectorize_extent=-1,  # disable vectorize
            unroll_max_steps=[0, 16, 64, 512, 1024],
            unroll_explicit=True,
        ),
    ],
    postprocs=[
        ms.postproc.rewrite_cooperative_fetch(),
        ms.postproc.rewrite_unbound_blocks(),
        ms.postproc.rewrite_parallel_vectorize_unroll(),
        ms.postproc.rewrite_reduction_block(),
        ms.postproc.disallow_dynamic_loops(),
        ms.postproc.verify_gpu_code(),
    ],
)


@pytest.mark.skip(reason="needs RPC")
def test_integration_matmul():
    os.environ["TVM_TRACKER_KEY"] = RPC_KEY
    sch = ms.autotune(
        task=ms.SearchTask(
            workload=te.create_prim_func(te_workload.matmul(1024, 1024, 1024)),
            target=TARGET,
            target_host=TARGET_HOST,
            task_name="cuda_matmul",
            log_file="./cuda_matmul.json",
        ),
        space=SPACE,
        strategy=ms.strategy.Evolutionary(
            total_measures=128,
            num_measures_per_iter=16,
            population=128,
            init_measured_ratio=0.2,
            genetic_algo_iters=10,
            p_mutate=0.85,
            mutator_probs={
                ms.mutator.mutate_tile_size(): 1.0,
            },
            cost_model=ms.XGBModel(
                num_warmup_samples=0,
            ),
            eps_greedy=0.05,
        ),
        measurer=ms.ProgramMeasurer(
            measure_callbacks=[
                ms.RecordToFile(),
            ]
        ),
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.mod))


if __name__ == "__main__":
    test_integration_matmul()
