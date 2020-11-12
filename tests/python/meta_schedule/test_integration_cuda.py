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
# pylint: disable=missing-function-docstring
import os

import pytest
import tvm
from tir_workload import matmul
from tvm import meta_schedule as ms

TARGET = tvm.target.Target("nvidia/rtx2080ti")
SPACE = ms.space.PostOrderApply(
    stages=[
        ms.rule.inline_pure_spatial(strict_mode=False),
        ms.rule.multi_level_tiling_and_fusion(
            structure="SSSRRSRS",
            must_cache_read=True,
            can_cache_write=True,
            must_cache_write=True,
            fusion_levels=[3],
            vector_load_max_len=4,
            tile_marks=["lazy_blockIdx.x", "lazy_vthread", "lazy_threadIdx.x"],
        ),
    ],
    postprocs=[
        ms.postproc.rewrite_vectorize(),
        ms.postproc.rewrite_cuda_thread_bind(warp_size=32),
        ms.postproc.verify_gpu_code(target=TARGET),
    ],
)


@pytest.mark.skip(reason="needs RPC")
def test_integration_matmul():
    os.environ["TVM_TRACKER_KEY"] = "test"
    sch = ms.autotune(
        task=ms.SearchTask(
            func=matmul,
            target=TARGET,
            task_name="cuda_matmul",
            filename="./cuda_matmul.json",
        ),
        space=SPACE,
        strategy=ms.strategy.Replay(num_iterations=32),
        measurer=ms.ProgramMeasurer(
            measure_callbacks=[
                ms.RecordToFile(),
            ]
        ),
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.sch.func))


if __name__ == "__main__":
    test_integration_matmul()
