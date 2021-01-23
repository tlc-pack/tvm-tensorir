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
import te_workload
import tvm
from tvm import meta_schedule as ms
from tvm import te

RPC_KEY = "jetson-agx-xavier"
TARGET = tvm.target.Target("nvidia/jetson-agx-xavier")
TARGET_HOST = tvm.target.Target("llvm -mcpu=carmel -mtriple=aarch64-linux-gnu")
SPACE = ms.space.PostOrderApply(
    stages=[
        ms.rule.inline_pure_spatial(strict_mode=False),
        ms.rule.multi_level_tiling(
            structure="SSSRRSRS",
            must_cache_read=True,
            cache_read_scope="shared",
            can_cache_write=True,
            must_cache_write=True,
            cache_write_scope="local",
            fusion_levels=[3],
            vector_load_max_len=4,
            tile_binds=["blockIdx.x", "vthread", "threadIdx.x"],
        ),
    ],
    postprocs=[
        ms.postproc.rewrite_cooperative_fetch(),
        ms.postproc.rewrite_unbound_blocks(),
        ms.postproc.verify_gpu_code(),
    ],
)


@pytest.mark.skip(reason="needs RPC")
def test_integration_matmul():
    os.environ["TVM_TRACKER_KEY"] = RPC_KEY
    sch = ms.autotune(
        task=ms.SearchTask(
            workload=te.create_func(te_workload.matmul(1024, 1024, 1024)),
            target=TARGET,
            target_host=TARGET_HOST,
            task_name="cuda_matmul",
            log_file="./cuda_matmul.json",
        ),
        space=SPACE,
        strategy=ms.strategy.Replay(num_trials=32),
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
