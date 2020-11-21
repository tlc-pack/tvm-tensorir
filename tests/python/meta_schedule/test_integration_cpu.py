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
import os

import pytest
import tvm
from tir_workload import matmul, matmul_relu
from tvm import meta_schedule as ms

TARGET = tvm.target.Target("llvm")
SPACE = ms.space.PostOrderApply(
    stages=[
        ms.rule.inline_pure_spatial(strict_mode=True),
        ms.rule.multi_level_tiling_and_fusion(
            structure="SSRSRS",
            must_cache_read=False,
            cache_read_scope="global",
            can_cache_write=True,
            must_cache_write=False,
            cache_write_scope="global",
            fusion_levels=[1, 2],
        ),
        ms.rule.mark_parallelize_outer(max_extent=256),
        ms.rule.mark_vectorize_inner(max_extent=32),
    ],
    postprocs=[
        ms.postproc.rewrite_parallel(),
        ms.postproc.rewrite_vectorize(),
    ],
)


@pytest.mark.skip(reason="needs RPC")
def test_matmul_post_order_apply():
    os.environ["TVM_TRACKER_KEY"] = "test"
    sch = ms.autotune(
        task=ms.SearchTask(
            func=matmul,
            target=TARGET,
            task_name="cpu_matmul",
            filename="./cpu_matmul.json",
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


@pytest.mark.skip(reason="needs RPC")
def test_matmul_relu_post_order_apply():
    os.environ["TVM_TRACKER_KEY"] = "test"
    sch = ms.autotune(
        task=ms.SearchTask(
            func=matmul_relu,
            target=TARGET,
            task_name="cpu_matmul_relu",
            filename="./cpu_matmul_relu.json",
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
    test_matmul_post_order_apply()
    test_matmul_relu_post_order_apply()
