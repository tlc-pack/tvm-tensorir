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
"""End to end resnet-18 CPU test"""
# pylint: disable=missing-function-docstring
import os

import numpy as np
import pytest
import tvm
import tvm.relay.testing
from tvm import meta_schedule as ms
from tvm import relay, te
from tvm.contrib import graph_runtime as runtime


# import logging
# logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion


def get_np_array(var, dtype):
    return np.random.randn(*[int(x) for x in var.type_annotation.shape]).astype(dtype)


def get_relay_conv2d(
        outc=128,
        inc=64,
        height=14,
        width=14,
        kh=3,
        kw=3,
        batch=1,
        pad=0,
        stride=1,
        dilation=1,
        layout="NHWC",
):
    dtype = "float32"
    if layout == "NHWC":
        kernel_layout = "HWIO"
        d = relay.var("data", shape=(batch, height, width, inc), dtype=dtype)
        w = relay.var("weight", shape=(kh, kw, inc, outc), dtype=dtype)
    elif layout == "NCHW":
        kernel_layout = "OIHW"
        d = relay.var("data", shape=(batch, inc, height, width), dtype=dtype)
        w = relay.var("weight", shape=(outc, inc, kh, kw), dtype=dtype)

    y = relay.nn.conv2d(
        d,
        w,
        padding=pad,
        kernel_size=(kh, kw),
        strides=(stride, stride),
        dilation=(dilation, dilation),
        channels=outc,
        groups=1,
        data_layout=layout,
        kernel_layout=kernel_layout,
    )
    mod = tvm.IRModule()
    mod["main"] = relay.Function([d, w], y)
    data, weight = get_np_array(d, dtype), get_np_array(w, dtype)
    return mod, data, weight


RPC_KEY = "test"
TARGET = tvm.target.Target("llvm")
TARGET_HOST = tvm.target.Target("llvm")
SPACE = ms.space.PostOrderApply(
    stages=[
        ms.rule.inline_pure_spatial(strict_mode=True),
        ms.rule.multi_level_tiling(
            structure="SSRSRS",
            must_cache_read=False,
            cache_read_scope="global",
            can_cache_write=True,
            must_cache_write=False,
            cache_write_scope="global",
            fusion_levels=[1, 2],
            consumer_inline_strict=True,
        ),
        ms.rule.random_compute_location(),
        ms.rule.parallelize_vectorize_unroll(
            max_jobs_per_core=16,
            max_vectorize_extent=32,
            unroll_max_steps=[0, 16, 64, 512],
            unroll_explicit=True,
        ),
    ],
    postprocs=[
        ms.postproc.rewrite_parallel_vectorize_unroll(),
        ms.postproc.rewrite_reduction_block(),
        ms.postproc.rewrite_layout()
    ],
)


@pytest.mark.skip(reason="needs RPC")
def test_end_to_end_resnet(log):
    os.environ["TVM_TRACKER_KEY"] = RPC_KEY
    mod, data, weight = get_relay_conv2d()

    ctx = tvm.context("llvm", 0)

    lib_std = relay.build_module.build(mod, TARGET, params={"weight": weight})

    with tvm.transform.PassContext(config={"relay.with_tir_schedule": True}):
        tir_func = relay.build_module.build_primfunc(mod, TARGET, params={"weight": weight})

    tuned_result = {}
    i = 0
    for target, func_map in tir_func.items():
        tuned_result[target] = {}
        for _, func in func_map.items():
            i += 1
            sch = ms.autotune(
                task=ms.SearchTask(
                    workload=func,
                    target=TARGET,
                    target_host=TARGET_HOST,
                    log_file=log,
                ),
                space=SPACE,
                strategy=ms.strategy.Evolutionary(
                    total_measures=192,
                    num_measures_per_iter=64,
                    population=2048,
                    init_measured_ratio=0.2,
                    genetic_algo_iters=10,
                    p_mutate=0.85,
                    mutator_probs={
                        ms.mutator.mutate_tile_size(): 0.95,
                        ms.mutator.mutate_compute_location(): 0.05,
                    },
                    cost_model=ms.XGBModel(
                        num_warmup_samples=0,
                    ),
                    eps_greedy=0.20,
                ),
                measurer=ms.ProgramMeasurer(
                    measure_callbacks=[
                        ms.RecordToFile(),
                    ]
                )
            )
            tuned_result[target][func] = sch.sch.module
    with ms.ApplyHistoryBest(log, SPACE):
        with tvm.transform.PassContext(opt_level=3, config={"relay.with_tir_schedule": True,
                                                            "relay.backend.use_meta_schedule": True}):
            lib = relay.build_module.build(mod, TARGET, params={"weight": weight}, tune_result=tuned_result)

    def run_module(lib):
        module = runtime.GraphModule(lib["default"](ctx))
        module.set_input("data", data)
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
        module.run()
        return module.get_output(0)

    std = run_module(lib_std).asnumpy()
    out = run_module(lib).asnumpy()
    np.testing.assert_allclose(out, std, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_end_to_end_resnet("layout_rewrite.json")
