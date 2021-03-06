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
from tvm import relay, te, auto_scheduler
from tvm.contrib import graph_runtime
from tvm.contrib.utils import tempdir


# import logging
# logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion


def get_np_array(var, dtype):
    return np.random.randn(*[int(x) for x in var.type_annotation.shape]).astype(dtype)


def get_relay_conv2d(
        outc=64,
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


def get_relay_dense(m=128, n=128, k=128):
    dtype = "float32"
    d = relay.var("data", shape=(m, k), dtype=dtype)
    w = relay.var("weight", shape=(n, k), dtype=dtype)
    y = relay.nn.dense(d, w, units=n)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([d, w], y)
    data, weight = get_np_array(d, dtype), get_np_array(w, dtype)
    return mod, data, weight


RPC_KEY = "raspi4b-aarch64"
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
        ms.postproc.rewrite_reduction_block(),
        ms.postproc.rewrite_parallel_vectorize_unroll(),
        ms.postproc.disallow_dynamic_loops(),
        ms.postproc.rewrite_layout()
    ],
)


@pytest.mark.skip(reason="needs RPC")
def tune_and_check(log, mod, data, weight):
    os.environ["TVM_TRACKER_KEY"] = RPC_KEY


    lib_std = relay.build_module.build(mod, TARGET, params={"weight": weight})

    with tvm.transform.PassContext(opt_level=3, config={"relay.with_tir_schedule": True,
                                                        "relay.backend.use_meta_schedule": True}):
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
                    total_measures=2048,
                    num_measures_per_iter=64,
                    population=2048,
                    init_measured_ratio=0.2,
                    genetic_algo_iters=10,
                    p_mutate=0.85,
                    mutator_probs={
                        ms.mutator.mutate_tile_size(): 0.90,
                        ms.mutator.mutate_compute_location(): 0.05,
                        ms.mutator.mutate_auto_unroll(): 0.03,
                        ms.mutator.mutate_parallel(max_jobs_per_core=16): 0.02
                    },
                    cost_model=ms.XGBModel(),
                    eps_greedy=0.25,
                ),
                measurer=ms.ProgramMeasurer(
                    measure_callbacks=[
                        ms.RecordToFile(),
                    ]
                ),
            )
    with ms.ApplyHistoryBest(log, SPACE):
        with tvm.transform.PassContext(opt_level=3, config={"relay.with_tir_schedule": True,
                                                            "relay.backend.use_meta_schedule": True}):
            lib = relay.build_module.build(mod, TARGET, params={"weight": weight}, tune_result={})
    use_ndk = False

    def run_module(lib, use_arm=False):
        if not use_arm:
            ctx = tvm.context("llvm", 0)
            module = graph_runtime.GraphModule(lib["default"](ctx))
            module.set_input("data", data)
            print("Evaluate inference time cost...")
            ftimer = module.module.time_evaluator("run", ctx, number=10, repeat=100, min_repeat_ms=50)
            prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
            print(prof_res)
            print(
                "Mean inference time (std dev): %.2f ms (%.2f ms)"
                % (np.mean(prof_res), np.std(prof_res))
            )
            module.run()
            return module.get_output(0)
        else:
            # Export library
            tmp = tempdir()
            if use_ndk:
                from tvm.contrib import ndk

                filename = "net.so"
                lib.export_library(tmp.relpath(filename), ndk.create_shared)
            else:
                filename = "net.tar"
                lib.export_library(tmp.relpath(filename))

            # Upload module to device
            print("Upload...")
            remote = auto_scheduler.utils.request_remote(RPC_KEY, "172.16.2.241", 4445, timeout=10000)
            remote.upload(tmp.relpath(filename))
            rlib = remote.load_module(filename)

            # Create graph runtime
            ctx = remote.cpu()
            module = graph_runtime.GraphModule(rlib["default"](ctx))
            data_tvm = tvm.nd.array((np.random.uniform(size=data.shape)).astype("float32"))
            module.set_input("data", data_tvm)
            # Evaluate
            print("Evaluate inference time cost...")
            ftimer = module.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=500)
            prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
            print(
                "Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
            )

            module.run()
            return module.get_output(0)

    std = run_module(lib_std, False).asnumpy()
    out = run_module(lib, False).asnumpy()
    np.testing.assert_allclose(out, std, rtol=1e-4, atol=1e-4)


def test_conv2d():
    mod, data, weight = get_relay_conv2d(height=32, width=32, batch=16)
    tune_and_check("conv2d.json", mod, data, weight)


def test_dense():
    mod, data, weight = get_relay_dense()
    tune_and_check("dense.json", mod, data, weight)


if __name__ == "__main__":
    test_conv2d()
    test_dense()
