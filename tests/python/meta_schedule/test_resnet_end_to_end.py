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


def get_network(name, batch_size, dtype="float32"):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "squeezenet_v1.1":
        mod, params = relay.testing.squeezenet.get_workload(
            batch_size=batch_size, version="1.1", dtype=dtype
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "simple":
        output_shape = (batch_size, 16, 224, 224)
        data = relay.var("data", relay.TensorType(input_shape, "float32"))
        weight = relay.var("weight")
        bn_gamma = relay.var("bn_gamma")
        bn_beta = relay.var("bn_beta")
        bn_mmean = relay.var("bn_mean")
        bn_mvar = relay.var("bn_var")

        simple_net = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )
        simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
        simple_net = relay.nn.relu(simple_net)
        simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)
        mod = tvm.IRModule.from_expr(simple_net)
        _net, params = relay.testing.create_workload(simple_net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


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
            consumer_inline_strict=True,
            fusion_levels=[1, 2],
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
    ],
)


@pytest.mark.skip(reason="needs RPC")
def test_end_to_end_resnet(log):
    os.environ["TVM_TRACKER_KEY"] = RPC_KEY
    mod, params, input_shape, output_shape = get_network("resnet-18", 1)

    ctx = tvm.context("llvm", 0)
    data = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    lib_std = relay.build_module.build(mod, TARGET, params=params)

    with tvm.transform.PassContext(config={"relay.with_tir_schedule": True}):
        tir_func = relay.build_module.build_primfunc(mod, TARGET, params=params)

    tuned_result = {}
    i = 0
    for target, func_map in tir_func.items():
        print(target)
        tuned_result[target] = {}
        for _, func in func_map.items():
            # print("func_name:", func_name)
            i += 1
            sch = ms.autotune(
                task=ms.SearchTask(
                    workload=func,
                    target=TARGET,
                    target_host=TARGET_HOST,
                    task_name="func" + str(i),
                    log_file=log,
                ),
                space=SPACE,
                strategy=ms.strategy.Evolutionary(
                    total_measures=32,
                    num_measures_per_iter=16,
                    population=16,
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
            tuned_result[target][func] = sch.mod

    with tvm.transform.PassContext(config={"relay.with_tir_schedule": True}):
        lib = relay.build_module.build(mod, TARGET, params=params, tune_result=tuned_result)

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
        return module.get_output(0, tvm.nd.empty(output_shape))

    std = run_module(lib_std).asnumpy()
    out = run_module(lib).asnumpy()
    np.testing.assert_allclose(out, std, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_end_to_end_resnet(None)
