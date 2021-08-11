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
"""End to end resnet-18 GPU test"""
# pylint: disable=missing-function-docstring
import os

import numpy as np
import pytest
import tvm
import tvm.relay.testing
from tvm import meta_schedule as ms
from tvm import relay, te, auto_scheduler
from tvm.contrib import graph_runtime as runtime
from tvm.contrib.utils import tempdir


# import logging
# logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion


def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    return mod, params, input_shape, output_shape


RPC_KEY = "rtx-3070"
network = "resnet-50"
batch_size = 1
layout = "NHWC"
target = tvm.target.Target("nvidia/geforce-rtx-2080-ti")
dtype = "float32"
TARGET_HOST = tvm.target.Target("llvm")
SPACE = ms.space.PostOrderApply(
    stages=[
        ms.rule.simplify_compute_with_const_tensor(),
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
        ms.rule.special_compute_location_gpu(),
        ms.rule.inline_pure_spatial(strict_mode=False),
        ms.rule.cross_thread_reduction(),
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
def test_end_to_end_resnet(log):
    os.environ["TVM_TRACKER_KEY"] = RPC_KEY
    mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)

    data = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    lib_std = relay.build_module.build(mod, target, params=params)
    tir_funcs = ms.extract_tasks(mod["main"], params, target)

    for func in tir_funcs.values():

        sch = ms.autotune(
            task=ms.SearchTask(
                workload=func,
                target=target,
                target_host=TARGET_HOST,
                log_file=log,
            ),
            space=SPACE,
            strategy=ms.strategy.Evolutionary(
                total_measures=16,
                num_measures_per_iter=16,
                population=2048,
                init_measured_ratio=0.2,
                genetic_algo_iters=4,
                p_mutate=0.85,
                mutator_probs={
                    ms.mutator.mutate_tile_size(): 0.90,
                    ms.mutator.mutate_auto_unroll(): 0.10,
                },
                cost_model=ms.XGBModel(xgb_eta=0.2),
                eps_greedy=0.25,
            ),
            measurer=ms.ProgramMeasurer(
                measure_callbacks=[
                    ms.RecordToFile(),
                ]
            )
        )

    with ms.ApplyHistoryBest(log, SPACE):
        with tvm.transform.PassContext(opt_level=3, config={"relay.with_tir_schedule": True,
                                                            "relay.backend.use_meta_schedule": True}):
            lib = relay.build_module.build(mod, target, params=params)

    def run_module(lib):
        ctx = tvm.context(str(target), 0)
        module = runtime.GraphModule(lib["default"](ctx))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("data", data_tvm)

        # Evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=500)
        prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res)))

        module.run()
        return module.get_output(0)

    std = run_module(lib_std).asnumpy()
    out = run_module(lib).asnumpy()
    np.testing.assert_allclose(out, std, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_end_to_end_resnet("resnet_cuda.json")
