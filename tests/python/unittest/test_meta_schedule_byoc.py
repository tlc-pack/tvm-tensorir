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
""" Test Meta Schedule Builder """
# pylint: disable=missing-docstring

import sys

import pytest
import tvm
from tvm import relay
from tvm.meta_schedule.arg_info import TensorInfo
from tvm.meta_schedule.builder import BuilderInput, LocalBuilder
from tvm.meta_schedule.runner import EvaluatorConfig, LocalRunner, RunnerInput
from tvm.meta_schedule.testing import get_network
from tvm.meta_schedule.testing.byoc_trt import (
    build_relay,
    build_relay_with_tensorrt,
    run_with_graph_executor,
)
from tvm.relay import testing
from tvm.relay.op.contrib import tensorrt
from tvm.target import Target
from tvm.tir import FloatImm

has_tensorrt_codegen = pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.tensorrt", True), reason="TensorRT codegen not available"
)
has_tensorrt_runtime = pytest.mark.skipif(
    not tensorrt.is_tensorrt_runtime_enabled(), reason="TensorRT runtime not available"
)

# conv2d+relu network
def get_conv2d_relu(
    data_shape,
    out_channels,
    kernel_size,
    strides,
    padding,
    dilation,
    groups,
    data_layout,
    kernel_layout,
    dtype,
):

    data = relay.var("data", relay.TensorType(data_shape, dtype))
    weight = relay.var("weight")

    net = relay.nn.conv2d(
        data=data,
        weight=weight,  # conv kernel
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        channels=out_channels,
        kernel_size=kernel_size,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )
    net = relay.add(net, net)
    net = relay.nn.relu(net)

    inputs = relay.analysis.free_vars(net)
    return relay.Function(inputs, net)


def verify_meta_schedule_with_tensorrt(
    mod,
    params,
    data_shape,
    use_meta_sched: bool = True,
    use_trt: bool = True,
    mode: str = "vm",
):
    if use_meta_sched:
        # With meta_schedule
        dev = "nvidia/geforce-rtx-2080"
        # Build
        builder = LocalBuilder(
            f_build=build_relay_with_tensorrt if use_trt else build_relay,
            timeout_sec=1000,
        )
        builder_input = BuilderInput(mod, Target(dev, host="llvm"), params)
        builder_result = builder.build([builder_input])[0]
        assert builder_result.error_msg is None, builder_result.error_msg
        assert builder_result.artifact_path is not None

        # Run
        runner_input = RunnerInput(
            builder_result.artifact_path,
            device_type="cuda",
            args_info=[TensorInfo("float32", data_shape)],
        )
        runner = LocalRunner(
            evaluator_config=EvaluatorConfig(
                number=5,
                repeat=2,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            ),
            f_run_evaluator=run_with_graph_executor,
        )

        # Run the module
        runner_future = runner.run([runner_input])[0]
        runner_result = runner_future.result()
        assert runner_result is not None
        assert runner_result.error_msg is None, runner_result.error_msg
        assert runner_result.run_secs is not None

        for result in runner_result.run_secs:
            if isinstance(result, FloatImm):
                result = result.value
            assert isinstance(result, float)
            assert result >= 0.0

    else:
        # Without meta_schedule
        if use_trt:
            mod, config = tensorrt.partition_for_tensorrt(mod)
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.ext.tensorrt.options": config}
            ):
                _func = relay.create_executor(
                    mode, mod=mod, device=tvm.cuda(0), target="cuda"
                ).evaluate()
        else:
            with tvm.transform.PassContext(opt_level=3):
                _func = relay.create_executor(
                    mode, mod=mod, device=tvm.cuda(0), target="cuda", params=params
                ).evaluate()


@has_tensorrt_codegen
def test_conv2d_relu():
    data_shape = (1, 1280, 14, 14)
    out_channels = 256
    kernel_size, strides, padding, dilation, groups = (1, 1), (1, 1), (0, 0, 0, 0), (1, 1), 1
    data_layout, kernel_layout = "NCHW", "OIHW"
    dtype = "float32"

    f = get_conv2d_relu(
        data_shape,
        out_channels,
        kernel_size,
        strides,
        padding,
        dilation,
        groups,
        data_layout,
        kernel_layout,
        dtype,
    )

    mod, params = testing.create_workload(f)
    verify_meta_schedule_with_tensorrt(mod, params, data_shape)


@has_tensorrt_codegen
@pytest.mark.parametrize(
    "model_name",
    ["resnet-50", "mobilenet"],
)
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("use_meta_sched", [True])
@pytest.mark.parametrize("use_trt", [True, False])
def test_relay_model(model_name: str, batch_size: int, use_meta_sched: bool, use_trt: bool):
    mod, params, input_shape, _oshape = get_network(
        name=model_name,
        batch_size=batch_size,
    )
    verify_meta_schedule_with_tensorrt(
        mod,
        params,
        input_shape,
        use_meta_sched=use_meta_sched,
        use_trt=use_trt,
        mode="vm",
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))