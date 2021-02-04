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

import numpy as np
import pytest
import tvm
from tvm import relay
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


def verify_workload(workload):
    print("Testing", workload)
    mod, params, input_shape, output_shape = get_network(workload, 1)

    target = "llvm"
    ctx = tvm.context(target, 0)
    data = np.random.uniform(-1, 1, size=input_shape).astype("float32")

    with tvm.transform.PassContext(config={"relay.with_tir_schedule": True}):
        lib = relay.build_module.build(mod, target, params=params)
    lib_std = relay.build_module.build(mod, target, params=params)

    def run_module(lib):
        module = runtime.GraphModule(lib["default"](ctx))
        module.set_input("data", data)
        module.run()
        return module.get_output(0, tvm.nd.empty(output_shape))

    out = run_module(lib).asnumpy()
    std = run_module(lib_std).asnumpy()
    np.testing.assert_allclose(out, std, rtol=1e-4, atol=1e-4)


@pytest.mark.skip("Heavy workload")
def test_workload():
    verify_workload("simple")
    verify_workload("resnet-18")
    verify_workload("mobilenet")
    verify_workload("vgg-19")


if __name__ == "__main__":
    test_workload()
