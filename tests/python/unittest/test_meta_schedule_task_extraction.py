import sys
from typing import Dict, List, Tuple

import pytest
import tvm
from tvm import meta_schedule as ms
from tvm.ir import IRModule
from tvm.meta_schedule.integration import ExtractedTask
from tvm.meta_schedule.testing import get_network, get_torch_network
from tvm.runtime import NDArray


@pytest.mark.parametrize("network_name", ["resnet", "resnet3d"])
@pytest.mark.parametrize("num_layers", [18, 34, 50, 101, 152])
@pytest.mark.parametrize("batch_size", [1, 8, 16])
@pytest.mark.parametrize("target", ["llvm", "cuda"])
def test_meta_schedule_extract_from_relay_network_with_num_layers(
    network_name: str, num_layers: int, batch_size: int, target: str
):
    mod, params, input_shape, output_shape = get_network(
        name=network_name + "-" + str(num_layers),
        batch_size=batch_size,
        layout="NHWC",
        dtype="float32",
    )
    extracted_tasks = ms.integration.extract_task(mod, params, target=target)


@pytest.mark.parametrize("network_name", ["mobilenet", "squeezenet_v1.1", "inception_v3", "mxnet"])
@pytest.mark.parametrize("batch_size", [1, 8, 16])
@pytest.mark.parametrize("target", ["llvm", "cuda"])
def test_meta_schedule_extract_from_relay_network(network_name: str, batch_size: int, target: str):
    layout = "NHWC"
    if network_name == "squeezenet_v1.1" or network_name == "mxnet":
        layout = "NCHW"

    mod, params, input_shape, output_shape = get_network(
        name=network_name,
        batch_size=batch_size,
        layout=layout,
        dtype="float32",
    )
    extracted_tasks = ms.integration.extract_task(mod, params, target=target)


@pytest.mark.parametrize(
    "network_name",
    [
        "resnet-50",
        "resnext50_32x4d",
        "mobilenet_v2",
        "shufflenet",
        "densenet-121",
        "vgg-16",
    ],
)
@pytest.mark.parametrize("batch_size", [1, 8, 16])
@pytest.mark.parametrize("target", ["llvm", "cuda"])
def test_meta_schedule_extract_from_torch_network(network_name: str, batch_size: int, target: str):
    layout = "NCHW"
    mod, params, input_shape, output_shape = get_torch_network(
        name=network_name,
        batch_size=batch_size,
        layout=layout,
        dtype="float32",
    )
    extracted_tasks = ms.integration.extract_task(mod, params, target=target)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
