from tvm import relay
from tvm.ir import IRModule
from typing import Dict, List, Tuple
from tvm.runtime import NDArray
import torch
import torchvision.models as models


# TVM name convention -> Torch convention
NETWORK_TO_TORCH_MODEL = {
    "resnet-50": "resnet50",
    "resnext50_32x4d": "resnext50_32x4d",
    "mobilenet_v2": "mobilenet_v2",
    "densenet-121": "densenet121",
    "vgg-16": "vgg16",
    "inception_v3": "inception_v3",
    "shufflenet": "shufflenet_v2_x1_0",
}


def get_torch_network(
    name: str,
    batch_size: int,
    layout: str = "NCHW",
    dtype: str = "float32",
) -> Tuple[IRModule, Dict[str, NDArray], Tuple[int, int, int, int], Tuple[int, int]]:

    assert dtype == "float32"
    assert name in NETWORK_TO_TORCH_MODEL

    model = getattr(models, NETWORK_TO_TORCH_MODEL[name])()

    if layout == "NHWC":
        input_shape = (224, 224, 3)
    elif layout == "NCHW":
        input_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape: Tuple[int, int, int, int] = (batch_size,) + input_shape
    output_shape: Tuple[int, int] = (batch_size, 1000)

    print(input_shape)
    input_data = torch.randn(input_shape).type(torch.float32)
    scripted_model = torch.jit.trace(model, input_data).eval()
    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params, input_shape, output_shape
