from tvm import relay
from tvm.ir import IRModule
from typing import Dict, List, Tuple
from tvm.runtime import NDArray
from enum import Enum

# Model types supported in Torchvision
class MODEL_TYPE(Enum):
    IMAGE_CLASSIFICATION = (1,)
    VIDEO_CLASSIFICATION = (2,)
    SEGMENTATION = (3,)
    OBJECT_DETECTION = (4,)


# Specify the type of each model
MODEL_TYPES = {
    # Image classification models
    "resnet50": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "alexnet": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "vgg16": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "squeezenet1_0": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "densenet121": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "densenet161": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "densenet169": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "densenet201": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "inception_v3": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "googlenet": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "shufflenet_v2_x1_0": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "mobilenet_v2": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "mobilenet_v3_large": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "mobilenet_v3_small": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "resnext50_32x4d": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "wide_resnet50_2": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "mnasnet1_0": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "efficientnet_b0": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "efficientnet_b1": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "efficientnet_b2": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "efficientnet_b3": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "efficientnet_b4": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "efficientnet_b5": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "efficientnet_b6": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "efficientnet_b7": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_y_400mf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_y_800mf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_y_1_6gf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_y_3_2gf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_y_8gf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_y_16gf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_y_32gf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_x_400mf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_x_800mf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_x_1_6gf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_x_3_2gf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_x_8gf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_x_16gf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "regnet_x_32gf": MODEL_TYPE.IMAGE_CLASSIFICATION,
    # Semantic Segmentation models
    "fcn_resnet50": MODEL_TYPE.SEGMENTATION,
    "fcn_resnet101": MODEL_TYPE.SEGMENTATION,
    "deeplabv3_resnet50": MODEL_TYPE.SEGMENTATION,
    "deeplabv3_resnet101": MODEL_TYPE.SEGMENTATION,
    "deeplabv3_mobilenet_v3_large": MODEL_TYPE.SEGMENTATION,
    "lraspp_mobilenet_v3_large": MODEL_TYPE.SEGMENTATION,
    # Object detection models
    # @Sung: Following networks are not runnable since Torch frontend cannot handle aten::remainder.
    #        "retinanet_resnet50_fpn", "keypointrcnn_resnet50_fpn",
    "fasterrcnn_resnet50_fpn": MODEL_TYPE.OBJECT_DETECTION,
    "fasterrcnn_mobilenet_v3_large_fpn": MODEL_TYPE.OBJECT_DETECTION,
    "fasterrcnn_mobilenet_v3_large_320_fpn": MODEL_TYPE.OBJECT_DETECTION,
    "retinanet_resnet50_fpn": MODEL_TYPE.OBJECT_DETECTION,
    "maskrcnn_resnet50_fpn": MODEL_TYPE.OBJECT_DETECTION,
    "keypointrcnn_resnet50_fpn": MODEL_TYPE.OBJECT_DETECTION,
    "ssd300_vgg16": MODEL_TYPE.OBJECT_DETECTION,
    "ssdlite320_mobilenet_v3_large": MODEL_TYPE.OBJECT_DETECTION,
    # Video classification
    "r3d_18": MODEL_TYPE.VIDEO_CLASSIFICATION,
    "mc3_18": MODEL_TYPE.VIDEO_CLASSIFICATION,
    "r2plus1d_18": MODEL_TYPE.VIDEO_CLASSIFICATION,
}


def get_torch_model(
    model_name: str,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, int],
    dtype: str = "float32",
) -> Tuple[IRModule, Dict[str, NDArray]]:
    """Load model from torch model zoo
    Parameters
    ----------
    model_name : str
        The name of the model to load
    input_shape: Tuple[int, ...]
        Tuple for input shape
    output_shape: Tuple[int, int]
        Tuple for output shape
    dtype: str
        Tensor data type
    """

    assert dtype == "float32"

    import torch
    import torchvision.models as models

    def do_trace(model, inp):
        model_trace = torch.jit.trace(model, inp)
        model_trace.eval()
        return model_trace

    # Load model from torchvision
    if MODEL_TYPES[model_name] == MODEL_TYPE.IMAGE_CLASSIFICATION:
        model = getattr(models, model_name)()
    elif MODEL_TYPES[model_name] == MODEL_TYPE.SEGMENTATION:
        model = getattr(models.segmentation, model_name)()
    elif MODEL_TYPES[model_name] == MODEL_TYPE.OBJECT_DETECTION:
        model = getattr(models.detection, model_name)()
    elif MODEL_TYPES[model_name] == MODEL_TYPE.VIDEO_CLASSIFICATION:
        model = getattr(models.video, model_name)()
    else:
        raise ValueError("Unsupported model in Torch model zoo.")

    # Setup input
    input_data = torch.randn(input_shape).type(torch.float32)
    shape_list = [("input0", input_shape)]

    # Get trace. Depending on the model type, wrapper may be necessary.
    if MODEL_TYPES[model_name] == MODEL_TYPE.SEGMENTATION:

        class TraceWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inp):
                out = self.model(inp)
                return out["out"]

        wrapped_model = TraceWrapper(model)
        wrapped_model.eval()
        with torch.no_grad():
            scripted_model = do_trace(wrapped_model, input_data)

    elif MODEL_TYPES[model_name] == MODEL_TYPE.OBJECT_DETECTION:

        def dict_to_tuple(out_dict):
            if "masks" in out_dict.keys():
                return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
            return out_dict["boxes"], out_dict["scores"], out_dict["labels"]

        class TraceWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inp):
                out = self.model(inp)
                return dict_to_tuple(out[0])

        wrapped_model = TraceWrapper(model)
        wrapped_model.eval()
        with torch.no_grad():
            out = wrapped_model(input_data)
            scripted_model = do_trace(wrapped_model, input_data)
    else:
        scripted_model = do_trace(model, input_data)

    # Convert torch model to relay module
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params
