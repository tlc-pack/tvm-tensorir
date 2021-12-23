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
# pylint: disable=missing-docstring
import sys
from typing import Tuple

import pytest

import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing import MODEL_TYPE, MODEL_TYPES, get_torch_model


@pytest.mark.skip("Skip because it runs too slowly as a unittest")
@pytest.mark.parametrize(
    "model_name",
    [
        # Image classification
        "resnet50",
        "alexnet",
        "vgg16",
        "squeezenet1_0",
        "densenet121",
        "densenet161",
        "densenet169",
        "densenet201",
        "inception_v3",
        "googlenet",
        "shufflenet_v2_x1_0",
        "mobilenet_v2",
        "mobilenet_v3_large",
        "mobilenet_v3_small",
        "resnext50_32x4d",
        "wide_resnet50_2",
        "mnasnet1_0",
        # Segmentation
        "fcn_resnet50",
        "fcn_resnet101",
        "deeplabv3_resnet50",
        "deeplabv3_resnet101",
        "deeplabv3_mobilenet_v3_large",
        "lraspp_mobilenet_v3_large",
        # Object detection
        "fasterrcnn_resnet50_fpn",
        "fasterrcnn_mobilenet_v3_large_fpn",
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "maskrcnn_resnet50_fpn",
        # video classification
        "r3d_18",
        "mc3_18",
        "r2plus1d_18",
    ],
)
@pytest.mark.parametrize("batch_size", [1, 8, 16])
@pytest.mark.parametrize("target", ["llvm", "cuda"])
def test_meta_schedule_extract_from_torch_model(model_name: str, batch_size: int, target: str):
    if model_name == "inception_v3" and batch_size == 1:
        pytest.skip("inception_v3 does not handle batch_size of 1")

    input_shape: Tuple[int, ...]
    if MODEL_TYPES[model_name] == MODEL_TYPE.IMAGE_CLASSIFICATION:
        input_shape = (batch_size, 3, 299, 299)
    elif MODEL_TYPES[model_name] == MODEL_TYPE.SEGMENTATION:
        input_shape = (batch_size, 3, 299, 299)
    elif MODEL_TYPES[model_name] == MODEL_TYPE.OBJECT_DETECTION:
        input_shape = (1, 3, 300, 300)
    elif MODEL_TYPES[model_name] == MODEL_TYPE.VIDEO_CLASSIFICATION:
        input_shape = (batch_size, 3, 3, 299, 299)
    else:
        raise ValueError("Unsupported model: " + model_name)

    output_shape: Tuple[int, int] = (batch_size, 1000)
    mod, params = get_torch_model(
        model_name=model_name,
        input_shape=input_shape,
        output_shape=output_shape,
        dtype="float32",
    )
    target = tvm.target.Target(target)
    ms.integration.extract_task_from_relay(mod, params=params, target=target)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
