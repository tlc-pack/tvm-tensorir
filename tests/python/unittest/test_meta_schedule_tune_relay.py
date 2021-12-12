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
import logging
import tempfile
from typing import List, Tuple

import pytest
from tvm.meta_schedule import ReplayTraceConfig
from tvm.meta_schedule.testing import MODEL_TYPE, MODEL_TYPES, get_torch_model
from tvm.meta_schedule.tune import tune_relay
from tvm.target.target import Target
from tvm.tir import Schedule

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@pytest.mark.skip("Integration test")
@pytest.mark.parametrize("model_name", ["resnet18"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("target", ["llvm --num-cores=16", "nvidia/geforce-rtx-3070"])
def test_meta_schedule_tune_relay(model_name: str, batch_size: int, target: str):
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

    with tempfile.TemporaryDirectory() as work_dir:
        target = Target(target)
        schs: List[Schedule] = tune_relay(
            mod=mod,
            params=params,
            target=target,
            config=ReplayTraceConfig(
                num_trials_per_iter=32,
                num_trials_total=32,
            ),
            work_dir=work_dir,
        )
        for i, sch in enumerate(schs):
            print("-" * 10 + f" Part {i}/{len(schs)} " + "-" * 10)
            if sch is None:
                print("No valid schedule found!")
            else:
                print(sch.mod.script())
                print(sch.trace)


if __name__ == """__main__""":
    test_meta_schedule_tune_relay("resnet18", 1, "llvm --num-cores=16")
    test_meta_schedule_tune_relay("resnet18", 1, "nvidia/geforce-rtx-3070")
