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
"""Integration test for CUDA with Tensor Core"""
# pylint: disable=missing-function-docstring
import te_workload
import tvm
from tvm import meta_schedule as ms
from tvm import te


TARGET = tvm.target.Target("nvidia/rtx2080ti")


def test_integration_conv2d_nchwc():
    workload = te_workload.conv2d_nchwc(
        n=1,
        h=98,
        w=98,
        ci=96,
        co=192,
        kh=3,
        kw=3,
        stride=1,
        in_type="float16",
        out_type="float32",
    )
    assert list(workload.shape) == [1, 12, 96, 96, 16]
    workload = te.create_func(workload)


if __name__ == "__main__":
    test_integration_conv2d_nchwc()
