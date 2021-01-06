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
"""Test for feature extraction"""
# pylint: disable=missing-function-docstring
import te_workload
from tvm import te
from tvm import meta_schedule as ms
import tvm


def test_meta_schedule_per_block_feature_cpu_matmul():
    func = te.create_func(te_workload.matmul(512, 512, 512))
    sch = ms.Schedule(func)
    print(tvm.script.asscript(sch.sch.func))
    print(ms.feature.calc_per_block_feature(sch))


def test_meta_schedule_per_block_feature_cpu_fusion():
    pass


def test_meta_schedule_per_block_feature_gpu():
    pass


if __name__ == "__main__":
    test_meta_schedule_per_block_feature_cpu_matmul()
    test_meta_schedule_per_block_feature_cpu_fusion()
    test_meta_schedule_per_block_feature_gpu()
