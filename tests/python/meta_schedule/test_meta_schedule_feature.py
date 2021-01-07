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
    names = ms.feature.per_bloc_feature_names()
    n_features = len(names)
    # Create schedule
    func = te.create_func(te_workload.matmul(512, 512, 512))
    sch = ms.Schedule(func)
    block = sch.get_block("C")
    i, j, k = sch.get_axes(block)
    i_o, i_i = sch.split(i, factors=[32, 16])
    j_o, j_i = sch.split(j, factors=[64, 8])
    sch.reorder(after_axes=[i_o, j_o, k, j_i, i_i])
    sch.vectorize(j_i)
    sch.parallel(i_o)
    sch.parallel(j_o)
    sch.sch.unroll(sch.evaluate(k))
    print(tvm.script.asscript(sch.sch.func))

    # feature = ms.feature.calc_per_block_feature(sch)
    # assert feature.shape == (2, n_features)
    # feature = feature[1]
    # # correspond the features with their names
    # feature_dict = {
    #     name: value
    #     for name, value in zip(names, feature)  # pylint: disable=unnecessary-comprehension
    # }
    # for name, value in feature_dict.items():
    #     print(name, value)


def test_meta_schedule_per_block_feature_cpu_fusion():
    pass


def test_meta_schedule_per_block_feature_gpu():
    pass


if __name__ == "__main__":
    test_meta_schedule_per_block_feature_cpu_matmul()
    test_meta_schedule_per_block_feature_cpu_fusion()
    test_meta_schedule_per_block_feature_gpu()
