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
""" Test meta schedule SearchStrategy """
# pylint: disable=missing-function-docstring

from tir_workload import matmul
from tvm import meta_schedule as ms


class FailBuilder(ms.PyBuilder):
    def __init__(self):
        super().__init__()

    def build(self, build_inputs: List[BuildInput]) -> List[BuildResult]:
        return [ms.BuildResult(artifact_path=None, error_msg="No return =^=")]


class PassBuilder(ms.PyBuilder):
    def __init__(self):
        super().__init__()

    def build(self, build_inputs: List[BuildInput]) -> List[BuildResult]:
        return [ms.BuildResult(artifact_path="/some/path", error_msg=None)]


def test_meta_schedule_build():
    def schedule_matmul(sch: ms.Schedule):
        block = sch.get_block("matmul")
        i, j, k = sch.get_axes(block=block)
        i_tiles = sch.sample_perfect_tile(i, n=4)
        j_tiles = sch.sample_perfect_tile(j, n=4)
        k_tiles = sch.sample_perfect_tile(k, n=2)
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
        k_0, k_1 = sch.split(loop=k, factors=k_tiles)
        sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)

    results = []
    temp = replay.generate_measure_candidates()
    while temp is not None:
        results += temp
        assert len(temp) == batch_size or len(results) == trials
        assert len(results) <= trials
        replay.notify_measure_results(temp)
        temp = replay.generate_measure_candidates()

    assert len(results) == trials


if __name__ == "__main__":
    test_meta_schedule_build()
