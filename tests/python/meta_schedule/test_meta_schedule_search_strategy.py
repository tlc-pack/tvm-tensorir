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
""" Test Meta Schedule SearchStrategy """
# pylint: disable=missing-function-docstring

import sys
import pytest

import tvm
from tvm import tir
from tvm.script import ty
from tvm.tir import Schedule
from tvm.meta_schedule import ReplaySearchStrategy, ScheduleFn

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off

@tvm.script.tir
class Matmul:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
        tir.func_attr({"global_symbol": "main"})
        A = tir.match_buffer(a, (1024, 1024), "float32")
        B = tir.match_buffer(b, (1024, 1024), "float32")
        C = tir.match_buffer(c, (1024, 1024), "float32")
        with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_tiles = sch.sample_perfect_tile(i, n=4)
    j_tiles = sch.sample_perfect_tile(j, n=4)
    k_tiles = sch.sample_perfect_tile(k, n=2)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def test_meta_schedule_search_strategy():
    trials = 100
    batch_size = 30

    space_generator = ScheduleFn(sch_fn=schedule_matmul)
    replay = ReplaySearchStrategy(trials, batch_size)
    replay.pre_tuning(design_spaces=space_generator.generate_design_space(Matmul()))

    results = []
    candidates = replay.generate_measure_candidates()
    while candidates is not None:
        results += candidates
        assert len(results) <= trials
        assert len(candidates) == batch_size or len(results) == trials
        for candidate in candidates:
            assert len(candidate.decisions) == 0
        replay.notify_runner_results(
            [(None, None) for candidate in candidates]
        )  # Assume runner return (None, None) for each candidate
        candidates = replay.generate_measure_candidates()

    assert len(results) == trials


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
