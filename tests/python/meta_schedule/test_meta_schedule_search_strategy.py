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
from typing import List

import sys

import pytest

import tvm
from tvm import tir
from tvm.meta_schedule import (
    PySearchStrategy,
    ReplayTrace,
    ScheduleFn,
    TuneContext,
    RunnerResult,
)
from tvm.script import ty
from tvm.tir.schedule import Schedule, Trace


MATMUL_M = 32

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument, unbalanced-tuple-unpacking
# fmt: off

@tvm.script.tir
class Matmul:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
        tir.func_attr({"global_symbol": "main"})
        A = tir.match_buffer(a, (32, 32), "float32")
        B = tir.match_buffer(b, (32, 32), "float32")
        C = tir.match_buffer(c, (32, 32), "float32")
        with tir.block([32, 32, tir.reduce_axis(0, 32)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _is_trace_equal(sch_1: Schedule, sch_2: Schedule) -> bool:
    trace_1 = Trace(sch_1.trace.insts, {})
    trace_2 = Trace(sch_2.trace.insts, {})
    return str(trace_1) == str(trace_2)


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_tiles = sch.sample_perfect_tile(i, n=4)
    j_tiles = sch.sample_perfect_tile(j, n=4)
    k_tiles = sch.sample_perfect_tile(k, n=2)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def test_meta_schedule_py_search_strategy():
    class MySearchStrategy(PySearchStrategy):
        """test class"""

        def initialize_with_tune_context(self, tune_context):
            raise Exception("TestSearchStrategy")

        def pre_tuning(self, design_spaces):
            pass

        def post_tuning(self):
            pass

        def generate_measure_candidates(self):
            pass

        def notify_runner_results(self, results):
            pass

    search_strategy = MySearchStrategy()
    with pytest.raises(Exception, match="TestSearchStrategy"):
        search_strategy.initialize_with_tune_context(TuneContext(mod=Matmul()))


def test_meta_schedule_replay_trace():
    num_trials_per_iter = 7
    num_trials_total = 20

    (example_sch,) = ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(Matmul())
    replay = ReplayTrace(num_trials_per_iter=num_trials_per_iter, num_trials_total=num_trials_total)
    tune_context = TuneContext(mod=Matmul())
    replay.initialize_with_tune_context(tune_context)

    num_trials_each_round: List[int] = []
    replay.pre_tuning([example_sch])
    while True:
        candidates = replay.generate_measure_candidates()
        if candidates is None:
            break
        num_trials_each_round.append(len(candidates))
        runner_results: List[RunnerResult] = []
        for candidate in candidates:
            assert _is_trace_equal(candidate.sch, example_sch)
            runner_results.append(RunnerResult(run_secs=[0.5, 0.4, 0.3], error_msg=None))
        replay.notify_runner_results(runner_results)
    replay.post_tuning()
    assert num_trials_each_round == [7, 7, 6]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
