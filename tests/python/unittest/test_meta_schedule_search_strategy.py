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
from tvm.ir.base import assert_structural_equal
from tvm.ir.module import IRModule
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule import space_generator
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.search_strategy.replay_func import ReplayFunc
from tvm.meta_schedule.space_generator import ScheduleFn
from tvm.meta_schedule.search_strategy import ReplayTrace
from tvm.meta_schedule.space_generator.space_generator import PySpaceGenerator

from tvm.script import tir as T
from tvm.tir import schedule
from tvm.tir.schedule import Schedule, Trace


MATMUL_M = 32

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument, unbalanced-tuple-unpacking
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (32, 32), "float32")
        B = T.match_buffer(b, (32, 32), "float32")
        C = T.match_buffer(c, (32, 32), "float32")
        with T.block([32, 32, T.reduce_axis(0, 32)], "matmul") as [vi, vj, vk]:
            with T.init():
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
    # TODO(@zxybazh): Change to `sample_perfect_tile` after upstreaming
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def test_meta_schedule_replay_trace():
    num_trials_per_iter = 7
    num_trials_total = 20

    (example_sch,) = ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(Matmul)
    replay_trace = ReplayTrace(
        num_trials_per_iter=num_trials_per_iter,
        num_trials_total=num_trials_total,
    )
    tune_context = TuneContext(mod=Matmul)
    replay_trace.initialize_with_tune_context(tune_context)

    num_trials_each_iter: List[int] = []
    replay_trace.pre_tuning([example_sch])
    while True:
        candidates = replay_trace.generate_measure_candidates()
        if candidates is None:
            break
        num_trials_each_iter.append(len(candidates))
        runner_results: List[RunnerResult] = []
        for candidate in candidates:
            assert _is_trace_equal(candidate.sch, example_sch)
            runner_results.append(RunnerResult(run_secs=[0.11, 0.41, 0.54], error_msg=None))
        replay_trace.notify_runner_results(runner_results)
    replay_trace.post_tuning()
    assert num_trials_each_iter == [7, 7, 6]


def test_meta_schedule_replay_func():
    def _schedule_matmul_part_0(sch: Schedule):
        block = sch.get_block("matmul")
        i, j, k = sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
        k_0, k_1 = sch.split(loop=k, factors=[32, 32])

    def _schedule_matmul_part_1(sch: Schedule):
        block = sch.get_block("matmul")
        i_0, i_1, i_2, i_3, j_0, j_1, j_2, j_3, k_0, k_1 = sch.get_loops(block=block)
        sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)

    num_trials_per_iter = 7
    num_trials_total = 20

    replay_func = ReplayFunc(
        num_trials_per_iter=num_trials_per_iter,
        num_trials_total=num_trials_total,
        space_generator=ScheduleFn(sch_fn=_schedule_matmul_part_1),
    )
    tune_context = TuneContext(mod=Matmul)
    replay_func.initialize_with_tune_context(tune_context)
    design_spaces = ScheduleFn(sch_fn=_schedule_matmul_part_0).generate_design_space(Matmul)

    num_trials_each_iter: List[int] = []
    replay_func.pre_tuning(design_spaces=design_spaces)
    (correct_sch,) = ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(Matmul)
    while True:
        candidates = replay_func.generate_measure_candidates()
        if candidates is None:
            break
        num_trials_each_iter.append(len(candidates))
        runner_results: List[RunnerResult] = []
        for candidate in candidates:
            assert_structural_equal(candidate.sch.mod, correct_sch.mod)
            runner_results.append(RunnerResult(run_secs=[0.11, 0.41, 0.54], error_msg=None))
        replay_func.notify_runner_results(runner_results)
    replay_func.post_tuning()
    assert num_trials_each_iter == [7, 7, 6]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
