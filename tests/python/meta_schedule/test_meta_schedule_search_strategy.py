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

from tvm.meta_schedule import (
    BuilderInput,
    RunnerInput,
    LocalBuilder,
    EvaluatorConfig,
    RPCConfig,
    RPCRunner,
    PySearchStrategy,
    TuneContext,
    ReplayTrace,
    ScheduleFn,
    TensorArgInfo,
)
from tvm.meta_schedule.testing import Server, Tracker
from tvm.meta_schedule.utils import get_global_func_with_default_on_worker

from tvm.script import ty
from tvm.target import Target
from tvm.tir import Schedule, FloatImm
from tvm.ir import IRModule


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


def _clean_build(artifact_path: str) -> None:
    f_clean_build = get_global_func_with_default_on_worker("meta_schedule.remove_build_dir", None)
    if f_clean_build is not None:
        f_clean_build(artifact_path)
    else:
        raise RuntimeError("Unable to find remove_build_dir function.")


def _compare_irmodule_similarity(mod1: IRModule, mod2: IRModule) -> bool:
    try:
        part1 = mod1["main"]
        part2 = mod2["main"]
    except KeyError:
        return False
    while part1 is not None and part2 is not None:
        if type(part1) != type(part2):  # pylint: disable=unidiomatic-typecheck
            return False
        try:
            try:
                part1 = part1.body
                part2 = part2.body
            except AttributeError:
                part1 = part1.block
                part2 = part2.block
        except AttributeError:
            break
    return True


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
    trials = 20
    batch_size = 7

    space_generator = ScheduleFn(sch_fn=schedule_matmul)
    tune_context = TuneContext(mod=Matmul())

    replay = ReplayTrace(batch_size, trials)
    replay.initialize_with_tune_context(tune_context)
    design_spaces = space_generator.generate_design_space(Matmul())
    replay.pre_tuning(design_spaces)

    builder = LocalBuilder()

    results = []

    with Tracker(silent=True) as tracker:
        with Server(tracker, silent=True) as server:
            candidates = replay.generate_measure_candidates()
            while candidates is not None:
                assert len(results) <= trials
                assert len(candidates) == batch_size or len(results) + len(candidates) == trials
                for candidate in candidates:
                    flag = True
                    # note that in this test, there is only 1 design space
                    for design_space in design_spaces:
                        flag |= _compare_irmodule_similarity(design_space.mod, candidate)
                    assert (
                        flag
                    ), f"The generated IRModule {candidate} does not match the design spaces."
                builder_results = builder.build(
                    [BuilderInput(mod, Target("llvm")) for mod in candidates]
                )
                for builder_result in builder_results:
                    assert builder_result.artifact_path is not None
                    assert builder_result.error_msg is None

                runner_inputs = [
                    RunnerInput(
                        builder_result.artifact_path,
                        "llvm",
                        [
                            TensorArgInfo("float32", [MATMUL_M, MATMUL_M]),
                            TensorArgInfo("float32", [MATMUL_M, MATMUL_M]),
                            TensorArgInfo("float32", [MATMUL_M, MATMUL_M]),
                        ],
                    )
                    for builder_result in builder_results
                ]

                rpc_config = RPCConfig(
                    tracker_host=tracker.host,
                    tracker_port=tracker.port,
                    tracker_key=server.key,
                    session_priority=1,
                    session_timeout_sec=100,
                )
                evaluator_config = EvaluatorConfig(
                    number=1,
                    repeat=1,
                    min_repeat_ms=0,
                    enable_cpu_cache_flush=False,
                )
                runner = RPCRunner(rpc_config, evaluator_config)

                # Run the module
                current_results = []
                runner_futures = runner.run(runner_inputs)
                for runner_future in runner_futures:
                    runner_result = runner_future.result()
                    assert runner_result.error_msg is None
                    for result in runner_result.run_sec:
                        if isinstance(result, FloatImm):
                            result = result.value
                        assert isinstance(result, float)
                        assert result >= 0.0
                    current_results.append(runner_result)

                for builder_result in builder_results:
                    _clean_build(builder_result.artifact_path)

                results += current_results
                replay.notify_runner_results(current_results)
                candidates = replay.generate_measure_candidates()

    assert len(results) == trials
    replay.post_tuning()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
