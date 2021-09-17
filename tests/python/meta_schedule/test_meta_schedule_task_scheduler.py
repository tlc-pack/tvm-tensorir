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
""" Test Meta Schedule Task Scheduler """

from typing import List

import sys
import random

import pytest

import tvm
from tvm import tir
from tvm.script import ty
from tvm.ir import IRModule
from tvm.tir import Schedule
from tvm.meta_schedule import (
    RoundRobin,
    PyBuilder,
    PyRunner,
    PyDatabase,
    BuilderInput,
    BuilderResult,
    RunnerInput,
    RunnerFuture,
    RunnerResult,
    TuneContext,
    TuningRecord,
    WorkloadToken,
    ScheduleFn,
    ReplayTrace,
)
from tvm.meta_schedule.utils import structural_hash


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


@tvm.script.tir
class MatmulModule:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, (1024, 1024), "float32")
        B = tir.match_buffer(b, (1024, 1024), "float32")
        C = tir.match_buffer(c, (1024, 1024), "float32")
        with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
class MatmulReluModule:
    def main(a: ty.handle, b: ty.handle, d: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, (1024, 1024), "float32")
        B = tir.match_buffer(b, (1024, 1024), "float32")
        D = tir.match_buffer(d, (1024, 1024), "float32")
        C = tir.alloc_buffer((1024, 1024), "float32")
        with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        with tir.block([1024, 1024], "relu") as [vi, vj]:
            D[vi, vj] = tir.max(C[vi, vj], 0.0)


@tvm.script.tir
class BatchMatmulModule:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, [16, 128, 128])
        B = tir.match_buffer(b, [16, 128, 128])
        C = tir.match_buffer(c, [16, 128, 128])
        with tir.block([16, 128, 128, tir.reduce_axis(0, 128)], "matmul") as [vn, vi, vj, vk]:
            with tir.init():
                C[vn, vi, vj] = 0.0
            C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


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


def _schedule_batch_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k, t = sch.get_loops(block=block)
    i_tiles = sch.sample_perfect_tile(i, n=4)
    j_tiles = sch.sample_perfect_tile(j, n=4)
    k_tiles = sch.sample_perfect_tile(k, n=2)
    t_tiles = sch.sample_perfect_tile(t, n=2)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    t_0, t_1 = sch.split(loop=t, factors=t_tiles)
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3, t_0, t_1)


class DummyRunnerFuture(RunnerFuture):
    def done(self) -> bool:
        return True

    def result(self) -> RunnerResult:
        return RunnerResult([random.uniform(5, 30) for _ in range(random.randint(1, 10))], None)


class DummyBuilder(PyBuilder):
    def build(self, build_inputs: List[BuilderInput]) -> List[BuilderResult]:
        return [BuilderResult("test_path", None) for _ in build_inputs]


class DummyRunner(PyRunner):
    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        return [DummyRunnerFuture() for _ in runner_inputs]


class DummyDatabase(PyDatabase):
    def __init__(self):
        super().__init__()
        self.records = []
        self.workload_reg = []

    def initialize_with_tune_context(self, tune_context: TuneContext) -> None:
        pass

    def add(self, record: TuningRecord) -> None:
        self.records.append(record)

    def get_top_k(self, workload: WorkloadToken, top_k: int) -> List[TuningRecord]:
        return list(
            filter(
                lambda x: x.workload.shash == workload.shash,
                sorted(self.records, key=lambda x: sum(x.run_secs) / len(x.run_secs)),
            )
        )[: int(top_k)]

    def lookup_or_add(self, mod: IRModule) -> WorkloadToken:
        for workload in self.workload_reg:
            if tvm.ir.structural_equal(workload.mod, mod):
                return workload
        workload = WorkloadToken(mod, structural_hash(mod), len(self.workload_reg))
        self.workload_reg.append(workload)
        return workload

    def __len__(self) -> int:
        return len(self.records)

    def print_results(self) -> None:
        print("\n".join([str(r) for r in self.records]))


def test_meta_schedule_task_scheduler_single():
    sch_fn = ScheduleFn(sch_fn=_schedule_matmul)
    num_trials_per_iter = 3
    num_trials_total = 10
    replay = ReplayTrace(num_trials_per_iter, num_trials_total)
    task = TuneContext(
        MatmulModule(),
        target=tvm.target.Target("llvm"),
        space_generator=sch_fn,
        search_strategy=replay,
        task_name="Test",
        rand_state=42,
    )
    database = DummyDatabase()
    round_robin = RoundRobin([task], DummyBuilder(), DummyRunner(), database)
    round_robin.tune()
    assert len(database) == num_trials_total


def test_meta_schedule_task_scheduler_multiple():
    num_trials_per_iter = 6
    num_trials_total = 101
    tasks = [
        TuneContext(
            MatmulModule(),
            target=tvm.target.Target("llvm"),
            space_generator=ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ReplayTrace(num_trials_per_iter, num_trials_total),
            task_name="Matmul",
            rand_state=42,
        ),
        TuneContext(
            MatmulReluModule(),
            target=tvm.target.Target("llvm"),
            space_generator=ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ReplayTrace(num_trials_per_iter, num_trials_total),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
        TuneContext(
            BatchMatmulModule(),
            target=tvm.target.Target("llvm"),
            space_generator=ScheduleFn(sch_fn=_schedule_batch_matmul),
            search_strategy=ReplayTrace(num_trials_per_iter, num_trials_total),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
    ]
    database = DummyDatabase()
    round_robin = RoundRobin(tasks, DummyBuilder(), DummyRunner(), database)
    round_robin.tune()
    assert len(database) == num_trials_total * len(tasks)
    print(database.workload_reg)
    for task in tasks:
        assert len(database.get_top_k(database.lookup_or_add(task.mod), 1e9)) == num_trials_total


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
