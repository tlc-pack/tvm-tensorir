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
""" Test Meta Schedule"""

import sys
import tempfile
import os.path as osp

import pytest

import tvm
from tvm import tir
from tvm.script import ty
from tvm.tir import Schedule
from tvm.meta_schedule import (
    RoundRobin,
    LocalBuilder,
    RPCRunner,
    TuneContext,
    ScheduleFn,
    ReplayTrace,
    JSONFileDatabase,
    RPCConfig,
    EvaluatorConfig,
)
from tvm.meta_schedule.testing import Server, Tracker

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


@tvm.script.tir
class MatmulModule:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, (32, 32), "float32")
        B = tir.match_buffer(b, (32, 32), "float32")
        C = tir.match_buffer(c, (32, 32), "float32")
        with tir.block([32, 32, tir.reduce_axis(0, 32)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
class MatmulReluModule:
    def main(a: ty.handle, b: ty.handle, d: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, (16, 16), "float32")
        B = tir.match_buffer(b, (16, 16), "float32")
        D = tir.match_buffer(d, (16, 16), "float32")
        C = tir.alloc_buffer((16, 16), "float32")
        with tir.block([16, 16, tir.reduce_axis(0, 16)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        with tir.block([16, 16], "relu") as [vi, vj]:
            D[vi, vj] = tir.max(C[vi, vj], 0.0)


@tvm.script.tir
class BatchMatmulModule:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, [4, 8, 8])
        B = tir.match_buffer(b, [4, 8, 8])
        C = tir.match_buffer(c, [4, 8, 8])
        with tir.block([4, 8, 8, tir.reduce_axis(0, 8)], "matmul") as [vn, vi, vj, vk]:
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


def test_meta_schedule_task_scheduler_single():
    num_trials_per_iter = 10
    num_trials_total = 30
    with tempfile.TemporaryDirectory() as tmpdir:
        record_path = osp.join(tmpdir, "records.json")
        workload_path = osp.join(tmpdir, "workloads.json")
        database = JSONFileDatabase(record_path, workload_path)

        with Tracker(silent=True) as tracker:
            with Server(tracker, silent=True) as server:
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

                tasks = [
                    TuneContext(
                        MatmulModule(),
                        target=tvm.target.Target("llvm"),
                        space_generator=ScheduleFn(sch_fn=_schedule_matmul),
                        search_strategy=ReplayTrace(num_trials_per_iter, num_trials_total),
                        task_name="Test",
                        rand_state=42,
                    )
                ]

                round_robin = RoundRobin(tasks, LocalBuilder(), runner, database)
                round_robin.tune()

                records = database.get_top_k(database.lookup_or_add(tasks[0].mod), num_trials_total)
                for record in records:
                    print(record.run_secs)

                assert len(database) == num_trials_total * len(tasks)


def test_meta_schedule_task_scheduler_multiple():
    num_trials_per_iter = 4
    num_trials_total = 40
    with tempfile.TemporaryDirectory() as tmpdir:
        record_path = osp.join(tmpdir, "records.json")
        workload_path = osp.join(tmpdir, "workloads.json")
        database = JSONFileDatabase(record_path, workload_path)

        with Tracker(silent=True) as tracker:
            with Server(tracker, silent=True) as server:
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

                round_robin = RoundRobin(tasks, LocalBuilder(), runner, database)
                round_robin.tune()
                assert len(database) == num_trials_total * len(tasks)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
