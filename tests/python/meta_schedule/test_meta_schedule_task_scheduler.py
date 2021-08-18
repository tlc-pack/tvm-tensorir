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
""" Test meta schedule TaskScheduler """

from typing import List
import random

from tir_workload import matmul
from tvm import meta_schedule as ms
from tvm.meta_schedule import PyTaskScheduler, TuneContext

# pylint: disable=missing-docstring
class TestTaskScheduler(PyTaskScheduler):
    def __init__(self, tune_contexts: List[TuneContext]):
        super().__init__(tune_contexts, None, None)
        self.results = []

    def sort_all_tasks(self) -> None:
        pass

    def tune_all_tasks(self) -> None:
        pass


def test_meta_schedule_py_task_scheduler():
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

    trials = 100
    batch_size = 25

    num = 20
    count = 0

    space_gen = ms.ScheduleFn(sch_fn=schedule_matmul)
    tune_ctxs = []
    for _ in range(num):
        trials = 100 * random.randint(1, 10)
        batch_size = random.randint(10, 50)
        tune_ctxs.append(
            ms.TuneContext(
                schedule_matmul,
                space_gen,
                ms.ReplaySearchStrategy(trials, batch_size),
                database=None,
                cost_model=None,
                target=None,
                post_procs=None,
                measure_callbacks=None,
                name="test",
                seed=42,
                num_threads=1,
                verbose=0,
            )
        )
        count += trials / batch_size
        if trials % batch_size != 0:
            count += 1

    task_scheduler = TestTaskScheduler(tune_ctxs)


if __name__ == "__main__":
    test_meta_schedule_py_task_scheduler()
