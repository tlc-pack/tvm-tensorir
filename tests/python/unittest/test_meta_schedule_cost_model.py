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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
from typing import List
import numpy as np
import re

import tvm
from tvm.script import tir as T
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.search_strategy import MeasureCandidate
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.cost_model import PyCostModel
from tvm.tir.schedule.schedule import Schedule


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


def test_meta_schedule_cost_model():
    class FancyCostModel(PyCostModel):
        def load(self, file_location: str) -> bool:
            return True

        def save(self, file_location: str) -> bool:
            return True

        def update(
            self,
            tune_context: TuneContext,
            candidates: List[MeasureCandidate],
            results: List[RunnerResult],
        ) -> None:
            pass

        def predict(
            self, tune_context: TuneContext, candidates: List[MeasureCandidate]
        ) -> np.ndarray:
            return [np.random.rand(10, 12)]

    model = FancyCostModel()
    assert model.save("fancy_test_location")
    assert model.load("fancy_test_location")
    model.update(TuneContext(), [], [])
    results = model.predict(TuneContext, [MeasureCandidate(Schedule(mod=Matmul), [])])
    assert len(results) == 1
    assert results[0].shape == (10, 12)


def test_meta_schedule_cost_model_as_string():
    class NotSoFancyCostModel(PyCostModel):
        def load(self, file_location: str) -> bool:
            return True

        def save(self, file_location: str) -> bool:
            return True

        def update(
            self,
            tune_context: TuneContext,
            candidates: List[MeasureCandidate],
            results: List[RunnerResult],
        ) -> None:
            pass

        def predict(
            self, tune_context: TuneContext, candidates: List[MeasureCandidate]
        ) -> np.ndarray:
            return np.random.rand(10, 12)

    cost_model = NotSoFancyCostModel()
    pattern = re.compile(r"NotSoFancyCostModel\(0x[a-f|0-9]*\)")
    assert pattern.match(str(cost_model))


if __name__ == "__main__":
    test_meta_schedule_cost_model()
    test_meta_schedule_cost_model_as_string()
