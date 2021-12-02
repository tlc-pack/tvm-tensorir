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
# pylint: disable=missing-docstring
import logging

import pytest
from tvm.meta_schedule import tune_tir
from tvm.meta_schedule.search_strategy import ReplayFuncConfig
from tvm.script import tir as T
from tvm.target.target import Target
from tvm.tir import Schedule


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [1024, 1024])
    B = T.match_buffer(b, [1024, 1024])
    C = T.match_buffer(c, [1024, 1024])
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


# pylint: enable=no-member,invalid-name,unused-variable


@pytest.mark.skip("Integration test")
def test_tune_matmul():
    sch: Schedule = tune_tir(
        mod=matmul,
        target=Target("llvm --num-cores=16"),
        config=ReplayFuncConfig(
            num_trials_per_iter=32,
            num_trials_total=1024,
        ),
    )
    if sch is None:
        print("No valid schedule found!")
    else:
        print(sch.mod.script())
        print(sch.trace)


if __name__ == """__main__""":
    test_tune_matmul()
