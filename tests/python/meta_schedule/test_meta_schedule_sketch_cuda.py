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
"""Test Ansor-like sketch generation in subgraphs in meta schedule"""
# pylint: disable=missing-function-docstring
from typing import List

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import ty


def _get_support(func: tir.PrimFunc, task_name: str, target):
    return ms.space.PostOrderApply(
        stages=[
            ms.rule.inline_pure_spatial(strict_mode=False),
            ms.rule.multi_level_tiling_and_fusion(
                structure="SSSRRSRS",
                must_cache_read=True,
                can_cache_write=True,
                must_cache_write=True,
                fusion_levels=[3],
            ),
        ]
    ).get_support(task=ms.SearchTask(func=func, task_name=task_name, target_host="llvm"))


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks
# fmt: off

@tvm.script.tir
def workload_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (512, 512), "float32")
    B = tir.match_buffer(b, (512, 512), "float32")
    C = tir.match_buffer(c, (512, 512), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([512, 512, tir.reduce_axis(0, 512)], "C") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


def test_meta_schedule_sketch_cuda_matmul():
    func = workload_matmul
    support = _get_support(func=func, task_name="matmul", target="cuda")
    for sch in support:
        print(tvm.script.asscript(sch.sch.func))


if __name__ == "__main__":
    test_meta_schedule_sketch_cuda_matmul()
