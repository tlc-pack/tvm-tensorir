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
""" Test for meta schedule analysis """
# pylint: disable=missing-function-docstring
import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.hybrid import ty

# pylint: disable=invalid-name,no-member


@tvm.hybrid.script
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "C") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])


@tvm.hybrid.script
def split_ewise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))

    for i in range(0, 16384):
        with tir.block([128, 128], "B") as [vi, vj]:
            tir.bind(vi, i // 128)
            tir.bind(vj, i % 128)
            B[vi, vj] = A[vi, vj] * 2.0


# pylint: enable=invalid-name,no-member


def test_meta_schedule_analysis_is_trivial_binding():
    sch = ms.Schedule(func=matmul)
    assert ms.analysis.is_trivial_binding(sch, sch.get_block("C"))
    sch = ms.Schedule(func=split_ewise)
    assert not ms.analysis.is_trivial_binding(sch, sch.get_block("B"))


def test_meta_schedule_analysis_get_iter_type():
    sch = ms.Schedule(func=matmul)
    assert ms.analysis.get_iter_type(sch, sch.get_block("C")) == [
        "spatial",
        "spatial",
        "reduce",
    ]
    sch = ms.Schedule(func=split_ewise)
    assert ms.analysis.get_iter_type(sch, sch.get_block("B")) == [
        "spatial",
        "spatial",
    ]


def test_meta_schedule_analysis_is_leaf():
    sch = ms.Schedule(func=matmul)
    assert ms.analysis.is_leaf(sch, sch.get_block("C"))


if __name__ == "__main__":
    test_meta_schedule_analysis_is_trivial_binding()
    test_meta_schedule_analysis_get_iter_type()
    test_meta_schedule_analysis_is_leaf()
