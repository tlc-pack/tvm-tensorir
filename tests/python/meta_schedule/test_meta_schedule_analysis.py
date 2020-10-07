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
import pytest

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.hybrid import ty
from tvm.ir import Op

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


@tvm.hybrid.script
def many_ewise(x: ty.handle, y: ty.handle) -> None:
    X = tir.match_buffer(x, (128, 128))
    Y = tir.match_buffer(y, (128, 128))
    A = tir.buffer_allocate((128, 128))
    with tir.block([128, 128], "A") as [vi, vj]:
        A[vi, vj] = X[vi, vj] * 2.0
    with tir.block([128, 128], "Y") as [vi, vj]:
        Y[vi, vj] = A[vi, vj] * 2.0


@tvm.hybrid.script
def split_ewise_multiple(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    for i in range(0, 16384):
        with tir.block([128, 128], "B") as [vi, vj]:
            tir.bind(vi, i // 128)
            tir.bind(vj, i % 128)
            B[vi, vj] = A[vi, vj] * 2.0
            C[vi, vj] = A[vi, vj] * 3.0


@tvm.hybrid.script
def apply_exp(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    for i in range(0, 16384):
        with tir.block([128, 128], "B") as [vi, vj]:
            tir.bind(vi, i // 128)
            tir.bind(vj, i % 128)
            B[vi, vj] = tir.exp(  # pylint: disable=unexpected-keyword-arg
                A[vi, vj],
                dtype="float32",
            )
            C[vi, vj] = tir.exp(  # pylint: disable=unexpected-keyword-arg
                A[vi, vj] * 2.0,
                dtype="float32",
            )


@tvm.hybrid.script
def with_predicate(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    for i, jo, ji in tir.grid(128, 10, 13):
        with tir.block([128, 128], "C") as [vi, vj]:
            tir.where(jo * 13 + ji < 128)
            tir.bind(vi, i)
            tir.bind(vj, jo * 13 + ji)
            C[vi, vj] = A[vi, vj] + 1.0


# pylint: enable=invalid-name,no-member


def test_meta_schedule_analysis_is_trivial_binding():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert ms.analysis.is_trivial_binding(sch.sch, sch.evaluate(block))
    sch = ms.Schedule(func=split_ewise)
    block = sch.get_block("B")
    assert not ms.analysis.is_trivial_binding(sch.sch, sch.evaluate(block))


def test_meta_schedule_analysis_get_block_var_types():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert ms.analysis.get_block_var_types(sch.sch, sch.evaluate(block)) == [
        "spatial",
        "spatial",
        "reduce",
    ]
    sch = ms.Schedule(func=split_ewise)
    block = sch.get_block("B")
    assert ms.analysis.get_block_var_types(sch.sch, sch.evaluate(block)) == [
        "spatial",
        "spatial",
    ]


def test_meta_schedule_analysis_is_spatial():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert not ms.analysis.is_spatial(sch.sch, sch.evaluate(block))
    sch = ms.Schedule(func=split_ewise)
    block = sch.get_block("B")
    assert ms.analysis.is_spatial(sch.sch, sch.evaluate(block))
    sch = ms.Schedule(func=many_ewise)
    block = sch.get_block("A")
    assert ms.analysis.is_spatial(sch.sch, sch.evaluate(block))
    block = sch.get_block("Y")
    assert ms.analysis.is_spatial(sch.sch, sch.evaluate(block))
    sch = ms.Schedule(func=split_ewise_multiple)
    block = sch.get_block("B")
    assert ms.analysis.is_spatial(sch.sch, sch.evaluate(block))


def test_meta_schedule_analysis_is_single_stmt_leaf():  # pylint: disable=invalid-name
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert ms.analysis.is_single_stmt_leaf(sch.sch, sch.evaluate(block))
    sch = ms.Schedule(func=split_ewise)
    block = sch.get_block("B")
    assert ms.analysis.is_single_stmt_leaf(sch.sch, sch.evaluate(block))
    sch = ms.Schedule(func=split_ewise_multiple)
    block = sch.get_block("B")
    assert not ms.analysis.is_single_stmt_leaf(sch.sch, sch.evaluate(block))


def test_meta_schedule_is_output_block():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert ms.analysis.is_output_block(sch.sch, sch.evaluate(block))
    sch = ms.Schedule(func=many_ewise)
    block = sch.get_block("A")
    assert not ms.analysis.is_output_block(sch.sch, sch.evaluate(block))


def test_meta_schedule_analysis_count_op():
    exp = Op.get("tir.exp")
    sch = ms.Schedule(func=apply_exp)
    block = sch.get_block("B")
    assert ms.analysis.count_op(sch.sch, sch.evaluate(block), exp) == 2
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert ms.analysis.count_op(sch.sch, sch.evaluate(block), exp) == 0


def test_meta_schedule_analysis_has_branch():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert not ms.analysis.has_branch(sch.sch, sch.evaluate(block))
    sch = ms.Schedule(func=with_predicate)
    block = sch.get_block("C")
    assert ms.analysis.has_branch(sch.sch, sch.evaluate(block))


def test_meta_schedule_is_elementwise_match():
    sch = ms.Schedule(func=many_ewise)
    block_a = sch.get_block("A")
    block_y = sch.get_block("Y")
    assert ms.analysis.is_elementwise_match(
        sch.sch,
        sch.evaluate(block_a),
        sch.evaluate(block_y),
    )


def test_meta_schedule_needs_multi_level_tiling():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert ms.analysis.needs_multi_level_tiling(sch.sch, sch.evaluate(block))
    sch = ms.Schedule(func=split_ewise)
    block = sch.get_block("B")
    assert not ms.analysis.needs_multi_level_tiling(sch.sch, sch.evaluate(block))


def test_meta_schedule_is_strictly_inlineable():
    sch = ms.Schedule(func=many_ewise)
    block = sch.get_block("A")
    assert ms.analysis.is_strictly_inlineable(sch.sch, sch.evaluate(block))
    sch = ms.Schedule(func=with_predicate)
    block = sch.get_block("C")
    assert not ms.analysis.is_strictly_inlineable(sch.sch, sch.evaluate(block))


if __name__ == "__main__":
    test_meta_schedule_analysis_is_trivial_binding()
    test_meta_schedule_analysis_get_block_var_types()
    test_meta_schedule_analysis_is_spatial()
    test_meta_schedule_analysis_is_single_stmt_leaf()
    test_meta_schedule_is_output_block()
    test_meta_schedule_analysis_count_op()
    test_meta_schedule_analysis_has_branch()
    test_meta_schedule_is_elementwise_match()
    test_meta_schedule_needs_multi_level_tiling()
    test_meta_schedule_is_strictly_inlineable()
