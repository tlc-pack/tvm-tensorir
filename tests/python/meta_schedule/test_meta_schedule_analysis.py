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
    assert ms.analysis.is_trivial_binding(sch, sch.get_block("C"))
    sch = ms.Schedule(func=split_ewise)
    assert not ms.analysis.is_trivial_binding(sch, sch.get_block("B"))


def test_meta_schedule_analysis_get_block_var_types():
    sch = ms.Schedule(func=matmul)
    assert ms.analysis.get_block_var_types(sch, sch.get_block("C")) == [
        "spatial",
        "spatial",
        "reduce",
    ]
    sch = ms.Schedule(func=split_ewise)
    assert ms.analysis.get_block_var_types(sch, sch.get_block("B")) == [
        "spatial",
        "spatial",
    ]


def test_meta_schedule_analysis_is_leaf_block():
    sch = ms.Schedule(func=matmul)
    assert ms.analysis.is_leaf_block(sch, sch.get_block("C"))


def test_meta_schedule_analysis_is_leaf_block_with_single_stmt():  # pylint: disable=invalid-name
    sch = ms.Schedule(func=matmul)
    assert ms.analysis.is_leaf_block_with_single_stmt(sch, sch.get_block("C"))
    sch = ms.Schedule(func=split_ewise)
    assert ms.analysis.is_leaf_block_with_single_stmt(sch, sch.get_block("B"))
    sch = ms.Schedule(func=split_ewise_multiple)
    assert not ms.analysis.is_leaf_block_with_single_stmt(sch, sch.get_block("B"))


def test_meta_schedule_analysis_get_buffer_store():
    sch = ms.Schedule(func=matmul)
    store = ms.analysis.get_buffer_store(sch, sch.get_block("C"))
    assert store.buffer.name == "C"
    sch = ms.Schedule(func=split_ewise)
    store = ms.analysis.get_buffer_store(sch, sch.get_block("B"))
    assert store.buffer.name == "B"


def test_meta_schedule_analysis_get_buffer_load():
    sch = ms.Schedule(func=matmul)
    load_a, load_b = ms.analysis.get_buffer_load(sch, sch.get_block("C"))
    assert {load_a.buffer.name, load_b.buffer.name} == {"A", "B"}
    sch = ms.Schedule(func=split_ewise)
    (load_a,) = ms.analysis.get_buffer_load(sch, sch.get_block("B"))
    assert load_a.buffer.name == "A"


def test_meta_schedule_analysis_count_op():
    exp = Op.get("tir.exp")
    sch = ms.Schedule(func=apply_exp)
    block = sch.get_block("B")
    assert ms.analysis.count_op(sch, block, exp) == 2
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert ms.analysis.count_op(sch, block, exp) == 0


def test_meta_schedule_analysis_has_branch():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert not ms.analysis.has_branch(sch, block)
    sch = ms.Schedule(func=with_predicate)
    block = sch.get_block("C")
    assert ms.analysis.has_branch(sch, block)


def test_meta_schedule_analysis_block_vars_used_in_store():  # pylint: disable=invalid-name
    sch = ms.Schedule(func=matmul)
    v_i, v_j = ms.analysis.block_vars_used_in_store(sch, sch.get_block("C"))
    assert v_i.name == "vi"
    assert v_j.name == "vj"
    sch = ms.Schedule(func=split_ewise)
    v_i, v_j = ms.analysis.block_vars_used_in_store(sch, sch.get_block("B"))
    assert v_i.name == "vi"
    assert v_j.name == "vj"


def test_meta_schedule_analysis_count_missing_block_vars():  # pylint: disable=invalid-name
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    load_a, load_b = ms.analysis.get_buffer_load(sch, block)
    store_c = ms.analysis.get_buffer_store(sch, block)
    block_vars = ms.analysis.block_vars_used_in_store(sch, block)
    assert ms.analysis.count_missing_block_vars(load_a, block_vars) == 1
    assert ms.analysis.count_missing_block_vars(load_b, block_vars) == 1
    assert ms.analysis.count_missing_block_vars(store_c, block_vars) == 0


def test_meta_schedule_analysis_inspect_load_indices():  # pylint: disable=invalid-name
    sch = ms.Schedule(func=matmul)
    result = ms.analysis.inspect_load_indices(sch, sch.get_block("C"))
    assert result is None
    sch = ms.Schedule(func=split_ewise)
    result = ms.analysis.inspect_load_indices(sch, sch.get_block("B"))
    assert result == (True, True, True)


def test_meta_schedule_analysis_has_reduce_block_var():
    sch = ms.Schedule(func=matmul)
    result = ms.analysis.has_reduce_block_var(sch, sch.get_block("C"))
    assert result
    sch = ms.Schedule(func=split_ewise)
    result = ms.analysis.has_reduce_block_var(sch, sch.get_block("B"))
    assert not result


def test_meta_schedule_needs_multi_level_tiling():
    sch = ms.Schedule(func=matmul)
    result = ms.analysis.needs_multi_level_tiling(sch, sch.get_block("C"))
    assert result
    sch = ms.Schedule(func=split_ewise)
    result = ms.analysis.needs_multi_level_tiling(sch, sch.get_block("B"))
    assert not result


def test_meta_schedule_do_multi_level_tiling():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    ms.analysis.do_multi_level_tiling(sch, block, "SSRSRS")
    assert len(sch.get_axes(block)) == 10


def test_meta_schedule_is_elementwise_match():
    sch = ms.Schedule(func=many_ewise)
    block_a = sch.get_block("A")
    block_y = sch.get_block("Y")
    assert ms.analysis.is_elementwise_match(sch, block_a, block_y)


@pytest.mark.xfail()
def test_meta_schedule_is_output_block():
    # TODO(@junrushao1994): need fix
    sch = ms.Schedule(func=matmul)
    print(tvm.hybrid.ashybrid(sch.sch.func))
    assert ms.analysis.is_output_block(sch, sch.get_block("C"))


if __name__ == "__main__":
    test_meta_schedule_analysis_is_trivial_binding()
    test_meta_schedule_analysis_get_block_var_types()
    test_meta_schedule_analysis_is_leaf_block()
    test_meta_schedule_analysis_is_leaf_block_with_single_stmt()
    test_meta_schedule_analysis_get_buffer_store()
    test_meta_schedule_analysis_get_buffer_load()
    test_meta_schedule_analysis_count_op()
    test_meta_schedule_analysis_has_branch()
    test_meta_schedule_analysis_block_vars_used_in_store()
    test_meta_schedule_analysis_count_missing_block_vars()
    test_meta_schedule_analysis_inspect_load_indices()
    test_meta_schedule_analysis_has_reduce_block_var()
    test_meta_schedule_needs_multi_level_tiling()
    test_meta_schedule_do_multi_level_tiling()
    test_meta_schedule_is_elementwise_match()
    test_meta_schedule_is_output_block()
