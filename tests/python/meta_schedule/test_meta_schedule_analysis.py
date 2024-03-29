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
import tir_tensor_intrin
import tvm
from tir_workload import batch_matmul, matmul, matmul_relu
from tvm import meta_schedule as ms
from tvm import tir
from tvm.ir import Op
from tvm.script import ty

# pylint: disable=invalid-name,no-member


@tvm.script.tir
def split_ewise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))

    for i in range(0, 16384):
        with tir.block([128, 128], "B") as [vi, vj]:
            tir.bind(vi, i // 128)
            tir.bind(vj, i % 128)
            B[vi, vj] = A[vi, vj] * 2.0


@tvm.script.tir
def many_ewise(x: ty.handle, y: ty.handle) -> None:
    X = tir.match_buffer(x, (128, 128))
    Y = tir.match_buffer(y, (128, 128))
    A = tir.alloc_buffer((128, 128))
    with tir.block([128, 128], "A") as [vi, vj]:
        A[vi, vj] = X[vi, vj] * 2.0
    with tir.block([128, 128], "Y") as [vi, vj]:
        Y[vi, vj] = A[vi, vj] * 2.0


@tvm.script.tir
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


@tvm.script.tir
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


@tvm.script.tir
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
    sch = tir.Schedule(mod=matmul)
    block = sch.get_block("matmul")
    assert ms.analysis.is_trivial_binding(sch.state, sch.get_sref(block))
    sch = tir.Schedule(mod=split_ewise)
    block = sch.get_block("B")
    assert not ms.analysis.is_trivial_binding(sch.state, sch.get_sref(block))


def test_meta_schedule_analysis_is_subroot_block():
    sch = tir.Schedule(mod=matmul)
    block = sch.get_block("matmul")
    assert ms.analysis.is_subroot_block(sch.state, sch.get_sref(block))


def test_meta_schedule_analysis_is_leaf_block():
    sch = tir.Schedule(mod=matmul)
    block = sch.get_block("matmul")
    assert ms.analysis.is_leaf_block(sch.state, sch.get_sref(block))


def test_meta_schedule_analysis_get_loop_iter_type():
    sch = tir.Schedule(mod=matmul)
    block = sch.get_block("matmul")
    i, j, k = [sch.get_sref(loop) for loop in sch.get_loops(block)]
    i = ms.analysis.get_loop_iter_type(sch.state, i)
    j = ms.analysis.get_loop_iter_type(sch.state, j)
    k = ms.analysis.get_loop_iter_type(sch.state, k)
    assert i == 0
    assert j == 0
    assert k == 2


def test_meta_schedule_analysis_get_block_var_types():
    sch = tir.Schedule(mod=matmul)
    block = sch.get_block("matmul")
    assert ms.analysis.get_block_var_types(sch.state, sch.get_sref(block)) == [
        "spatial",
        "spatial",
        "reduce",
    ]
    sch = tir.Schedule(mod=split_ewise)
    block = sch.get_block("B")
    assert ms.analysis.get_block_var_types(sch.state, sch.get_sref(block)) == [
        "spatial",
        "spatial",
    ]


def test_meta_schedule_analysis_is_spatial():
    sch = tir.Schedule(mod=matmul)
    block = sch.get_block("matmul")
    assert not ms.analysis.is_spatial(sch.state, sch.get_sref(block))
    sch = tir.Schedule(mod=split_ewise)
    block = sch.get_block("B")
    assert ms.analysis.is_spatial(sch.state, sch.get_sref(block))
    sch = tir.Schedule(mod=many_ewise)
    block = sch.get_block("A")
    assert ms.analysis.is_spatial(sch.state, sch.get_sref(block))
    block = sch.get_block("Y")
    assert ms.analysis.is_spatial(sch.state, sch.get_sref(block))
    sch = tir.Schedule(mod=split_ewise_multiple)
    block = sch.get_block("B")
    assert ms.analysis.is_spatial(sch.state, sch.get_sref(block))


def test_meta_schedule_is_output_block():
    sch = tir.Schedule(mod=matmul)
    block = sch.get_block("matmul")
    assert ms.analysis.is_output_block(sch.state, sch.get_sref(block))
    sch = tir.Schedule(mod=many_ewise)
    block = sch.get_block("A")
    assert not ms.analysis.is_output_block(sch.state, sch.get_sref(block))


def test_meta_schedule_analysis_count_op():
    exp = Op.get("tir.exp")
    sch = tir.Schedule(mod=apply_exp)
    block = sch.get_block("B")
    assert ms.analysis.count_op(sch.state, sch.get_sref(block), exp) == 2
    sch = tir.Schedule(mod=matmul)
    block = sch.get_block("matmul")
    assert ms.analysis.count_op(sch.state, sch.get_sref(block), exp) == 0


def test_meta_schedule_analysis_has_branch():
    sch = tir.Schedule(mod=matmul)
    block = sch.get_block("matmul")
    assert not ms.analysis.has_branch(sch.state, sch.get_sref(block))
    sch = tir.Schedule(mod=with_predicate)
    block = sch.get_block("C")
    assert ms.analysis.has_branch(sch.state, sch.get_sref(block))


def test_meta_schedule_is_elementwise_match():
    sch = tir.Schedule(mod=many_ewise)
    block_a = sch.get_block("A")
    block_y = sch.get_block("Y")
    assert ms.analysis.is_elementwise_match(
        sch.state,
        sch.get_sref(block_a),
        sch.get_sref(block_y),
    )


def test_meta_schedule_needs_multi_level_tiling():
    sch = tir.Schedule(mod=matmul)
    block = sch.get_block("matmul")
    assert ms.analysis.needs_multi_level_tiling(sch.state, sch.get_sref(block))
    sch = tir.Schedule(mod=split_ewise)
    block = sch.get_block("B")
    assert not ms.analysis.needs_multi_level_tiling(sch.state, sch.get_sref(block))


def test_meta_schedule_is_strictly_inlineable():
    sch = tir.Schedule(mod=many_ewise)
    block = sch.get_block("A")
    assert ms.analysis.is_strictly_inlineable(sch.state, sch.get_sref(block))
    sch = tir.Schedule(mod=with_predicate)
    block = sch.get_block("C")
    assert not ms.analysis.is_strictly_inlineable(sch.state, sch.get_sref(block))


def test_meta_schedule_get_tensorize_loop_mapping():
    sch = tir.Schedule(batch_matmul)
    block = sch.get_block(name="update")
    assert (
        ms.analysis.get_tensorize_loop_mapping(
            sch.state, sch.get_sref(block), tir_tensor_intrin.tensorcore_desc
        )
        is not None
    )
    sch = tir.Schedule(batch_matmul)
    block = sch.get_block(name="update")
    assert (
        ms.analysis.get_tensorize_loop_mapping(
            sch.state, sch.get_sref(block), tir_tensor_intrin.dot_product_desc
        )
        is not None
    )


def test_meta_schedule_analysis_count_flop():
    result = ms.analysis.count_flop(matmul)
    expected = 2 * 1024 ** 3
    assert abs(result - expected) < 0.5
    result = ms.analysis.count_flop(matmul_relu)
    expected = 2 * 1024 ** 3 + 1024 ** 2
    assert abs(result - expected) < 0.5


if __name__ == "__main__":
    test_meta_schedule_analysis_is_trivial_binding()
    test_meta_schedule_analysis_is_subroot_block()
    test_meta_schedule_analysis_is_leaf_block()
    test_meta_schedule_analysis_get_loop_iter_type()
    test_meta_schedule_analysis_get_block_var_types()
    test_meta_schedule_analysis_is_spatial()
    test_meta_schedule_is_output_block()
    test_meta_schedule_analysis_count_op()
    test_meta_schedule_analysis_has_branch()
    test_meta_schedule_is_elementwise_match()
    test_meta_schedule_needs_multi_level_tiling()
    test_meta_schedule_is_strictly_inlineable()
    test_meta_schedule_get_tensorize_loop_mapping()
    test_meta_schedule_analysis_count_flop()
