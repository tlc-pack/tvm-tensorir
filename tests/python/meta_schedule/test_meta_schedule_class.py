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
""" Test for meta schedule class """
import pytest

# pylint: disable=missing-function-docstring
import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import ty

# pylint: disable=invalid-name,no-member,unused-variable


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "C") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])


@tvm.script.tir
def matmul_relu(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    D = tir.match_buffer(d, (1024, 1024), "float32")
    C = tir.buffer_allocate((1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])
    with tir.block([1024, 1024], "relu") as [vi, vj]:
        D[vi, vj] = tir.max(C[vi, vj], 0.0)


@tvm.script.tir
def matmul_relu_fused(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    D = tir.match_buffer(d, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C = tir.buffer_allocate([1024, 1024], elem_offset=0, align=128, offset_factor=1)
        for i0 in range(0, 1024):
            for i1 in range(0, 1024):
                for i2 in range(0, 1024):
                    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [
                        vi,
                        vj,
                        vk,
                    ]:
                        tir.bind(vi, i0)
                        tir.bind(vj, i1)
                        tir.bind(vk, i2)
                        tir.reads(
                            [
                                C[vi : (vi + 1), vj : (vj + 1)],
                                A[vi : (vi + 1), vk : (vk + 1)],
                                B[vk : (vk + 1), vj : (vj + 1)],
                            ]
                        )
                        tir.writes([C[vi : (vi + 1), vj : (vj + 1)]])
                        reducer.step(C[vi, vj], (A[vi, vk] * B[vk, vj]))
                for ax0 in range(0, 1):
                    for ax1 in range(0, 1):
                        with tir.block([1024, 1024], "relu") as [vi_1, vj_1]:
                            tir.bind(vi_1, i0)
                            tir.bind(vj_1, i1)
                            tir.reads([C[vi_1 : (vi_1 + 1), vj_1 : (vj_1 + 1)]])
                            tir.writes([D[vi_1 : (vi_1 + 1), vj_1 : (vj_1 + 1)]])
                            D[vi_1, vj_1] = tir.max(C[vi_1, vj_1], tir.float32(0))


@tvm.script.tir
def matmul_cache_write(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    C = tir.match_buffer(c, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [1024, 1024], elem_offset=0, align=128, offset_factor=1)
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    # body
    with tir.block([], "root") as []:
        tir.reads([])
        tir.writes([])
        C_local = tir.buffer_allocate(
            [1024, 1024], elem_offset=0, scope="local", align=128, offset_factor=1
        )
        for i0 in range(0, 1024):
            for i1 in range(0, 1024):
                for i2 in range(0, 1024):
                    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "C") as [
                        vi,
                        vj,
                        vk,
                    ]:
                        tir.bind(vi, i0)
                        tir.bind(vj, i1)
                        tir.bind(vk, i2)
                        tir.reads(
                            [
                                C_local[vi : (vi + 1), vj : (vj + 1)],
                                A[vi : (vi + 1), vk : (vk + 1)],
                                B[vk : (vk + 1), vj : (vj + 1)],
                            ]
                        )
                        tir.writes([C_local[vi : (vi + 1), vj : (vj + 1)]])
                        reducer.step(C_local[vi, vj], (A[vi, vk] * B[vk, vj]))
        for ax0 in range(0, 1024):
            for ax1 in range(0, 1024):
                with tir.block([1024, 1024], "") as [v0, v1]:
                    tir.bind(v0, ax0)
                    tir.bind(v1, ax1)
                    tir.reads([C_local[v0 : (v0 + 1), v1 : (v1 + 1)]])
                    tir.writes([C[v0 : (v0 + 1), v1 : (v1 + 1)]])
                    C[v0, v1] = C_local[v0, v1]


@tvm.script.tir
def elementwise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.buffer_allocate((1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    with tir.block([1024, 1024], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] + 1.0
    with tir.block([1024, 1024], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] * 2.0


@tvm.script.tir
def elementwise_inlined(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    with tir.block([1024, 1024], "C") as [vi, vj]:
        C[vi, vj] = (A[vi, vj] + 1.0) * 2.0


# pylint: enable=invalid-name,no-member,unused-variable


def test_meta_schedule_creation():
    sch = ms.Schedule(func=matmul)
    assert tvm.ir.structural_equal(sch.orig_func, sch.sch.func)
    assert len(sch.trace) == 0


def test_meta_schedule_copy():
    sch = ms.Schedule(func=matmul)
    i, j, k = sch.get_axes(sch.get_block("C"))
    sch_copy = sch.copy(seed=42)
    assert not sch.evaluate(i).same_as(sch_copy.evaluate(i))
    assert not sch.evaluate(j).same_as(sch_copy.evaluate(j))
    assert not sch.evaluate(k).same_as(sch_copy.evaluate(k))
    assert sch.evaluate(i).stmt.same_as(sch_copy.evaluate(i).stmt)
    assert sch.evaluate(j).stmt.same_as(sch_copy.evaluate(j).stmt)
    assert sch.evaluate(k).stmt.same_as(sch_copy.evaluate(k).stmt)
    i_0, i_1 = sch.split(i, [2, 512])
    j_0, j_1 = sch_copy.split(j, [4, 256])

    assert sch.evaluate(i_0).stmt.extent == 2
    assert sch.evaluate(i_1).stmt.extent == 512
    with pytest.raises(IndexError):
        sch_copy.evaluate(i_0)
    with pytest.raises(IndexError):
        sch_copy.evaluate(i_1)

    with pytest.raises(IndexError):
        sch.evaluate(j_0)
    with pytest.raises(IndexError):
        sch.evaluate(j_1)
    assert sch_copy.evaluate(j_0).stmt.extent == 4
    assert sch_copy.evaluate(j_1).stmt.extent == 256


def test_meta_schedule_sample_tile_factor():
    from functools import reduce  # pylint: disable=import-outside-toplevel

    where = [1, 2, 4]
    sch = ms.Schedule(func=matmul)
    i, _, _ = sch.get_axes(sch.get_block("C"))
    factors = sch.sample_tile_factor(n_splits=4, loop=i, where=where)
    factors = [sch.evaluate(i) for i in factors]
    for i in factors[1:]:
        assert i in where
    prod = reduce(lambda x, y: x * y, factors)
    assert prod == 1024


def test_meta_schedule_sample_perfect_tile():
    from functools import reduce  # pylint: disable=import-outside-toplevel

    sch = ms.Schedule(func=matmul)
    i, _, _ = sch.get_axes(sch.get_block("C"))
    factors = sch.sample_perfect_tile(n_splits=4, loop=i)
    factors = [sch.evaluate(i) for i in factors]
    prod = reduce(lambda x, y: x * y, factors)
    assert prod == 1024


def test_meta_schedule_sample_fusible_loops():
    sch = ms.Schedule(func=matmul)
    loops = sch.get_axes(block=sch.get_block(name="C"))
    result = sch.sample_fusible_loops(
        loops=loops,
        loop_types=[0, 0, 0],
        max_extent=1024,
        include_overflow_loop=True,
        order="outer_to_inner",
    )
    assert sch.evaluate(result) == 2
    result = sch.sample_fusible_loops(
        loops=loops,
        loop_types=[0, 0, 0],
        max_extent=1023,
        include_overflow_loop=True,
        order="outer_to_inner",
    )
    assert sch.evaluate(result) == 1
    result = sch.sample_fusible_loops(
        loops=loops,
        loop_types=[0, 0, 0],
        max_extent=1024,
        include_overflow_loop=True,
        order="inner_to_outer",
    )
    assert sch.evaluate(result) == 2
    result = sch.sample_fusible_loops(
        loops=loops,
        loop_types=[0, 0, 0],
        max_extent=1023,
        include_overflow_loop=True,
        order="inner_to_outer",
    )
    assert sch.evaluate(result) == 1


def test_meta_schedule_get_only_consumer():
    sch = ms.Schedule(func=matmul_relu)
    block = sch.get_block("matmul")
    consumer = sch.get_only_consumer(block)
    assert tvm.ir.structural_equal(
        sch.evaluate(consumer).stmt,
        sch.evaluate(sch.get_block("relu")).stmt,
    )


def test_meta_schedule_get_block():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    assert tvm.ir.structural_equal(
        sch.evaluate(block).stmt,
        matmul.body.block.body.body.body.body.block,
    )


def test_meta_schedule_get_axes():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block("C")
    axes = sch.get_axes(block)
    i_0, i_1, i_2 = [sch.evaluate(i).stmt for i in axes]
    assert tvm.ir.structural_equal(i_0, matmul.body.block.body)
    assert tvm.ir.structural_equal(i_1, matmul.body.block.body.body)
    assert tvm.ir.structural_equal(i_2, matmul.body.block.body.body.body)


def test_meta_schedule_get_root_blocks():
    sch = ms.Schedule(func=matmul)
    blocks = sch.get_root_blocks()
    assert len(blocks) == 1
    sch = ms.Schedule(func=matmul_relu)
    blocks = sch.get_root_blocks()
    assert len(blocks) == 2


def test_meta_schedule_get_leaf_blocks():
    sch = ms.Schedule(func=matmul)
    blocks = sch.get_leaf_blocks()
    assert len(blocks) == 1
    sch = ms.Schedule(func=matmul_relu)
    blocks = sch.get_leaf_blocks()
    assert len(blocks) == 2


def test_meta_schedule_fuse():
    sch = ms.Schedule(func=matmul)
    block = sch.get_block(name="C")
    i, j, _ = sch.get_axes(block)
    sch.fuse(loops=[i, j])
    assert len(sch.get_axes(block)) == 2


def test_meta_schedule_mark_parallel():
    def check_annotation(sch, loop):
        loop = sch.evaluate(loop).stmt
        assert len(loop.annotations) == 1
        (ann,) = loop.annotations
        assert ann.attr_key == "loop_type"
        assert ann.value == "lazy_parallel"

    sch = ms.Schedule(func=matmul)
    block = sch.get_block(name="C")
    axes = sch.get_axes(block)
    sch.mark_parallel(axes, tvm.ir.Range(0, 3))
    block = sch.get_block(name="C")
    i, j, k = sch.get_axes(block)
    check_annotation(sch, i)
    check_annotation(sch, j)
    check_annotation(sch, k)


def test_meta_schedule_mark_vectorize():
    def check_annotation(sch, loop):
        loop = sch.evaluate(loop).stmt
        assert len(loop.annotations) == 1
        (ann,) = loop.annotations
        assert ann.attr_key == "loop_type"
        assert ann.value == "lazy_vectorize"

    sch = ms.Schedule(func=matmul)
    block = sch.get_block(name="C")
    axes = sch.get_axes(block)
    sch.mark_vectorize(axes, tvm.ir.Range(0, 3))
    block = sch.get_block(name="C")
    i, j, k = sch.get_axes(block)
    check_annotation(sch, i)
    check_annotation(sch, j)
    check_annotation(sch, k)


def test_meta_schedule_split():
    sch = ms.Schedule(func=matmul)
    i, _, _ = sch.get_axes(sch.get_block("C"))
    i_0, i_1, i_2 = [sch.evaluate(i).stmt for i in sch.split(loop=i, factors=[4, 8, 32])]
    assert tvm.ir.structural_equal(i_0, sch.sch.func.body.block.body)
    assert tvm.ir.structural_equal(i_1, sch.sch.func.body.block.body.body)
    assert tvm.ir.structural_equal(i_2, sch.sch.func.body.block.body.body.body)


def test_meta_schedule_reorder():
    sch = ms.Schedule(func=matmul)
    i_0, i_1, i_2 = sch.get_axes(sch.get_block("C"))
    sch.reorder(after_axes=[i_2, i_1, i_0])
    i_0, i_1, i_2 = [sch.evaluate(i).stmt for i in [i_0, i_1, i_2]]

    tir_sch = tir.create_schedule(func=matmul)
    ti_0, ti_1, ti_2 = tir_sch.get_axes(tir_sch.get_block("C"))
    tir_sch.reorder(ti_2, ti_1, ti_0)

    assert tvm.ir.structural_equal(i_0, ti_0.stmt)
    assert tvm.ir.structural_equal(i_1, ti_1.stmt)
    assert tvm.ir.structural_equal(i_2, ti_2.stmt)


def test_meta_schedule_reverse_compute_at():
    sch = ms.Schedule(func=matmul_relu)
    relu_block = sch.get_block("relu")
    matmul_block = sch.get_block("matmul")
    _, i_1, _ = sch.get_axes(matmul_block)
    sch.reverse_compute_at(relu_block, i_1)
    assert tvm.ir.structural_equal(sch.sch.func, matmul_relu_fused)


def test_meta_schedule_compute_inline():
    sch = ms.Schedule(func=elementwise)
    block = sch.get_block(name="B")
    sch.compute_inline(block=block)
    assert tvm.ir.structural_equal(sch.sch.func, elementwise_inlined)


def test_meta_schedule_cache_write():
    sch = ms.Schedule(func=matmul)
    sch.cache_write(sch.get_block("C"), storage_scope="local")
    assert tvm.ir.structural_equal(sch.sch.func, matmul_cache_write)


def test_meta_schedule_mutate_decision():
    sch = ms.Schedule(func=matmul)
    i, j, _ = sch.get_axes(sch.get_block("C"))
    sch.sample_perfect_tile(n_splits=4, loop=i)
    sch.sample_perfect_tile(n_splits=3, loop=j)
    i_inst = sch.trace[-2]
    j_inst = sch.trace[-1]
    sch.mutate_decision(i_inst, [1, 1, 1, 1024])
    assert list(sch.decisions[i_inst]) == [1, 1, 1, 1024]
    sch.mutate_decision(j_inst, None)
    assert not j_inst in sch.decisions


def test_meta_schedule_resample():
    from functools import reduce  # pylint: disable=import-outside-toplevel

    sch = ms.Schedule(func=matmul)
    i, j, _ = sch.get_axes(sch.get_block("C"))
    i_tiles = sch.sample_perfect_tile(n_splits=4, loop=i)
    j_tiles = sch.sample_perfect_tile(n_splits=3, loop=j)
    i_inst = sch.trace[-2]
    j_inst = sch.trace[-1]
    for _ in range(10):
        sch.resample()
        i_eval = [sch.evaluate(i) for i in i_tiles]
        j_eval = [sch.evaluate(j) for j in j_tiles]
        i_dec = list(sch.decisions[i_inst])
        j_dec = list(sch.decisions[j_inst])
        assert i_eval == i_dec
        assert j_eval == j_dec
        assert reduce(lambda x, y: x * y, i_eval) == 1024
        assert reduce(lambda x, y: x * y, j_eval) == 1024


def test_meta_schedule_replay_decision():
    from collections import defaultdict  # pylint: disable=import-outside-toplevel

    sch = ms.Schedule(func=matmul)
    i, j, _ = sch.get_axes(sch.get_block("C"))
    i_tiles = sch.sample_perfect_tile(n_splits=4, loop=i)
    j_tiles = sch.sample_perfect_tile(n_splits=3, loop=j)
    i_inst = sch.trace[-2]
    j_inst = sch.trace[-1]
    sch.split(loop=i, factors=i_tiles)
    sch.split(loop=j, factors=j_tiles)
    i_cnt = defaultdict(int)
    j_cnt = defaultdict(int)
    for _ in range(100):
        # set i to deterministic
        sch.mutate_decision(i_inst, [1, 1, 1, 1024])
        # set i to random
        sch.mutate_decision(j_inst, None)
        # go!
        sch.replay_decision()
        i_eval = [sch.evaluate(i) for i in i_tiles]
        j_eval = [sch.evaluate(j) for j in j_tiles]
        i_cnt[str(i_eval)] += 1
        j_cnt[str(j_eval)] += 1
    assert len(i_cnt) == 1
    assert len(j_cnt) > 1


if __name__ == "__main__":
    test_meta_schedule_creation()
    test_meta_schedule_copy()
    test_meta_schedule_sample_tile_factor()
    test_meta_schedule_sample_perfect_tile()
    test_meta_schedule_sample_fusible_loops()
    test_meta_schedule_get_only_consumer()
    test_meta_schedule_get_block()
    test_meta_schedule_get_axes()
    test_meta_schedule_get_root_blocks()
    test_meta_schedule_get_leaf_blocks()
    test_meta_schedule_fuse()
    test_meta_schedule_mark_parallel()
    test_meta_schedule_mark_vectorize()
    test_meta_schedule_split()
    test_meta_schedule_reorder()
    test_meta_schedule_reverse_compute_at()
    test_meta_schedule_compute_inline()
    test_meta_schedule_cache_write()
    # test_meta_schedule_decompose_reduction()
    test_meta_schedule_mutate_decision()
    test_meta_schedule_resample()
    test_meta_schedule_replay_decision()
