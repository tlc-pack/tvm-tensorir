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
# pylint: disable=missing-function-docstring
import tvm
from tvm.hybrid import ty
from tvm import tir
from tvm import meta_schedule as ms


# pylint: disable=invalid-name,no-member


@tvm.hybrid.script
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "C") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])


# pylint: enable=invalid-name,no-member


def _print_prim_func(prim_func):
    print(tvm.hybrid.ashybrid(prim_func))


def test_meta_schedule_creation():
    sch = ms.Schedule(func=matmul)
    assert tvm.ir.structural_equal(sch.orig_func, sch.sch.func)
    assert len(sch.trace) == 0


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


def test_meta_schedule_split():
    sch = ms.Schedule(func=matmul)
    i, _, _ = sch.get_axes(sch.get_block("C"))
    i_0, i_1, i_2 = [
        sch.evaluate(i).stmt for i in sch.split(loop=i, factors=[4, 8, 32])
    ]
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


def test_meta_schedule_sample_tile_factor():
    from functools import reduce  # pylint: disable=import-outside-toplevel

    where = [1, 2, 4]
    sch = ms.Schedule(func=matmul)
    i, _, _ = sch.get_axes(sch.get_block("C"))
    factors = sch.sample_tile_factor(n=4, loop=i, where=where)
    factors = [sch.evaluate(i) for i in factors]
    for i in factors[1:]:
        assert i in where
    prod = reduce(lambda x, y: x * y, factors)
    assert prod == 1024


if __name__ == "__main__":
    test_meta_schedule_creation()
    test_meta_schedule_get_block()
    test_meta_schedule_get_axes()
    test_meta_schedule_split()
    test_meta_schedule_reorder()
    test_meta_schedule_sample_tile_factor()
