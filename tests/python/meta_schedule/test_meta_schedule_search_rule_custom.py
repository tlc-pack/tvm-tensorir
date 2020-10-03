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
""" Test meta schedule SearchRule """
# pylint: disable=missing-function-docstring

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.hybrid import ty
from tvm.meta_schedule.search import RulePackedArgs

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


@ms.register_rule("do_nothing")
def do_nothing(sch: ms.Schedule, _block: ms.BlockRV):
    return sch


@ms.register_rule("do_mlt")
def do_mlt(sch: ms.Schedule, block: ms.BlockRV):
    TILING_FORMAT = "SSRSRS"  # pylint: disable=invalid-name
    spatial_indices = [i for i, c in enumerate(TILING_FORMAT) if c == "S"]
    reduce_indices = [i for i, c in enumerate(TILING_FORMAT) if c == "R"]
    order = [list() for _ in TILING_FORMAT]
    axes = sch.get_axes(block=block)
    iter_types = ms.analysis.get_block_var_types(sch, block)
    assert len(axes) == len(iter_types)
    for axis, iter_type in zip(axes, iter_types):
        for expect_iter_type, indices in [
            ("spatial", spatial_indices),
            ("reduce", reduce_indices),
        ]:
            if iter_type == expect_iter_type:
                tiles = sch.sample_perfect_tile(n=len(indices), loop=axis)
                splits = sch.split(loop=axis, factors=tiles)
                for i, split in zip(indices, splits):
                    order[i].append(split)
    sch.reorder(after_axes=sum(order, []))
    return sch


def test_meta_schedule_rule_do_nothing():
    sch = ms.Schedule(func=matmul)
    args: RulePackedArgs = do_nothing(sch, sch.get_block("C"))
    assert not args.skipped
    assert len(args.proceed) == 1
    assert sch.same_as(args.proceed[0])


def test_meta_schedule_rule_do_mlt():
    sch = ms.Schedule(func=matmul)
    args: RulePackedArgs = do_mlt(sch, sch.get_block("C"))
    assert not args.skipped
    assert len(args.proceed) == 1
    sch: tir.Schedule = args.proceed[0].sch
    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        i.stmt.extent for i in sch.get_axes(sch.get_block("C"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


def test_meta_schedule_rule_composite_0():
    rule = ms.SearchRule.compose(
        name="composed",
        rules=[
            do_nothing,
            do_mlt,
        ],
    )
    sch = ms.Schedule(func=matmul)
    args: RulePackedArgs = rule(sch, sch.get_block("C"))
    assert not args.skipped
    assert len(args.proceed) == 1
    sch: tir.Schedule = args.proceed[0].sch
    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        i.stmt.extent for i in sch.get_axes(sch.get_block("C"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


def test_meta_schedule_rule_composite_1():
    rule = ms.SearchRule.compose(
        name="composed",
        rules=[
            do_mlt,
            do_nothing,
        ],
    )
    sch = ms.Schedule(func=matmul)
    args: RulePackedArgs = rule(sch, sch.get_block("C"))
    assert not args.skipped
    assert len(args.proceed) == 1
    sch: tir.Schedule = args.proceed[0].sch
    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        i.stmt.extent for i in sch.get_axes(sch.get_block("C"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


if __name__ == "__main__":
    test_meta_schedule_rule_do_nothing()
    test_meta_schedule_rule_do_mlt()
    test_meta_schedule_rule_composite_0()
    test_meta_schedule_rule_composite_1()
