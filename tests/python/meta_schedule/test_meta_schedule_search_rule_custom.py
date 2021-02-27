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

from tir_workload import matmul
from tvm import meta_schedule as ms
from tvm import tir


@ms.rule.register_rule("do_nothing")
def do_nothing(_task: ms.SearchTask, sch: ms.Schedule, _block: ms.BlockRV):
    return sch


@ms.rule.register_rule("do_mlt")
def do_mlt(_task: ms.SearchTask, sch: ms.Schedule, block: ms.BlockRV):
    TILING_FORMAT = "SSRSRS"  # pylint: disable=invalid-name
    spatial_indices = [i for i, c in enumerate(TILING_FORMAT) if c == "S"]
    reduce_indices = [i for i, c in enumerate(TILING_FORMAT) if c == "R"]
    order = [list() for _ in TILING_FORMAT]
    axes = sch.get_axes(block=block)
    iter_types = ms.analysis.get_block_var_types(sch.sch.state, sch.evaluate(block))
    assert len(axes) == len(iter_types)
    for axis, iter_type in zip(axes, iter_types):
        for expect_iter_type, indices in [
            ("spatial", spatial_indices),
            ("reduce", reduce_indices),
        ]:
            if iter_type == expect_iter_type:
                tiles = sch.sample_perfect_tile(n_splits=len(indices), loop=axis)
                splits = sch.split(loop=axis, factors=tiles)
                for i, split in zip(indices, splits):
                    order[i].append(split)
    sch.reorder(after_axes=sum(order, []))
    return sch


def test_meta_schedule_rule_do_nothing():
    task = ms.SearchTask(workload=matmul)
    sch = ms.Schedule(func=matmul)
    args = do_nothing(task, sch, sch.get_block("matmul"))
    assert len(args) == 1
    assert sch.same_as(args[0])


def test_meta_schedule_rule_do_mlt():
    task = ms.SearchTask(workload=matmul)
    sch = ms.Schedule(func=matmul)
    args = do_mlt(task, sch, sch.get_block("matmul"))
    assert len(args) == 1
    sch = args[0].sch
    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        i.stmt.extent for i in sch.get_axes(sch.get_block("matmul"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


def test_meta_schedule_rule_composite_0():
    rule = ms.rule.compose(
        name="composed",
        rules=[
            do_nothing,
            do_mlt,
        ],
    )
    task = ms.SearchTask(workload=matmul)
    sch = ms.Schedule(func=matmul)
    args = rule(task, sch, sch.get_block("matmul"))
    assert len(args) == 1
    sch: tir.Schedule = args[0].sch
    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        i.stmt.extent for i in sch.get_axes(sch.get_block("matmul"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


def test_meta_schedule_rule_composite_1():
    rule = ms.rule.compose(
        name="composed",
        rules=[
            do_mlt,
            do_nothing,
        ],
    )
    task = ms.SearchTask(workload=matmul)
    sch = ms.Schedule(func=matmul)
    args = rule(task, sch, sch.get_block("matmul"))
    assert len(args) == 1
    sch: tir.Schedule = args[0].sch
    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        i.stmt.extent for i in sch.get_axes(sch.get_block("matmul"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


if __name__ == "__main__":
    test_meta_schedule_rule_do_nothing()
    test_meta_schedule_rule_do_mlt()
    test_meta_schedule_rule_composite_0()
    test_meta_schedule_rule_composite_1()
