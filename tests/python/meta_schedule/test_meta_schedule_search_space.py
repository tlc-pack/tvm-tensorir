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
""" Test meta schedule SearchSpace """
# pylint: disable=missing-function-docstring

from tir_workload import matmul
from tvm import meta_schedule as ms


def test_meta_schedule_search_space_schedule_fn():
    def schedule_matmul(sch: ms.Schedule):
        block = sch.get_block("matmul")
        i, j, k = sch.get_loops(block=block)
        i_tiles = sch.sample_perfect_tile(i, n=4)
        j_tiles = sch.sample_perfect_tile(j, n=4)
        k_tiles = sch.sample_perfect_tile(k, n=2)
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
        k_0, k_1 = sch.split(loop=k, factors=k_tiles)
        sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)

    task = ms.SearchTask(matmul)
    space = ms.space.ScheduleFn(schedule_matmul)
    sch = space.sample_schedule(task)

    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        sch.get_sref(i).stmt.extent for i in sch.get_loops(sch.get_block("matmul"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


def test_meta_schedule_search_space_post_order_apply():
    @ms.rule.register_rule("do_mlt")
    def do_mlt(_task, sch: ms.Schedule, block: ms.BlockRV):
        TILING_FORMAT = "SSRSRS"  # pylint: disable=invalid-name
        spatial_indices = [i for i, c in enumerate(TILING_FORMAT) if c == "S"]
        reduce_indices = [i for i, c in enumerate(TILING_FORMAT) if c == "R"]
        order = [list() for _ in TILING_FORMAT]
        axes = sch.get_loops(block=block)
        iter_types = ms.analysis.get_block_var_types(sch.state, sch.get_sref(block))
        assert len(axes) == len(iter_types)
        for axis, iter_type in zip(axes, iter_types):
            for expect_iter_type, indices in [
                ("spatial", spatial_indices),
                ("reduce", reduce_indices),
            ]:
                if iter_type == expect_iter_type:
                    tiles = sch.sample_perfect_tile(axis, n=len(indices))
                    splits = sch.split(loop=axis, factors=tiles)
                    for i, split in zip(indices, splits):
                        order[i].append(split)
        sch.reorder(*sum(order, []))
        return sch

    task = ms.SearchTask(matmul)
    space = ms.space.PostOrderApply(stages=[do_mlt])
    sch = space.sample_schedule(task)
    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        sch.get_sref(i).stmt.extent for i in sch.get_loops(sch.get_block("matmul"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


if __name__ == "__main__":
    test_meta_schedule_search_space_schedule_fn()
    test_meta_schedule_search_space_post_order_apply()
