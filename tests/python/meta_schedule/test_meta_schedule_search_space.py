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


# pylint: enable=invalid-name,no-member


def test_meta_schedule_schedule_fn():
    def schedule_matmul(sch):
        block = sch.get_block(name="C")
        i, j, k = sch.get_axes(block=block)
        i_tiles = sch.sample_tile_factor(n=4, loop=i, where=[1, 2, 4])
        j_tiles = sch.sample_tile_factor(n=4, loop=j, where=[1, 2, 4])
        k_tiles = sch.sample_tile_factor(n=2, loop=k, where=[1, 2, 4])
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
        k_0, k_1 = sch.split(loop=k, factors=k_tiles)
        sch.reorder(after_axes=[i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3])

    task = ms.SearchTask(matmul)
    space = ms.search.ScheduleFn(schedule_matmul)
    sch = space.sample_schedule(task)

    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        sch.evaluate(i).stmt.extent for i in sch.get_axes(sch.get_block("C"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


def test_meta_schedule_post_order_apply():
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
                    tiles = sch.sample_tile_factor(
                        n=len(indices), loop=axis, where=[1, 2, 4]
                    )
                    splits = sch.split(loop=axis, factors=tiles)
                    for i, split in zip(indices, splits):
                        order[i].append(split)
        sch.reorder(after_axes=sum(order, []))
        return sch

    task = ms.SearchTask(matmul)
    space = ms.PostOrderApply(rule=do_mlt)
    sch = space.sample_schedule(task)
    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        sch.evaluate(i).stmt.extent for i in sch.get_axes(sch.get_block("C"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


if __name__ == "__main__":
    test_meta_schedule_schedule_fn()
    test_meta_schedule_post_order_apply()
