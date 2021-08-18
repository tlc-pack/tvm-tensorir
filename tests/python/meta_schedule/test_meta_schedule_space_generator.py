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
from tvm.tir.schedule import Schedule

from tvm.meta_schedule import ScheduleFn


def test_meta_schedule_space_generator_schedule_fn():
    def schedule_matmul(sch: Schedule):
        block = sch.get_block("matmul")
        i, j, k = sch.get_loops(block=block)
        i_tiles = sch.sample_perfect_tile(i, n=4)
        j_tiles = sch.sample_perfect_tile(j, n=4)
        k_tiles = sch.sample_perfect_tile(k, n=2)
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
        k_0, k_1 = sch.split(loop=k, factors=k_tiles)
        sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)

    space_gen = ScheduleFn(sch_fn=schedule_matmul)
    (sch,) = space_gen.generate(workload=matmul)

    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        sch.get_sref(i).stmt.extent for i in sch.get_loops(sch.get_block("matmul"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


def test_meta_schedule_design_space_generator_union():
    def schedule_matmul(sch: Schedule):
        block = sch.get_block("matmul")
        i, j, k = sch.get_loops(block=block)
        i_tiles = sch.sample_perfect_tile(i, n=4)
        j_tiles = sch.sample_perfect_tile(j, n=4)
        k_tiles = sch.sample_perfect_tile(k, n=2)
        i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
        j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
        k_0, k_1 = sch.split(loop=k, factors=k_tiles)
        sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)

    space_gen = ScheduleFn(sch_fn=schedule_matmul)
    space_gen_union = ms.SpaceGeneratorUnion([space_gen, space_gen])
    ret = space_gen_union.generate(workload=matmul)
    assert len(ret) == 2
    sch = ret[0]

    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        sch.get_sref(i).stmt.extent for i in sch.get_loops(sch.get_block("matmul"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024

    sch = ret[1]

    i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = [
        sch.get_sref(i).stmt.extent for i in sch.get_loops(sch.get_block("matmul"))
    ]
    assert i_0 * i_1 * i_2 * i_3 == 1024
    assert j_0 * j_1 * j_2 * j_3 == 1024
    assert k_0 * k_1 == 1024


if __name__ == "__main__":
    test_meta_schedule_space_generator_schedule_fn()
    test_meta_schedule_design_space_generator_union()
