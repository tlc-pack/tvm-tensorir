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
"""Test for meta schedule mutators"""
from tir_workload import matmul

# pylint: disable=missing-function-docstring
from tvm import meta_schedule as ms


def test_meta_schedule_mutate_tile_size():
    def verify(changed, old_tiles, new_tiles, length):
        from functools import reduce  # pylint: disable=import-outside-toplevel

        assert reduce(lambda x, y: x * y, new_tiles) == length
        if not changed:
            return
        n = sum(1 for old, new in zip(old_tiles, new_tiles) if old != new)
        if n != 2:
            print(n, old_tiles, new_tiles)
        assert n == 2

    n_trial = 100
    n_fails = 0

    task = ms.SearchTask(func=matmul)
    mutator = ms.mutator.mutate_tile_size()
    sch = ms.Schedule(func=matmul)
    i, j, k = sch.get_axes(sch.get_block("matmul"))
    i_tiles = sch.sample_perfect_tile(n_splits=4, loop=i)
    j_tiles = sch.sample_perfect_tile(n_splits=4, loop=j)
    k_tiles = sch.sample_perfect_tile(n_splits=2, loop=k)
    i_eval = [int(sch.evaluate(i)) for i in i_tiles]
    j_eval = [int(sch.evaluate(j)) for j in j_tiles]
    k_eval = [int(sch.evaluate(k)) for k in k_tiles]

    verify(False, None, i_eval, length=1024)
    verify(False, None, j_eval, length=1024)
    verify(False, None, k_eval, length=1024)

    for _ in range(n_trial):
        new_sch: ms.Schedule = mutator.apply(task=task, sch=sch)
        if new_sch is None:
            assert False
            n_fails += 1
            continue
        new_i_eval = [int(new_sch.evaluate(i)) for i in i_tiles]
        new_j_eval = [int(new_sch.evaluate(j)) for j in j_tiles]
        new_k_eval = [int(new_sch.evaluate(k)) for k in k_tiles]
        i_changed = int(i_eval != new_i_eval)
        j_changed = int(j_eval != new_j_eval)
        k_changed = int(k_eval != new_k_eval)
        assert i_changed + j_changed + k_changed == 1
        verify(i_changed, i_eval, new_i_eval, length=1024)
        verify(j_changed, j_eval, new_j_eval, length=1024)
        verify(k_changed, k_eval, new_k_eval, length=1024)
        sch = new_sch
        i_eval = new_i_eval
        j_eval = new_j_eval
        k_eval = new_k_eval

    assert n_fails <= n_trial * 0.01


if __name__ == "__main__":
    test_meta_schedule_mutate_tile_size()
