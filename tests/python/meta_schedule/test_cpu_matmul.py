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
import tvm
from tvm.hybrid import ty
from tvm import tir
from tvm import meta_schedule as ms


@tvm.hybrid.script
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "C") as [vi, vj, vk]:
        reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])


def _get_prim_func_from_hybrid(hybrid_func):
    module = tvm.hybrid.create_module({"hybrid_func": hybrid_func})
    prim_func = module["hybrid_func"]
    assert isinstance(prim_func, tvm.tir.PrimFunc)
    return prim_func


def _print_prim_func(prim_func):
    print(tvm.hybrid.ashybrid(prim_func))


def test_matmul_tiling():
    sch = ms.Schedule(_get_prim_func_from_hybrid(matmul))
    block = sch.get_block(name="C")
    i, j, k = sch.get_axes(block=block)
    i_tiles = sch.sample_tile_factor(n=4, loop=i, where=[1, 2, 4])
    j_tiles = sch.sample_tile_factor(n=4, loop=j, where=[1, 2, 4])
    k_tiles = sch.sample_tile_factor(n=2, loop=k, where=[1, 2, 4])
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    sch.reorder(after_axes=[i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3])
    _print_prim_func(sch.sch.func)
    for _ in range(1000):
        sch.replay_once()
        _print_prim_func(sch.sch.func)


if __name__ == "__main__":
    test_matmul_tiling()
