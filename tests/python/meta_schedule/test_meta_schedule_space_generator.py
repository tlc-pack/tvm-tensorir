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
""" Test Meta Schedule SpaceGenerator """
# pylint: disable=missing-function-docstring

import sys
import math

import pytest

import tvm
from tvm import tir
from tvm.script import ty

from tvm.tir.schedule import Schedule, Trace
from tvm.meta_schedule import ScheduleFn, SpaceGeneratorUnion
from tvm.ir import IRModule
from tvm.tir import PrimFunc


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks
# fmt: off

@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


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


def _check_correct(trace: Trace):
    for inst in trace.decisions:
        assert math.prod(trace.decisions[inst]) == 1024


def _to_irmodule(prim_func: PrimFunc) -> IRModule:
    mod = tvm.IRModule()
    mod["main"] = prim_func
    return mod


def test_meta_schedule_space_generator_schedule_fn():
    space_generator = ScheduleFn(sch_fn=schedule_matmul)
    mod = _to_irmodule(matmul)
    design_spaces = space_generator.generate_design_space(mod)
    assert len(design_spaces) == 1
    (trace,) = design_spaces
    _check_correct(trace)


def test_meta_schedule_design_space_generator_union():
    space_generator = ScheduleFn(sch_fn=schedule_matmul)
    mod = _to_irmodule(matmul)
    space_generator_union = SpaceGeneratorUnion([space_generator, space_generator])
    design_spaces = space_generator_union.generate_design_space(mod)
    assert len(design_spaces) == 2
    for design_space in design_spaces:
        _check_correct(design_space)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
