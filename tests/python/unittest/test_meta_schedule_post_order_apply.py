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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

from typing import List
import pytest
import math

import tvm
from tvm._ffi.base import py2cerror
from tvm.script import tir as T
from tvm.tir.schedule import Schedule, BlockRV
from tvm.target import Target

from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.meta_schedule.schedule_rule import PyScheduleRule
from tvm.meta_schedule.utils import _get_hex_address


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        with T.block([1024, 1024, T.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

@tvm.script.ir_module
class DuplicateMatmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        with T.block([1024, 1024, T.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    @T.prim_func
    def test(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "test"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        with T.block([1024, 1024, T.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _check_correct(schedule: Schedule):
    trace = schedule.trace
    for inst in trace.decisions:
        assert math.prod(trace.decisions[inst]) == 1024


class WowSoFancyScheduleRule(PyScheduleRule):
    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
        new_sch = sch.copy()
        i, j, k = new_sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = new_sch.split(loop=i, factors=[2, 4, 64, 2])
        j_0, j_1, j_2, j_3 = new_sch.split(loop=j, factors=[4, 64, 2, 2])
        k_0, k_1 = new_sch.split(loop=k, factors=[32, 32])
        new_sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
        return [new_sch]


class DoubleScheduleRule(PyScheduleRule):
    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
        new_sch = sch.copy()
        i, j, k = new_sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = new_sch.split(loop=i, factors=[4, 64, 2, 2])
        j_0, j_1, j_2, j_3 = new_sch.split(loop=j, factors=[2, 4, 64, 2])
        k_0, k_1 = new_sch.split(loop=k, factors=[32, 32])
        new_sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
        result = [new_sch]
        new_sch = sch.copy()
        i, j, k = new_sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = new_sch.split(loop=i, factors=[4, 64, 2, 2])
        j_0, j_1, j_2, j_3 = new_sch.split(loop=j, factors=[2, 4, 64, 2])
        k_0, k_1 = new_sch.split(loop=k, factors=[32, 32])
        new_sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
        result.append(new_sch)
        return result


class ReorderScheduleRule(PyScheduleRule):
    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        pass

    def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
        new_sch = sch.copy()
        i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = new_sch.get_loops(block=block)
        new_sch.reorder(i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3, i_0, j_0)
        result = [new_sch]
        new_sch = sch.copy()
        i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3 = new_sch.get_loops(block=block)
        new_sch.reorder(i_1, j_3, i_0, j_0, j_1, k_0, i_2, j_2, k_1, i_3)
        result.append(new_sch)
        return result


def test_meta_schedule_post_order_apply():
    mod = Matmul
    context = TuneContext(
        mod=mod, target=Target("llvm"), task_name="Test Task", sch_rules=[WowSoFancyScheduleRule()]
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 1
    try:
        tvm.ir.assert_structural_equal(mod, schs[0].mod)
        raise Exception("The schedule rule did not change the schedule.")
    except (ValueError):
        _check_correct(schs[0])


def test_meta_schedule_post_order_apply_double():
    mod = Matmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Double Rules Task",
        sch_rules=[DoubleScheduleRule()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 2
    for sch in schs:
        try:
            tvm.ir.assert_structural_equal(mod, sch.mod)
            raise Exception("The schedule rule did not change the schedule.")
        except (ValueError):
            _check_correct(sch)


def test_meta_schedule_post_order_apply_multiple():
    mod = Matmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Double Rules Task",
        sch_rules=[DoubleScheduleRule(), ReorderScheduleRule()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    schs = post_order_apply.generate_design_space(mod)
    assert len(schs) == 4
    for sch in schs:
        try:
            tvm.ir.assert_structural_equal(mod, sch.mod)
            raise Exception("The schedule rule did not change the schedule.")
        except (ValueError):
            _check_correct(sch)


@pytest.mark.xfail
def test_meta_schedule_post_order_apply_duplicate_matmul():
    mod = DuplicateMatmul
    context = TuneContext(
        mod=mod,
        target=Target("llvm"),
        task_name="Duplicate Matmul Task",
        sch_rules=[WowSoFancyScheduleRule()],
    )
    post_order_apply = PostOrderApply()
    post_order_apply.initialize_with_tune_context(context)
    schs = post_order_apply.generate_design_space(mod)


if __name__ == "__main__":
    test_meta_schedule_post_order_apply()
    test_meta_schedule_post_order_apply_duplicate_matmul()
