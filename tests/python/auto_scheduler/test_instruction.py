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
""" Testing tvm.auto_scheduler.AccessAnalysis. """
import tvm
from tvm.auto_scheduler.instruction import DeclIntVar, SplitInnerToOuter, Reorder, ComputeAtOffset, CursorMoveOffset


def test_decl_int_var():
    var = tvm.tir.Var("i_1", "int32")
    inst = DeclIntVar(var, [1, 2, 4, 8])
    inst = str(inst)
    assert inst == "i_1 = DeclIntVarNode(choices=[1, 2, 4, 8])"


def test_split_inner_to_outer_with_none():
    factors = [tvm.tir.Var("j_" + str(i), "int32") for i in range(4)]
    inferred = factors[0]
    factors[0] = None
    inst = SplitInnerToOuter(2, factors, inferred)
    inst = str(inst)
    assert inst == "j_0 = SplitInnerToOuter(loop_id=2, factors=[None, j_1, j_2, j_3])"


def test_split_inner_to_outer_without_none():
    factors = [tvm.tir.Var("j_" + str(i), "int32") for i in range(4)]
    inferred = None
    inst = SplitInnerToOuter(2, factors, inferred)
    inst = str(inst)
    assert inst == "SplitInnerToOuter(loop_id=2, factors=[j_0, j_1, j_2, j_3])"


def test_reorder():
    inst = Reorder([0, 4, 1, 5, 8, 2, 6, 9, 3, 7])
    inst = str(inst)
    assert inst == "Reorder(after_ids=[0, 4, 1, 5, 8, 2, 6, 9, 3, 7])"


def test_compute_at_offset():
    inst = ComputeAtOffset(-1, 1)
    inst = str(inst)
    assert inst == "ComputeAt(offset=-1, loop_id=1)"


def test_cursor_move_offset():
    inst = CursorMoveOffset(+8)
    inst = str(inst)
    assert inst == "CursorMove(offset=8)"


if __name__ == "__main__":
    test_decl_int_var()
    test_split_inner_to_outer_without_none()
    test_split_inner_to_outer_with_none()
    test_reorder()
    test_compute_at_offset()
    test_cursor_move_offset()
