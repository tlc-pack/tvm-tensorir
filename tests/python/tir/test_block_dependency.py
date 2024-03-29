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
# pylint: disable=missing-function-docstring,missing-module-docstring
import pytest
import tvm
from tvm import tir
from tvm.script import ty

import util

# pylint: disable=no-member,invalid-name,unused-variable


@tvm.script.tir
def test_WAR(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))

    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = B[vi, vj] + 1.0
        with tir.block([128, 128], "B") as [vi, vj]:
            B[vi, vj] = A[vi, vj] * 2.0


# pylint: enable=no-member,invalid-name,unused-variable

# pylint: disable=invalid-name


def test_element_wise_dependency():
    func = util.element_wise_stmt()
    s = tir.Schedule(func, debug_mode=True)
    root = s.get_sref(s.get_block("root"))
    block_b = s.get_sref(s.get_block("B"))
    block_c = s.get_sref(s.get_block("C"))
    # Check get_deps_by_dst
    (predecessor_c,) = s.state.get_block_scope(root).get_deps_by_dst(block_c)
    assert predecessor_c.src.same_as(block_b)
    assert predecessor_c.kind == tir.schedule.DepKind.RAW
    # Check get_deps_by_src
    (successor_b,) = s.state.get_block_scope(root).get_deps_by_src(block_b)
    assert successor_b.dst.same_as(block_c)
    assert predecessor_c.kind == tir.schedule.DepKind.RAW


def test_matmul_dependency():
    func = util.matmul_stmt_original()
    s = tir.Schedule(func, debug_mode=True)
    root = s.get_sref(s.get_block("root"))
    init = s.get_sref(s.get_block("init"))
    update = s.get_sref(s.get_block("update"))
    # Check predecessors
    p0, p1 = s.state.get_block_scope(root).get_deps_by_dst(update)
    assert p0.src.same_as(init)
    assert p1.src.same_as(init)
    # WAW and RAW
    assert (p0.kind == tir.schedule.DepKind.RAW and p1.kind == tir.schedule.DepKind.WAW) or (
        p0.kind == tir.schedule.DepKind.WAW and p1.kind == tir.schedule.DepKind.RAW
    )
    # Check successors
    p0, p1 = s.state.get_block_scope(root).get_deps_by_src(init)
    assert p0.dst == update
    assert p1.dst == update
    # WAW and RAW
    assert (p0.kind == tir.schedule.DepKind.RAW and p1.kind == tir.schedule.DepKind.WAW) or (
        p0.kind == tir.schedule.DepKind.WAW and p1.kind == tir.schedule.DepKind.RAW
    )


def test_WAR_dependency():
    mod = tvm.script.create_module({"test_WAR": test_WAR})
    func = mod["test_WAR"]
    s = tir.Schedule(func, debug_mode=True)
    root = s.get_sref(s.get_block("root"))
    b0 = s.get_sref(s.get_block("C"))
    b1 = s.get_sref(s.get_block("B"))
    (e,) = s.state.get_block_scope(root).get_deps_by_src(b0)
    assert e.kind == tir.schedule.DepKind.WAR
    assert e.dst.same_as(b1)
    (e,) = s.state.get_block_scope(root).get_deps_by_dst(b1)
    assert e.kind == tir.schedule.DepKind.WAR
    assert e.src.same_as(b0)


if __name__ == "__main__":
    test_element_wise_dependency()
    test_matmul_dependency()
    test_WAR_dependency()
