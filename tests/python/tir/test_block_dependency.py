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

import pytest
import tvm
from tvm import tir
from tvm.hybrid import ty
import util


def test_element_wise_dependency():
    func = util.element_wise_stmt()
    s = tir.create_schedule(func)

    block_B = s.get_block("B")
    block_C = s.get_block("C")
    predecessor_c = s.get_predecessors(block_C)
    assert len(predecessor_c) == 1
    assert predecessor_c[0].dst == block_B
    assert predecessor_c[0].type == 0

    successor_b = s.get_successors(block_B)
    assert len(successor_b) == 1
    assert successor_b[0].dst == block_C
    assert predecessor_c[0].type == 0


def test_matmul_dependency():
    func = util.matmul_stmt_original()
    s = tir.create_schedule(func)
    buffer_c = func.buffer_map[func.params[2]]

    block_C = s.get_block(buffer_c)
    assert len(block_C) == 2
    init, update = block_C

    predecessor_update = s.get_predecessors(update)
    assert len(predecessor_update) == 2
    assert predecessor_update[0].dst == init
    assert predecessor_update[1].dst == init
    # Both WAW and RAW
    assert predecessor_update[0].type + predecessor_update[1].type == 1

    successor_init = s.get_successors(init)
    assert len(successor_init) == 2
    assert successor_init[0].dst == update
    assert successor_init[1].dst == update
    # Both WAW and RAW
    assert successor_init[0].type + successor_init[1].type == 1


@tvm.hybrid.script
def test_WAR(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.buffer_bind(a, (128, 128))
    B = tir.buffer_bind(b, (128, 128))
    C = tir.buffer_bind(c, (128, 128))

    with tir.block({}, A[0: 128, 0: 128], [B[0: 128, 0: 128], C[0: 128, 0: 128]], name="root"):
        for i, j in tir.grid(128, 128):
            with tir.block({vi(0, 128): i, vj(0, 128): j}, reads=B[vi, vj], writes=C[vi, vj],
                           name="C"):
                C[vi, vj] = B[vi, vj] + 1.0
            with tir.block({vi(0, 128): i, vj(0, 128): j}, reads=A[vi, vj], writes=B[vi, vj],
                           name="B"):
                B[vi, vj] = A[vi, vj] * 2.0


def test_WAR_dependency():
    mod = tvm.hybrid.create_module({"test_WAR": test_WAR})
    func = mod["test_WAR"]
    with pytest.raises(TypeError) as excinfo:
        tir.create_schedule(func)
    assert "WAR dependency is not allowed" in str(excinfo.value)


if __name__ == "__main__":
    test_element_wise_dependency()
    test_matmul_dependency()
    test_WAR_dependency()
