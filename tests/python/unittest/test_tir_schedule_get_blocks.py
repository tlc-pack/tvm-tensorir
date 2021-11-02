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
import sys

import pytest
import tvm
from tvm import tir
from tvm.script import tir as T

# pylint: disable=no-member,invalid-name,unused-variable


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])
    for i, j in T.grid(128, 128):
        with T.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = T.float32(0)
        for k in range(0, 128):
            with T.block([128, 128, T.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


def test_get_child_blocks():
    s = tir.Schedule(matmul, debug_mask="all")
    init = s.get_block("init")
    update = s.get_block("update")
    # loop
    blocks = s.get_child_blocks(s.get_loops(init)[0])
    assert len(blocks) == 2
    assert s.get(init) == s.get(blocks[0])
    assert s.get(update) == s.get(blocks[1])
    # block
    root = s.get_block("root")
    blocks = s.get_child_blocks(root)
    assert len(blocks) == 2
    assert s.get(init) == s.get(blocks[0])
    assert s.get(update) == s.get(blocks[1])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
