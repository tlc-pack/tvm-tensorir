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
from tvm import te
import util


def test_element_wise_dependency():
    func, tensors, tensor_map, buffers = util.element_wise_stmt()
    s = te.create_schedule(func)

    block_B = s.get_block("B")
    block_C = s.get_block("C")
    predecessor_c = s.get_predecessors(block_C)
    assert len(predecessor_c) == 1
    assert predecessor_c[0] == block_B

    successor_b = s.get_successors(block_B)
    assert len(successor_b) == 1
    assert successor_b[0] == block_C


def test_matmul_dependency():
    func, tensors, tensor_map, buffers = util.matmul_stmt()
    s = te.create_schedule(func)
    buffer_c = buffers[2]

    block_C = s.get_block(buffer_c)
    assert len(block_C) == 2
    init, update = block_C

    predecessor_update = s.get_predecessors(update)
    assert len(predecessor_update) == 1
    assert predecessor_update[0] == init

    successor_init = s.get_successors(init)
    assert len(successor_init) == 1
    assert successor_init[0] == update


if __name__ == "__main__":
    test_element_wise_dependency()
    test_matmul_dependency()
