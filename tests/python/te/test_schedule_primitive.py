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


def test_fuse():
    m, n = 128, 128
    func, tensors, tensor_map, _ = util.element_wise_stmt(m, n)

    # schedule
    s = te.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(B)
    s.fuse(outer, inner)
    outer, inner = s.get_axes(C)
    s.fuse(outer, inner)

    assert isinstance(s.func.body, tvm.stmt.TeBlock)
    assert isinstance(s.func.body.body[0], tvm.stmt.Loop)
    assert s.func.body.body[0].extent.value == m * n
    assert isinstance(s.func.body.body[1], tvm.stmt.Loop)
    assert s.func.body.body[1].extent.value == m * n

    util.check_correctness(func, s.func, tensors, tensor_map)

def test_split():
    m, n = 128, 128
    func, tensors, tensor_map, _ = util.element_wise_stmt(m, n)

    # schedule
    s = te.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(B)
    s.split(outer, factor=8)
    outer, inner = s.get_axes(C)
    s.split(inner, nparts=10)

    assert isinstance(s.func.body, tvm.stmt.TeBlock)
    assert isinstance(s.func.body.body[0], tvm.stmt.Loop)
    assert s.func.body.body[0].extent.value == m // 8
    assert isinstance(s.func.body.body[0].body, tvm.stmt.Loop)
    assert s.func.body.body[0].body.extent.value == 8
    assert isinstance(s.func.body.body[0].body.body.body, tvm.stmt.TeBlock)
    assert isinstance(s.func.body.body[0].body.body.body.predicate, tvm.expr.UIntImm)
    assert s.func.body.body[0].body.body.body.predicate.value == 1

    assert isinstance(s.func.body.body[1].body, tvm.stmt.Loop)
    assert s.func.body.body[1].body.extent.value == 10
    assert isinstance(s.func.body.body[1].body.body, tvm.stmt.Loop)
    assert s.func.body.body[1].body.body.extent.value == (n + 9) // 10
    assert isinstance(s.func.body.body[1].body.body.body, tvm.stmt.TeBlock)
    assert not isinstance(s.func.body.body[1].body.body.body.predicate, tvm.expr.IntImm)

    util.check_correctness(func, s.func, tensors, tensor_map)

if __name__ == "__main__":
    test_fuse()
    test_split()