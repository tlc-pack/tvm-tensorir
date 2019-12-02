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


def test_create_schedule():
    func, tensors, tensor_map = util.element_wise_stmt()
    s = te.create_schedule(func)
    print(s.func)


def test_block_axis():
    func, tensors, tensor_map = util.matmul_stmt()
    s = te.create_schedule(func)
    init = s.get_block("init")
    update = s.get_block("update")
    assert len(s.get_axes(init)) == 2
    assert len(s.get_axes(update)) == 3


def test_fuse():
    m, n = 128, 128
    func, tensors, tensor_map = util.element_wise_stmt(m, n)

    # schedule
    s = te.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(B)
    s.fuse(outer, inner)
    outer, inner = s.get_axes(C)
    s.fuse(outer, inner)

    util.check_correctness(func, s.func, tensors, tensor_map)


def test_split():
    m, n = 128, 128
    func, tensors, tensor_map = util.element_wise_stmt(m, n)

    # schedule
    s = te.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(B)
    s.split(outer, factor=8)
    outer, inner = s.get_axes(C)
    s.split(inner, nparts=10)

    util.check_correctness(func, s.func, tensors, tensor_map)


def test_compute_inline():
    m, n = 128, 128
    func, tensors, tensor_map = util.element_wise_stmt(m, n)

    # schedule
    s = te.create_schedule(func)
    B = s.get_block("B")
    s.compute_inline(B)

    util.check_correctness(func, s.func, tensors, tensor_map)


if __name__ == "__main__":
    # test_create_schedule()
    # test_block_axis()
    test_fuse()
    # test_split()
    # test_compute_inline()
