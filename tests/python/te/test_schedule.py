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
import numpy as np


def test_create_schedule():
    func, tensors, tensor_map = util.element_wise_stmt()
    te.create_schedule(func)


def test_block_axis():
    func, tensors, tensor_map = util.matmul_stmt()
    s = te.create_schedule(func)
    init = s.get_block("init")
    update = s.get_block("update")
    assert len(s.get_axes(init)) == 2
    assert len(s.get_axes(update)) == 3


def test_fuse():
    m, n = 128, 128
    func, tensors, tensor_map = util.element_wise_stmt()

    # build
    lower_func = tvm.lower(tvm.ir_pass.TeLower(func, tensor_map), tensors)
    build_func = tvm.build(lower_func)
    a_np = np.random.uniform(size=(m, n)).astype("float32")
    a = tvm.nd.array(a_np)
    c1 = tvm.nd.array(np.zeros((m, n)).astype("float32"))
    build_func(a, c1)

    # schedule
    s = te.create_schedule(func)
    B = s.get_block("B")
    C = s.get_block("C")
    outer, inner = s.get_axes(B)
    s.fuse(outer, inner)
    outer, inner = s.get_axes(C)
    s.fuse(outer, inner)
    func = s.func

    # build
    lower_func = tvm.lower(tvm.ir_pass.TeLower(func, tensor_map), tensors)
    build_func = tvm.build(lower_func)
    c2 = tvm.nd.array(np.zeros((m, n)).astype("float32"))
    build_func(a, c2)

    tvm.testing.assert_allclose(c1.asnumpy(), c2.asnumpy(), rtol=1e-6)


if __name__ == "__main__":
    test_create_schedule()
    test_block_axis()
    test_fuse()
