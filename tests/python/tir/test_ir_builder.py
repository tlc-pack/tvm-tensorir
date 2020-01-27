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
import numpy as np
import util


def test_element_wise():
    m, n = 16, 16
    func, tensors, tensor_map, _ = util.element_wise_stmt(m, n)
    func = tvm.ir_pass.TeLower(func, tensor_map)
    print(func)
    lower_func = tvm.lower(func, tensors)
    func = tvm.build(lower_func)
    a_np = np.random.uniform(size=(m, n)).astype("float32")
    a = tvm.nd.array(a_np)
    c = tvm.nd.array(np.zeros((m, n)).astype("float32"))
    func(a, c)
    tvm.testing.assert_allclose(c.asnumpy(), a_np * 2 + 1, rtol=1e-6)


def test_matmul():
    m, n, l = 16, 16, 16
    func, tensors, tensor_map, _ = util.matmul_stmt(m, n, l)
    func = tvm.ir_pass.TeLower(func, tensor_map)
    print(func)
    lower_func = tvm.lower(func, tensors)
    func = tvm.build(lower_func)
    a_np = np.random.uniform(size=(m, l)).astype("float32")
    b_np = np.random.uniform(size=(n, l)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((m, n)).astype("float32"))
    func(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np.dot(a_np, b_np.transpose()), rtol=1e-6)


if __name__ == "__main__":
    test_element_wise()
    test_matmul()
