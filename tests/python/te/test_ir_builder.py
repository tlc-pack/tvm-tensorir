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


def test_ir_builder():
    ib = tvm.ir_builder.create()
    a = tvm.var("a")
    b = tvm.var("b")
    c = tvm.var("c")
    A = ib.declare_buffer((16, 16), "float32", name="A")
    B = ib.declare_buffer((16, 16), "float32", name="B")
    C = ib.declare_buffer((16, 16), "float32", name="C")
    dom = tvm.make.range_by_min_extent(0, 16)
    with ib.function([b, c, a], [B, C, A]) as tensors:
        with ib.loop_range(0, 16, name="i") as i:
            with ib.loop_range(0, 16, name="j") as j:
                bv_i = ib.iter_var(dom, name="vi")
                bv_j = ib.iter_var(dom, name="vj")
                vi = bv_i.var
                vj = bv_j.var
                with ib.block([bv_i, bv_j], [i, j], [], A[vi:vi+1, vj:vj+1], name="init"):
                    A[vi, vj] = 0.0
                with ib.loop_range(0, 16, name="k") as k:
                    ii = ib.iter_var(dom, name="vi")
                    jj = ib.iter_var(dom, name="vj")
                    kk = ib.iter_var(dom, iter_type="reduce", name="vk")
                    vi = ii.var
                    vj = jj.var
                    vk = kk.var
                    reads = [A[vi:vi+1, vj:vj+1], B[vi:vi+1, vk:vk+1], C[vj:vj+1, vk:vk+1]]
                    writes = [A[vi:vi+1, vj:vj+1]]
                    with ib.block([ii, jj, kk], [i, j, k], reads, writes, name="update"):
                        A[vi, vj] = A[vi, vj] + B[vi, vk] * C[vj, vk]

    stmt = ib.get()
    print(stmt)
    stmt = tvm.ir_pass.TeLower(stmt, tensors)
    print(stmt)
    lower_func = tvm.lower(stmt, tensors)
    func = tvm.build(lower_func)
    a_np = np.random.uniform(size=(16, 16)).astype("float32")
    b_np = np.random.uniform(size=(16, 16)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((16, 16)).astype("float32"))
    func(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np.dot(a_np, b_np.transpose()), rtol=1e-6)


if __name__ == "__main__":
    test_ir_builder()
