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

def test_simple_print():

    ib = tvm.ir_builder.create()
    A = ib.allocate_buffer((16, 16), "float32", name="A")
    B = ib.allocate_buffer((16, 16), "float32", name="B")
    C = ib.allocate_buffer((16, 16), "float32", name="C")
    dom = tvm.make.range_by_min_extent(0, 16)
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

    print(ib.get())

if __name__ == "__main__":
    test_simple_print()
