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


def element_wise_stmt(m=128, n=128):
    ib = tvm.ir_builder.create()
    a = tvm.var("a")
    c = tvm.var("c")
    A = ib.declare_buffer((m, n), "float32", name="A")
    B = ib.allocate_buffer((m, n), "float32", name="B")
    C = ib.declare_buffer((m, n), "float32", name="C")
    buffer_map = { a: A, c: C }
    dom_i = tvm.make.range_by_min_extent(0, m)
    dom_j = tvm.make.range_by_min_extent(0, n)

    with ib.loop_range(0, m, name="i0") as i:
        with ib.loop_range(0, n, name="j0") as j:
            bv_i = ib.iter_var(dom_i, name="vi")
            bv_j = ib.iter_var(dom_j, name="vj")
            vi = bv_i.var
            vj = bv_j.var
            with ib.block([bv_i, bv_j], [i, j], A[vi:vi + 1, vj:vj + 1], B[vi:vi + 1, vj:vj + 1], name="B"):
                B[vi, vj] = A[vi, vj] * 2

    with ib.loop_range(0, m, name="i1") as i:
        with ib.loop_range(0, n, name="j1") as j:
            bv_i = ib.iter_var(dom_i, name="vi")
            bv_j = ib.iter_var(dom_j, name="vj")
            vi = bv_i.var
            vj = bv_j.var
            with ib.block([bv_i, bv_j], [i, j], B[vi:vi + 1, vj:vj + 1], C[vi:vi + 1, vj:vj + 1], name="C"):
                C[vi, vj] = B[vi, vj] + 1

    stmt = ib.get()
    func, tensors, tensor_map = ib.function([a, c], buffer_map, stmt)
    return func, tensors, tensor_map


def matmul_stmt(m=128, n=128, l=128):
    ib = tvm.ir_builder.create()
    a = tvm.var("a")
    b = tvm.var("b")
    c = tvm.var("c")
    A = ib.declare_buffer((m, l), "float32", name="A")
    B = ib.declare_buffer((n, l), "float32", name="B")
    C = ib.declare_buffer((m, n), "float32", name="C")
    buffer_map = { a: A, b: B, c: C }
    dom_i = tvm.make.range_by_min_extent(0, m)
    dom_j = tvm.make.range_by_min_extent(0, n)
    dom_k = tvm.make.range_by_min_extent(0, l)

    with ib.loop_range(0, m, name="i") as i:
        with ib.loop_range(0, n, name="j") as j:
            bv_i = ib.iter_var(dom_i, name="vi")
            bv_j = ib.iter_var(dom_j, name="vj")
            vi = bv_i.var
            vj = bv_j.var
            with ib.block([bv_i, bv_j], [i, j], [], C[vi:vi + 1, vj:vj + 1], name="init"):
                C[vi, vj] = 0.0
            with ib.loop_range(0, l, name="k") as k:
                ii = ib.iter_var(dom_i, name="vi")
                jj = ib.iter_var(dom_j, name="vj")
                kk = ib.iter_var(dom_k, iter_type="reduce", name="vk")
                vi = ii.var
                vj = jj.var
                vk = kk.var
                reads = [C[vi:vi + 1, vj:vj + 1], A[vi:vi + 1, vk:vk + 1], B[vj:vj + 1, vk:vk + 1]]
                writes = [C[vi:vi + 1, vj:vj + 1]]
                with ib.block([ii, jj, kk], [i, j, k], reads, writes, name="update"):
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    stmt = ib.get()
    func, tensors, tensor_map = ib.function([a, b, c], buffer_map, stmt)
    return func, tensors, tensor_map
