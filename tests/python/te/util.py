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
import topi
import numpy as np


def element_wise_stmt(m=128, n=128):
    ib = tvm.ir_builder.create()
    a = tvm.var("a")
    c = tvm.var("c")
    A = ib.declare_buffer((m, n), "float32", name="A")
    C = ib.declare_buffer((m, n), "float32", name="C")
    buffer_map = {a: A, c: C}
    dom_i = tvm.make.range_by_min_extent(0, m)
    dom_j = tvm.make.range_by_min_extent(0, n)

    with ib.block([], [], A[0:m, 0:n], C[0:m, 0:n], name="root"):
        B = ib.allocate_buffer((m, n), "float32", name="B")

        with ib.loop_range(0, m, name="i0") as i:
            with ib.loop_range(0, n, name="j0") as j:
                bv_i = ib.iter_var(dom_i, name="vi0")
                bv_j = ib.iter_var(dom_j, name="vj0")
                vi = bv_i.var
                vj = bv_j.var
                with ib.block([bv_i, bv_j], [i, j], A[vi:vi + 1, vj:vj + 1], B[vi:vi + 1, vj:vj + 1],
                        name="B"):
                    B[vi, vj] = A[vi, vj] * 2

        with ib.loop_range(0, m, name="i1") as i:
            with ib.loop_range(0, n, name="j1") as j:
                bv_i = ib.iter_var(dom_i, name="vi1")
                bv_j = ib.iter_var(dom_j, name="vj1")
                vi = bv_i.var
                vj = bv_j.var
                with ib.block([bv_i, bv_j], [i, j], B[vi:vi + 1, vj:vj + 1], C[vi:vi + 1, vj:vj + 1],
                        name="C"):
                    C[vi, vj] = B[vi, vj] + 1

    stmt = ib.get()
    func, tensors, tensor_map = ib.function([a, c], buffer_map, stmt)
    return func, tensors, tensor_map, [A._buffer, B._buffer, C._buffer]


def matmul_stmt(m=128, n=128, l=128):
    ib = tvm.ir_builder.create()
    a = tvm.var("a")
    b = tvm.var("b")
    c = tvm.var("c")
    A = ib.declare_buffer((m, l), "float32", name="A")
    B = ib.declare_buffer((n, l), "float32", name="B")
    C = ib.declare_buffer((m, n), "float32", name="C")
    buffer_map = {a: A, b: B, c: C}
    dom_i = tvm.make.range_by_min_extent(0, m)
    dom_j = tvm.make.range_by_min_extent(0, n)
    dom_k = tvm.make.range_by_min_extent(0, l)

    with ib.block([], [], [A[0:m, 0:l], B[0:n, 0:l]], C[0:m, 0:n], name="root"):
        with ib.loop_range(0, m, name="i") as i:
            with ib.loop_range(0, n, name="j") as j:
                bv_i = ib.iter_var(dom_i, name="vi0")
                bv_j = ib.iter_var(dom_j, name="vj0")
                vi = bv_i.var
                vj = bv_j.var
                with ib.block([bv_i, bv_j], [i, j], [], C[vi:vi + 1, vj:vj + 1], name="init"):
                    C[vi, vj] = 0.0
                with ib.loop_range(0, l, name="k") as k:
                    ii = ib.iter_var(dom_i, name="vi1")
                    jj = ib.iter_var(dom_j, name="vj1")
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
    return func, tensors, tensor_map, [A._buffer, B._buffer, C._buffer]


def check_correctness(func1, func2, args, tensor_map=None, target='llvm'):
    if isinstance(func1, tvm.container.TeFunction):
        func1 = tvm.ir_pass.TeLower(func1, tensor_map)
    if isinstance(func2, tvm.container.TeFunction):
        func2 = tvm.ir_pass.TeLower(func2, tensor_map)

    if isinstance(target, tuple) or isinstance(target, list):
        target1, target2 = target
    else:
        target1 = target2 = target

    func1 = tvm.build(tvm.lower(func1, args), target=target1)
    func2 = tvm.build(tvm.lower(func2, args), target=target2)

    ctx1 = tvm.context(target1)
    ctx2 = tvm.context(target2)

    bufs1 = [
        tvm.nd.array(np.array(np.random.randn(*topi.util.get_const_tuple(x.shape)), dtype=x.dtype),
            ctx=ctx1) for x in args]
    bufs2 = [tvm.nd.array(x, ctx=ctx2) for x in bufs1]
    func1(*bufs1)
    func2(*bufs2)
    bufs1_np = [x.asnumpy() for x in bufs1]
    bufs2_np = [x.asnumpy() for x in bufs2]

    for x, y in zip(bufs1_np, bufs2_np):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)
