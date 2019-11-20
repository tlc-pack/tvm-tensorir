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
import util
from tvm import tir
from tvm.tir.hybrid import from_source


def test_matmul():
    func = util.matmul_stmt_original()
    rt_func = from_source(tvm.tir.hybrid.ashybrid(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body.block, tir.stmt.Block)
    assert isinstance(rt_func.body.block.body, tir.stmt.Loop)
    assert isinstance(rt_func.body.block.body.body, tir.stmt.Loop)
    assert isinstance(rt_func.body.block.body.body.body, tir.stmt.SeqStmt)
    assert isinstance(rt_func.body.block.body.body.body[0].block, tir.stmt.Block)
    assert isinstance(rt_func.body.block.body.body.body[1], tir.stmt.Loop)
    assert isinstance(rt_func.body.block.body.body.body[1].body.block, tir.stmt.Block)


def test_element_wise():
    func = util.element_wise_stmt()
    rt_func = from_source(tvm.tir.hybrid.ashybrid(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body.block, tir.stmt.Block)
    assert isinstance(rt_func.body.block.body, tir.stmt.SeqStmt)
    assert isinstance(rt_func.body.block.body[0], tir.stmt.Loop)
    assert isinstance(rt_func.body.block.body[0].body, tir.stmt.Loop)
    assert isinstance(rt_func.body.block.body[0].body.body.block, tir.stmt.Block)

    assert isinstance(rt_func.body.block.body[1], tir.stmt.Loop)
    assert isinstance(rt_func.body.block.body[1].body, tir.stmt.Loop)
    assert isinstance(rt_func.body.block.body[1].body.body.block, tir.stmt.Block)


def test_predicate():
    func = util.predicate_stmt()
    rt_func = from_source(tvm.tir.hybrid.ashybrid(func, True))
    tvm.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body.block, tir.stmt.Block)
    assert isinstance(rt_func.body.block.body, tir.stmt.Loop)
    assert isinstance(rt_func.body.block.body.body, tir.stmt.Loop)
    assert isinstance(rt_func.body.block.body.body.body, tir.stmt.Loop)
    assert isinstance(rt_func.body.block.body.body.body.body.block, tir.stmt.Block)


def test_functions():
    test_matmul()
    test_element_wise()
    test_predicate()


@tvm.tir.hybrid.script
class MyModule:
    def matmul(a, b, c):
        A = buffer_bind(a, (128, 128), "float32")
        B = buffer_bind(b, (128, 128), "float32")
        C = buffer_bind(c, (128, 128), "float32")

        with block({}, reads=[A[0: 128, 0: 128], B[0: 128, 0: 128]], writes=C[0: 128, 0: 128],
                   name="root"):
            for i in range(0, 128):
                for j in range(0, 128):
                    with block({vi(0, 128): i, vj(0, 128): j}, reads=[],
                               writes=C[vi: vi + 1, vj: vj + 1],
                               name="init"):
                        C[vi, vj] = float32(0)
                    for k in range(0, 128):
                        with block(
                                {vi(0, 128): i, vj(0, 128): j, vk(0, 128, iter_type="reduce"): k},
                                reads=[C[vi: vi + 1, vj: vj + 1], A[vi: vi + 1, vk: vk + 1],
                                       B[vj: vj + 1, vk: vk + 1]],
                                writes=[C[vi: vi + 1, vj: vj + 1]], name="update"):
                            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

    def element_wise(a, c):
        A = buffer_bind(a, (128, 128), "float32")
        C = buffer_bind(c, (128, 128), "float32")

        with block({}, A[0: 128, 0: 128], C[0: 128, 0: 128], name="root"):
            B = buffer_allocate((128, 128), "float32")

            for i in range(0, 128):
                for j in range(0, 128):
                    with block({vi(0, 128): i, vj(0, 128): j}, A[vi: vi + 1, vj: vj + 1],
                               B[vi: vi + 1, vj: vj + 1],
                               name="B"):
                        B[vi, vj] = A[vi, vj] * 2

            for i in range(0, 128):
                for j in range(0, 128):
                    with block({vi(0, 128): i, vj(0, 128): j}, B[vi: vi + 1, vj: vj + 1],
                               C[vi: vi + 1, vj: vj + 1],
                               name="C"):
                        C[vi, vj] = B[vi, vj] + 1


def test_module_class_based():
    mod = MyModule()
    rt_mod = from_source(tvm.tir.hybrid.ashybrid(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod)


if __name__ == '__main__':
    test_functions()
    test_module_class_based()
