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
""" Testing tvm.auto_scheduler.AccessAnalysis. """
import tvm
from tvm.hybrid import ty
from tvm import tir


@tvm.hybrid.script
def _matmul_with_relu(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    A = tir.buffer_bind(a, (1024, 1024), "float32")
    B = tir.buffer_bind(b, (1024, 1024), "float32")
    D = tir.buffer_bind(d, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    with tir.block({},
                   writes=[D[0:1024, 0:1024]],
                   reads=[A[0:1024, 0:1024], B[0:1024, 0:1024]],
                   name="root"):
        C = tir.buffer_allocate((128, 128), "float32")

        for i, j, k in tir.grid(1024, 1024, 1024):
            with tir.block({vi(0, 1024): i, vj(0, 1024): j, vk(0, 1024, iter_type="reduce"): k},
                           writes=C[vi, vj], reads=[C[vi, vj], A[vi, vk],B[vj, vk]], name="C"):
                reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])

        for i, j in tir.grid(1024, 1024):
            with tir.block({vi(0, 1024): i, vj(0, 1024): j}, writes=D[vi, vj], reads=C[vi, vj], name="D"):
                D[vi, vj] = tir.max(C[vi, vj], 1.0)


def test_matmul_with_relu():
    module = tvm.hybrid.create_module({"hybrid_func": _matmul_with_relu})
    func = module["hybrid_func"]
    assert isinstance(func, tvm.tir.PrimFunc)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    analysis = tvm.auto_scheduler.access_analysis.analyze(loop_tree)
    # Check root
    root_pattern = analysis[loop_tree]
    assert isinstance(
        root_pattern, tvm.auto_scheduler.access_analysis.DummyAccessPattern)
    # Check C (matmul)
    c_pattern = analysis[loop_tree.children[0]]
    assert isinstance(
        c_pattern, tvm.auto_scheduler.access_analysis.LeafAccessPattern)
    assert c_pattern.num_stmts == 1
    assert c_pattern.has_branch == 0
    assert c_pattern.has_expensive_op == 0
    assert c_pattern.all_trivial_store == 1
    assert len(c_pattern.block_vars_in_trivial_store) == 2
    assert c_pattern.block_vars_in_trivial_store[0].name == "vi"
    assert c_pattern.block_vars_in_trivial_store[1].name == "vj"
    assert c_pattern.lsmap_exists == 0
    assert c_pattern.lsmap_surjective == 0
    assert c_pattern.lsmap_injective == 0
    assert c_pattern.lsmap_ordered == 0
    assert c_pattern.num_axes_reuse == 2
    # Check D (relu)
    d_pattern = analysis[loop_tree.children[1]]
    assert isinstance(
        d_pattern, tvm.auto_scheduler.access_analysis.LeafAccessPattern)
    assert d_pattern.num_stmts == 1
    assert d_pattern.has_branch == 0
    assert d_pattern.has_expensive_op == 0
    assert d_pattern.all_trivial_store == 1
    assert len(d_pattern.block_vars_in_trivial_store) == 2
    assert d_pattern.block_vars_in_trivial_store[0].name == "vi"
    assert d_pattern.block_vars_in_trivial_store[1].name == "vj"
    assert d_pattern.lsmap_exists == 1
    assert d_pattern.lsmap_surjective == 1
    assert d_pattern.lsmap_injective == 1
    assert d_pattern.lsmap_ordered == 1
    assert d_pattern.num_axes_reuse == 0


if __name__ == "__main__":
    test_matmul_with_relu()
