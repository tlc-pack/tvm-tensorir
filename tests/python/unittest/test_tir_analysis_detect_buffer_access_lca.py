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
from tvm import tir
from tvm.script import ty


@tvm.script.tir
def buffer_laod_store_func(a: ty.handle, b: ty.handle, c: ty.handle, d: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.match_buffer(b, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    D = tir.match_buffer(d, (128, 128), "float32")
    with tir.block([128, 128]) as [i, j]:
        A[i, j] = tir.float32(0)
    with tir.block([32, 32, tir.reduce_axis(0, 32)]) as [i, j, k]:
        with tir.init():
            for ii, jj in tir.grid(4, 4):
                B[i * 4 + ii, j * 4 + jj] = A[i * 4 + ii, j * 4 + jj]
        for ii, jj in tir.grid(4, 4):
            for kk in range(0, 4):
                B[i * 4 + ii, j * 4 + jj] += C[i * 4 + ii, k * 4 + kk]
            for kk in range(0, 4):
                B[i * 4 + ii, j * 4 + jj] += D[j * 4 + jj, k * 4 + kk] * C[i * 4 + ii, k * 4 + kk]


@tvm.script.tir
def buffer_opaque_access(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 16], "float32")
    B = tir.match_buffer(b, [16, 16], "float32")
    C = tir.match_buffer(c, [16, 16], "float32")

    with tir.block([]):
        tir.reads(A[0:16, 0:16])
        tir.writes(B[0:16, 0:16])
        for i, j in tir.grid(16, 16):
            tir.store(A.data, i * 16 + j, 1)
        for i in range(0, 16):
            for j in range(0, 16):
                tir.evaluate(tir.load("float32", A.data, i * 16 + j))
            for j in range(0, 16):
                tir.evaluate(
                    tir.tvm_fill_fragment(B.data, 16, 16, 16, 0, tir.float32(0), dtype="handle")
                )

    for i, j in tir.grid(16, 16):
        with tir.block([16, 16]) as [vi, vj]:
            tir.bind(vi, i)
            tir.bind(vj, j)
            C[vi, vj] = B[vi, vj]


def test_buffer_laod_store():
    func = buffer_laod_store_func
    A, B, C, D = [func.buffer_map[x] for x in func.params]
    lca = tir.analysis.detect_buffer_access_lca(func)

    # LCA of Buffer A is root
    root_block = func.body.block
    assert lca[A] == func.body.block

    # LCA of Buffer B is reduction block
    reduce_block = root_block.body[1].body.body.body.block
    assert lca[B] == reduce_block

    # LCA of Buffer C is the second loop kk
    loop_jj = reduce_block.body.body
    assert lca[C] == loop_jj

    # LCA of Buffer D is loop jj
    loop_kk = loop_jj.body[1]
    assert lca[D] == loop_kk


def test_opaque_access():
    func = buffer_opaque_access
    A, B, C = [func.buffer_map[x] for x in func.params]
    lca = tir.analysis.detect_buffer_access_lca(func)

    # LCA of Buffer B is root
    root_block = func.body.block
    assert lca[B] == func.body.block

    # LCA of Buffer A is the opaque block
    opaque_block = root_block.body[0].block
    print(opaque_block)
    print(lca[A])
    assert lca[A] == opaque_block

    # LCA of Buffer C is the correspond block
    assert lca[C] == root_block.body[1].body.body.block


if __name__ == "__main__":
    # test_buffer_laod_store()
    test_opaque_access()
