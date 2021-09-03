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
from tvm._ffi.base import TVMError
from tvm.script import ty


@tvm.script.tir
def elementwise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [4, 4], 'float32')
    B = tir.match_buffer(b, [4, 4], 'float32')

    for i in tir.serial(2, 4):
        for j in tir.serial(1, 4):
            with tir.block([2, 3], 'B') as [vi, vj]:
                tir.bind(vi, i - 2)
                tir.bind(vj, j - 1)
                B[vi, vj] = A[vi, vj]


@tvm.script.tir
def elementwise_normalized(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [4, 4], 'float32')
    B = tir.match_buffer(b, [4, 4], 'float32')

    with tir.block([2, 3], 'B') as [vi, vj]:
        B[vi, vj] = A[vi, vj]


@tvm.script.tir
def multilevel(a: ty.handle, offset: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [100], 'float32')
    Off = tir.match_buffer(offset, [10], 'int32')
    B = tir.match_buffer(b, [9], 'float32')

    for i in tir.serial(1, 10):
        with tir.block([9], 'i') as [vi]:
            tir.bind(vi, i - 1)
            with tir.init():
                B[vi] = 0.
            with tir.block([tir.reduce_axis(Off[vi], Off[vi + 1])], 'j') as [vj]:
                B[vi] = B[vi] + A[vj]


@tvm.script.tir
def multilevel_normalized(a: ty.handle, offset: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [100], 'float32')
    Off = tir.match_buffer(offset, [10], 'int32')
    B = tir.match_buffer(b, [9], 'float32')

    for i in tir.serial(0, 9):
        with tir.block([9], 'i') as [vi]:
            tir.bind(vi, (i + 1) - 1)
            with tir.init():
                B[vi] = 0.
            for j in tir.serial(0, Off[vi + 1] - Off[vi]):
                with tir.block([tir.reduce_axis(Off[vi], Off[vi + 1])], 'j') as [vj]:
                    tir.bind(vj, j + Off[vi])
                    B[vi] = B[vi] + A[vj]


@tvm.script.tir
def spmm(a_indptr: ty.handle, a_indices: ty.handle, a_data: ty.handle, b: ty.handle, c: ty.handle) -> None:
    m = tir.var('int32')
    k = tir.var('int32')
    n = tir.var('int32')
    nnz = tir.var('int32')
    indptr = tir.match_buffer(a_indptr, [m + 1], 'int32')
    indices = tir.match_buffer(a_indices, [nnz], 'int32')
    A = tir.match_buffer(a_data, [nnz], 'float32')
    B = tir.match_buffer(b, [m, n], 'float32')
    C = tir.match_buffer(c, [k, n], 'float32')
    with tir.block([m, k], 'spmm_outer') as [vi, vj]:
        with tir.init():
            C[vi, vj] = 0.
        with tir.block([tir.reduce_axis(indptr[vi], indptr[vi + 1])], 'spmm_inner') as [vk]:
            C[vi, vj] = C[vi, vj] + A[vk] * B[indices[vk], vj]


@tvm.script.tir
def spmm_normalized(a_indptr: ty.handle, a_indices: ty.handle, a_data: ty.handle, b: ty.handle, c: ty.handle) -> None:
    m = tir.var('int32')
    k = tir.var('int32')
    n = tir.var('int32')
    nnz = tir.var('int32')
    indptr = tir.match_buffer(a_indptr, [m + 1], 'int32')
    indices = tir.match_buffer(a_indices, [nnz], 'int32')
    A = tir.match_buffer(a_data, [nnz], 'float32')
    B = tir.match_buffer(b, [m, n], 'float32')
    C = tir.match_buffer(c, [k, n], 'float32')
    with tir.block([m, k], 'spmm_outer') as [vi, vj]:
        with tir.init():
            C[vi, vj] = 0.
        for k in tir.serial(0, indptr[vi + 1] - indptr[vi]):
            with tir.block([tir.reduce_axis(indptr[vi], indptr[vi + 1])], 'spmm_inner') as [vk]:
                tir.bind(vk, k + indptr[vi])
                C[vi, vj] = C[vi, vj] + A[vk] * B[indices[vk], vj]


def test_elementwise():
    sch = tir.Schedule(elementwise, debug_mode=True, traced=True)
    i, j = sch.get_loops(sch.get_block('B'))
    sch.normalize(i, j)
    f = tvm.tir.transform.Simplify()(sch.mod)['main']
    tvm.ir.assert_structural_equal(elementwise_normalized, f)


def test_multi_level():
    sch = tir.Schedule(multilevel, debug_mode=True, traced=True)
    blk_j = sch.get_block('j')
    j, = sch.get_loops(blk_j)
    blk_i = sch.get_block('i')
    i, = sch.get_loops(blk_i)
    sch.normalize(i)
    sch.normalize(j)
    f = sch.mod['main']
    tvm.ir.assert_structural_equal(multilevel_normalized, f)


def test_spmm():
    sch = tir.Schedule(spmm, debug_mode=True)
    blk_outer = sch.get_block('spmm_outer')
    blk_inner = sch.get_block('spmm_inner')
    i, j = sch.get_loops(blk_outer)
    k, = sch.get_loops(blk_inner)
    try:
        sch.normalize(i, k)
    except TVMError:
        pass
    else:
        assert "Should throw error"
    sch.normalize(k)
    tvm.ir.assert_structural_equal(spmm_normalized, sch.mod['main'])


if __name__ == "__main__":
    test_elementwise()
    test_multi_level()
    test_spmm()
