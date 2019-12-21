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


def replace_ir_builder():
    m, n = 128, 128
    func, tensors, tensor_map, (A, B, C) = util.element_wise_stmt(m, n)
    s = te.create_schedule(func)
    ib = tvm.ir_builder.create()

    A._builder = ib
    B._builder = ib
    with ib.loop_range(0, m * n, name="i0") as fused:
        bv = ib.iter_var((0, m * n), name="vi0")
        v = bv.var
        i = tvm.truncdiv(v, n)
        j = tvm.truncmod(v, n)
        with ib.block([bv], [fused], A[i:i + 1, j:j + 1], B[i:i + 1, j:j + 1],
                name="B"):
            B[i, j] = A[i, j] * 2

    target = ib.get()

    return s, func, target, tensors, tensor_map


def replace_root_ir_builder():
    m, n = 128, 128
    func, tensors, tensor_map, (A, B, C) = util.element_wise_stmt(m, n)
    s = te.create_schedule(func)
    ib = tvm.ir_builder.create()

    A._builder = ib
    C._builder = ib
    dom_i = tvm.make.range_by_min_extent(0, m)
    dom_j = tvm.make.range_by_min_extent(0, n)
    with ib.block([], [], A[0:m, 0:n], C[0:m, 0:n], name="root"):
        with ib.loop_range(0, m, name="i") as i:
            with ib.loop_range(0, n, name="j") as j:
                bv_i = ib.iter_var(dom_i, name="vi")
                bv_j = ib.iter_var(dom_j, name="vj")
                vi = bv_i.var
                vj = bv_j.var
                with ib.block([bv_i, bv_j], [i, j], A[vi:vi + 1, vj:vj + 1], B[vi:vi + 1, vj:vj + 1],
                        name="B"):
                    C[vi, vj] = A[vi, vj] * 2 + 1

    target = ib.get()

    return s, func, target, tensors, tensor_map


def test_replace_direct_write():
    s, func, target, tensors, tensor_map = replace_ir_builder()

    old_hash = s.func.__hash__()
    sref = s.get_sref(s.func.body.body.seq[0])
    s.replace(sref, target)
    assert old_hash == s.func.__hash__()

    util.check_correctness(func, s.func, tensors, tensor_map)


def test_replace_copy():
    s, func, target, tensors, tensor_map = replace_ir_builder()

    old_hash = s.func.__hash__()
    func_ref = s.func
    sref = s.get_sref(s.func.body.body.seq[0])
    s.replace(sref, target)
    assert old_hash != s.func.__hash__()
    assert old_hash == func_ref.__hash__()

    util.check_correctness(func, s.func, tensors, tensor_map)


def test_replace_partial_copy():
    s, func, target, tensors, tensor_map = replace_ir_builder()

    old_hash = s.func.__hash__()
    func_ref = s.func.body.body
    ref_old_hash = func_ref.__hash__()
    sref = s.get_sref(s.func.body.body.seq[0])
    other_part_hash = s.func.body.body.seq[1].__hash__()
    s.replace(sref, target)
    assert old_hash == s.func.__hash__()
    assert ref_old_hash != s.func.body.body.__hash__()
    assert other_part_hash == s.func.body.body.seq[1].__hash__()

    util.check_correctness(func, s.func, tensors, tensor_map)


def test_replace_root_write():
    s, func, target, tensors, tensor_map = replace_root_ir_builder()

    old_hash = s.func.__hash__()
    sref = s.get_sref(s.func.body)
    s.replace(sref, target)
    assert old_hash == s.func.__hash__()
    assert isinstance(s.func.body.body, tvm.stmt.Loop)

    util.check_correctness(func, s.func, tensors, tensor_map)


def test_replace_root_copy():
    s, func, target, tensors, tensor_map = replace_root_ir_builder()

    old_hash = s.func.__hash__()
    func_ref = s.func
    sref = s.get_sref(s.func.body)
    s.replace(sref, target)
    assert old_hash != s.func.__hash__()
    assert old_hash == func_ref.__hash__()

    util.check_correctness(func, s.func, tensors, tensor_map)


def test_replace_remove_sref():
    s, func, target, tensors, tensor_map = replace_root_ir_builder()

    sref = s.get_sref(s.func.body.body.seq[0])
    s.replace(s.get_sref(s.func.body), target)

    assert te.get_stmt(sref) is None

    util.check_correctness(func, s.func, tensors, tensor_map)


if __name__ == "__main__":
    test_replace_direct_write()
    test_replace_copy()
    test_replace_partial_copy()
    test_replace_root_write()
    test_replace_root_copy()
    test_replace_remove_sref()
