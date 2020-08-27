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
""" Testing tvm.auto_scheduler.LoopState for creation, dumping to human readable format. """
import tvm
from tvm.hybrid import ty
from tvm import tir

# pylint: disable=invalid-name,line-too-long,undefined-variable


@tvm.hybrid.script
def _matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.buffer_bind(a, [1024, 1024])
    B = tir.buffer_bind(b, [1024, 1024])
    C = tir.buffer_bind(c, [1024, 1024])
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    for i, j, k in tir.grid(1024, 1024, 1024):
        with tir.block("C", [1024, 1024, tir.reduce_axis(0, 1024)]) as [vi, vj, vk]:
            reducer.step(C[vi, vj], A[vi, vk] * B[vj, vk])


@tvm.hybrid.script
def _matmul_packed(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.buffer_bind(a, (1024, 1024), "float32")
    B = tir.buffer_bind(b, (1024, 1024), "float32")
    C = tir.buffer_bind(c, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    packedB = tir.buffer_allocate((1024 // 32, 1024, 32), "float32")
    for i, j, k in tir.grid(1024 // 32, 1024, 32):
        with tir.block("packed", [1024 // 32, 1024, 32]) as [vi, vj, vk]:
            packedB[vi, vj, vk] = B[vj, vi * 32 + vk]

    for i, j, k in tir.grid(1024, 1024, 1024):
        with tir.block("C", [1024, 1024, tir.reduce_axis(0, 1024)]) as [vi, vj, vk]:
            reducer.step(C[vi, vj], A[vi, vk] * packedB[vj // 32, vk, vj % 32])


@tvm.hybrid.script
def _fused_ewise(a: ty.handle, c: ty.handle) -> None:
    A = tir.buffer_bind(a, (128, 128), "float32")
    C = tir.buffer_bind(c, (128, 128), "float32")

    B = tir.buffer_allocate((128, 128), "float32")

    for i in range(0, 128 * 128):
        with tir.block("B", [128, 128]) as [vi, vj]:
            tir.bind(vi, i // 128)
            tir.bind(vj, i % 128)
            B[vi, vj] = A[vi, vj] * tir.float32(2.0)

    for j in range(0, 128 * 128):
        with tir.block("C", [128, 128]) as [vi, vj]:
            tir.bind(vi, j // 128)
            tir.bind(vj, j % 128)
            C[vi, vj] = B[vi, vj] + tir.float32(1.0)


@tvm.hybrid.script
def _split_ewise(a: ty.handle, c: ty.handle) -> None:
    A = tir.buffer_bind(a, (128, 128), "float32")
    C = tir.buffer_bind(c, (128, 128), "float32")
    B = tir.buffer_allocate((128, 128), "float32")

    for io, ii, j in tir.grid(8, 16, 128):
        with tir.block("B", [128, 128]) as [vi, vj]:
            tir.bind(vi, io * 16 + ii)
            tir.bind(vj, j)
            B[vi, vj] = A[vi, vj] * tir.float32(2.0)

    for i, jo, ji in tir.grid(128, 10, 13):
        with tir.block("C", [128, 128]) as [vi, vj]:
            tir.where(jo * 13 + ji < 128)
            tir.bind(vi, i)
            tir.bind(vj, jo * 13 + ji)
            C[vi, vj] = B[vi, vj] + tir.float32(1.0)


@tvm.hybrid.script
def _split_fuse_ewise(a: ty.handle, c: ty.handle) -> None:
    C = tir.buffer_bind(c, (128, 128), "float32")
    A = tir.buffer_bind(a, (128, 128), "float32")

    B = tir.buffer_allocate((128, 128), "float32")
    for i, j in tir.grid(128, 128):
        with tir.block("B", [128, 128]) as [vi, vj]:
            tir.bind(vi, ((tir.floordiv(i, 16) * 16) + tir.floormod(i, 16)))
            tir.bind(vj, j)
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in tir.grid(128, 130):
        with tir.block("C", [128, 128]) as [vi, vj]:
            tir.where((((tir.floordiv(j, 13) * 13) + tir.floormod(j, 13)) < 128))
            tir.bind(vi, i)
            tir.bind(vj, ((tir.floordiv(j, 13) * 13) + tir.floormod(j, 13)))
            C[vi, vj] = B[vi, vj] + 1.0


# pylint: enable=invalid-name,line-too-long,undefined-variable


def _get_func_from_hybrid(hybrid_func):
    module = tvm.hybrid.create_module({"hybrid_func": hybrid_func})
    func = module["hybrid_func"]
    assert isinstance(func, tvm.tir.PrimFunc)
    return func


def _check_and_remove_first_line(result):
    result = result.strip().splitlines()
    first_line = result[0].strip()
    assert first_line.startswith("LoopTree(")
    assert first_line.endswith("):")
    return "\n".join(map(str.rstrip, result[1:])).strip()


def test_matmul():
    func = _get_func_from_hybrid(_matmul)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i data_par(1024)
  for j data_par(1024)
    for k reduce(1024)
      ReduceStep(C) from (A, B)
""".strip()
    # schedule: blocking
    sch = tvm.tir.create_schedule(func)
    update = sch.get_block("C")
    i, j, k = sch.get_axes(update)
    i_o, i_i = sch.split(i, 32)
    j_o, j_i = sch.split(j, 32)
    k_o, k_i = sch.split(k, 4)
    sch.reorder(i_o, j_o, k_o, k_i, i_i, j_i)
    func = sch.func
    # check again
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(sch.func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i_outer data_par(32)
  for j_outer data_par(32)
    for k_outer reduce(256)
      for k_inner reduce(4)
        for i_inner data_par(32)
          for j_inner data_par(32)
            ReduceStep(C) from (A, B)
""".strip()
    # decompose reduction
    sch.decompose_reduction(update, j_o)
    # print(tvm.hybrid.ashybrid(sch.func))
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(sch.func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i_outer special(32)
  for j_outer_init data_par(32)
    for i_inner_init data_par(32)
      for j_inner_init data_par(32)
        BufferStore(C) from ()
  for j_outer data_par(32)
    for k_outer reduce(256)
      for k_inner reduce(4)
        for i_inner data_par(32)
          for j_inner data_par(32)
            BufferStore(C) from (A, B, C)
""".strip()
    # reorder
    sch = tvm.tir.create_schedule(func)
    update = sch.get_block("C")
    i_o, j_o, k_o, k_i, i_i, j_i = sch.get_axes(update)
    sch.reorder(i_o, j_o, k_o, i_i, k_i, j_i)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(sch.func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i_outer data_par(32)
  for j_outer data_par(32)
    for k_outer reduce(256)
      for i_inner data_par(32)
        for k_inner reduce(4)
          for j_inner data_par(32)
            ReduceStep(C) from (A, B)
""".strip()
    # decompose reduction
    sch.decompose_reduction(update, j_o)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(sch.func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i_outer special(32)
  for j_outer_init data_par(32)
    for i_inner_init data_par(32)
      for j_inner_init data_par(32)
        BufferStore(C) from ()
  for j_outer data_par(32)
    for k_outer reduce(256)
      for i_inner data_par(32)
        for k_inner reduce(4)
          for j_inner data_par(32)
            BufferStore(C) from (A, B, C)
""".strip()


def test_matmul_packed():
    func = _get_func_from_hybrid(_matmul_packed)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i data_par(32)
  for j data_par(1024)
    for k data_par(32)
      BufferStore(packedB) from (B)
for i data_par(1024)
  for j data_par(1024)
    for k reduce(1024)
      ReduceStep(C) from (A, packedB)
""".strip()

    sch = tvm.tir.create_schedule(func)
    update = sch.get_block("C")
    i, j, k = sch.get_axes(update)
    i_o, i_i = sch.split(i, 32)
    j_o, j_i = sch.split(j, 32)
    k_o, k_i = sch.split(k, 4)
    sch.reorder(i_o, j_o, k_o, i_i, k_i, j_i)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(sch.func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i data_par(32)
  for j data_par(1024)
    for k data_par(32)
      BufferStore(packedB) from (B)
for i_outer data_par(32)
  for j_outer data_par(32)
    for k_outer reduce(256)
      for i_inner data_par(32)
        for k_inner reduce(4)
          for j_inner data_par(32)
            ReduceStep(C) from (A, packedB)
""".strip()


def test_fuse_ewise():
    func = _get_func_from_hybrid(_fused_ewise)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i data_par(16384)
  BufferStore(B) from (A)
for j data_par(16384)
  BufferStore(C) from (B)
""".strip()


def test_split_ewise():
    func = _get_func_from_hybrid(_split_ewise)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for io data_par(8)
  for ii data_par(16)
    for j data_par(128)
      BufferStore(B) from (A)
for i data_par(128)
  for jo data_par(10)
    for ji data_par(13)
      BufferStore(C) from (B)
""".strip()


def test_split_fuse_ewise():
    func = _get_func_from_hybrid(_split_fuse_ewise)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i data_par(128)
  for j data_par(128)
    BufferStore(B) from (A)
for i data_par(128)
  for j data_par(130)
    BufferStore(C) from (B)
""".strip()


if __name__ == "__main__":
    test_matmul()
    test_matmul_packed()
    test_fuse_ewise()
    test_split_ewise()
    test_split_fuse_ewise()
