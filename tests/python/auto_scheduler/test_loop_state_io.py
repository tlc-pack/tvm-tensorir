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
""" Testing tvm.auto_scnheduler.LoopState for creation, dumping to human readable format. """
import tvm
from tvm.hybrid import ty

# pylint: disable=invalid-name,line-too-long,undefined-variable


@tvm.hybrid.script
def _matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = buffer_bind(a, (1024, 1024), "float32")
    B = buffer_bind(b, (1024, 1024), "float32")
    C = buffer_bind(c, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))
    with tir.block({}, reads=[A[0: 1024, 0: 1024], B[0: 1024, 0: 1024]], writes=C[0: 1024, 0: 1024],
                   name="root"):
        for i in tir.grid(0, 1024):
            for j in tir.grid(0, 1024):
                for k in tir.grid(0, 1024):
                    with tir.block({vi(0, 1024): i, vj(0, 1024): j, vk(0, 1024, iter_type="reduce"): k},
                                   reads=[C[vi, vj], A[vi, vk],
                                          B[vj, vk]], writes=[C[vi, vj]],
                                   name="C"):
                        reducer.step(C[vi, vj], A[vi, vk] * B[vj, vk])


@tvm.hybrid.script
def _matmul_packed(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = buffer_bind(a, (1024, 1024), "float32")
    B = buffer_bind(b, (1024, 1024), "float32")
    C = buffer_bind(c, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    with tir.block({}, reads=[A[0: 1024, 0: 1024], B[0: 1024, 0: 1024]], writes=C[0: 1024, 0: 1024],
                   name="root"):
        packedB = tir.buffer_allocate((1024 // 32, 1024, 32), "float32")
        for i in tir.grid(0, 1024 // 32):
            for j in tir.grid(0, 1024):
                for k in tir.grid(0, 32):
                    with tir.block({vi(0, 1024 // 32): i, vj(0, 1024): j, vk(0, 32): k},
                                   reads=B[vj: vj + 1, vi * 32 +
                                           vk: vi * 32 + vk + 1],
                                   writes=packedB[vi: vi + 1, vj: vj + 1, vk: vk + 1], name="packed"):
                        packedB[vi, vj, vk] = B[vj, vi * 32 + vk]

        for i in tir.grid(0, 1024):
            for j in tir.grid(0, 1024):
                for k in tir.grid(0, 1024):
                    with tir.block({vi(0, 1024): i, vj(0, 1024): j, vk(0, 1024, iter_type="reduce"): k},
                                   reads=[C[vi: vi + 1, vj: vj + 1], A[vi: vi + 1, vk: vk + 1],
                                          packedB[vj // 32: vj // 32 + 1, vk: vk + 1, vj % 32: vj % 32 + 1]],
                                   writes=[C[vi: vi + 1, vj: vj + 1]], name="C"):
                        reducer.step(C[vi, vj], A[vi, vk] *
                                     packedB[vj // 32, vk, vj % 32])


@tvm.hybrid.script
def _fused_ewise(a: ty.handle, c: ty.handle) -> None:
    A = buffer_bind(a, (128, 128), "float32")
    C = buffer_bind(c, (128, 128), "float32")

    with tir.block({}, A[0: 128, 0: 128], C[0: 128, 0: 128], name="root"):
        B = tir.buffer_allocate((128, 128), "float32")

        for i in tir.grid(0, 128 * 128):
            with tir.block({vi(0, 128): i // 128, vj(0, 128): i % 128},
                           reads=A[vi: vi + 1, vj: vj + 1], writes=B[vi: vi + 1, vj: vj + 1], name="B"):
                B[vi, vj] = A[vi, vj] * tir.float32(2.0)

        for j in tir.grid(0, 128 * 128):
            with tir.block({vi(0, 128): j // 128, vj(0, 128): j % 128},
                           reads=B[vi: vi + 1, vj: vj + 1], writes=C[vi: vi + 1, vj: vj + 1], name="C"):
                C[vi, vj] = B[vi, vj] + tir.float32(1.0)


@tvm.hybrid.script
def _split_ewise(a: ty.handle, c: ty.handle) -> None:
    A = buffer_bind(a, (128, 128), "float32")
    C = buffer_bind(c, (128, 128), "float32")

    with tir.block({}, A[0: 128, 0: 128], C[0: 128, 0: 128], name="root"):
        B = buffer_allocate((128, 128), "float32")

        for io in tir.grid(0, 8):
            for ii in tir.grid(0, 16):
                for j in tir.grid(0, 128):
                    with tir.block({vi(0, 128): io * 16 + ii, vj(0, 128): j},
                                   reads=A[vi: vi + 1, vj: vj +
                                           1], writes=B[vi: vi + 1, vj: vj + 1],
                                   name="B"):
                        B[vi, vj] = A[vi, vj] * tir.float32(2.0)

        for i in tir.grid(0, 128):
            for jo in tir.grid(0, 10):
                for ji in tir.grid(0, 13):
                    with tir.block({vi(0, 128): i, vj(0, 128): jo * 13 + ji},
                                   reads=B[vi: vi + 1, vj: vj +
                                           1], writes=C[vi: vi + 1, vj: vj + 1],
                                   predicate=jo * 13 + ji < 128, name="C"):
                        C[vi, vj] = B[vi, vj] + tir.float32(1.0)


@tvm.hybrid.script
def _split_fuse_ewise(a: ty.handle, c: ty.handle) -> None:
    C = buffer_bind(c, (128, 128), "float32")
    A = buffer_bind(a, (128, 128), "float32")
    with tir.block({}, writes=[C[0:128, 0:128]], reads=[A[0:128, 0:128]], name="root"):
        B = tir.buffer_allocate((128, 128), "float32")
        for i in tir.grid(0, 128):
            for j in tir.grid(0, 128):
                with tir.block({vi(0, 128): ((floordiv(i, 16) * 16) + floormod(i, 16)), vj(0, 128): j},
                               writes=[B[vi:(vi + 1), vj:(vj + 1)]],
                               reads=[A[vi:(vi + 1), vj:(vj + 1)]], name="B"):
                    B[vi, vj] = A[vi, vj] * 2.0
        for i in tir.grid(0, 128):
            for j in tir.grid(0, 130):
                with tir.block({vi(0, 128): i, vj(0, 128): ((floordiv(j, 13) * 13) + floormod(j, 13))},
                               writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                               reads=[B[vi:(vi + 1), vj:(vj + 1)]],
                               predicate=(((floordiv(j, 13) * 13) + floormod(j, 13)) < 128), name="C"):
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
for i data_par[0, 1024)
  for j data_par[0, 1024)
    for k reduce[0, 1024)
      C = ...
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
for i_outer data_par[0, 32)
  for j_outer data_par[0, 32)
    for k_outer reduce[0, 256)
      for k_inner reduce[0, 4)
        for i_inner data_par[0, 32)
          for j_inner data_par[0, 32)
            C = ...
""".strip()
    # decompose reduction
    sch.decompose_reduction(update, j_o)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(sch.func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i_outer special[0, 32)
  for j_outer_init data_par[0, 32)
    for i_inner_init data_par[0, 32)
      for j_inner_init data_par[0, 32)
        C = ...
  for j_outer data_par[0, 32)
    for k_outer reduce[0, 256)
      for k_inner reduce[0, 4)
        for i_inner data_par[0, 32)
          for j_inner data_par[0, 32)
            C = ...
""".strip()
    # reorder
    sch = tvm.tir.create_schedule(func)
    update = sch.get_block("C")
    i_o, j_o, k_o, k_i, i_i, j_i = sch.get_axes(update)
    sch.reorder(i_o, j_o, k_o, i_i, k_i, j_i)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(sch.func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i_outer data_par[0, 32)
  for j_outer data_par[0, 32)
    for k_outer reduce[0, 256)
      for i_inner data_par[0, 32)
        for k_inner reduce[0, 4)
          for j_inner data_par[0, 32)
            C = ...
""".strip()
    # decompose reduction
    sch.decompose_reduction(update, j_o)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(sch.func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i_outer special[0, 32)
  for j_outer_init data_par[0, 32)
    for i_inner_init data_par[0, 32)
      for j_inner_init data_par[0, 32)
        C = ...
  for j_outer data_par[0, 32)
    for k_outer reduce[0, 256)
      for i_inner data_par[0, 32)
        for k_inner reduce[0, 4)
          for j_inner data_par[0, 32)
            C = ...
""".strip()


def test_matmul_packed():
    func = _get_func_from_hybrid(_matmul_packed)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i data_par[0, 32)
  for j data_par[0, 1024)
    for k data_par[0, 32)
      packedB = ...
for i data_par[0, 1024)
  for j data_par[0, 1024)
    for k reduce[0, 1024)
      C = ...
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
for i data_par[0, 32)
  for j data_par[0, 1024)
    for k data_par[0, 32)
      packedB = ...
for i_outer data_par[0, 32)
  for j_outer data_par[0, 32)
    for k_outer reduce[0, 256)
      for i_inner data_par[0, 32)
        for k_inner reduce[0, 4)
          for j_inner data_par[0, 32)
            C = ...
""".strip()


def test_fuse_ewise():
    func = _get_func_from_hybrid(_fused_ewise)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i data_par[0, 16384)
  B = ...
for j data_par[0, 16384)
  C = ...
""".strip()


def test_split_ewise():
    func = _get_func_from_hybrid(_split_ewise)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for io data_par[0, 8)
  for ii data_par[0, 16)
    for j data_par[0, 128)
      B = ...
for i data_par[0, 128)
  for jo data_par[0, 10)
    for ji data_par[0, 13)
      C = ...
""".strip()


def test_split_fuse_ewise():
    func = _get_func_from_hybrid(_split_fuse_ewise)
    loop_tree = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    str_repr = _check_and_remove_first_line(str(loop_tree))
    assert str_repr == """
for i data_par[0, 128)
  for j data_par[0, 128)
    B = ...
for i data_par[0, 128)
  for j data_par[0, 130)
    C = ...
""".strip()


if __name__ == "__main__":
    test_matmul()
    test_matmul_packed()
    test_fuse_ewise()
    test_split_ewise()
    test_split_fuse_ewise()
