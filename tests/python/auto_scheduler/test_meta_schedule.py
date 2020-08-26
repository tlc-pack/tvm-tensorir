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
""" Testing tvm.auto_scheduler.MetaSchedule. """
import tvm
from tvm.hybrid import ty
from tvm import tir


@tvm.hybrid.script
def _matmul_with_relu(a: ty.handle, b: ty.handle, d: ty.handle) -> None:
    A = tir.buffer_bind(a, (1024, 1024), "float32")
    B = tir.buffer_bind(b, (1024, 1024), "float32")
    D = tir.buffer_bind(d, (1024, 1024), "float32")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    with tir.block():
        tir.block_name("root")
        C = tir.buffer_allocate((128, 128), "float32")

        for i, j, k in tir.grid(1024, 1024, 1024):
            with tir.block(1024, 1024, tir.reduce_axis(0, 1024)) as [vi, vj, vk]:
                tir.block_name("C")
                reducer.step(C[vi, vj], A[vi, vk] * B[vk, vj])

        for i, j in tir.grid(1024, 1024):
            with tir.block(1024, 1024) as [vi, vj]:
                tir.block_name("D")
                D[vi, vj] = tir.max(C[vi, vj], 1.0)


def _get_meta_schedule_from_hybrid(hybrid_func):
    module = tvm.hybrid.create_module({"hybrid_func": hybrid_func})
    func = module["hybrid_func"]
    assert isinstance(func, tvm.tir.PrimFunc)
    meta_ir = tvm.auto_scheduler.LoopTree.from_prim_func(func)
    return tvm.auto_scheduler.MetaSchedule(meta_ir)


def _check_and_remove_first_line(result):
    result = result.strip().splitlines()
    first_line = result[0].strip()
    assert first_line.startswith("LoopTree(")
    assert first_line.endswith("):")
    return "\n".join(map(str.rstrip, result[1:])).strip()


def test_creation():
    sch = _get_meta_schedule_from_hybrid(_matmul_with_relu)
    assert _check_and_remove_first_line(str(sch.meta_ir)) == """
for i data_par(1024)
  for j data_par(1024)
    for k reduce(1024)
      ReduceStep(C) from (A, B)
for i data_par(1024)
  for j data_par(1024)
    BufferStore(D) from (C)
""".strip()
    sch.meta_ir.validate()


def test_decl_int_var():
    sch = _get_meta_schedule_from_hybrid(_matmul_with_relu)
    i_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.1.extent")
    i_2_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.2.extent")
    i_3_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.3.extent")
    j_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.1.extent")
    j_2_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.2.extent")
    j_3_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.3.extent")
    k_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="k.1.extent")

    assert _check_and_remove_first_line(str(sch.meta_ir)) == """
for i data_par(1024)
  for j data_par(1024)
    for k reduce(1024)
      ReduceStep(C) from (A, B)
for i data_par(1024)
  for j data_par(1024)
    BufferStore(D) from (C)
""".strip()
    assert str(
        sch.declared_vars) == "[i.1.extent, i.2.extent, i.3.extent, j.1.extent, j.2.extent, j.3.extent, k.1.extent]"
    assert sch.declared_vars[0].same_as(i_1_extent)
    assert sch.declared_vars[1].same_as(i_2_extent)
    assert sch.declared_vars[2].same_as(i_3_extent)
    assert sch.declared_vars[3].same_as(j_1_extent)
    assert sch.declared_vars[4].same_as(j_2_extent)
    assert sch.declared_vars[5].same_as(j_3_extent)
    assert sch.declared_vars[6].same_as(k_1_extent)
    sch.meta_ir.validate()


def test_multi_level_tiling_without_reorder():
    sch = _get_meta_schedule_from_hybrid(_matmul_with_relu)
    i_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.1.extent")
    i_2_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.2.extent")
    i_3_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.3.extent")
    j_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.1.extent")
    j_2_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.2.extent")
    j_3_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.3.extent")
    k_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="k.1.extent")
    k_0_extent = sch.split_inner_to_outer(
        loop_id=2, factors=[None, k_1_extent], name_hint="k.0.extent")
    j_0_extent = sch.split_inner_to_outer(
        loop_id=1, factors=[None, j_1_extent, j_2_extent, j_3_extent], name_hint="j.0.extent")
    i_0_extent = sch.split_inner_to_outer(
        loop_id=0, factors=[None, i_1_extent, i_2_extent, i_3_extent], name_hint="i.0.extent")

    assert _check_and_remove_first_line(str(sch.meta_ir)) == """
for i.0 data_par(i.0.extent)
  for i.1 data_par(i.1.extent)
    for i.2 data_par(i.2.extent)
      for i.3 data_par(i.3.extent)
        for j.0 data_par(j.0.extent)
          for j.1 data_par(j.1.extent)
            for j.2 data_par(j.2.extent)
              for j.3 data_par(j.3.extent)
                for k.0 reduce(k.0.extent)
                  for k.1 reduce(k.1.extent)
                    ReduceStep(C) from (A, B)
for i data_par(1024)
  for j data_par(1024)
    BufferStore(D) from (C)
""".strip()
    assert str(
        sch.declared_vars) == "[i.1.extent, i.2.extent, i.3.extent, j.1.extent, j.2.extent, j.3.extent, k.1.extent, k.0.extent, j.0.extent, i.0.extent]"
    assert sch.declared_vars[0].same_as(i_1_extent)
    assert sch.declared_vars[1].same_as(i_2_extent)
    assert sch.declared_vars[2].same_as(i_3_extent)
    assert sch.declared_vars[3].same_as(j_1_extent)
    assert sch.declared_vars[4].same_as(j_2_extent)
    assert sch.declared_vars[5].same_as(j_3_extent)
    assert sch.declared_vars[6].same_as(k_1_extent)
    assert sch.declared_vars[7].same_as(k_0_extent)
    assert sch.declared_vars[8].same_as(j_0_extent)
    assert sch.declared_vars[9].same_as(i_0_extent)
    sch.meta_ir.validate()


def test_multi_level_tiling_with_reorder():
    sch = _get_meta_schedule_from_hybrid(_matmul_with_relu)
    i_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.1.extent")
    i_2_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.2.extent")
    i_3_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.3.extent")
    j_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.1.extent")
    j_2_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.2.extent")
    j_3_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.3.extent")
    k_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="k.1.extent")
    k_0_extent = sch.split_inner_to_outer(
        loop_id=2, factors=[None, k_1_extent], name_hint="k.0.extent")
    j_0_extent = sch.split_inner_to_outer(
        loop_id=1, factors=[None, j_1_extent, j_2_extent, j_3_extent], name_hint="j.0.extent")
    i_0_extent = sch.split_inner_to_outer(
        loop_id=0, factors=[None, i_1_extent, i_2_extent, i_3_extent], name_hint="i.0.extent")
    sch.reorder(after_ids=[0, 4, 1, 5, 8, 2, 6, 9, 3, 7])
    assert _check_and_remove_first_line(str(sch.meta_ir)) == """
for i.0 data_par(i.0.extent)
  for j.0 data_par(j.0.extent)
    for i.1 data_par(i.1.extent)
      for j.1 data_par(j.1.extent)
        for k.0 reduce(k.0.extent)
          for i.2 data_par(i.2.extent)
            for j.2 data_par(j.2.extent)
              for k.1 reduce(k.1.extent)
                for i.3 data_par(i.3.extent)
                  for j.3 data_par(j.3.extent)
                    ReduceStep(C) from (A, B)
for i data_par(1024)
  for j data_par(1024)
    BufferStore(D) from (C)
""".strip()
    assert str(
        sch.declared_vars) == "[i.1.extent, i.2.extent, i.3.extent, j.1.extent, j.2.extent, j.3.extent, k.1.extent, k.0.extent, j.0.extent, i.0.extent]"
    assert sch.declared_vars[0].same_as(i_1_extent)
    assert sch.declared_vars[1].same_as(i_2_extent)
    assert sch.declared_vars[2].same_as(i_3_extent)
    assert sch.declared_vars[3].same_as(j_1_extent)
    assert sch.declared_vars[4].same_as(j_2_extent)
    assert sch.declared_vars[5].same_as(j_3_extent)
    assert sch.declared_vars[6].same_as(k_1_extent)
    assert sch.declared_vars[7].same_as(k_0_extent)
    assert sch.declared_vars[8].same_as(j_0_extent)
    assert sch.declared_vars[9].same_as(i_0_extent)
    sch.meta_ir.validate()


def test_multi_level_tiling_with_fusion():
    sch = _get_meta_schedule_from_hybrid(_matmul_with_relu)
    i_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.1.extent")
    i_2_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.2.extent")
    i_3_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="i.3.extent")
    j_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.1.extent")
    j_2_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.2.extent")
    j_3_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="j.3.extent")
    k_1_extent = sch.decl_int_var(choices=[1, 2, 4], name_hint="k.1.extent")
    k_0_extent = sch.split_inner_to_outer(
        loop_id=2, factors=[None, k_1_extent], name_hint="k.0.extent")
    j_0_extent = sch.split_inner_to_outer(
        loop_id=1, factors=[None, j_1_extent, j_2_extent, j_3_extent], name_hint="j.0.extent")
    i_0_extent = sch.split_inner_to_outer(
        loop_id=0, factors=[None, i_1_extent, i_2_extent, i_3_extent], name_hint="i.0.extent")
    sch.reorder(after_ids=[0, 4, 1, 5, 8, 2, 6, 9, 3, 7])
    sch.cursor_move_offset(1)
    v_0_extent = sch.split_inner_to_outer(
        loop_id=1, factors=[j_0_extent, None])
    v_1_extent = sch.split_inner_to_outer(
        loop_id=0, factors=[i_0_extent, None])
    sch.reorder(after_ids=[0, 2, 1, 3])
    sch.cursor_move_offset(-1)
    sch.compute_at_offset(offset=1, loop_id=1)

    assert _check_and_remove_first_line(str(sch.meta_ir)) == """
for i.0 data_par(i.0.extent)
  for j.0 data_par(j.0.extent)
    for i.1 data_par(i.1.extent)
      for j.1 data_par(j.1.extent)
        for k.0 reduce(k.0.extent)
          for i.2 data_par(i.2.extent)
            for j.2 data_par(j.2.extent)
              for k.1 reduce(k.1.extent)
                for i.3 data_par(i.3.extent)
                  for j.3 data_par(j.3.extent)
                    ReduceStep(C) from (A, B)
    for i.1 data_par(v.1)
      for j.1 data_par(v.0)
        BufferStore(D) from (C)
""".strip()

    assert str(
        sch.declared_vars) == "[i.1.extent, i.2.extent, i.3.extent, j.1.extent, j.2.extent, j.3.extent, k.1.extent, k.0.extent, j.0.extent, i.0.extent, v.0, v.1]"
    assert sch.declared_vars[0].same_as(i_1_extent)
    assert sch.declared_vars[1].same_as(i_2_extent)
    assert sch.declared_vars[2].same_as(i_3_extent)
    assert sch.declared_vars[3].same_as(j_1_extent)
    assert sch.declared_vars[4].same_as(j_2_extent)
    assert sch.declared_vars[5].same_as(j_3_extent)
    assert sch.declared_vars[6].same_as(k_1_extent)
    assert sch.declared_vars[7].same_as(k_0_extent)
    assert sch.declared_vars[8].same_as(j_0_extent)
    assert sch.declared_vars[9].same_as(i_0_extent)
    assert sch.declared_vars[10].same_as(v_0_extent)
    assert sch.declared_vars[11].same_as(v_1_extent)
    sch.meta_ir.validate()


if __name__ == "__main__":
    test_creation()
    test_decl_int_var()
    test_multi_level_tiling_without_reorder()
    test_multi_level_tiling_with_reorder()
    test_multi_level_tiling_with_fusion()
