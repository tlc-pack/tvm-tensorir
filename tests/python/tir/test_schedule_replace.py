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
# pylint: disable=missing-function-docstring,missing-module-docstring

import gc

import tvm
from tvm import tir

import util


def replace_ir_builder(deep_copy=False, realize=False):
    func = util.element_wise_stmt()
    new_func = tvm.script.from_source(tvm.script.asscript(func))
    s = tir.ScheduleState(new_func, debug_mode=True)
    # The target stmt
    target = tvm.tir.Block([], [], [], [], {}, [], "", "target", s.func.body.block.body[1])
    if realize:
        target = tvm.tir.BlockRealize([], 1, target)
    if deep_copy:
        target.__setstate__(target.__getstate__())

    # It's important to collect garbage explicitly to make
    # sure that there is only one reference of the function
    gc.collect()
    return s, target


def replace_ir_builder_with_opaque():
    pass  # TODO
    # func = tvm.script.from_source(tvm.script.asscript(util.block_in_opaque_block))
    # s = tir.Schedule(func, debug_mode=True)
    # gc.collect()
    # return s


def test_replace_direct_write0():
    s, target = replace_ir_builder(realize=True)

    old_hash = s.func.__hash__()
    sref = s.get_sref(s.func.body.block.body[1])
    s.replace(sref, target)

    # There is no other reference so the AST node can be write directly
    assert old_hash == s.func.__hash__()
    # Check the replaced part is equal to the target
    tvm.ir.assert_structural_equal(s.func.body.block.body[1], target)
    # The target reuse the stmt of the sref, so the sref won't be none
    assert sref.stmt is not None


def test_replace_direct_write1():
    s, target = replace_ir_builder(realize=True)

    old_hash = s.func.body.block.body.__hash__()
    hold_ref = s.func.body.block.body[1]
    sref = s.get_sref(s.func.body.block.body[1])
    s.replace(sref, target)

    # There is no other reference so the AST node can be write directly
    assert old_hash == s.func.body.block.body.__hash__()
    assert not tvm.ir.structural_equal(hold_ref.body, target)
    # Check the replaced part is equal to the target
    tvm.ir.assert_structural_equal(s.func.body.block.body[1], target)
    # The target reuse the sref's stmt, so the sref won't be none
    assert sref.stmt is not None


def test_replace_copy():
    s, target = replace_ir_builder(deep_copy=True, realize=True)

    old_hash = s.func.__hash__()
    # We hold another reference of func
    old_func = s.func
    sref = s.get_sref(s.func.body.block.body[0])
    s.replace(sref, target)

    # We need to copy the whole func to remain the old_func unchanged
    assert old_hash != s.func.__hash__()
    assert not tvm.ir.structural_equal(old_func.body, s.func.body)
    assert old_hash == old_func.__hash__()
    # Check the replaced part is equal to the target
    tvm.ir.assert_structural_equal(s.func.body.block.body[0], target)
    # The replaced AST node will be deleted, so the ref will be None
    assert sref.stmt is None


def test_replace_partial_copy0():
    s, target = replace_ir_builder(deep_copy=True, realize=True)

    func_old_hash = s.func.__hash__()
    hold_ref = s.func.body.block.body[0]
    ref_old_hash = hold_ref.__hash__()
    sref = s.get_sref(s.func.body.block.body[0].body)
    other_part_hash = s.func.body.block.body[1].__hash__()
    s.replace(sref, target)

    # The hold stmt will not change but copy a new one
    assert ref_old_hash != s.func.body.block.body[0].__hash__()
    assert not tvm.ir.structural_equal(hold_ref.body, target)
    # The function and the other part stmt can be directly write
    assert func_old_hash == s.func.__hash__()
    assert other_part_hash == s.func.body.block.body[1].__hash__()
    # Check the replaced part is equal to the target
    tvm.ir.assert_structural_equal(s.func.body.block.body[0].body, target)
    # The replaced AST node will be deleted, so the ref will be None
    assert sref.stmt is None


def test_replace_partial_copy1():
    s, target = replace_ir_builder(deep_copy=True)

    func_old_hash = s.func.__hash__()
    hold_ref = s.func.body.block.body[0].body
    stmt_old_hash = s.func.body.block.body[0].__hash__()
    sref = s.get_sref(s.func.body.block.body[0].body.body.block)
    other_part_hash = s.func.body.block.body[1].__hash__()
    s.replace(sref, target)

    # The father stmt will change since there is only one reference
    assert stmt_old_hash == s.func.body.block.body[0].__hash__()
    assert not tvm.ir.structural_equal(hold_ref.body, target)
    # The function and the other part stmt can be directly write
    assert func_old_hash == s.func.__hash__()
    assert other_part_hash == s.func.body.block.body[1].__hash__()
    # Check the replaced part is equal to the target
    tvm.ir.assert_structural_equal(s.func.body.block.body[0].body.body.block, target)
    # The replaced AST node will be deleted, so the ref will be None
    assert sref.stmt is None


def test_replace_root_write():
    s, target = replace_ir_builder()

    old_hash = s.func.__hash__()
    sref = s.get_sref(s.func.body.block)
    s.replace(sref, target)
    # Check no copy and the new body equals to target
    assert old_hash == s.func.__hash__()
    tvm.ir.assert_structural_equal(s.func.body.block, target)


def test_replace_root_copy0():
    s, target = replace_ir_builder(deep_copy=True)

    old_hash = s.func.__hash__()
    func_ref = s.func
    sref = s.get_sref(s.func.body.block)
    s.replace(sref, target)
    # Check the new body equals to target
    assert old_hash != s.func.__hash__()
    tvm.ir.assert_structural_equal(s.func.body.block, target)
    # Check the original func remains unchanged
    assert old_hash == func_ref.__hash__()
    assert not tvm.ir.structural_equal(func_ref.body, target)


def test_replace_root_copy1():
    s, target = replace_ir_builder(deep_copy=True, realize=True)

    old_hash = s.func.body.block.__hash__()
    func_ref = s.func.body.block
    sref = s.get_sref(s.func.body.block.body[0])
    s.replace(sref, target)
    # Check the new body equals to target
    assert old_hash != s.func.body.block.__hash__()
    tvm.ir.assert_structural_equal(s.func.body.block.body[0], target)
    # Check the original func remains unchanged
    assert old_hash == func_ref.__hash__()
    assert not tvm.ir.structural_equal(func_ref.body, target)


def test_replace_block_remap():
    func = util.element_wise_stmt()
    s = tir.Schedule(func, debug_mode=True)
    # The target stmt
    target = util.matmul_stmt_original().body.block.body.body.body[0].block
    sref = s.state.get_sref(s.module.body.block.body[0].body.body.block)
    s.state.replace(sref, target, {target: s.module.body.block.body[0].body.body.block})
    sref_new = s.get_block("init")
    # Check the original sref has been remapped
    assert sref.__hash__() == sref_new.__hash__()
    tvm.ir.assert_structural_equal(sref.stmt, target)


def test_replace_block_in_opaque_block():
    pass  # TODO
    # s = replace_ir_builder_with_opaque()
    # root_hash = s.func.__hash__()
    # for_loop = s.func.body.block.body.body.block.body[1].then_case.block.body
    # sref = s.get_sref(for_loop)
    # new_for_loop = tir.Loop(
    #     loop_var=for_loop.loop_var,
    #     min_val=0,
    #     extent=128,
    #     annotations=[],
    #     body=tir.Evaluate(0),
    # )
    # s.replace(sref, new_for_loop)
    # assert root_hash == s.func.__hash__()
    # tvm.ir.assert_structural_equal(sref.stmt, new_for_loop)


if __name__ == "__main__":
    test_replace_direct_write0()
    test_replace_direct_write1()
    test_replace_copy()
    test_replace_partial_copy0()
    test_replace_partial_copy1()
    test_replace_root_write()
    test_replace_root_copy0()
    test_replace_root_copy1()
    test_replace_block_remap()
    test_replace_block_in_opaque_block()
