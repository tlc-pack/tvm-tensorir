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

import tvm
from tvm import tir
from tvm.script import ty
from tvm.tir.schedule.state import CachedFlags
from tvm.tir.stmt_functor import post_order_visit

# pylint: disable=no-member,invalid-name,unused-variable


@tvm.script.tir
def elementwise(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    C = tir.match_buffer(c, (128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    with tir.block([128, 128], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0
    with tir.block([128, 128], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "init") as [vi, vj]:
            C[vi, vj] = tir.float32(0)
        for k in range(0, 128):
            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
def block_in_opaque_block(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128), "float32")
    B = tir.match_buffer(b, (128, 128), "float32")
    with tir.block([128], "B") as vi:
        tir.reads([A[0:128, 0:128]])
        tir.writes([B[0:128, 0:128]])
        B[vi, 0] = A[vi, 0]
        if A[vi, 0] == 0.0:
            with tir.block([], "C"):
                tir.reads([A[0:128, 0:128]])
                tir.writes([B[0:128, 0:128]])
                with tir.block([128], "D") as vj:
                    B[vi, vj] = A[vi, vj] * 3.0
        else:
            with tir.block([], "E"):
                tir.reads([A[0:128, 0:128]])
                tir.writes([B[0:128, 0:128]])
                with tir.block([128], "F") as vj:
                    B[vi, vj] = A[vi, vj] * 2.0


@tvm.script.tir
def write_after_read(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (128, 128))
    B = tir.match_buffer(b, (128, 128))
    C = tir.match_buffer(c, (128, 128))
    for i, j in tir.grid(128, 128):
        with tir.block([128, 128], "C") as [vi, vj]:
            C[vi, vj] = B[vi, vj] + 1.0
        with tir.block([128, 128], "B") as [vi, vj]:
            B[vi, vj] = A[vi, vj] * 2.0


# pylint: enable=no-member,invalid-name,unused-variable


def _get_block(s: tir.ScheduleState, name_hint: str) -> tir.StmtSRef:
    result = None

    def f_visit(node):
        nonlocal result
        if isinstance(node, tvm.tir.Block) and node.name_hint == name_hint:
            result = node

    func = s.mod["main"]
    post_order_visit(func.body, f_visit)
    assert result is not None and isinstance(result, tvm.tir.Block)
    return s.get_sref(result)


def test_cached_flag_elementwise():
    s = tir.ScheduleState(elementwise, debug_mode=True)
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: disable=protected-access


def test_cached_flag_matmul():
    s = tir.ScheduleState(matmul, debug_mode=True)
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "init")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "update")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: disable=protected-access


def test_cached_flag_block_in_opaque_block():
    s = tir.ScheduleState(block_in_opaque_block, debug_mode=True)
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "E")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "F")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    # pylint: disable=protected-access


def test_cached_flag_write_after_read():
    s = tir.ScheduleState(write_after_read, debug_mode=True)
    # pylint: disable=protected-access
    assert s._get_cached_flags(_get_block(s, "B")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "C")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=True,
    )
    assert s._get_cached_flags(_get_block(s, "root")) == CachedFlags(
        affine_binding=True,
        region_cover=True,
        stage_pipeline=False,
    )
    # pylint: disable=protected-access


if __name__ == "__main__":
    test_cached_flag_elementwise()
    test_cached_flag_matmul()
    test_cached_flag_block_in_opaque_block()
    test_cached_flag_write_after_read()
