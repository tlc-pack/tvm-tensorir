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
from tvm.script import tir as T
import sys
import pytest


@tvm.script.ir_module
class Transpose:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for ty in T.thread_binding(8, thread="threadIdx.y"):
                with T.block():
                    A_shared_dyn = T.alloc_buffer([16, 128], dtype="float32", scope="shared.dyn")
                    with T.block("A_shared"):
                        T.block_attr({"auto_copy": 1})
                        for ax0, ax1 in T.grid(128, 16):
                            A_shared_dyn[ax1, ax0] = A[ax0, ax1]
                    with T.block("B"):
                        for ax1, ax0 in T.grid(16, 128):
                            T.block_attr({"auto_copy": 1})
                            B[ax1, ax0] = A_shared_dyn[ax1, ax0]


@tvm.script.ir_module
class GlobalToShared:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer([128, 128], dtype="float32", scope="shared.dyn")
                            with T.block("A_shared"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax0, ax1] = A[bx * 128 + ax0, by * 128 + ax1]
                            with T.block("B"):
                                for ax0, ax1 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]


@tvm.script.ir_module
class SharedToGlobal:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer([128, 128], dtype="float32", scope="shared.dyn")
                            with T.block("A_shared"):
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax1, ax0] = A[bx * 128 + ax0, by * 128 + ax1]
                            with T.block("B"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                for ax1, ax0 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax1, ax0]


@tvm.script.ir_module
class GlobalToSharedWithLocalStage:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer([128, 128], dtype="float32", scope="shared.dyn")
                            with T.block("A_shared"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16, "local_stage": True})
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax0, ax1] = A[bx * 128 + ax0, by * 128 + ax1]
                            with T.block("B"):
                                for ax0, ax1 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]


@tvm.script.ir_module
class SharedToWmma:
    @T.prim_func
    def main() -> None:
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer([128, 128], dtype="float16", scope="shared.dyn")
                            A_wmma = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_a")
                            with T.block("A_wmma"):
                                T.block_attr({"auto_copy": 1})
                                for ax0, ax1 in T.grid(128, 128):
                                    A_wmma[ax0, ax1] = A_shared_dyn[ax0, ax1]


@tvm.script.ir_module
class WmmaToShared:
    @T.prim_func
    def main() -> None:
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            C_accum = T.alloc_buffer([128, 128], dtype="float32", scope="wmma.accumulator")
                            C_shared = T.alloc_buffer([128, 128], dtype="float32", scope="shared.dyn")
                            with T.block("C_shared"):
                                T.block_attr({"auto_copy": 1})
                                for ax0, ax1 in T.grid(128, 128):
                                    C_shared[ax0, ax1] = C_accum[ax0, ax1]


@tvm.script.ir_module
class WmmaToGlobal:
    @T.prim_func
    def main(c: T.handle) -> None:
        C = T.match_buffer(c, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution": True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            C_accum = T.alloc_buffer([128, 128], dtype="float32", scope="wmma.accumulator")
                            with T.block("C_global"):
                                T.block_attr({"auto_copy": 1, "vector_bytes": 16})
                                for ax0, ax1 in T.grid(128, 128):
                                    C[bx * 128 + ax0, by * 128 + ax1] = C_accum[ax0, ax1]

@tvm.script.ir_module
class TransformedGlobalToShared:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution":True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer([128, 128], dtype="float32", strides=[128, 1], scope="shared.dyn")
                            with T.block("A_shared"):
                                T.block_attr({"auto_copy":1, "vector_bytes":16})
                                for outer in T.serial(16):
                                    for ty_1 in T.thread_binding(8, thread="threadIdx.y"):
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            for vec in T.vectorized(4):
                                                A_shared_dyn[(((outer * 8 + ty_1) * 32 + tx) * 4 + vec) // 128 % 128, (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) % 128] = A[bx * 128 + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) // 128 % 128, by * 128 + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) % 128]
                            with T.block("B"):
                                for ax0, ax1 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]

@tvm.script.ir_module
class TransformedSharedToGlobal:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution":True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer([128, 128], dtype="float32", strides=[129, 1], scope="shared.dyn")
                            with T.block("A_shared"):
                                T.reads(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                T.writes(A_shared_dyn[0 : 128, 0 : 128])
                                for ax0, ax1 in T.grid(128, 128):
                                    A_shared_dyn[ax1, ax0] = A[bx * 128 + ax0, by * 128 + ax1]
                            with T.block("B"):
                                T.block_attr({"auto_copy":1, "vector_bytes":16})
                                for outer in T.serial(16):
                                    for ty_1 in T.thread_binding(8, thread="threadIdx.y"):
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            for vec in T.vectorized(4):
                                                B[bx * 128 + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) // 128 % 128, by * 128 + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) % 128] = A_shared_dyn[(((outer * 8 + ty_1) * 32 + tx) * 4 + vec) % 128, (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) // 128 % 128]

@tvm.script.ir_module
class TransformedGlobalToSharedWithLocalStage:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [1024, 1024])
        B = T.match_buffer(b, [1024, 1024])
        with T.block("root"):
            T.block_attr({"warp_execution":True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer([128, 128], dtype="float32", strides=[128, 1], scope="shared.dyn")
                            with T.block("A_shared"):
                                T.reads(A[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                T.writes(A_shared_dyn[0 : 128, 0 : 128])
                                T.block_attr({"auto_copy":1, "local_stage":True, "vector_bytes":16})
                                A_local = T.alloc_buffer([16, 4], dtype="float32", scope="local")
                                for ty_1 in T.thread_binding(8, thread="threadIdx.y"):
                                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0, ax1, ax2, ax3, ax4 in T.grid(1, 16, 1, 1, 1):
                                            for vec in T.vectorized(4):
                                                A_local[ax0 * 16 + ax1 + ax2, (ax3 + ax4) * 4 + vec] = A[((bx % 8 + ax0) * 16 + ax1) * 8 + (ty_1 % 128 + ax2), ((by % 8 + ax3) * 32 + (tx % 32 + ax4)) * 4 + vec]
                                        for serial in T.serial(16):
                                            for vec in T.vectorized(4):
                                                A_shared_dyn[(((serial * 8 + ty_1) * 32 + tx) * 4 + vec) // 128 % 128, (((serial * 8 + ty_1) * 32 + tx) * 4 + vec) % 128] = A_local[(serial * 8 + (tx * 4 + vec) // 128 + ty_1) % 128 // 8 + (((tx * 4 + vec) // 128 + ty_1) % 8 - ty_1 % 128), ((tx * 4 + vec) % 128 // 4 - tx % 32) * 4 + vec % 4]
                            with T.block("B"):
                                for ax0, ax1 in T.grid(128, 128):
                                    B[bx * 128 + ax0, by * 128 + ax1] = A_shared_dyn[ax0, ax1]

@tvm.script.ir_module
class TransformedSharedToWmma:
    @T.prim_func
    def main() -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        # body
        with T.block("root"):
            T.block_attr({"warp_execution":True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            A_shared_dyn = T.alloc_buffer([128, 128], dtype="float16", strides=[136, 1], scope="shared.dyn")
                            A_wmma = T.alloc_buffer([128, 128], dtype="float16", scope="wmma.matrix_a")
                            with T.block("C_shared"):
                                T.reads(A_shared_dyn[0 : 128, 0 : 128])
                                T.writes(A_wmma[0 : 128, 0 : 128])
                                T.block_attr({"auto_copy":1})
                                for ax00, ax10 in T.grid(8, 8):
                                    with T.block("wmma_load"):
                                        T.reads(A_shared_dyn[ax00 * 16 : ax00 * 16 + 16, ax10 * 16 : ax10 * 16 + 16])
                                        T.writes(A_wmma[ax00 * 16 : ax00 * 16 + 16, ax10 * 16 : ax10 * 16 + 16])
                                        src = T.match_buffer(A_shared_dyn[ax00 * 16 : ax00 * 16 + 16, ax10 * 16 : ax10 * 16 + 16], [16, 16], dtype="float16", strides=[s1, s0], scope="shared.dyn", offset_factor=16)
                                        tgt = T.match_buffer(A_wmma[ax00 * 16 : ax00 * 16 + 16, ax10 * 16 : ax10 * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_a", offset_factor=16)
                                        T.evaluate(T.tvm_load_matrix_sync(tgt.data, 16, 16, 16, tgt.elem_offset // 256 + tgt.elem_offset % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float16"), src.data, src.elem_offset, s1 * 16, 1, dtype="handle"), s1, "row_major", dtype="handle"))

@tvm.script.ir_module
class TransformedWmmaToShared:
    @T.prim_func
    def main() -> None:
        s0 = T.var("int32")
        s1 = T.var("int32")
        # body
        with T.block("root"):
            T.block_attr({"warp_execution":True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            C_accum = T.alloc_buffer([128, 128], dtype="float32", scope="wmma.accumulator")
                            C_shared = T.alloc_buffer([128, 128], dtype="float32", strides=[136, 1], scope="shared.dyn")
                            with T.block("A_wmma"):
                                T.reads(C_accum[0 : 128, 0 : 128])
                                T.writes(C_shared[0 : 128, 0 : 128])
                                T.block_attr({"auto_copy":1})
                                for ax00, ax10 in T.grid(8, 8):
                                    with T.block("wmma_store"):
                                        T.reads(C_accum[ax00 * 16 : ax00 * 16 + 16, ax10 * 16 : ax10 * 16 + 16])
                                        T.writes(C_shared[ax00 * 16 : ax00 * 16 + 16, ax10 * 16 : ax10 * 16 + 16])
                                        src = T.match_buffer(C_accum[ax00 * 16 : ax00 * 16 + 16, ax10 * 16 : ax10 * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                                        tgt = T.match_buffer(C_shared[ax00 * 16 : ax00 * 16 + 16, ax10 * 16 : ax10 * 16 + 16], [16, 16], dtype="float32", strides=[s1, s0], scope="shared.dyn", offset_factor=16)
                                        T.evaluate(T.tvm_store_matrix_sync(src.data, 16, 16, 16, src.elem_offset // 256 + src.elem_offset % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float32"), tgt.data, tgt.elem_offset, s1 * 16, 2, dtype="handle"), s1, "row_major", dtype="handle"))

@tvm.script.ir_module
class TransformedWmmaToGlobal:
    @T.prim_func
    def main(c: T.handle) -> None:
        C = T.match_buffer(c, [1024, 1024])
        s0 = T.var("int32")
        s1 = T.var("int32")
        # body
        with T.block("root"):
            T.block_attr({"warp_execution":True})
            for bx in T.thread_binding(8, thread="blockIdx.x"):
                for by in T.thread_binding(8, thread="blockIdx.y"):
                    for ty in T.thread_binding(8, thread="threadIdx.y"):
                        with T.block():
                            C_accum = T.alloc_buffer([128, 128], dtype="float32", scope="wmma.accumulator")
                            with T.block("C_global"):
                                T.reads(C_accum[0 : 128, 0 : 128])
                                T.writes(C[bx * 128 : bx * 128 + 128, by * 128 : by * 128 + 128])
                                T.block_attr({"auto_copy":1, "vector_bytes":16})
                                C_shared_dyn = T.alloc_buffer([16, 128], dtype="float32", strides=[136, 1], scope="shared.dyn")
                                for ax00 in T.serial(8):
                                    for ax10 in T.serial(8):
                                        with T.block("wmma_store"):
                                            T.reads(C_accum[ax00 * 16 : ax00 * 16 + 16, ax10 * 16 : ax10 * 16 + 16])
                                            T.writes(C_shared_dyn[((ax00 // 8 + bx) % 8 - bx % 8 + (ax00 % 8 - ax00 % 64)) * 16 : ((ax00 // 8 + bx) % 8 - bx % 8 + (ax00 % 8 - ax00 % 64)) * 16 + 16, (((ax10 // 8 + by) % 8 - by % 8) * 8 + ax10 % 8) * 16 : (((ax10 // 8 + by) % 8 - by % 8) * 8 + ax10 % 8) * 16 + 16])
                                            src = T.match_buffer(C_accum[ax00 * 16 : ax00 * 16 + 16, ax10 * 16 : ax10 * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                                            tgt = T.match_buffer(C_shared_dyn[((ax00 // 8 + bx) % 8 - bx % 8 + (ax00 % 8 - ax00 % 64)) * 16 : ((ax00 // 8 + bx) % 8 - bx % 8 + (ax00 % 8 - ax00 % 64)) * 16 + 16, (((ax10 // 8 + by) % 8 - by % 8) * 8 + ax10 % 8) * 16 : (((ax10 // 8 + by) % 8 - by % 8) * 8 + ax10 % 8) * 16 + 16], [16, 16], dtype="float32", strides=[s1, s0], scope="shared.dyn", offset_factor=16)
                                            T.evaluate(T.tvm_store_matrix_sync(src.data, 16, 16, 16, src.elem_offset // 256 + src.elem_offset % 256 // 16, T.tvm_access_ptr(T.type_annotation(dtype="float32"), tgt.data, tgt.elem_offset, s1 * 16, 2, dtype="handle"), s1, "row_major", dtype="handle"))
                                    for outer in T.serial(2):
                                        for ty_1 in T.thread_binding(8, thread="threadIdx.y"):
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                for vec in T.vectorized(4):
                                                    C[((bx % 8 + 0) * 8 + (ax00 % 64 + 0)) * 16 + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) // 16 // 8 % 16, ((by % 8 + 0) * 8 + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) // 16 % 8) * 16 + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) % 16] = C_shared_dyn[(0 + 0) * 16 + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) // 16 // 8 % 16, (0 * 8 + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) // 16 % 8) * 16 + (((outer * 8 + ty_1) * 32 + tx) * 4 + vec) % 16]


def _check(original, transformed):
    mod = tvm.tir.transform.LowerAutoCopy()(original)
    tvm.ir.assert_structural_equal(mod, transformed, True)


def test_coalesce_vectorize():
    _check(GlobalToShared, TransformedGlobalToShared)


def test_inverse():
    _check(SharedToGlobal, TransformedSharedToGlobal)


def test_local_stage():
    _check(GlobalToSharedWithLocalStage, TransformedGlobalToSharedWithLocalStage)


def test_rewrite_shared_to_wmma():
    _check(SharedToWmma, TransformedSharedToWmma)


def test_rewrite_wmma_to_shared():
    _check(WmmaToShared, TransformedWmmaToShared)


def test_rewrite_wmma_to_global():
    _check(WmmaToGlobal, TransformedWmmaToGlobal)


def verify_single_allocation(stmt, alloc_size=None):
    num_alloc = [0]
    alloc_extents = []

    def verify(n):
        if (
            isinstance(n, tvm.tir.Allocate)
            and n.buffer_var.type_annotation.storage_scope == "shared.dyn"
        ):
            num_alloc[0] += 1
            alloc_extents.append(n.extents[0])

    tvm.tir.stmt_functor.post_order_visit(stmt, verify)
    assert num_alloc[0] == 1

    if alloc_size:
        assert alloc_extents[0] == alloc_size


def test_auto_padding():
    mod = tvm.tir.transform.LowerAutoCopy()(Transpose)
    mod = tvm.tir.transform.FlattenBuffer()(mod)
    verify_single_allocation(mod['main'].body, 16 * 130)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
