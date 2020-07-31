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
from tvm.tir.hybrid import ty
import util


def test_matmul():
    func = util.matmul_stmt_original()

    assert isinstance(func.body.block, tir.stmt.Block)
    assert isinstance(func.body.block.body, tir.stmt.Loop)
    assert isinstance(func.body.block.body.body, tir.stmt.Loop)
    assert isinstance(func.body.block.body.body.body, tir.stmt.SeqStmt)
    assert isinstance(func.body.block.body.body.body[0].block, tir.stmt.Block)
    assert isinstance(func.body.block.body.body.body[1], tir.stmt.Loop)
    assert isinstance(func.body.block.body.body.body[1].body.block, tir.stmt.Block)


def test_element_wise():
    func = util.element_wise_stmt()

    assert isinstance(func.body.block, tir.stmt.Block)
    assert isinstance(func.body.block.body, tir.stmt.SeqStmt)
    assert isinstance(func.body.block.body[0], tir.stmt.Loop)
    assert isinstance(func.body.block.body[0].body, tir.stmt.Loop)
    assert isinstance(func.body.block.body[0].body.body.block, tir.stmt.Block)

    assert isinstance(func.body.block.body[1], tir.stmt.Loop)
    assert isinstance(func.body.block.body[1].body, tir.stmt.Loop)
    assert isinstance(func.body.block.body[1].body.body.block, tir.stmt.Block)


def test_predicate():
    func = util.predicate_stmt()

    assert isinstance(func.body.block, tir.stmt.Block)
    assert isinstance(func.body.block.body, tir.stmt.Loop)
    assert isinstance(func.body.block.body.body, tir.stmt.Loop)
    assert isinstance(func.body.block.body.body.body, tir.stmt.Loop)
    assert isinstance(func.body.block.body.body.body.body.block, tir.stmt.Block)


@tvm.tir.hybrid.script
class Module:
    def default_function(args: ty.handle, arg_type_ids: ty.handle, num_args: ty.int32, out_ret_value: ty.handle, out_ret_tcode: ty.handle) -> ty.int32:
        # function attr dict
        tir.func_attr({"tir.is_entry_func":True, "global_symbol":"default_function", "calling_conv":1})
        # body
        tir.Assert((num_args == 3), "default_function: num_args should be 3")
        arg0: ty.handle = tir.tvm_struct_get(args, 0, 12, dtype="handle")
        arg0_code: ty.int32 = tir.load("int32", arg_type_ids, 0)
        arg1: ty.handle = tir.tvm_struct_get(args, 1, 12, dtype="handle")
        arg1_code: ty.int32 = tir.load("int32", arg_type_ids, 1)
        arg2: ty.handle = tir.tvm_struct_get(args, 2, 12, dtype="handle")
        arg2_code: ty.int32 = tir.load("int32", arg_type_ids, 2)
        A: ty.handle = tir.tvm_struct_get(arg0, 0, 1, dtype="handle")
        tir.attr(A, "storage_alignment", 128)
        arg0_shape: ty.handle = tir.tvm_struct_get(arg0, 0, 2, dtype="handle")
        arg0_strides: ty.handle = tir.tvm_struct_get(arg0, 0, 3, dtype="handle")
        dev_id: ty.int32 = tir.tvm_struct_get(arg0, 0, 9, dtype="int32")
        B: ty.handle = tir.tvm_struct_get(arg1, 0, 1, dtype="handle")
        tir.attr(B, "storage_alignment", 128)
        arg1_shape: ty.handle = tir.tvm_struct_get(arg1, 0, 2, dtype="handle")
        arg1_strides: ty.handle = tir.tvm_struct_get(arg1, 0, 3, dtype="handle")
        C: ty.handle = tir.tvm_struct_get(arg2, 0, 1, dtype="handle")
        tir.attr(C, "storage_alignment", 128)
        arg2_shape: ty.handle = tir.tvm_struct_get(arg2, 0, 2, dtype="handle")
        arg2_strides: ty.handle = tir.tvm_struct_get(arg2, 0, 3, dtype="handle")
        tir.Assert(((((arg0_code == 3) or (arg0_code == 13)) or (arg0_code == 7)) or (arg0_code == 4)), "default_function: Expect arg[0] to be pointer")
        tir.Assert(((((arg1_code == 3) or (arg1_code == 13)) or (arg1_code == 7)) or (arg1_code == 4)), "default_function: Expect arg[1] to be pointer")
        tir.Assert(((((arg2_code == 3) or (arg2_code == 13)) or (arg2_code == 7)) or (arg2_code == 4)), "default_function: Expect arg[2] to be pointer")
        tir.Assert((2 == tir.tvm_struct_get(arg0, 0, 4, dtype="int32")), "arg0.ndim is expected to equal 2")
        tir.Assert((2 == tir.tvm_struct_get(arg0, 0, 4, dtype="int32")), "arg0.ndim is expected to equal 2")
        tir.Assert((((tir.tvm_struct_get(arg0, 0, 5, dtype="uint8") == tir.uint8(2)) and (tir.tvm_struct_get(arg0, 0, 6, dtype="uint8") == tir.uint8(32))) and (tir.tvm_struct_get(arg0, 0, 7, dtype="uint16") == tir.uint16(1))), "arg0.dtype is expected to be float32")
        tir.Assert((1024 == tir.cast("int32", tir.load("int64", arg0_shape, 0))), "Argument arg0.shape[0] has an unsatisfied constraint")
        tir.Assert((1024 == tir.cast("int32", tir.load("int64", arg0_shape, 1))), "Argument arg0.shape[1] has an unsatisfied constraint")
        if not (tir.isnullptr(arg0_strides, dtype="bool")):
            tir.Assert(((1 == tir.cast("int32", tir.load("int64", arg0_strides, 1))) and (1024 == tir.cast("int32", tir.load("int64", arg0_strides, 0)))), "arg0.strides: expected to be compact array")
            tir.evaluate(0)
        tir.Assert((tir.uint64(0) == tir.tvm_struct_get(arg0, 0, 8, dtype="uint64")), "Argument arg0.byte_offset has an unsatisfied constraint")
        tir.Assert((1 == tir.tvm_struct_get(arg0, 0, 10, dtype="int32")), "Argument arg0.device_type has an unsatisfied constraint")
        tir.Assert((2 == tir.tvm_struct_get(arg1, 0, 4, dtype="int32")), "arg1.ndim is expected to equal 2")
        tir.Assert((2 == tir.tvm_struct_get(arg1, 0, 4, dtype="int32")), "arg1.ndim is expected to equal 2")
        tir.Assert((((tir.tvm_struct_get(arg1, 0, 5, dtype="uint8") == tir.uint8(2)) and (tir.tvm_struct_get(arg1, 0, 6, dtype="uint8") == tir.uint8(32))) and (tir.tvm_struct_get(arg1, 0, 7, dtype="uint16") == tir.uint16(1))), "arg1.dtype is expected to be float32")
        tir.Assert((1024 == tir.cast("int32", tir.load("int64", arg1_shape, 0))), "Argument arg1.shape[0] has an unsatisfied constraint")
        tir.Assert((1024 == tir.cast("int32", tir.load("int64", arg1_shape, 1))), "Argument arg1.shape[1] has an unsatisfied constraint")
        if not (tir.isnullptr(arg1_strides, dtype="bool")):
            tir.Assert(((1 == tir.cast("int32", tir.load("int64", arg1_strides, 1))) and (1024 == tir.cast("int32", tir.load("int64", arg1_strides, 0)))), "arg1.strides: expected to be compact array")
            tir.evaluate(0)
        tir.Assert((tir.uint64(0) == tir.tvm_struct_get(arg1, 0, 8, dtype="uint64")), "Argument arg1.byte_offset has an unsatisfied constraint")
        tir.Assert((1 == tir.tvm_struct_get(arg1, 0, 10, dtype="int32")), "Argument arg1.device_type has an unsatisfied constraint")
        tir.Assert((dev_id == tir.tvm_struct_get(arg1, 0, 9, dtype="int32")), "Argument arg1.device_id has an unsatisfied constraint")
        tir.Assert((2 == tir.tvm_struct_get(arg2, 0, 4, dtype="int32")), "arg2.ndim is expected to equal 2")
        tir.Assert((2 == tir.tvm_struct_get(arg2, 0, 4, dtype="int32")), "arg2.ndim is expected to equal 2")
        tir.Assert((((tir.tvm_struct_get(arg2, 0, 5, dtype="uint8") == tir.uint8(2)) and (tir.tvm_struct_get(arg2, 0, 6, dtype="uint8") == tir.uint8(32))) and (tir.tvm_struct_get(arg2, 0, 7, dtype="uint16") == tir.uint16(1))), "arg2.dtype is expected to be float32")
        tir.Assert((1024 == tir.cast("int32", tir.load("int64", arg2_shape, 0))), "Argument arg2.shape[0] has an unsatisfied constraint")
        tir.Assert((1024 == tir.cast("int32", tir.load("int64", arg2_shape, 1))), "Argument arg2.shape[1] has an unsatisfied constraint")
        if not (tir.isnullptr(arg2_strides, dtype="bool")):
            tir.Assert(((1 == tir.cast("int32", tir.load("int64", arg2_strides, 1))) and (1024 == tir.cast("int32", tir.load("int64", arg2_strides, 0)))), "arg2.strides: expected to be compact array")
            tir.evaluate(0)
        tir.Assert((tir.uint64(0) == tir.tvm_struct_get(arg2, 0, 8, dtype="uint64")), "Argument arg2.byte_offset has an unsatisfied constraint")
        tir.Assert((1 == tir.tvm_struct_get(arg2, 0, 10, dtype="int32")), "Argument arg2.device_type has an unsatisfied constraint")
        tir.Assert((dev_id == tir.tvm_struct_get(arg2, 0, 9, dtype="int32")), "Argument arg2.device_id has an unsatisfied constraint")
        tir.attr(0, "compute_scope", "default_function_compute_")
        for i_outer in tir.range(0, 32):
            for j_outer_init in tir.range(0, 32):
                for i_inner_init in tir.range(0, 32):
                    for j_inner_init in tir.range(0, 32):
                        tir.store(C, ((((i_outer*32768) + (i_inner_init*1024)) + (j_outer_init*32)) + j_inner_init), tir.float32(0))
            for j_outer in tir.range(0, 32):
                for k_outer in tir.range(0, 256):
                    for k_inner in tir.range(0, 4):
                        for i_inner in tir.range(0, 32):
                            for j_inner in tir.range(0, 32):
                                tir.store(C, ((((i_outer*32768) + (i_inner*1024)) + (j_outer*32)) + j_inner), tir.call_llvm_pure_intrin(tir.uint32(97), tir.uint32(3), tir.load("float32", A, ((((i_outer*32768) + (i_inner*1024)) + (k_outer*4)) + k_inner)), tir.load("float32", B, ((((k_outer*4096) + (k_inner*1024)) + (j_outer*32)) + j_inner)), tir.load("float32", C, ((((i_outer*32768) + (i_inner*1024)) + (j_outer*32)) + j_inner)), dtype="float32"))
    __tvm_meta__ = {
        "root": 1,
        "nodes": [
            {
                "type_key": ""
            },
            {
                "type_key": "Map",
                "keys": [
                    "Target"
                ],
                "data": [2]
            },
            {
                "type_key": "Array",
                "data": [3]
            },
            {
                "type_key": "Target",
                "attrs": {
                    "attrs": "10",
                    "id": "4",
                    "keys": "9",
                    "tag": "8"
                }
            },
            {
                "type_key": "TargetId",
                "attrs": {
                    "default_keys": "6",
                    "device_type": "1",
                    "name": "5"
                }
            },
            {
                "type_key": "runtime.String",
                "repr_str": "llvm"
            },
            {
                "type_key": "Array",
                "data": [7]
            },
            {
                "type_key": "runtime.String",
                "repr_str": "cpu"
            },
            {
                "type_key": "runtime.String"
            },
            {
                "type_key": "Array",
                "data": [7]
            },
            {
                "type_key": "Map"
            }
        ],
        "b64ndarrays": [],
        "attrs": {"tvm_version": "0.7.dev1"}
    }


def test_matmul_lower():
    mod = Module()
    rt_mod = tir.hybrid.from_source(tvm.tir.hybrid.ashybrid(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod)


@tvm.tir.hybrid.script
class Module2:
    def default_function(A: ty.handle, W: ty.handle, Conv: ty.handle) -> None:
        # function attr dict
        tir.func_attr({"global_symbol":"default_function", "tir.noalias":True})
        # var definition
        blockIdx_x = tir.var("int32")
        blockIdx_y = tir.var("int32")
        blockIdx_z = tir.var("int32")
        threadIdx_x = tir.var("int32")
        threadIdx_y = tir.var("int32")
        threadIdx_z = tir.var("int32")
        # buffer definition
        Apad_shared = tir.buffer_decl([16, 16, 16, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1)
        Apad_shared_wmma_matrix_a = tir.buffer_decl([16, 16, 16, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1)
        BA = tir.buffer_decl([16, 16], dtype="float16", scope="wmma.matrix_a", align=32, offset_factor=256)
        BB = tir.buffer_decl([16, 16], dtype="float16", scope="wmma.matrix_b", align=32, offset_factor=256)
        BC = tir.buffer_decl([16, 16], scope="wmma.accumulator", align=32, offset_factor=256)
        Conv_wmma_accumulator = tir.buffer_decl([16, 14, 14, 32, 16, 16], elem_offset=0, align=128, offset_factor=1)
        W_shared = tir.buffer_decl([3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1)
        W_shared_wmma_matrix_b = tir.buffer_decl([3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1)
        buffer = tir.buffer_decl([16, 16], dtype="float16", scope="shared", align=32, offset_factor=256)
        buffer_1 = tir.buffer_decl([16, 16], dtype="float16", scope="wmma.matrix_a", align=32, offset_factor=256)
        buffer_2 = tir.buffer_decl([16, 16], dtype="float16", scope="shared", align=32, offset_factor=256)
        buffer_3 = tir.buffer_decl([16, 16], dtype="float16", scope="wmma.matrix_b", align=32, offset_factor=256)
        buffer_4 = tir.buffer_decl([16, 16], scope="wmma.accumulator", align=32, offset_factor=256)
        buffer_5 = tir.buffer_decl([16, 16], align=32, offset_factor=256)
        A_1 = tir.buffer_bind(A, [16, 14, 14, 16, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1)
        W_1 = tir.buffer_bind(W, [3, 3, 16, 32, 16, 16], dtype="float16", elem_offset=0, align=128, offset_factor=1)
        Conv_1 = tir.buffer_bind(Conv, [16, 14, 14, 32, 16, 16], elem_offset=0, align=128, offset_factor=1)
        # body
        tir.attr(Conv_1, "realize_scope", "")
        tir.realize(Conv_1[0:16, 0:14, 0:14, 0:32, 0:16, 0:16])
        tir.attr(tir.iter_var(blockIdx_z, None, "ThreadIndex", "blockIdx.z"), "thread_extent", 196)
        tir.attr(tir.iter_var(blockIdx_x, None, "ThreadIndex", "blockIdx.x"), "thread_extent", 2)
        tir.attr(tir.iter_var(blockIdx_y, None, "ThreadIndex", "blockIdx.y"), "thread_extent", 4)
        tir.attr(tir.iter_var(threadIdx_y, None, "ThreadIndex", "threadIdx.y"), "thread_extent", 4)
        tir.attr(tir.iter_var(threadIdx_z, None, "ThreadIndex", "threadIdx.z"), "thread_extent", 2)
        tir.attr(Conv_wmma_accumulator, "realize_scope", "wmma.accumulator")
        tir.realize(Conv_wmma_accumulator[((blockIdx_x*8) + (threadIdx_y*2)):(((blockIdx_x*8) + (threadIdx_y*2)) + 2), tir.floordiv(blockIdx_z, 14):(tir.floordiv(blockIdx_z, 14) + 1), tir.floormod(blockIdx_z, 14):(tir.floormod(blockIdx_z, 14) + 1), ((blockIdx_y*8) + (threadIdx_z*4)):(((blockIdx_y*8) + (threadIdx_z*4)) + 4), 0:16, 0:16])
        for n_c_init in tir.range(0, 2):
            for o_c_init in tir.range(0, 4):
                tir.attr([BC, Conv_wmma_accumulator], "buffer_bind_scope", tir.tvm_tuple((n_c_init + ((blockIdx_x*8) + (threadIdx_y*2))), 1, tir.floordiv(blockIdx_z, 14), 1, tir.floormod(blockIdx_z, 14), 1, (o_c_init + ((blockIdx_y*8) + (threadIdx_z*4))), 1, 0, 16, 0, 16, dtype="handle"))
                tir.evaluate(tir.tvm_fill_fragment(BC.data, 16, 16, 16, tir.floordiv(BC.elem_offset, 256), tir.float32(0), dtype="handle"))
        for ic_outer in tir.range(0, 8):
            for kh in tir.range(0, 3):
                tir.attr(Apad_shared, "realize_scope", "shared")
                tir.realize(Apad_shared[(blockIdx_x*8):((blockIdx_x*8) + 8), (tir.floordiv(blockIdx_z, 14) + kh):((tir.floordiv(blockIdx_z, 14) + kh) + 1), tir.floormod(blockIdx_z, 14):(tir.floormod(blockIdx_z, 14) + 3), (ic_outer*2):((ic_outer*2) + 2), 0:16, 0:16])
                for ax2 in tir.range(0, 3):
                    for ax3 in tir.range(0, 2):
                        for ax4_ax5_fused_outer in tir.range(0, 8):
                            tir.attr(tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"), "thread_extent", 32)
                            Apad_shared[((threadIdx_z + (threadIdx_y*2)) + (blockIdx_x*8)), (tir.floordiv(blockIdx_z, 14) + kh), (ax2 + tir.floormod(blockIdx_z, 14)), (ax3 + (ic_outer*2)), tir.floordiv((threadIdx_x + (ax4_ax5_fused_outer*32)), 16), tir.floormod((threadIdx_x + (ax4_ax5_fused_outer*32)), 16)] = tir.if_then_else((((((tir.floordiv(blockIdx_z, 14) + kh) >= 1) and (((tir.floordiv(blockIdx_z, 14) + kh) - 1) < 14)) and ((ax2 + tir.floormod(blockIdx_z, 14)) >= 1)) and (((ax2 + tir.floormod(blockIdx_z, 14)) - 1) < 14)), A_1[((threadIdx_z + (threadIdx_y*2)) + (blockIdx_x*8)), ((tir.floordiv(blockIdx_z, 14) + kh) - 1), ((ax2 + tir.floormod(blockIdx_z, 14)) - 1), (ax3 + (ic_outer*2)), tir.floordiv((threadIdx_x + (ax4_ax5_fused_outer*32)), 16), tir.floormod((threadIdx_x + (ax4_ax5_fused_outer*32)), 16)], tir.float16(0), dtype="float16")
                tir.attr(W_shared, "realize_scope", "shared")
                tir.realize(W_shared[kh:(kh + 1), 0:3, (ic_outer*2):((ic_outer*2) + 2), (blockIdx_y*8):((blockIdx_y*8) + 8), 0:16, 0:16])
                for ax1 in tir.range(0, 3):
                    for ax2_1 in tir.range(0, 2):
                        tir.attr(tir.iter_var(threadIdx_x, None, "ThreadIndex", "threadIdx.x"), "thread_extent", 32)
                        for ax4_ax5_fused_inner in tir.range(0, 8, "vectorized"):
                            W_shared[kh, ax1, (ax2_1 + (ic_outer*2)), ((threadIdx_z + (threadIdx_y*2)) + (blockIdx_y*8)), tir.floordiv((ax4_ax5_fused_inner + (threadIdx_x*8)), 16), tir.floormod((ax4_ax5_fused_inner + (threadIdx_x*8)), 16)] = W_1[kh, ax1, (ax2_1 + (ic_outer*2)), ((threadIdx_z + (threadIdx_y*2)) + (blockIdx_y*8)), tir.floordiv((ax4_ax5_fused_inner + (threadIdx_x*8)), 16), tir.floormod((ax4_ax5_fused_inner + (threadIdx_x*8)), 16)]
                for ic_inner in tir.range(0, 2):
                    for kw in tir.range(0, 3):
                        tir.attr(Apad_shared_wmma_matrix_a, "realize_scope", "wmma.matrix_a")
                        tir.realize(Apad_shared_wmma_matrix_a[((blockIdx_x*8) + (threadIdx_y*2)):(((blockIdx_x*8) + (threadIdx_y*2)) + 2), (tir.floordiv(blockIdx_z, 14) + kh):((tir.floordiv(blockIdx_z, 14) + kh) + 1), (kw + tir.floormod(blockIdx_z, 14)):((kw + tir.floormod(blockIdx_z, 14)) + 1), ((ic_outer*2) + ic_inner):(((ic_outer*2) + ic_inner) + 1), 0:16, 0:16])
                        for ax0 in tir.range(0, 2):
                            tir.attr([buffer, Apad_shared], "buffer_bind_scope", tir.tvm_tuple((ax0 + ((blockIdx_x*8) + (threadIdx_y*2))), 1, (tir.floordiv(blockIdx_z, 14) + kh), 1, (kw + tir.floormod(blockIdx_z, 14)), 1, ((ic_outer*2) + ic_inner), 1, 0, 16, 0, 16, dtype="handle"))
                            tir.attr([buffer_1, Apad_shared_wmma_matrix_a], "buffer_bind_scope", tir.tvm_tuple((ax0 + ((blockIdx_x*8) + (threadIdx_y*2))), 1, (tir.floordiv(blockIdx_z, 14) + kh), 1, (kw + tir.floormod(blockIdx_z, 14)), 1, ((ic_outer*2) + ic_inner), 1, 0, 16, 0, 16, dtype="handle"))
                            tir.evaluate(tir.tvm_load_matrix_sync(buffer_1.data, 16, 16, 16, tir.floordiv(buffer_1.elem_offset, 256), tir.tvm_access_ptr(tir.type_annotation(dtype="float16"), buffer.data, buffer.elem_offset, 256, 1, dtype="handle"), 16, "row_major", dtype="handle"))
                        tir.attr(W_shared_wmma_matrix_b, "realize_scope", "wmma.matrix_b")
                        tir.realize(W_shared_wmma_matrix_b[kh:(kh + 1), kw:(kw + 1), ((ic_outer*2) + ic_inner):(((ic_outer*2) + ic_inner) + 1), ((blockIdx_y*8) + (threadIdx_z*4)):(((blockIdx_y*8) + (threadIdx_z*4)) + 4), 0:16, 0:16])
                        for ax3_1 in tir.range(0, 4):
                            tir.attr([buffer_2, W_shared], "buffer_bind_scope", tir.tvm_tuple(kh, 1, kw, 1, ((ic_outer*2) + ic_inner), 1, (ax3_1 + ((blockIdx_y*8) + (threadIdx_z*4))), 1, 0, 16, 0, 16, dtype="handle"))
                            tir.attr([buffer_3, W_shared_wmma_matrix_b], "buffer_bind_scope", tir.tvm_tuple(kh, 1, kw, 1, ((ic_outer*2) + ic_inner), 1, (ax3_1 + ((blockIdx_y*8) + (threadIdx_z*4))), 1, 0, 16, 0, 16, dtype="handle"))
                            tir.evaluate(tir.tvm_load_matrix_sync(buffer_3.data, 16, 16, 16, tir.floordiv(buffer_3.elem_offset, 256), tir.tvm_access_ptr(tir.type_annotation(dtype="float16"), buffer_2.data, buffer_2.elem_offset, 256, 1, dtype="handle"), 16, "row_major", dtype="handle"))
                        for n_c in tir.range(0, 2):
                            for o_c in tir.range(0, 4):
                                tir.attr([BA, Apad_shared_wmma_matrix_a], "buffer_bind_scope", tir.tvm_tuple((n_c + ((blockIdx_x*8) + (threadIdx_y*2))), 1, (tir.floordiv(blockIdx_z, 14) + kh), 1, (tir.floormod(blockIdx_z, 14) + kw), 1, ((ic_outer*2) + ic_inner), 1, 0, 16, 0, 16, dtype="handle"))
                                tir.attr([BB, W_shared_wmma_matrix_b], "buffer_bind_scope", tir.tvm_tuple(kh, 1, kw, 1, ((ic_outer*2) + ic_inner), 1, (o_c + ((blockIdx_y*8) + (threadIdx_z*4))), 1, 0, 16, 0, 16, dtype="handle"))
                                tir.attr([BC, Conv_wmma_accumulator], "buffer_bind_scope", tir.tvm_tuple((n_c + ((blockIdx_x*8) + (threadIdx_y*2))), 1, tir.floordiv(blockIdx_z, 14), 1, tir.floormod(blockIdx_z, 14), 1, (o_c + ((blockIdx_y*8) + (threadIdx_z*4))), 1, 0, 16, 0, 16, dtype="handle"))
                                tir.evaluate(tir.tvm_mma_sync(BC.data, tir.floordiv(BC.elem_offset, 256), BA.data, tir.floordiv(BA.elem_offset, 256), BB.data, tir.floordiv(BB.elem_offset, 256), BC.data, tir.floordiv(BC.elem_offset, 256), dtype="handle"))
        for n_inner in tir.range(0, 2):
            for o_inner in tir.range(0, 4):
                tir.attr([buffer_4, Conv_wmma_accumulator], "buffer_bind_scope", tir.tvm_tuple(((((blockIdx_x*4) + threadIdx_y)*2) + n_inner), 1, tir.floordiv(blockIdx_z, 14), 1, tir.floormod(blockIdx_z, 14), 1, ((((blockIdx_y*2) + threadIdx_z)*4) + o_inner), 1, 0, 16, 0, 16, dtype="handle"))
                tir.attr([buffer_5, Conv_1], "buffer_bind_scope", tir.tvm_tuple(((((blockIdx_x*4) + threadIdx_y)*2) + n_inner), 1, tir.floordiv(blockIdx_z, 14), 1, tir.floormod(blockIdx_z, 14), 1, ((((blockIdx_y*2) + threadIdx_z)*4) + o_inner), 1, 0, 16, 0, 16, dtype="handle"))
                tir.evaluate(tir.tvm_store_matrix_sync(buffer_4.data, 16, 16, 16, tir.floordiv(buffer_4.elem_offset, 256), tir.tvm_access_ptr(tir.type_annotation(dtype="float32"), buffer_5.data, buffer_5.elem_offset, 256, 2, dtype="handle"), 16, "row_major", dtype="handle"))


def test_opt_conv_tensorcore_lower():
    mod = Module2()
    rt_mod = tir.hybrid.from_source(tvm.tir.hybrid.ashybrid(mod, True))
    tvm.ir.assert_structural_equal(mod, rt_mod)


if __name__ == '__main__':
    # test_matmul()
    # test_element_wise()
    # test_predicate()
    test_matmul_lower()
    test_opt_conv_tensorcore_lower()
