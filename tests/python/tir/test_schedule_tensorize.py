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
import numpy as np
import tvm
import tvm.testing
from tvm import tir
from tvm.script import ty

import util

# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,redundant-keyword-arg

@tvm.script.tir
def desc_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = tir.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = tir.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        for i, j, k in tir.grid(16, 16, 16):
            with tir.block([16, 16, tir.reduce_axis(0, 16)], "update") as [vii, vjj, vkk]:
                tir.bind(vii, vi + i)
                tir.bind(vjj, vj + j)
                tir.bind(vkk, vk + k)
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@tvm.script.tir
def intrin_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = tir.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = tir.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        # These access region must be explicitly stated. Otherwise the auto-completed region starts from (0, 0) instead of (vi, vj)
        tir.reads([A[vi: vi+16, vk: vk+16], B[vj: vj+16, vk: vk+16], C[vi:vi+16, vj:vj+16]])
        tir.writes([C[vi: vi+16, vj: vj+16]])
        for i, j, k in tir.grid(16, 16, 16):
            with tir.block([16, 16, tir.reduce_axis(0, 16)], "update") as [vii, vjj, vkk]:
                tir.bind(vii, i)
                tir.bind(vjj, j)
                tir.bind(vkk, k)
                C[vii, vjj] = C[vii, vjj] + B[vjj, vkk] * A[vii, vkk]



@tvm.script.tir
def lower_intrin_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), align=128, offset_factor=1)
    B = tir.match_buffer(b, (16, 16), align=128, offset_factor=1)
    C = tir.match_buffer(c, (16, 16), align=128, offset_factor=1)

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        tir.reads([C[vi:vi + 16, vj:vj + 16], A[vi:vi + 16, vk:vk + 16], B[vj:vj + 16, vk:vk + 16]])
        tir.writes(C[vi:vi + 16, vj:vj + 16])
        tir.evaluate(tir.tvm_mma_sync(C.data, C.elem_offset // 256,
                                      A.data, A.elem_offset // 256,
                                      B.data, B.elem_offset // 256,
                                      C.data, C.elem_offset // 256,
                                      dtype="handle"))


@tvm.script.tir
def tensorized_func(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    C = tir.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [128, 128], elem_offset=0, align=128, offset_factor=1)
    A = tir.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
    # body
    for i_outer, j_outer in tir.grid(8, 8):
        for i_inner_init, j_inner_init in tir.grid(16, 16):
            with tir.block([128, 128], "init") as [vi_init, vj_init]:
                tir.bind(vi_init, ((i_outer * 16) + i_inner_init))
                tir.bind(vj_init, ((j_outer * 16) + j_inner_init))
                C[vi_init, vj_init] = tir.float32(0)
        for k_outer in tir.grid(8):
            with tir.block([8, 8, tir.reduce_axis(0, 8)], "update") as [vi, vj, vk]:
                tir.bind(vi, i_outer)
                tir.bind(vj, j_outer)
                tir.bind(vk, k_outer)
                tir.reads([C[vi*16:vi*16 + 16, vj*16:vj*16 + 16], A[vi*16:vi*16 + 16, vk*16:vk*16 + 16], B[vj*16:vj*16 + 16, vk*16:vk*16 + 16]])
                tir.writes(C[vi*16:vi*16 + 16, vj*16:vj*16 + 16])
                A_elem_offset = tir.var('int32')
                B_elem_offset = tir.var('int32')
                C_elem_offset = tir.var('int32')
                A_sub = tir.match_buffer(A[vi*16:vi*16+16, vk*16:vk*16+16], [16, 16], elem_offset=A_elem_offset)
                B_sub = tir.match_buffer(B[vj*16:vj*16+16, vk*16:vk*16+16], [16, 16], elem_offset=B_elem_offset)
                C_sub = tir.match_buffer(C[vi*16:vi*16+16, vj*16:vj*16+16], [16, 16], elem_offset=C_elem_offset)
                tir.evaluate(
                    tir.tvm_mma_sync(C_sub.data, tir.floordiv(C_sub.elem_offset, 256),
                                     A_sub.data, tir.floordiv(A_sub.elem_offset, 256),
                                     B_sub.data, tir.floordiv(B_sub.elem_offset, 256),
                                     C_sub.data, tir.floordiv(C_sub.elem_offset, 256),
                                     dtype="handle"))


@tvm.script.tir
def batch_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [16, 128, 128])
    B = tir.match_buffer(b, [16, 128, 128])
    C = tir.match_buffer(c, [16, 128, 128])

    for i, j, k in tir.grid(16, 128, 128):
        with tir.block([16, 128, 128]) as [vn, vi, vj]:
            C[vn, vi, vj] = tir.float32(0)

    with tir.block([16, 128, 128, tir.reduce_axis(0, 128)], "update") as [vn, vi, vj, vk]:
        C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


@tvm.script.tir
def tensorized_batch_matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    C = tir.match_buffer(c, [16, 128, 128])
    B = tir.match_buffer(b, [16, 128, 128])
    A = tir.match_buffer(a, [16, 128, 128])

    with tir.block([16, 128, 128]) as [vn, vi, vj]:
        C[vn, vi, vj] = tir.float32(0)
    # body
    for n in range(0, 16):
        for i, j, k in tir.grid(8, 8, 8):
            with tir.block([16, 8, 8, tir.reduce_axis(0, 8)], "update") as [vn, vi, vj, vk]:
                tir.bind(vn, n)
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.reads([C[vn:vn + 1, vi*16:vi*16 + 16, vj*16:vj*16 + 16], A[vn:vn + 1, vi*16:vi*16 + 16, vk*16:vk*16 + 16],
                           B[vn:vn + 1, vj*16:vj*16 + 16, vk*16:vk*16 + 16]])
                tir.writes(C[vn:vn + 1, vi*16:vi*16 + 16, vj*16:vj*16 + 16])
                A_elem_offset = tir.var('int32')
                B_elem_offset = tir.var('int32')
                C_elem_offset = tir.var('int32')
                A_sub = tir.match_buffer(A[vn:vn + 1, vi*16:vi*16+16,vk*16:vk*16+16], (16, 16), elem_offset=A_elem_offset)
                B_sub = tir.match_buffer(B[vn:vn + 1, vj*16:vj*16+16,vk*16:vk*16+16], (16, 16), elem_offset=B_elem_offset)
                C_sub = tir.match_buffer(C[vn:vn + 1, vi*16:vi*16+16,vj*16:vj*16+16], (16, 16), elem_offset=C_elem_offset)
                tir.evaluate(
                    tir.tvm_mma_sync(C_sub.data, tir.floordiv(C_sub.elem_offset, 256),
                                     A_sub.data, tir.floordiv(A_sub.elem_offset, 256),
                                     B_sub.data, tir.floordiv(B_sub.elem_offset, 256),
                                     C_sub.data, tir.floordiv(C_sub.elem_offset, 256),
                                     dtype="handle"))


@tvm.script.tir
def batch_matmul_dot_product(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [1, 4, 4], "float32")
    B = tir.match_buffer(b, [1, 4, 4], "float32")
    C = tir.match_buffer(c, [1, 4, 4], "float32")

    t = tir.var("int32")
    tir.attr(tir.iter_var(t, None, "DataPar", ""), "pragma_import_llvm",
             "; ModuleID = '/tmp/tmpur44d1nu/input0.cc'\n\
source_filename = \"/tmp/tmpur44d1nu/input0.cc\"\n\
target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\"\n\
target triple = \"x86_64-pc-linux-gnu\"\n\
\n\
; Function Attrs: noinline nounwind optnone uwtable\n\
define dso_local i32 @vec4add(float* %0, i32 %1, float* %2, i32 %3, float* %4, i32 %5) #0 {\n\
  %7 = alloca float*, align 8\n\
  %8 = alloca i32, align 4\n\
  %9 = alloca float*, align 8\n\
  %10 = alloca i32, align 4\n\
  %11 = alloca float*, align 8\n\
  %12 = alloca i32, align 4\n\
  %13 = alloca i32, align 4\n\
  store float* %0, float** %7, align 8\n\
  store i32 %1, i32* %8, align 4\n\
  store float* %2, float** %9, align 8\n\
  store i32 %3, i32* %10, align 4\n\
  store float* %4, float** %11, align 8\n\
  store i32 %5, i32* %12, align 4\n\
  store i32 0, i32* %13, align 4\n\
  br label %14\n\
\n\
14:                                               ; preds = %39, %6\n\
  %15 = load i32, i32* %13, align 4\n\
  %16 = icmp slt i32 %15, 4\n\
  br i1 %16, label %17, label %42\n\
\n\
17:                                               ; preds = %14\n\
  %18 = load float*, float** %9, align 8\n\
  %19 = load i32, i32* %13, align 4\n\
  %20 = load i32, i32* %10, align 4\n\
  %21 = add nsw i32 %19, %20\n\
  %22 = sext i32 %21 to i64\n\
  %23 = getelementptr inbounds float, float* %18, i64 %22\n\
  %24 = load float, float* %23, align 4\n\
  %25 = load float*, float** %11, align 8\n\
  %26 = load i32, i32* %13, align 4\n\
  %27 = load i32, i32* %12, align 4\n\
  %28 = add nsw i32 %26, %27\n\
  %29 = sext i32 %28 to i64\n\
  %30 = getelementptr inbounds float, float* %25, i64 %29\n\
  %31 = load float, float* %30, align 4\n\
  %32 = fmul float %24, %31\n\
  %33 = load float*, float** %7, align 8\n\
  %34 = load i32, i32* %8, align 4\n\
  %35 = sext i32 %34 to i64\n\
  %36 = getelementptr inbounds float, float* %33, i64 %35\n\
  %37 = load float, float* %36, align 4\n\
  %38 = fadd float %37, %32\n\
  store float %38, float* %36, align 4\n\
  br label %39\n\
\n\
39:                                               ; preds = %17\n\
  %40 = load i32, i32* %13, align 4\n\
  %41 = add nsw i32 %40, 1\n\
  store i32 %41, i32* %13, align 4\n\
  br label %14\n\
\n\
42:                                               ; preds = %14\n\
  ret i32 0\n\
}\n\
\n\
attributes #0 = { noinline nounwind optnone uwtable \"correctly-rounded-divide-sqrt-fp-math\"=\"false\" \"disable-tail-calls\"=\"false\" \"frame-pointer\"=\"all\" \"less-precise-fpmad\"=\"false\" \"min-legal-vector-width\"=\"0\" \"no-infs-fp-math\"=\"false\" \"no-jump-tables\"=\"false\" \"no-nans-fp-math\"=\"false\" \"no-signed-zeros-fp-math\"=\"false\" \"no-trapping-math\"=\"true\" \"stack-protector-buffer-size\"=\"8\" \"target-cpu\"=\"x86-64\" \"target-features\"=\"+cx8,+fxsr,+mmx,+sse,+sse2,+x87\" \"unsafe-fp-math\"=\"false\" \"use-soft-float\"=\"false\" }\n\
\n\
!llvm.module.flags = !{!0}\n\
!llvm.ident = !{!1}\n\
\n\
!0 = !{i32 1, !\"wchar_size\", i32 4}\n\
!1 = !{!\"Ubuntu clang version 11.0.0-++20200928083541+eb83b551d3e-1~exp1~20200928184208.110\"}\n\
\n\
             ")

    for i, j, k in tir.grid(1, 4, 4):
        with tir.block([1, 4, 4]) as [vn, vi, vj]:
            C[vn, vi, vj] = tir.float32(0)

    with tir.block([1, 4, 4, tir.reduce_axis(0, 4)], "update") as [vn, vi, vj, vk]:
        C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


@tvm.script.tir
def dot_product_desc(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (4,))
    B = tir.match_buffer(b, (4,))
    C = tir.match_buffer(c, (1,))

    with tir.block([tir.reduce_axis(0, 4)], "root") as [v0]:
        tir.bind(v0, 0)
        for i in range(0, 4):
            with tir.block([tir.reduce_axis(0, 4)], "update") as [vi]:
                tir.bind(vi, v0 + i)
                C[0] = C[0] + A[vi] * B[vi]


@tvm.script.tir
def dot_product_impl(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (4,), offset_factor=1)
    B = tir.match_buffer(b, (4,), offset_factor=1)
    C = tir.match_buffer(c, (1,), offset_factor=1)

    with tir.block([tir.reduce_axis(0, 4)], "root") as [v0]:
        tir.bind(v0, 0)
        tir.reads([C[0 : 1], A[v0 : v0 + 4], B[v0 : v0 + 4]])
        tir.writes([C[0 : 1]])
        tir.evaluate(tir.call_extern("vec4add", C.data, C.elem_offset, A.data, A.elem_offset, B.data, B.elem_offset, dtype="int32"))


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name,redundant-keyword-arg
# fmt: on

# pylint: disable=invalid-name


def test_tensorize_gemm():
    func = util.matmul_stmt()
    tensor_intrin = tvm.tir.TensorIntrin(desc_func, intrin_func)
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    update = s.get_block("update")
    i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction(update, ko)
    s.tensorize(ii, tensor_intrin)

    func = tvm.build(s.mod["main"])
    a_np = np.random.uniform(size=(128, 128)).astype("float32")
    b_np = np.random.uniform(size=(128, 128)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((128, 128)).astype("float32"))
    func(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np.dot(a_np, b_np.transpose()), rtol=1e-6)


def test_tensorize_buffer_bind():
    func = util.matmul_stmt()
    # schedule
    s = tir.Schedule(func, debug_mode=True)
    update = s.get_block("update")
    i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction(update, ko)
    tensor_intrin = tvm.tir.TensorIntrin(desc_func, lower_intrin_func)
    s.tensorize(ii, tensor_intrin)
    tvm.ir.assert_structural_equal(tensorized_func, s.mod["main"])


def test_high_dim_tensorize():
    s = tir.Schedule(batch_matmul, debug_mode=True)
    update = s.get_block("update")
    _, i, j, k = s.get_loops(update)
    io, ii = s.split(i, factors=[None, 16])
    jo, ji = s.split(j, factors=[None, 16])
    ko, ki = s.split(k, factors=[None, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    tensor_intrin = tvm.tir.TensorIntrin(desc_func, lower_intrin_func)
    s.tensorize(ii, tensor_intrin)
    tvm.ir.assert_structural_equal(tensorized_batch_matmul, s.mod["main"])


def test_tensorize_dot_product():
    dot_prod = tvm.tir.TensorIntrin(dot_product_desc, dot_product_impl)
    s = tir.Schedule(batch_matmul_dot_product, debug_mode=True)
    C = s.get_block("update")
    _, _, _, k = s.get_loops(C)
    _, ki = s.split(k, factors=[None, 4])
    s.tensorize(ki, dot_prod)
    target = "llvm"
    ctx = tvm.device(target, 0)
    a_np = np.random.uniform(size=(1, 4, 4)).astype("float32")
    b_np = np.random.uniform(size=(1, 4, 4)).astype("float32")
    a = tvm.nd.array(a_np)
    b = tvm.nd.array(b_np)
    c = tvm.nd.array(np.zeros((1, 4, 4), dtype="float32"), ctx)
    func = tvm.build(s.mod["main"], target=target)
    func(a, b, c)
    tvm.testing.assert_allclose(
        c.asnumpy(),
        np.matmul(a.asnumpy(), b.asnumpy().transpose(0, 2, 1)),
        rtol=1e-5,
    )


if __name__ == "__main__":
    test_tensorize_gemm()
    test_tensorize_buffer_bind()
    test_high_dim_tensorize()
    # test_tensorize_dot_product()
