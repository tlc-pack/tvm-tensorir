import tvm
import tvm.testing
from tvm.script import tir
import numpy as np

import os
def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

TASK='gemm'
USE_MANUAL_CODE=True
@tvm.register_func('tvm_callback_cuda_postproc', override=True)
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/gemm_local_stage_double_buffer.cu").read()
    return code
M = N = K = 1024

@tir.prim_func
def func(a: tir.handle, b: tir.handle, c: tir.handle) -> None:
    s0 = tir.var("int32")
    s0_1 = tir.var("int32")
    s0_2 = tir.var("int32")
    s1 = tir.var("int32")
    s1_1 = tir.var("int32")
    s1_2 = tir.var("int32")
    A = tir.match_buffer(a, [1024, 1024], dtype="float16")
    B = tir.match_buffer(b, [1024, 1024], dtype="float16")
    C = tir.match_buffer(c, [1024, 1024], dtype="float32")
    # body
    # with tir.block("root")

    A_shared = tir.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
    A_shared_local = tir.alloc_buffer([2, 8], dtype="float16", scope="local")
    B_shared = tir.alloc_buffer([1024, 1024], dtype="float16", scope="shared")
    B_shared_local = tir.alloc_buffer([2, 8], dtype="float16", scope="local")

    A_shared_wmma_matrix_a = tir.alloc_buffer([1024, 1024], dtype="float16", scope="wmma.matrix_a")
    B_shared_wmma_matrix_b = tir.alloc_buffer([1024, 1024], dtype="float16", scope="wmma.matrix_b")
    C_wmma_accumulator = tir.alloc_buffer([1024, 1024], dtype="float32", scope="wmma.accumulator")
    for i0_0_0_i1_0_0_fused in tir.thread_binding(0, 8, thread="blockIdx.x"):
        for i0_0_1_i1_0_1_fused in tir.thread_binding(0, 8, thread="blockIdx.y"):
            for i0_0_2_i1_0_2_fused in tir.thread_binding(0, 8, thread="threadIdx.y"):
                for i0_0_4_init, i1_0_4_init in tir.grid(4, 2):
                    with tir.block([64, 64], "blockized_C_init") as [io, jo]:
                        tir.bind(io, i0_0_0_i1_0_0_fused * 8 + tir.floordiv(i0_0_2_i1_0_2_fused, 4) * 4 + i0_0_4_init)
                        tir.bind(jo, i0_0_1_i1_0_1_fused * 8 + tir.floormod(i0_0_2_i1_0_2_fused, 4) * 2 + i1_0_4_init)
                        tir.reads([])
                        tir.writes([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                        with tir.block([1, 1], "blockized_C_init") as [i_inito, j_inito]:
                            tir.bind(i_inito, 0)
                            tir.bind(j_inito, 0)
                            tir.reads([])
                            tir.writes([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                            C_1 = tir.match_buffer(C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                            tir.evaluate(tir.tvm_fill_fragment(C_1.data, 16, 16, 16, tir.floordiv(C_1.elem_offset, 256) + tir.floordiv(tir.floormod(C_1.elem_offset, 256), 16), tir.float32(0), dtype="handle"))
                for i2_0_0 in tir.serial(0, 32, annotations={'pipeline_scope': 2}):
                    for ax0_ax1_fused_1 in tir.thread_binding(0, 8, thread="threadIdx.y"):
                        for ax0_ax1_fused_2 in tir.thread_binding(0, 32, thread="threadIdx.x"):
                            for ax0 in tir.serial(0, 2):
                                for ax1 in tir.vectorized(0, 8):
                                    with tir.block([2, 8], "A_shared_local") as [v0, v1]:
                                        tir.bind(v0, ax0)
                                        tir.bind(v1, ax1)
                                        tir.reads([A[i0_0_0_i1_0_0_fused * 128 + v0 * 64 + ax0_ax1_fused_1 * 8 + tir.floordiv(ax0_ax1_fused_2, 4), i2_0_0 * 32 + tir.floormod(ax0_ax1_fused_2, 4) * 8 + v1]])
                                        tir.writes([A_shared_local[v0, v1]])
                                        A_shared_local[v0, v1] = A[i0_0_0_i1_0_0_fused * 128 + v0 * 64 + ax0_ax1_fused_1 * 8 + tir.floordiv(ax0_ax1_fused_2, 4), i2_0_0 * 32 + tir.floormod(ax0_ax1_fused_2, 4) * 8 + v1]
                            for ax0_ax1_fused_0 in tir.serial(0, 2):
                                for ax0_ax1_fused_3 in tir.vectorized(0, 8):
                                    with tir.block([1024, 1024], "A_shared") as [v0, v1]:
                                        tir.bind(v0, i0_0_0_i1_0_0_fused * 128 + ax0_ax1_fused_0 * 64 + ax0_ax1_fused_1 * 8 + tir.floordiv(ax0_ax1_fused_2, 4))
                                        tir.bind(v1, i2_0_0 * 32 + tir.floormod(ax0_ax1_fused_2, 4) * 8 + ax0_ax1_fused_3)
                                        tir.reads([A_shared_local[tir.floordiv(v0 - i0_0_0_i1_0_0_fused * 128, 64), tir.floormod(v1 - i2_0_0 * 32, 8)]])
                                        tir.writes([A_shared[v0, v1]])
                                        tir.block_attr({"buffer_dim_align":[[0, 0, 32, 8]]})
                                        A_shared[v0, v1] = A_shared_local[tir.floordiv(v0 - i0_0_0_i1_0_0_fused * 128, 64), tir.floormod(v1 - i2_0_0 * 32, 8)]
                    for ax0_ax1_fused_1 in tir.thread_binding(0, 8, thread="threadIdx.y"):
                        for ax0_ax1_fused_2 in tir.thread_binding(0, 32, thread="threadIdx.x"):
                            for ax0 in tir.serial(0, 2):
                                for ax1 in tir.vectorized(0, 8):
                                    with tir.block([2, 8], "B_shared_local") as [v0, v1]:
                                        tir.bind(v0, ax0)
                                        tir.bind(v1, ax1)
                                        tir.reads([B[i2_0_0 * 32 + v0 * 16 + ax0_ax1_fused_1 * 2 + tir.floordiv(ax0_ax1_fused_2, 16), tir.floormod(ax0_ax1_fused_2, 16) * 8 + v1]])
                                        tir.writes([B_shared_local[v0, v1]])
                                        B_shared_local[v0, v1] = B[i2_0_0 * 32 + v0 * 16 + ax0_ax1_fused_1 * 2 + tir.floordiv(ax0_ax1_fused_2, 16), i0_0_1_i1_0_1_fused * 128 + tir.floormod(ax0_ax1_fused_2, 16) * 8 + v1]
                            for ax0_ax1_fused_0 in tir.serial(0, 2):
                                for ax0_ax1_fused_3 in tir.vectorized(0, 8):
                                    with tir.block([1024, 1024], "B_shared") as [v0, v1]:
                                        tir.bind(v0, i2_0_0 * 32 + ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + tir.floordiv(ax0_ax1_fused_2, 16))
                                        tir.bind(v1, i0_0_1_i1_0_1_fused * 128 + tir.floormod(ax0_ax1_fused_2, 16) * 8 + ax0_ax1_fused_3)
                                        tir.reads([B_shared_local[tir.floordiv(v0 - i2_0_0 * 32, 16), tir.floormod(v1 - i0_0_1_i1_0_1_fused * 128, 8)]])
                                        tir.writes([B_shared[v0, v1]])
                                        tir.block_attr({"buffer_dim_align":[[0, 0, 32, 8]]})
                                        B_shared[v0, v1] = B_shared_local[tir.floordiv(v0 - i2_0_0 * 32, 16), tir.floormod(v1 - i0_0_1_i1_0_1_fused * 128, 8)]

                    for i2_0_1 in tir.serial(0, 2, annotations={'pipeline_scope': 2}):
                        for ax0_0, ax1_0 in tir.grid(4, 1):
                            with tir.block([64, 64], "blockized_A_shared_wmma.matrix_a") as [v0o, v1o]:
                                tir.bind(v0o, i0_0_0_i1_0_0_fused * 8 + tir.floordiv(i0_0_2_i1_0_2_fused, 4) * 4 + ax0_0)
                                tir.bind(v1o, i2_0_0 * 2 + i2_0_1)
                                tir.reads([A_shared[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16]])
                                tir.writes([A_shared_wmma_matrix_a[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16]])
                                A_1 = tir.match_buffer(A_shared[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16], [16, 16], dtype="float16", strides=[s1, s0], scope="shared", offset_factor=16)
                                C_2 = tir.match_buffer(A_shared_wmma_matrix_a[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_a", offset_factor=16)
                                tir.evaluate(tir.tvm_load_matrix_sync(C_2.data, 16, 16, 16, tir.floordiv(C_2.elem_offset, 256) + tir.floordiv(tir.floormod(C_2.elem_offset, 256), 16), tir.tvm_access_ptr(tir.type_annotation(dtype="float16"), A_1.data, A_1.elem_offset, s1 * 16, 1, dtype="handle"), s1, "row_major", dtype="handle"))
                        for ax0_0, ax1_0 in tir.grid(1, 2):
                            with tir.block([64, 64], "blockized_B_shared_wmma.matrix_b") as [v0o, v1o]:
                                tir.bind(v0o, i2_0_0 * 2 + i2_0_1)
                                tir.bind(v1o, i0_0_1_i1_0_1_fused * 8 + tir.floormod(i0_0_2_i1_0_2_fused, 4) * 2 + ax1_0)
                                tir.reads([B_shared[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16]])
                                tir.writes([B_shared_wmma_matrix_b[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16]])
                                A_2 = tir.match_buffer(B_shared[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16], [16, 16], dtype="float16", strides=[s1_1, s0_1], scope="shared", offset_factor=16)
                                C_3 = tir.match_buffer(B_shared_wmma_matrix_b[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_b", offset_factor=16)
                                tir.evaluate(tir.tvm_load_matrix_sync(C_3.data, 16, 16, 16, tir.floordiv(C_3.elem_offset, 256) + tir.floordiv(tir.floormod(C_3.elem_offset, 256), 16), tir.tvm_access_ptr(tir.type_annotation(dtype="float16"), A_2.data, A_2.elem_offset, s1_1 * 16, 1, dtype="handle"), s1_1, "row_major", dtype="handle"))
                        for i0_0_3, i1_0_3, i2_0_2, i0_0_4, i1_0_4 in tir.grid(1, 1, 1, 4, 2):
                            with tir.block([64, 64, tir.reduce_axis(0, 64)], "blockized_C_update") as [io, jo, ko]:
                                tir.bind(io, i0_0_0_i1_0_0_fused * 8 + tir.floordiv(i0_0_2_i1_0_2_fused, 4) * 4 + i0_0_4)
                                tir.bind(jo, i0_0_1_i1_0_1_fused * 8 + tir.floormod(i0_0_2_i1_0_2_fused, 4) * 2 + i1_0_4)
                                tir.bind(ko, i2_0_0 * 2 + i2_0_1)
                                tir.reads([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], A_shared_wmma_matrix_a[io * 16 : io * 16 + 16, ko * 16 : ko * 16 + 16], B_shared_wmma_matrix_b[ko * 16 : ko * 16 + 16, jo * 16 : jo * 16 + 16]])
                                tir.writes([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                                with tir.block([1, 1, tir.reduce_axis(0, 1)], "blockized_C") as [io_1, jo_1, ko_1]:
                                    tir.bind(io_1, 0)
                                    tir.bind(jo_1, 0)
                                    tir.bind(ko_1, 0)
                                    tir.reads([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], A_shared_wmma_matrix_a[io * 16 : io * 16 + 16, ko * 16 : ko * 16 + 16], B_shared_wmma_matrix_b[ko * 16 : ko * 16 + 16, jo * 16 : jo * 16 + 16]])
                                    tir.writes([C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                                    A_3 = tir.match_buffer(A_shared_wmma_matrix_a[io * 16 : io * 16 + 16, ko * 16 : ko * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_a", offset_factor=16)
                                    B_1 = tir.match_buffer(B_shared_wmma_matrix_b[ko * 16 : ko * 16 + 16, jo * 16 : jo * 16 + 16], [16, 16], dtype="float16", scope="wmma.matrix_b", offset_factor=16)
                                    C_4 = tir.match_buffer(C_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                                    tir.evaluate(tir.tvm_mma_sync(C_4.data, tir.floordiv(C_4.elem_offset, 256) + tir.floordiv(tir.floormod(C_4.elem_offset, 256), 16), A_3.data, tir.floordiv(A_3.elem_offset, 256) + tir.floordiv(tir.floormod(A_3.elem_offset, 256), 16), B_1.data, tir.floordiv(B_1.elem_offset, 256) + tir.floordiv(tir.floormod(B_1.elem_offset, 256), 16), C_4.data, tir.floordiv(C_4.elem_offset, 256) + tir.floordiv(tir.floormod(C_4.elem_offset, 256), 16), dtype="handle"))
                for ax0_0, ax1_0 in tir.grid(4, 2):
                    with tir.block([64, 64], "blockized_C_wmma.accumulator") as [v0o, v1o]:
                        tir.bind(v0o, i0_0_0_i1_0_0_fused * 8 + tir.floordiv(i0_0_2_i1_0_2_fused, 4) * 4 + ax0_0)
                        tir.bind(v1o, i0_0_1_i1_0_1_fused * 8 + tir.floormod(i0_0_2_i1_0_2_fused, 4) * 2 + ax1_0)
                        tir.reads([C_wmma_accumulator[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16]])
                        tir.writes([C[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16]])
                        A_4 = tir.match_buffer(C_wmma_accumulator[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16], [16, 16], dtype="float32", scope="wmma.accumulator", offset_factor=16)
                        C_5 = tir.match_buffer(C[v0o * 16 : v0o * 16 + 16, v1o * 16 : v1o * 16 + 16], [16, 16], dtype="float32", strides=[s1_2, s0_2], offset_factor=16)
                        tir.evaluate(tir.tvm_store_matrix_sync(A_4.data, 16, 16, 16, tir.floordiv(A_4.elem_offset, 256) + tir.floordiv(tir.floormod(A_4.elem_offset, 256), 16), tir.tvm_access_ptr(tir.type_annotation(dtype="float32"), C_5.data, C_5.elem_offset, s1_2 * 16, 2, dtype="handle"), s1_2, "row_major", dtype="handle"))


def main():
    print('Script')
    mod = func
    print(mod.script())
    print('Text')
    print(mod)
    print('Lower')
    print(tvm.lower(mod, None, simple_mode=True))

    dev = tvm.device("cuda", 0)
    a_np = np.random.uniform(size=(N, K)).astype("float16")
    b_np = np.random.uniform(size=(K, M)).astype("float16")
    c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
    f = tvm.build(mod, target="cuda", name="dense")
    print(f.imported_modules[0].get_source())
    f(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=100)
    gflops = (N*M*K) * 2 / 1e9
    time_ms = evaluator(a, b, c).mean * 1e3
    print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))

if __name__=='__main__':
    main()
