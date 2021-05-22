import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.search_rule import SearchRule
from tvm.script import ty
import numpy as np
import os

@tvm.script.tir
def dyn_mm(a: ty.handle, b: ty.handle, c: ty.handle, M: ty.int32, N: ty.int32) -> None:
    A = tir.match_buffer(a, (M, 768), "float32")
    B = tir.match_buffer(b, (768, N), "float32")
    C = tir.match_buffer(c, (M, N), "float32")
    with tir.block([M, N, tir.reduce_axis(0, 768)], "matmul") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

M, N, K = 1024, 768, 768
target = "llvm"
ctx = tvm.context(target, 0)


a_np = np.random.uniform(size=(M, K)).astype("float32")
b_np = np.random.uniform(size=(K, N)).astype("float32")
a = tvm.nd.array(a_np)
b = tvm.nd.array(b_np)
c = tvm.nd.array(np.zeros((M, N)).astype("float32"))


def build_and_test(mod):
    build_func = tvm.build(mod["main"], target=target)
    build_func(a, b, c, M, N)
    np.testing.assert_allclose(c.asnumpy(), np.matmul(a.asnumpy(), b.asnumpy()), rtol=1e-5)
    evaluator = build_func.time_evaluator(build_func.entry_name, ctx, number=1)
    print(tvm.script.asscript(mod))
    return evaluator(a, b, c, M, N).mean


def optimal_static_schedule(sch):
    b1 = sch.get_block(name="matmul")
    b2 = sch.cache_write(block=b1, i=0, storage_scope="global")
    l3, l4, l5 = sch.get_axes(block=b2)
    v6, v7, v8, v9 = sch.sample_perfect_tile(n=4, loop=l3, max_innermost_factor=16, decision=[-1, 1, 8, 4])
    l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9])
    v14, v15, v16, v17 = sch.sample_perfect_tile(n=4, loop=l4, max_innermost_factor=16, decision=[-1, 1, 16, 16])
    l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17])
    v22, v23 = sch.sample_perfect_tile(n=2, loop=l5, max_innermost_factor=16, decision=[-1, 4])
    l24, l25 = sch.split(loop=l5, factors=[v22, v23])
    sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)
    sch.reverse_compute_at(block=b1, loop=l18)
    b26 = sch.get_block(name="C_global")
    sch.mark_block(block=b26, ann_key="auto_vectorize_extent", ann_val=32)
    b27 = sch.get_block(name="matmul")
    sch.mark_block(block=b27, ann_key="auto_parallel_extent", ann_val=96)
    sch.mark_block(block=b27, ann_key="auto_vectorize_extent", ann_val=32)
    v28 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
    sch.mark_block(block=b27, ann_key="auto_unroll_explicit", ann_val=v28)
    return sch


def manual_schedule():
    func = dyn_mm
    sch = ms.Schedule(func, debug_mode=True)
    sch = optimal_static_schedule(sch)
    print("time: %fms" % (build_and_test(sch.mod) * 1000))

@tvm.script.tir
class Module:
    def main(a: ty.handle, b: ty.handle, c: ty.handle, M: ty.int32, N: ty.int32) -> None:
        C = tir.match_buffer(c, [M, N], elem_offset=0, align=128, offset_factor=1)
        A = tir.match_buffer(a, [M, 768], elem_offset=0, align=128, offset_factor=1)
        B = tir.match_buffer(b, [768, N], elem_offset=0, align=128, offset_factor=1)
        # body
        with tir.block([], "root") as []:
            tir.reads([])
            tir.writes([])
            C_global = tir.buffer_allocate([M, N], elem_offset=0, align=128, offset_factor=1)
            for i0_outer_outer_outer, i1_outer_outer_outer in tir.grid(((tir.floordiv(((tir.floordiv(((M + 4) - 1), 4) + 8) - 1), 8) + 1) - 1),
                                                                       ((tir.floordiv(((tir.floordiv(((N + 16) - 1), 16) + 16) - 1), 16) + 1) - 1)):
                for i0_outer_outer_inner, i1_outer_outer_inner, i2_outer, i0_outer_inner, i1_outer_inner, i2_inner, i0_inner, i1_inner in tir.grid(1, 1, 192, 8, 16, 4, 4, 16):
                    with tir.block([M, N, tir.reduce_axis(0, 768)], "matmul") as [vi, vj, vk]:
                        tir.bind(vi, ((((i0_outer_outer_outer*8) + i0_outer_inner)*4) + i0_inner))
                        tir.bind(vj, ((((i1_outer_outer_outer*16) + i1_outer_inner)*16) + i1_inner))
                        tir.bind(vk, ((i2_outer*4) + i2_inner))
                        tir.reads([C_global[vi:(vi + 1), vj:(vj + 1)], A[vi:(vi + 1), vk:(vk + 1)], B[vk:(vk + 1), vj:(vj + 1)]])
                        tir.writes([C_global[vi:(vi + 1), vj:(vj + 1)]])
                        tir.block_attr({"auto_parallel_extent":"96", "auto_vectorize_extent":"32", "auto_unroll_explicit":"0"})
                        with tir.init():
                            C_global[vi, vj] = tir.float32(0)
                        C_global[vi, vj] = (C_global[vi, vj] + (A[vi, vk]*B[vk, vj]))
                for ax0, ax1 in tir.grid(32, 256):
                    with tir.block([M, N], "C_global") as [v0, v1]:
                        tir.bind(v0, ((i0_outer_outer_outer*32) + ax0))
                        tir.bind(v1, ((i1_outer_outer_outer*256) + ax1))
                        tir.reads([C_global[v0:(v0 + 1), v1:(v1 + 1)]])
                        tir.writes([C[v0:(v0 + 1), v1:(v1 + 1)]])
                        tir.block_attr({"auto_vectorize_extent":"32"})
                        C[v0, v1] = C_global[v0, v1]


def manual_kernel():
    mod = Module()
    rt_mod = tvm.script.from_source(tvm.script.asscript(mod, True))
    print("time: %fms" % (build_and_test(mod) * 1000))


manual_schedule()
manual_kernel()
