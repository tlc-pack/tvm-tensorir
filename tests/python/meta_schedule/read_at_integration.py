import tvm
from tvm import te, tir
from tvm.script import tir as T
import numpy as np
import tir_tensor_intrin
import te_workload

# @T.prim_func
# def original_matmul(a: T.handle, b: T.handle, c: T.handle) ->None:
#     A = T.match_buffer(a, [1024, 1024], "float16")
#     B = T.match_buffer(b, [1024, 1024], "float16")
#     C = T.match_buffer(c, [1024, 1024], "float32")
#
#     for i, j, k in T.grid(1024, 1024, 1024):
#         with T.block([1024, 1024, T.reduce_axis(0, 1024)], "C") as [vi,vj,vk]:
#             T.bind(vi,i)
#             T.bind(vj,j)
#             T.bind(vk, k)
#             with T.init():
#                 C[vi, vj] = 0.0
#             C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def cuda_matmul(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=undefined-loop-variable
    A = T.match_buffer(a, [1024, 1024], "float32")
    B = T.match_buffer(b, [1024, 1024], "float32")
    C = T.match_buffer(c, [1024, 1024], "float32")

    for bx in T.thread_binding(0, 8, thread="blockIdx.x"):
        for by in T.thread_binding(0, 8, thread="blockIdx.y"):
            for ty in T.thread_binding(0, 8, thread="threadIdx.y"):
                for tx in T.thread_binding(0, 32, thread="threadIdx.x"):
                    for k0 in T.serial(0, 32):
                        for k1 in T.serial(0, 2):
                            for i0, j0 in T.grid(4, 2):
                                for i1, j1, k2 in T.grid(16, 16, 16):
                                    with T.block([1024, 1024, T.reduce_axis(0, 1024)], "C") as [vi,vj,vk]:
                                        T.bind(vi, bx*128 + T.floordiv(ty, 4)*64+i0*16+i1 )
                                        T.bind(vj, by*128+T.floormod(ty, 4)*32+j0*16+j1)
                                        T.bind(vk, k0*32+k1*16+k2)
                                        with T.init():
                                            C[vi, vj] = 0.0
                                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test():
    workload = te_workload.matmul_fp16(n=1024, m=1024, k=1024)
    workload = te.create_prim_func(workload)
    sch = tir.Schedule(workload)
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    # Step 1. Rule-Auto-Tensorize
    # pylint: disable=invalid-name
    i, i_tc = sch.split(i, factors=[None, 16])
    j, j_tc = sch.split(j, factors=[None, 16])
    k, k_tc = sch.split(k, factors=[None, 16])
    sch.reorder(
        # fmt: off
        i, j, k,
        # tensor core
        i_tc, j_tc, k_tc,
        # fmt: on
    )
    block_inner = sch.blockize(i_tc)
    block_outer, block_inner = block_inner, block
    del block
    # Step 2. Rule-Multi-Level-Tiling
    i_factors = [8, 1, 2, 1, 4]
    j_factors = [1, 8, 4, 1, 2]
    i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
    j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
    k_factors = [32, 2, 1]
    k0, k1, k2 = sch.split(k, factors=k_factors)
    # pylint: enable=invalid-name
    sch.reorder(
        # fmt: off
        i0, j0,   # S => blockIdx.x
        i1, j1,   # S => blockIdx.y
        i2, j2,   # S => threadIdx.y
        # cache_write here
        k0,       # R
        # vectorized cooperative fetching here
        k1,       # R
        i3, j3,   # S
        k2,       # R
        i4, j4,
        # S
        # fmt: on
    )
    block_idx = sch.fuse(i0, j0)
    block_idy = sch.fuse(i1, j1)
    thread_idy = sch.fuse(i2, j2)
    sch.bind(block_idx, "blockIdx.x")
    sch.bind(block_idy, "blockIdx.y")
    sch.bind(thread_idy, "threadIdx.y")

    by, _bx, ty,  k0, k1,_,_,_, _i, _j, = sch.get_loops(block_outer)
    print(sch.mod["main"].script())
    block_wmma_a = sch.read_at(k1, block_outer, 1, "wmma.matrix_a", False)
    block_wmma_b = sch.read_at(k1, block_outer, 2, "wmma.matrix_b", False)
    sch.read_at(k0, block_wmma_a, 0, "shared", False)
    sch.read_at(k0, block_wmma_b, 0, "shared", False)
    sch.write_at(ty, block_outer, 0, "wmma.accumulator", False)
    loop = sch.get_loops(block_outer)[3]
    block_init_c = sch.decompose_reduction(block_outer, loop)
    block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

    loop = sch.get_loops(block_inner)[-3]
    sch.tensorize(loop, "wmma_sync")
    loop = sch.get_loops(block_init_c_inner)[-2]
    sch.tensorize(loop, "wmma_fill")
    sch.annotate
    print(sch.mod["main"].script())



test()