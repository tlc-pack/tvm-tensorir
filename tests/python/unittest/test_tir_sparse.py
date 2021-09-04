import tvm
from tvm import tir
from tvm.script import ty


@tvm.script.tir
def spmm_tir(a_indptr: ty.handle, a_indices: ty.handle, a_data: ty.handle, b: ty.handle, c: ty.handle) -> None:
    m = tir.var('int32')
    n = tir.var('int32')
    k = tir.var('int32')
    nnz = tir.var('int32')
    A_indptr = tir.match_buffer(a_indptr, [m + 1], 'int32')
    A_indices = tir.match_buffer(a_indices, [nnz], 'int32')
    A = tir.match_buffer(a_data, [nnz], 'float32')
    B = tir.match_buffer(b, [k, n], 'float32')
    C = tir.match_buffer(c, [m, n], 'float32')
    with tir.block([m, n], 'spmm_outer') as [vi, vj]:
        with tir.init():
            C[vi, vj] = 0.
        with tir.block([tir.reduce_axis(A_indptr[vi], A_indptr[vi + 1])], 'spmm_inner') as [vk]:
            C[vi, vj] = C[vi, vj] + A[vk] * B[A_indices[vk], vj]


"""
@tvm.script.tir
def spmm_sparse_tir(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    m = tir.var('int32')
    n = tir.var('int32')
    k = tir.var('int32')
    A = tir.sp.match_buffer(a, [m, k], ['dense', ('sparse', None)], 'int32', 'float32')
    B = tir.match_buffer(b, [k, n], 'float32')
    C = tir.match_buffer(c, [m, n], 'float32')
    for i, j in tir.grid(m, n):
        for k in tir.sp.iter(A[i]):
            with tir.block([m, n, tir.reduce_axis(0, k)], 'spmm') as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                with tir.init():
                    C[vi, vj] = 0.
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
"""
 

@tvm.script.tir
def embedding_update(a: ty.handle, grad_out: ty.handle, grad_in: ty.handle) -> None:
    m = tir.var('int32')  # number of tokens
    n = tir.var('int32')  # feature size
    k = tir.var('int32')  # dictionary size
    A = tir.sp.match_buffer(a, [m, k], ['dense', ('sparse', None)], 'int32', 'float32')
    B = tir.match_buffer(grad_out, [m, n], 'float32')
    C = tir.match_buffer(grad_in, [k, n], 'float32')
    for i, j in tir.grid(m, n):
        for k in tir.sp.iter(A[i]):
            with tir.block([m, n, k]) as [vi, vj, vk]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                C[vk, vj] = C[vk, vj] + A[vi, vk] * B[vi, vj]


"""
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-f16
@tvm.script.tir
def sparse_tensor_core_desr(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.sp.match_buffer(a, [16, 4, 4], ['dense', 'dense', ('sparse', 2)], 'int8', 'float16', align=128, offset_factor=256, scope='wmma.matrix_a')
    B = tir.match_buffer(b, [16, 16], 'float16', align=128, offset_factor=256, scope='wmma.matrix_b')
    C = tir.match_buffer(c, [16, 16], 'float32', align=128, offset_factor=256, scope='wmma.matrix_accumulator')

    for i, j, k in tir.grid(16, 16, 4):
        for l in tir.sp.iter(A[i, j]):
            with tir.block([16, 16, tir.reduce_axis(0, 4), tir.reduce_axis(0, 4)], "root") as [vi, vj, vk, vl]:
                tir.bind(vi, i)
                tir.bind(vj, j)
                tir.bind(vk, k)
                tir.bind(vl, l)
                with tir.init():
                    C[vi, vj] = 0.
                C[vi, vj] = C[vi, vj] + A[vi, vk, vl] * B[vk * 4 + vl, vj]
"""


if __name__ == "__main__":
    pass
