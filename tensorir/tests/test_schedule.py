import numpy as np
import tvm
import topi
from tvm import tensorir
from tvm import ir_pass, register_func
from common import check_correctness

def test_decompile_fuse():
    N = M = K = 128

    # a tiled gemm
    A = tvm.placeholder((N, K), name='A')
    B = tvm.placeholder((M, K), name='B')
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k), name='C')
    D = topi.nn.relu(C)

    s = tvm.create_schedule([D.op])
    io, jo, ii, ji = s[C].tile(C.op.axis[0], C.op.axis[1], 8, 8)
    ko, ki = s[C].split(C.op.reduce_axis[0], 8)
    s[C].reorder(jo, io, ko, ji, ii, ki)

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        init, reduction, relu = s.blocks()

        jo, io, ji_init, ii_init = s.axis(init)
        _, _, ko, ji, ii, ki = s.axis(reduction)

        ji, ii, ko = s.reorder(ji, ii, ko)

        s.fuse(ji_init, ii_init)
        s.fuse(ko, ki)

        stmt = s.to_halide()
        return stmt

    with tvm.build_config(add_lower_pass=[(0, _schedule_pass)]):
        tvm.build(s, [A, B, D], 'llvm')

    check_correctness(s, [A, B, D], _schedule_pass)

def test_inline():
    N = M = 128

    A = tvm.placeholder((N, M), name='A')
    B = tvm.compute((N, M), lambda i, j: A[i][j] + 1, name='B')
    C = tvm.compute((N, M), lambda i, j: B[i][j] * 2, name='C')
    D = tvm.compute((N, M), lambda i, j: C[i][j] - 3, name='D')

    s = tvm.create_schedule([C.op, D.op])
    s[D].split(s[D].op.axis[0], 8)

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        B, C, D = s.blocks()
        i, j = s.axis(C)

        io, ii = s.split(i, 16)  # useless split

        s.compute_inline(C)

        stmt = s.to_halide()
        return stmt

    check_correctness(s, [A, D], _schedule_pass)

def test_compute_at():
    N = M = K = 128

    A = tvm.placeholder((N, M, K), name='A')
    B = tvm.compute((N, M, K), lambda i, j, k: A[i, j, k] + 1, name='B')
    C = tvm.compute((N, M, K), lambda i, j, k: B[i, j, k] * 2, name='C')
    D = tvm.compute((N, M, K), lambda i, j, k: C[i, j, k] - 3, name='D')

    s = tvm.create_schedule([D.op])
    i, j, k = C.op.axis
    s[C].split(j, 8)

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        B, C, D = s.blocks()
        i, j, k = s.axis(D)

        s.compute_at(C, i)
        s.compute_root(C)

        stmt = s.to_halide()
        return stmt

    check_correctness(s, [A, D], _schedule_pass)

def test_unroll():
    N = 4
    M = 4

    A = tvm.placeholder((N, M), name='A')
    B = tvm.compute((N, M), lambda i, j: A[tvm.if_then_else(j % 2 == 0, 0, 1)][j], name='B')

    s = tvm.create_schedule([B.op])

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)
        B, = s.blocks()
        i, j = s.axis(B)

        blocks = s.unroll(j)

        stmt = s.to_halide()
        return stmt

    check_correctness(s, [A, B], _schedule_pass)

def test_from_unroll():
    N = 4
    M = 4

    A = tvm.placeholder((N, M), name='A')
    B = tvm.compute((N, M), lambda i, j: A[tvm.if_then_else(j % 2 == 0, 0, 1)][j], name='B')

    s = tvm.create_schedule([B.op])

    s[B].unroll(B.op.axis[1])

    def _schedule_pass(stmt):
        stmt = ir_pass.UnrollLoop(stmt, -1, -1, -1, True)

        s = tensorir.create_schedule(stmt)
        i, = s.axis(s.blocks()[0])
        s.unroll(i)

        stmt = s.to_halide()
        return stmt

    check_correctness(s, [A, B], _schedule_pass)


def test_rank_zero():
    A = tvm.placeholder((), name='A')
    B = tvm.compute((), lambda : A(), name='B')

    # test invariant
    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)
        return  s.to_halide()

    s = tvm.create_schedule([B.op])
    check_correctness(s, [A, B], _schedule_pass)


def test_blockize():
    N = M = K = 4

    A = tvm.placeholder((N, M, K), name='A')
    B = tvm.compute((N, M, K), lambda i, j, k: A[i, j, k] + 1.0, name='B')

    # test invariant
    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        B, = s.blocks()
        i, j, k = s.axis(B)
        Bb = s.blockize(j)
        s.unblockize(Bb)

        stmt = s.to_halide()
        return stmt

    s = tvm.create_schedule([B.op])
    check_correctness(s, [A, B], _schedule_pass)

    # test compilation
    def _schedule_pass2(stmt):
        s = tensorir.create_schedule(stmt)

        B, = s.blocks()
        i, j, k = s.axis(B)
        Bb = s.blockize(j)

        stmt = s.to_halide()
        return stmt

    s = tvm.create_schedule([B.op])
    check_correctness(s, [A, B], _schedule_pass2)

def test_partial_tile():
    pass

def test_from_gpu():
    """Test naive translation of GPU code"""
    N = M = 128

    A = tvm.placeholder((N, M), name='A')
    B = tvm.compute((N, M), lambda i, j: A[i][j] + 1, name='B')

    s = tvm.create_schedule([B.op])
    s[B].bind(B.op.axis[0], tvm.thread_axis('blockIdx.x'))
    s[B].bind(B.op.axis[1], tvm.thread_axis('threadIdx.x'))
    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        stmt = s.to_halide()
        return stmt

    check_correctness(s, [A, B], _schedule_pass, 'cuda')

def test_bind():
    """Test schedule primitive bind"""
    N = M = 128

    A = tvm.placeholder((N, M), name='A')
    B = tvm.compute((N, M), lambda i, j: A[i][j] + 1, name='B')

    s = tvm.create_schedule([B.op])

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        B, = s.blocks()
        i, j = s.axis(B)

        bx = tvm.thread_axis('blockIdx.x')
        tx = tvm.thread_axis('threadIdx.x')
        s.bind(i, bx)
        s.bind(j, tx)

        stmt = s.to_halide()
        return stmt

    check_correctness(s, [A, B], _schedule_pass, ['cuda', 'llvm'])


if __name__ == "__main__":
    test_decompile_fuse()
    test_inline()
    test_compute_at()
    test_unroll()
    test_from_unroll()
    test_rank_zero()
    test_blockize()

    #test_partial_tile()

    test_from_gpu()
    test_bind()

