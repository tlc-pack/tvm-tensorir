import tvm
import topi
from tvm import tensorir
from tvm import ir_pass

def get_phase0(s, args, simple_mode=True):
    """get statement after phase 0"""
    ret = []

    def fetch_pass(stmt):
        ret.append(stmt)
        return stmt

    with tvm.build_config(add_lower_pass=[(0, fetch_pass)]):
        tvm.lower(s, args, simple_mode=simple_mode)

    return ret[0]


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
    stmt = get_phase0(s, [A, B, D])

    # schedule
    s = tensorir.create_schedule(stmt)

    init, reduction, relu = s.blocks()

    jo, io, ji_init, ii_init = s.axis(init)
    _, _, ko, ji, ii, ki = s.axis(reduction)

    ji, ko = s.reorder(ko, ji)
    ii, ko = s.reorder(ko, ii)

    s.fuse(ji_init, ii_init)
    s.fuse(ko, ki)

    print(s.root)


def test_inline():
    N = M = 128

    # 
    A = tvm.placeholder((N, M), name='A')
    B = tvm.compute((N, M), lambda i, j: A[i][j] + 1, name='B')
    C = tvm.compute((N, M), lambda i, j: B[i][j] * 2, name='C')
    D = tvm.compute((N, M), lambda i, j: C[i][j] - 3, name='D')

    s = tvm.create_schedule([C.op, D.op])
    s[D].split(s[D].op.axis[0], 8)
    stmt = get_phase0(s, [A, C, D])

    print(stmt)

    s = tensorir.create_schedule(stmt)

    B, C, D = s.blocks()
    i, j = s.axis(C)

    io, ii = s.split(i, 16)  # useless split

    s.compute_inline(C)

    print(s.root)
    print(s.to_halide())


def test_compute_at():
    N = M = K = 128

    # 
    A = tvm.placeholder((N, M, K), name='A')
    B = tvm.compute((N, M, K), lambda i, j, k: A[i, j, k] + 1, name='B')
    C = tvm.compute((N, M, K), lambda i, j, k: B[i, j, k] * 2, name='C')
    D = tvm.compute((N, M, K), lambda i, j, k: C[i, j, k] - 3, name='D')

    s = tvm.create_schedule([C.op, D.op])
    i, j, k = C.op.axis
    s[C].split(i, 8)
    stmt = get_phase0(s, [A, C, D])

    s = tensorir.create_schedule(stmt)
    print(s.root)

    B, C, D = s.blocks()
    i, j, k = s.axis(D)
    s.compute_at(C, i)

    print(s.root)

def test_unroll():
    N = 128
    M = 8

    A = tvm.placeholder((N, M), name='A')
    B = tvm.compute((N, M), lambda i, j: A[tvm.if_then_else(j % 2 == 0, 0, 1)][j], name='B')

    s = tvm.create_schedule([B.op])
    s[B].unroll(B.op.axis[1])
    stmt = get_phase0(s, [A, B])

    s = tensorir.create_schedule(stmt)

    print(s.root)
    B, = s.blocks()
    i, j = s.axis(B)

    blocks = s.unroll(j)

    print(ir_pass.CanonicalSimplify(s.to_halide()))


def test_tile():
    N = M = K = 128

    # a vanilla gemm
    A = tvm.placeholder((N, K), name='A')
    B = tvm.placeholder((M, K), name='B')
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k), name='C')

    s = tvm.create_schedule([C.op])
    stmt = get_phase0(s, [A, B, C])

    # schedule
    print(stmt)

    s = tensorir.create_schedule(stmt)
    init, reduction = s.statements()

    i, j, k = s.axis(reduction)

    io, ii = s.split(i, 8)
    jo, ji = s.split(j, 8)
    ko, ki = s.split(k, 8)
    s.reorder(io, jo, ko, ji, ii, ki)    # automatic done : move init part

    stmt = s.gen_stmt()
    print(stmt)

def test_partial_tile():
    N = M = K = 128

    # a vanilla gemm
    A = tvm.placeholder((N, K), name='A')
    B = tvm.placeholder((M, K), name='B')
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k), name='C')

    s = tvm.create_schedule([C.op])
    stmt = get_phase0(s, [A, B, C])

    # schedule
    print(stmt)

    s = tensorir.create_schedule(stmt)
    init, reduction = s.statements()

    i, j, k = s.axis(reduction)

    io, ii = s.split(i, 7)   # partial tile binded to io
    jo, ji = s.split(j, 7)
    ko, ki = s.split(k, 7)
    s.reorder(io, jo, ko, ji, ii, ki)

def test_gpu():
    N = M = 128

    # 
    A = tvm.placeholder((N, M), name='A')
    B = tvm.compute((N, M), lambda i, j: A[i][j] + 1, name='B')
    C = tvm.compute((N, M), lambda i, j: B[i][j] * 2, name='C')

    s = tvm.create_schedule([C.op])
    stmt = get_phase0(s, [A, B, C])

    #
    s = tensorir.create_schedule(stmt)
    B, C = s.blocks()

    i, j = s.axis(B)
    s.move(C, j)

    s.bind(i, tvm.thread_axis('blockIdx.x'))
    s.bind(j, tvm.thread_axis('threadIdx.x'))

if __name__ == "__main__":
    test_decompile_fuse()
    test_inline()
    test_compute_at()
    test_unroll()
    #test_tile()
    #test_partial_tile()
    #test_gpu()
    #test_memory_scope()

