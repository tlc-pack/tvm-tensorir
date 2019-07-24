import tvm
from tvm import tensorir
from tvm import ir_pass, register_func

from test_schedule import check_correctness

@register_func
def vadd_for_test(c, a, b):
    c[:] = a.asnumpy() + b.asnumpy()

def intrin_vadd(n):
    X = tvm.placeholder((n,), name='X')
    Y = tvm.placeholder((n,), name='Y')
    Z = tvm.compute(X.shape, lambda i: X[i] + Y[i], name='Z')

    def intrin_func(ins, outs):  # (List[TensorRegion]) -> Stmt or BlockTreeNode
        ib = tvm.ir_builder.create()

        Bx = ins[0].emit_buffer_bind(ib)
        By = ins[1].emit_buffer_bind(ib)
        Bz = outs[0].emit_buffer_bind(ib)

        ib.emit(tvm.call_packed('vadd_for_test', Bz, Bx, By))
        return ib.get()

    return tensorir.decl_tensor_intrin(Z.op, intrin_func, 'intrin_vadd')

def test_tensorize_vadd():
    N = 128
    M = 128

    A = tvm.placeholder((N, M), name='A')
    B = tvm.placeholder((N, M), name='B')
    C = tvm.compute((N, M), lambda i, j: A[i][j] + B[i][j], name='C')

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        C, = s.blocks()
        i, j = s.axis(C)

        jo, ji = s.split(j, 8)

        bb = s.blockize(ji)
        s.tensorize(bb, intrin_vadd(8))

        stmt = s.to_halide()
        return stmt

    s = tvm.create_schedule([C.op])
    check_correctness(s, [A, B, C], _schedule_pass)

@register_func
def dot_for_test(c, a, b):
    c[:] = c.asnumpy() + a.asnumpy().dot(b.asnumpy())

def intrin_dot(n):
    X = tvm.placeholder((n,), name='X')
    Y = tvm.placeholder((n,), name='Y')
    k = tvm.reduce_axis((0, n), name='k')
    Z = tvm.compute((1,), lambda i: tvm.sum(X[k] * Y[k], axis=k), name='Z')

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()

        Bx = ins[0].emit_buffer_bind(ib)
        By = ins[1].emit_buffer_bind(ib)
        Bz = outs[0].emit_buffer_bind(ib)

        ib.emit(tvm.call_packed('dot_for_test', Bz, Bx, By))
        return ib.get()

    return tensorir.decl_tensor_intrin(Z.op, intrin_func, 'dot')

def test_tensorize_dot():
    N = M = K = 32

    A = tvm.placeholder((N, K), name='A')
    B = tvm.placeholder((M, K), name='B')
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i][k] * B[j][k], axis=k), name='C')

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        init, update = s.blocks()
        i, j, k = s.axis(update)

        ko, ki = s.split(k, 16)

        B = s.blockize(ki)
        s.tensorize(B, intrin_dot(16))

        stmt = s.to_halide()
        return stmt

    s = tvm.create_schedule([C.op])
    check_correctness(s, [A, B, C], _schedule_pass)


def intrin_composed_gemm(N, M, K):
    X = tvm.placeholder((N, K), name='X')
    Y = tvm.placeholder((M, K), name='Y')
    k = tvm.reduce_axis((0, N), name='k')
    Z = tvm.compute((N, M), lambda i, j: tvm.sum(X[i, k] * Y[j, k], axis=k), name='Z')

    DOT_K = 4
    dot = intrin_dot(DOT_K)

    def intrin_func(ins, outs):
        ib = tvm.ir_builder.create()


        C, = outs
        A, B, _ = ins

        with ib.axis_tree_node('i', 0, N) as i:
            with ib.axis_tree_node('j', 0, M) as j:
                with ib.axis_tree_node('ko', 0, K // DOT_K) as ko:
                    block = dot([A[i:i+1, ko*DOT_K:ko*DOT_K+DOT_K], B[j:j+1, ko*DOT_K:ko*DOT_K+DOT_K]], [C[i:i+1, j:j+1],])
                    ib.emit(block)
        return ib.get()

    return tensorir.decl_tensor_intrin(Z.op, intrin_func, 'dot')


def test_tensorize_composed_gemm():
    N, M, K = 32, 32, 32

    A = tvm.placeholder((N, K), name='A')
    B = tvm.placeholder((M, K), name='B')
    k = tvm.reduce_axis((0, N), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k), name='C')

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        init, update = s.blocks()
        i, j, k = s.axis(update)

        io, ii = s.split(i, 8)
        jo, ji = s.split(j, 8)
        ko, ki = s.split(k, 8)

        s.reorder(ii, jo)
        s.compute_at(init, jo)

        s.reorder(ji, ko)
        s.reorder(ii, ko)

        B = s.blockize(ii)
        B = s.tensorize(B, intrin_composed_gemm(8, 8, 8))

        stmt = s.to_halide()
        return stmt

    s = tvm.create_schedule([C.op])
    check_correctness(s, [A, B, C], _schedule_pass)

def test_untensorize():
    pass

def test_schedule_after_tensorize():
    N = 128
    M = 128

    A = tvm.placeholder((N, M), name='A')
    B = tvm.placeholder((N, M), name='B')
    C = tvm.compute((N, M), lambda i, j: A[i][j] + B[i][j], name='C')
    D = tvm.compute((N, M), lambda i, j: C[i][j] + B[i][j], name='D')

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

#        C, = s.blocks()
#        i, j = s.axis(C)
#
#        jo, ji = s.split(j, 8)
#
#        bb = s.blockize(ji)
#        bb = s.tensorize(bb, intrin_vadd(8))
#
        stmt = s.to_halide()
        return stmt

    s = tvm.create_schedule([D.op])
    check_correctness(s, [A, B, C, D], _schedule_pass)


def test_usage_in_compute_dsl():
    pass

if __name__ == "__main__":
    test_tensorize_vadd()
    test_tensorize_dot()
    test_tensorize_composed_gemm()
    test_untensorize()

    test_schedule_after_tensorize()
    test_usage_in_compute_dsl()

