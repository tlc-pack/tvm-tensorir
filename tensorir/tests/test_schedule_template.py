import tvm
import topi
import re
from tvm import tensorir

from test_common import check_correctness

def test_fused_matmul():
    N = M = K = 16

    def generate_compute_dsl(with_bias, with_relu, with_double, with_input_preprocess):
        A = tvm.placeholder((N, K), name='A')
        B = tvm.placeholder((M, K), name='B')
        bias = tvm.placeholder((M,), name='bias')

        bufs = [A, B]

        if with_input_preprocess:
            A = topi.abs(A)

        k = tvm.reduce_axis((0, N), name='k')
        C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k), name='C')

        if with_bias:
            C = tvm.compute((N, M), lambda i, j: C[i, j] + bias[j], name='C')
            bufs.append(bias)

        if with_relu:
            C = topi.nn.relu(C)

        if with_double:
            C = C * 2.0

        bufs.append(C)
        return tvm.create_schedule([C.op]), bufs

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        C_update = s.reduction_blocks()[0]
        C_init = s.predecessor(C_update)[0]

        i, j, k = s.axis(C_update)
        io, ii = s.split(i, 8)
        jo, ji = s.split(j, 8)
        ko, ki = s.split(k, 8)
        io, jo, ko, ii, ji, ki = s.reorder(io, jo, ko, ii, ji, ki)

        # inline elemwise/broadcast
        s.inline_all_injective()

        # fuse the last
        b = s.output_blocks()[0]
        if b != C_update:
            s.compute_after(b, jo)

        stmt = s.to_halide()
        return stmt

    for with_bias in [True, False]:
        for with_relu in [True, False]:
            for with_double in [True, False]:
                for with_input_preprocess in [True, False]:
                    s, bufs = generate_compute_dsl(with_bias, with_relu, 
                                                   with_double, with_input_preprocess)
                    stmt = check_correctness(s, bufs, _schedule_pass, return_stmt=True)
                    stmt = str(stmt)

                    if with_bias or with_relu or with_double:
                        assert(len(re.findall("\nfor", stmt)) == 1)   # assert post elemwise is inlined
                    if with_input_preprocess:
                        assert " + (fabs(A" in stmt     # assert input pre-processing in inlined

if __name__ == "__main__":
    test_fused_matmul()

