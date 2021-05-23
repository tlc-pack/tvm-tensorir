"""
Workload Definition
"""
import tvm
from tvm import te, topi

import logging
import numpy as np
logger = logging.getLogger(__name__)

from ..shared import CUTarget, CUDevice


def _builtin_dense_kernel(B, I, H, f):
    X = te.placeholder((B, I), name='X')
    W = te.placeholder((H, I), name='W')
    Y = f(X, W, transa=False, transb=True)
    sched = te.create_schedule(Y.op)
    cublas_kernel = tvm.build(sched, [X, W, Y], CUTarget)
    return cublas_kernel, Y


class cuBLASDenseFixture:
    __slots__ = 'B', 'I', 'H', 'cublas_kernel', \
                'X_np', 'W_np', \
                'Y', 'Y_np_expected'

    def __init__(self, B, I, H):
        self.B, self.I, self.H = B, I, H
        self.X_np = np.random.uniform(-0.1, 0.1, size=(B, I)).astype(np.float32)
        self.W_np = np.random.uniform(-0.1, 0.1, size=(H, I)).astype(np.float32)

        self.cublas_kernel, self.Y = _builtin_dense_kernel(B=B, I=I, H=H, f=tvm.contrib.cublas.matmul)

        module_data = [tvm.nd.array(self.X_np, device=CUDevice),
                       tvm.nd.array(self.W_np, device=CUDevice),
                       tvm.nd.array(np.empty(shape=topi.utils.get_const_tuple(self.Y.shape),
                                             dtype=np.float32), device=CUDevice)]
        self.cublas_kernel(*module_data)
        self.Y_np_expected = module_data[-1].asnumpy()


class cuTLASSDenseFixture:
    __slots__ = 'B', 'I', 'H', 'cutlass_kernel', 'X_np', 'W_np', 'Y', 'Y_np_expected'

    def __init__(self, cublas_fixture):
        self.B, self.I, self.H = cublas_fixture.B, cublas_fixture.I, cublas_fixture.H
        self.X_np = cublas_fixture.X_np
        self.W_np = cublas_fixture.W_np

        self.cutlass_kernel, self.Y = _builtin_dense_kernel(B=self.B, I=self.I, H=self.H, \
                                                            f=tvm.contrib.cutlass.matmul)

        module_data = [tvm.nd.array(self.X_np, device=CUDevice),
                       tvm.nd.array(self.W_np, device=CUDevice),
                       tvm.nd.array(np.empty(shape=topi.utils.get_const_tuple(self.Y.shape),
                                             dtype=np.float32), device=CUDevice)]
        self.cutlass_kernel(*module_data)
        self.Y_np_expected = module_data[-1].asnumpy()

        np.testing.assert_allclose(self.Y_np_expected,
                                   cublas_fixture.Y_np_expected,
                                   rtol=1e-3, atol=1e-3)
