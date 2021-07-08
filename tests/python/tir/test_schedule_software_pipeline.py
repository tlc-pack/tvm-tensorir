import numpy as np
import pytest
import tvm
from tvm import tir, testing
from tvm.script import ty

import util


def test_cuda_pipeline():
    device = "cuda"
    ctx = tvm.context(device, 0)
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return

    s = tir.Schedule(util.matmul_stmt())
    C = s.get_block('update')
    i, j, k = s.get_axes(C)
    io, ii = s.split(i, factors=[8, 16])
    jo, ji = s.split(j, factors=[8, 16])
    ko, ki = s.split(k, factors=[8, 16])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.bind(io, 'blockIdx.x')
    s.bind(jo, 'threadIdx.x')

    A_shared = s.cache_read(C, 1, 'shared')
    B_shared = s.cache_read(C, 2, 'shared')
    s.compute_at(A_shared, ko)
    s.compute_at(B_shared, ko)

    for load in [A_shared, B_shared]:
        _, tt = s.split(s.fuse(s.get_axes(load)[-2]), factors=[None, 8])
        s.bind(tt, 'threadIdx.x')
    s.software_pipeline(ko, 2)

    f = tvm.build(s.mod['main'], None, target='cuda')
    a_np = np.random.uniform(size=(128, 128)).astype("float32")
    b_np = np.random.uniform(size=(128, 128)).astype("float32")
    a = tvm.nd.array(a_np, ctx=ctx)
    b = tvm.nd.array(b_np, ctx=ctx)
    c = tvm.nd.array(np.zeros((128, 128), dtype="float32"), ctx=ctx)
    f(a, b, c)
    c_np = np.matmul(a_np, b_np.T)
    tvm.testing.assert_allclose(c.asnumpy(), c_np, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    test_cuda_pipeline()