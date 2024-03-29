# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-function-docstring,missing-module-docstring
import numpy as np
import scipy.sparse as sp
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import te, tir, topi
from tvm.script import ty

# pylint: disable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name


@tvm.script.tir
def sparse_dense_bsr(
    x: ty.handle,
    data: ty.handle,
    indices: ty.handle,
    indptr: ty.handle,
    bsrmm: ty.handle,
    N_blocks: ty.int32,
) -> None:
    M = tir.var("int32")
    K = tir.var("int32")
    num_blocks = tir.var("int32")
    bs_r = tir.var("int32")
    bs_c = tir.var("int32")

    X = tir.match_buffer(x, [M, K], "float32")
    W_data = tir.match_buffer(data, [num_blocks, bs_r, bs_c], "float32")
    W_indices = tir.match_buffer(indices, [num_blocks], "int32")
    W_indptr = tir.match_buffer(indptr, [N_blocks + 1], "int32")
    BSRmm = tir.match_buffer(bsrmm, [M, N_blocks * bs_r], "float32")
    BSRmm_block = tir.alloc_buffer([M, N_blocks, bs_r], "float32")
    for ax0, ax1, ax2 in tir.grid(M, N_blocks, bs_r):
        with tir.block([M, N_blocks, bs_r], "bsr_par") as [i, nb_j, j]:
            with tir.init():
                BSRmm_block[i, nb_j, j] = 0.0
            for ax3, ax4 in tir.grid(W_indptr[nb_j + 1] - W_indptr[nb_j], bs_c):
                with tir.block(
                    [
                        tir.reduce_axis(0, W_indptr[nb_j + 1] - W_indptr[nb_j]),
                        tir.reduce_axis(0, bs_c),
                    ],
                    "bsr_reduce",
                ) as [k, c]:
                    BSRmm_block[i, nb_j, j] = (
                        BSRmm_block[i, nb_j, j]
                        + W_data[k + W_indptr[nb_j], j, c]
                        * X[i, bs_c * W_indices[k + W_indptr[nb_j]] + c]
                    )

    for i, j in tir.grid(M, N_blocks * bs_r):
        with tir.block([M, N_blocks * bs_r], "bsr_block") as [m, n]:
            BSRmm[m, n] = BSRmm_block[m, tir.floordiv(n, bs_r), tir.floormod(n, bs_r)]


# pylint: enable=no-member,invalid-name,unused-variable,line-too-long,redefined-outer-name
# pylint: disable=invalid-name,cell-var-from-loop


def schedule_sparse_dense_llvm(func):
    s = tir.Schedule(func, debug_mode=True)
    bsr_par = s.get_block("bsr_par")
    bsr_block = s.get_block("bsr_block")
    i, j = s.get_loops(bsr_block)
    data = s.mod["main"].params[1]
    jo, ji = s.split(j, factors=[None, s.mod["main"].buffer_map[data].shape[1]])
    s.compute_at(bsr_par, ji)
    s.vectorize(ji)
    i_jo = s.fuse(i, jo)
    s.parallel(i_jo)
    return s.mod["main"]


_sparse_dense_implement_tir = {
    "llvm": schedule_sparse_dense_llvm,
}

_sparse_dense_implement_te = {
    "generic": (topi.nn.sparse_dense, topi.generic.schedule_sparse_dense),
    "cpu": (topi.nn.sparse_dense, topi.x86.schedule_sparse_dense),
}


def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype):
    import itertools  # pylint: disable=import-outside-toplevel

    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)
    ]
    for r, c in chosen_blocks:
        Y[r : r + BS_R, c : c + BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (M // BS_R + 1,)
    return s


def test_sparse_dense():
    for _ in range(20):
        BS_R = np.random.randint(1, 16)
        BS_C = np.random.randint(1, 16)
        M = np.random.randint(1, 32)
        N = int(np.random.randint(1, 16) * BS_R)
        K = int(np.random.randint(1, 16) * BS_C)
        density = np.clip(np.random.random(), 0.1, 0.9)
        X_np = np.random.randn(M, K).astype("float32")
        W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
        W_np = W_sp_np.todense()
        Y_np = np.array(X_np.dot(W_np.T))
        W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype))
        W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype))
        W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype))
        X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))

        print("M =", M, "N =", N, "K =", K, "BS_R =", BS_R, "BS_C = ", BS_C)

        def check_device(device):
            ctx = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print("Skip because %s is not enabled" % device)
                return
            print("Running on target: %s" % device)
            # te schedule
            with tvm.target.Target(device):
                f_compute, f_schedule = tvm.topi.testing.dispatch(
                    device, _sparse_dense_implement_te
                )
                Y = f_compute(X, W_data, W_indices, W_indptr)
                s = f_schedule([Y])
                func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
                Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), device=ctx)
                func(
                    tvm.nd.array(X_np, device=ctx),
                    tvm.nd.array(W_sp_np.data, device=ctx),
                    tvm.nd.array(W_sp_np.indices, device=ctx),
                    tvm.nd.array(W_sp_np.indptr, device=ctx),
                    Y_tvm,
                )
                tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)
                evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
                print(
                    "sparse dense te schedule: %f ms"
                    % (
                        evaluator(
                            tvm.nd.array(X_np, device=ctx),
                            tvm.nd.array(W_sp_np.data, device=ctx),
                            tvm.nd.array(W_sp_np.indices, device=ctx),
                            tvm.nd.array(W_sp_np.indptr, device=ctx),
                            Y_tvm,
                        ).mean
                        * 1e3
                    )
                )
            # tir schedule
            with tvm.target.Target(device):
                func = sparse_dense_bsr
                x, data, _, _, _, N_blocks = func.params
                func = func.specialize(
                    {
                        x: tir.decl_buffer([M, K]),
                        data: tir.decl_buffer(W_data.shape),
                        N_blocks: N // BS_R,
                    }
                )
                func = _sparse_dense_implement_tir[device](func)
                func = tvm.build(func)
                Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), device=ctx)
                func(
                    tvm.nd.array(X_np, device=ctx),
                    tvm.nd.array(W_sp_np.data, device=ctx),
                    tvm.nd.array(W_sp_np.indices, device=ctx),
                    tvm.nd.array(W_sp_np.indptr, device=ctx),
                    Y_tvm,
                )
                tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-5, rtol=1e-5)
                evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
                print(
                    "sparse dense tir schedule: %f ms"
                    % (
                        evaluator(
                            tvm.nd.array(X_np, device=ctx),
                            tvm.nd.array(W_sp_np.data, device=ctx),
                            tvm.nd.array(W_sp_np.indices, device=ctx),
                            tvm.nd.array(W_sp_np.indptr, device=ctx),
                            Y_tvm,
                        ).mean
                        * 1e3
                    )
                )

        for device in ["llvm"]:
            check_device(device)


if __name__ == "__main__":
    test_sparse_dense()
