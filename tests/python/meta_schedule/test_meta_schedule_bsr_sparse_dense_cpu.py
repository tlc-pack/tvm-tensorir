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
# pylint: disable=missing-function-docstring
import os

import numpy as np
import pytest
import scipy.sparse as sp
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import meta_schedule as ms
from tvm import te, tir, topi
from tvm._ffi.base import TVMError
from tvm.script import ty

RPC_KEY = "test"

# fmt: off
# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks

@tvm.script.tir
def sparse_dense_bsr(x: ty.handle, data: ty.handle, indices: ty.handle, indptr: ty.handle, bsrmm: ty.handle, N_blocks: ty.int32) -> None:
    M = tir.var("int32")
    K = tir.var("int32")
    num_blocks = tir.var("int32")
    bs_r = tir.var("int32")
    bs_c = tir.var("int32")
    tir.func_attr({"flop_ct": 2 * M * num_blocks * bs_r * K, "tir.noalias": True})

    X = tir.match_buffer(x, [M, K], "float32")
    W_data = tir.match_buffer(data, [num_blocks, bs_r, bs_c], "float32")
    W_indices = tir.match_buffer(indices, [num_blocks], "int32")
    W_indptr = tir.match_buffer(indptr, [N_blocks + 1], "int32")
    BSRmm = tir.match_buffer(bsrmm, [M, N_blocks*bs_r], "float32")

    for i, j in tir.grid(M, N_blocks*bs_r):
        for block_offset, k in tir.grid(W_indptr[tir.floordiv(j, bs_r) + 1] - W_indptr[tir.floordiv(j, bs_r)], bs_c):
            with tir.block([M, N_blocks*bs_r, tir.reduce_axis(0, W_indptr[j + 1] - W_indptr[j]), tir.reduce_axis(0, bs_c)], "sparse_dense") as [m, n, offset, kk]:
                with tir.init():
                    BSRmm[m, n] = tir.float32(0)
                BSRmm[m, n] = BSRmm[m, n] + W_data[offset + W_indptr[tir.floordiv(n, bs_r)], tir.floormod(n, bs_r), kk]*X[m, bs_c*W_indices[offset+W_indptr[tir.floordiv(n, bs_r)]]+kk]


_sparse_dense_implement_te = {
    "generic": (topi.nn.sparse_dense, topi.generic.schedule_sparse_dense),
    "cpu": (topi.nn.sparse_dense, topi.x86.schedule_sparse_dense),
}


def meta_schedule_sparse_dense_llvm(func, f_create_args):
    def schedule_sparse_dense(s: tir.Schedule):
        sparse_dense = s.get_block("sparse_dense")
        sparse_dense_local = s.cache_write(sparse_dense, 0, "local")
        i, j, offset, k = s.get_loops(sparse_dense_local)
        i_tiles = s.sample_perfect_tile(n_splits=4, loop=i)
        j_tiles = s.sample_perfect_tile(n_splits=4, loop=j)
        i_0, i_1, i_2, i_3 = s.split(i, i_tiles)
        j_0, j_1, j_2, j_3 = s.split(j, j_tiles)
        s.reorder([i_0, j_0, i_1, j_1, i_2, j_2, offset, k, i_3, j_3])
        s.reverse_compute_at(sparse_dense, j_1)
        outer_fused = s.fuse(s.get_loops(sparse_dense)[:4])
        s.parallel(outer_fused)
        s.mark_loop(outer_fused, "auto_unroll_max_step", tir.IntImm("int32", 512))
        s.mark_loop(outer_fused, "unroll_explicit", tir.IntImm("int32", 1))
        s.decompose_reduction(sparse_dense_local, offset)
        s.vectorize(j_3)
        try:
            j_init = s.get_loops(s.get_block("sparse_dense_init"))[-1]
            s.vectorize(j_init)
        except:  # pylint: disable=bare-except
            pass

    task = ms.SearchTask(func, task_name=sparse_dense_bsr.__qualname__)
    runner = ms.measure.RPCRunner(f_create_args=f_create_args)
    measurer = ms.measure.ProgramMeasurer(runner=runner)
    space = ms.space.ScheduleFn(schedule_sparse_dense)
    strategy = ms.strategy.Replay(num_trials=200)
    sch = ms.autotune(task=task, space=space, strategy=strategy, measurer=measurer, verbose=False)
    return sch


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
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r : r + BS_R, c : c + BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (M // BS_R + 1,)
    return s


@pytest.mark.skip(reason="needs RPC")
def test_sparse_dense():
    os.environ["TVM_TRACKER_KEY"] = RPC_KEY
    for _ in range(1):
        # BERT
        M = 128
        N = 3072
        K = 768
        BS_R = 16
        BS_C = 1
        density = 0.15

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
                fcompute, fschedule = tvm.topi.testing.dispatch(device, _sparse_dense_implement_te)
                Y = fcompute(X, W_data, W_indices, W_indptr)
                s = fschedule([Y])
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

            # auto tir schedule
            with tvm.target.Target(device):
                func = sparse_dense_bsr
                x, data, _, _, _, N_blocks = func.params
                func = func.specialize(x, tir.decl_buffer([M, K]))
                func = func.specialize(data, tir.decl_buffer(W_data.shape))
                func = func.specialize(N_blocks, N // BS_R).remove_const_param(N_blocks)

                def f_create_args(ctx):
                    X = tvm.nd.array(X_np, device=ctx)
                    W_data = tvm.nd.array(W_sp_np.data, device=ctx)
                    W_indices = tvm.nd.array(W_sp_np.indices, device=ctx)
                    W_indptr = tvm.nd.array(W_sp_np.indptr, device=ctx)
                    Y = tvm.nd.array(Y_np, device=ctx)
                    return [X, W_data, W_indices, W_indptr, Y]

                sch = meta_schedule_sparse_dense_llvm(func, f_create_args)
                func = sch.mod

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
                    "sparse dense auto tir schedule: %f ms"
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
