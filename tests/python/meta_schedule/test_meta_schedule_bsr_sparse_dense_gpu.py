import numpy as np
import scipy.sparse as sp

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import te
from tvm import tir
from tvm import topi
from tvm import meta_schedule as ms
from tvm.script import ty
from tvm.topi.util import get_const_tuple
from tvm.topi.cuda.sparse import pad_sparse_matrix


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


@tvm.script.tir
def transpose(x: ty.handle, x_t: ty.handle) -> None:
    tir.func_attr({"tir.noalias": True})
    M = tir.var("int32")
    K = tir.var("int32")
    num_blocks = tir.var("int32")
    bs_r = tir.var("int32")
    bs_c = tir.var("int32")
    tir.func_attr({"flop_ct": 2 * M * num_blocks * bs_r * K, "tag": "sparse_dense_bsr", "tir.noalias": True})

    X = tir.match_buffer(x, [M, K], "float32")
    X_t = tir.match_buffer(x_t, [K, M], "float32")

    for i, j in tir.grid(K, M):
        with tir.block([K, M], 'X_t') as [k, m]:
            X_t[k, m] = X[m, k]


def schedule_transpose_cuda(func):
    s = tir.create_schedule(func)
    warp_size = int(tvm.target.Target.current(allow_none=False).thread_warp_size)
    X_t = s.get_block('X_t')
    m, n = s.get_axes(X_t)
    mo, mi = s.split(m, factor=warp_size)
    no, ni = s.split(n, factor=warp_size)
    s.reorder(mo, no, mi, ni)
    s.bind(mo, tir.thread_axis('blockIdx.x'))
    s.bind(no, tir.thread_axis('blockIdx.y'))
    c = s.cache_read(X_t, 0, 'shared')
    s.compute_at(c, no)
    s.bind(ni, tir.thread_axis('threadIdx.x'))
    # use 4 warps
    m_i_o, m_i_i = s.split(mi, nparts=4)
    s.bind(m_i_o, tir.thread_axis('threadIdx.y'))

    m_c_i, n_c_i = s.get_axes(c)[-2:]
    s.bind(n_c_i, tir.thread_axis('threadIdx.x'))
    m_c_i_o, m_c_i_i = s.split(m_c_i, nparts=4)
    s.bind(m_c_i_o, tir.thread_axis('threadIdx.y'))
    return s


def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype):
    import itertools

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


def meta_schedule_sparse_dense_cuda(func, f_create_args):
    def schedule_sparse_dense(s: ms.Schedule):
        sparse_dense = s.get_block('sparse_dense')
        m, n, bb, k = s.get_axes(sparse_dense)
        k_tiles = s.sample_perfect_tile(loop=k, n_splits=2)
        ko, ki = s.split(k, k_tiles)
        sparse_dense_rf = s.rfactor(ki, 2) # rf is in global scope
        sparse_dense_rf_local = s.cache_write(sparse_dense_rf, 0, 'local')
        ki_rf, m_rf, n_rf, bb_rf, ko_rf = s.get_axes(sparse_dense_rf_local)
        s.reorder([m_rf, n_rf, ki_rf, bb_rf, ko_rf])
        s.bind(m_rf, 'blockIdx.x')
        s.bind(n_rf, 'blockIdx.y')
        s.bind(ki_rf, 'threadIdx.x')
        s.compute_inline(sparse_dense_rf)
        s.reverse_compute_at(sparse_dense, n_rf)
        all_reduce_axis = s.get_axes(sparse_dense)[-1]
        s.bind(all_reduce_axis, 'threadIdx.x')
        s.decompose_reduction(sparse_dense_rf_local, bb_rf)

    task = ms.SearchTask(func, task_name=sparse_dense_bsr.__qualname__, target='cuda -arch=sm_75')
    runner = ms.measure.RPCRunner(f_create_args=f_create_args)
    measurer = ms.measure.ProgramMeasurer(runner=runner)
    space = ms.space.ScheduleFn(schedule_sparse_dense)
    strategy = ms.strategy.Replay(num_trials=200)
    sch = ms.autotune(task=task, space=space, strategy=strategy, measurer=measurer, verbose=True)
    return sch


@tvm.script.tir
def sparse_dense_bsr_padded(x_t: ty.handle, data: ty.handle, indices: ty.handle, indptr: ty.handle, bsrmm: ty.handle, N_blocks: ty.int32) -> None:
    # sparsed dense bsr on CUDA, ported from TOPI.
    tir.func_attr({"tir.noalias": True})
    M = tir.var("int32")
    K = tir.var("int32")
    num_blocks = tir.var("int32")
    bs_r = tir.var("int32")
    bs_c = tir.var("int32")
    tir.func_attr({"flop_ct": 2 * M * num_blocks * bs_r * K, "tag": "sparse_dense_bsr", "tir.noalias": True})

    X_t = tir.match_buffer(x_t, [K, M], "float32")
    W_data = tir.match_buffer(data, [num_blocks, bs_r, bs_c], "float32")
    W_indices = tir.match_buffer(indices, [num_blocks], "int32")
    W_indptr = tir.match_buffer(indptr, [N_blocks + 1], "int32")
    BSRmm = tir.match_buffer(bsrmm, [M, N_blocks*bs_r], "float32")

    warp_size : ty.int32 = 32
    M_blocks : ty.int32 = M // bs_r

    with tir.block([], "root") as []:
        for ax0_outer in range(0, tir.floordiv(M // bs_r + warp_size - 1, warp_size), annotation={"loop_type": "blockIdx.x"}):
            for ax1_outer in range(0, N_blocks, annotation={"loop_type": "blockIdx.y"}):
                with tir.block([tir.floordiv(M // bs_r + warp_size - 1, warp_size), N_blocks], "GPU_Block", exec_scope="gpu_block") as [bx, by]:
                    w_indices_cache = tir.buffer_allocate((warp_size,), "int32", scope="warp")
                    w_data_cache = tir.buffer_allocate((warp_size, bs_r, bs_c), "float32", scope="warp")
                    for ax0_inner in range(0, 32, annotation={"loop_type": "threadIdx.x"}):
                        for ax1_inner in range(0, 1, annotation={"loop_type": "threadIdx.y"}):
                            with tir.block([32, 1], "GPU_Thread", exec_scope="gpu_thread") as [tx, ty]:
                                m_index : ty.int32 = bx * warp_size + tx
                                n_index : ty.int32 = by + ty
                                data_cache = tir.buffer_allocate((warp_size, bs_r, bs_c), "float32", name="data_cache", scope="local")
                                # zero block
                                block = tir.buffer_allocate([bs_r, bs_r], 'float32', scope='local')
                                for i in range(0, bs_r, annotation={"loop_type": "unroll"}):
                                    for j in range(0, bs_r, annotation={"loop_type": "unroll"}):
                                        with tir.block([bs_r, bs_r], 'block_init') as [x, y]:
                                            block[x, y] = tir.float32(0)

                                # compute into thread local storage using warp_size chunks
                                for ax4 in range(0, (W_indptr[n_index + 1] - W_indptr[n_index]) // warp_size):
                                    with tir.block([(W_indptr[n_index + 1] - W_indptr[n_index]) // warp_size], 'rowlength_bo') as [bb]:
                                        # cache indices
                                        elem_idx : ty.int32 = W_indptr[n_index] + bb * warp_size + tx
                                        w_indices_cache[tx] = W_indices[elem_idx]
                                        # cache dense matrix
                                        # each thread has a row
                                        for bi in range(0, warp_size):
                                            for x in range(0, bs_r, annotation={"loop_type": "unroll"}):
                                                for z in range(0, bs_c, annotation={"loop_type": "unroll"}):
                                                    # This memory acces should be out of bounds when
                                                    # m_index >= mb (which occurs when the dense matrix
                                                    # rows % 32 != 0), but it seems to work just fine...
                                                    data_cache[bi, x, z] = X_t[w_indices_cache[bi] * bs_c + z, (bx * 32 + tx) * bs_r + x]
                                        # cache w_data
                                        for y in range(0, bs_r, annotation={"loop_type": "unroll"}):
                                            for z in range(0, bs_c, annotation={"loop_type": "unroll"}):
                                                w_data_cache[tx, y, z] = W_data[elem_idx, y, z]

                                        for i in range(0, warp_size):
                                            # thread local block matmul
                                            for x in range(0, bs_r, annotation={"loop_type": "unroll"}):
                                                for y in range(0, bs_r, annotation={"loop_type": "unroll"}):
                                                    for z in range(0, bs_c, annotation={"loop_type": "unroll"}):
                                                        block[x, y] = block[x, y] + data_cache[i, x, z] * w_data_cache[i, y, z]

                                # store results
                                for i in range(0, bs_r, annotation={"loop_type": "unroll"}):
                                   for j in range(0, bs_r, annotation={"loop_type": "unroll"}):
                                        with tir.block([bs_r, bs_r], 'BSRmm') as [x, y]:
                                            if m_index < M_blocks and n_index < N_blocks:
                                                BSRmm[(bx * 32 + tx) * bs_r + x, (by + ty) * bs_r + y] = block[x, y]


def test_sparse_dense_padded():
    M = 512
    N = 3072
    K = 768
    BS_R = 16
    BS_C = 1
    density = 0.15

    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
    W_sp_np = pad_sparse_matrix(W_sp_np, 32)

    W_np = W_sp_np.todense()
    Y_np = np.array(X_np.dot(W_np.T))

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype), name='W_data')
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype), name='W_indices')
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype), name='W_indptr')
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))

    print("M =", M, "N =", N, "K =", K, "BS_R =", BS_R, "BS_C = ", BS_C)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        # te schedule
        with tvm.target.Target(device):
            Y = topi.cuda.sparse_dense_padded(X, W_data, W_indices, W_indptr)
            s = topi.cuda.schedule_sparse_dense_padded([Y])
            func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
            Y_tvm = tvm.nd.array( np.zeros(Y_np.shape, dtype=Y_np.dtype), ctx=ctx)
            func(
                tvm.nd.array(X_np, ctx=ctx),
                tvm.nd.array(W_sp_np.data, ctx=ctx),
                tvm.nd.array(W_sp_np.indices, ctx=ctx),
                tvm.nd.array(W_sp_np.indptr, ctx=ctx),
                Y_tvm,
            )
            tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)
            evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
            print("sparse dense te schedule: %f ms" % (evaluator(tvm.nd.array(X_np, ctx=ctx),
                                                                 tvm.nd.array(W_sp_np.data, ctx=ctx),
                                                                 tvm.nd.array(W_sp_np.indices, ctx=ctx),
                                                                 tvm.nd.array(W_sp_np.indptr, ctx=ctx),
                                                                 Y_tvm).mean * 1e3))

        # tir schedule
        with tvm.target.Target(device):
            func = transpose
            func = func.specialize(func.params[0], tir.decl_buffer([M, K]))
            s = schedule_transpose_cuda(func)
            func = tvm.build(s.func)
            X_t_tvm = tvm.nd.array(np.zeros(X_np.T.shape, dtype=X_np.dtype), ctx=ctx)
            func(tvm.nd.array(X_np, ctx=ctx), X_t_tvm)
            tvm.testing.assert_allclose(X_t_tvm.asnumpy(), X_np.T, atol=1e-5, rtol=1e-5)
            evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
            print("transpose tir schedule: %f ms" % (evaluator(tvm.nd.array(X_np, ctx=ctx),
                                                                  X_t_tvm).mean * 1e3))
            func = sparse_dense_bsr_padded
            x_t, data, _, _, _, N_blocks = func.params
            func = func.specialize(x_t, tir.decl_buffer([K, M]))
            func = func.specialize(data, tir.decl_buffer(W_data.shape))
            func = func.specialize(N_blocks, N // BS_R).remove_const_param(N_blocks)

            s = tir.create_schedule(func)
            func = tvm.build(s.func)
            Y_tvm = tvm.nd.array(np.zeros(Y_np.shape, dtype=Y_np.dtype), ctx=ctx)
            func(
                tvm.nd.array(X_np.T, ctx=ctx),
                tvm.nd.array(W_sp_np.data, ctx=ctx),
                tvm.nd.array(W_sp_np.indices, ctx=ctx),
                tvm.nd.array(W_sp_np.indptr, ctx=ctx),
                Y_tvm
            )

            tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-5, rtol=1e-5)
            evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
            print("sparse dense tir schedule: %f ms" % (evaluator(tvm.nd.array(X_np.T, ctx=ctx),
                                                                  tvm.nd.array(W_sp_np.data, ctx=ctx),
                                                                  tvm.nd.array(W_sp_np.indices, ctx=ctx),
                                                                  tvm.nd.array(W_sp_np.indptr, ctx=ctx),
                                                                  Y_tvm).mean * 1e3))
    for device in ["cuda"]:
        check_device(device)


def test_sparse_dense():
    M = 512
    N = 3072
    K = 768
    BS_R = 16
    BS_C = 16
    density = 0.15

    X_np = np.random.randn(M, K).astype("float32")
    W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
    W_sp_np = pad_sparse_matrix(W_sp_np, 32)

    W_np = W_sp_np.todense()
    Y_np = np.array(X_np.dot(W_np.T))

    W_data = te.placeholder(shape=W_sp_np.data.shape, dtype=str(W_sp_np.data.dtype), name='W_data')
    W_indices = te.placeholder(shape=W_sp_np.indices.shape, dtype=str(W_sp_np.indices.dtype), name='W_indices')
    W_indptr = te.placeholder(shape=W_sp_np.indptr.shape, dtype=str(W_sp_np.indptr.dtype), name='W_indptr')
    X = te.placeholder(shape=X_np.shape, dtype=str(X_np.dtype))

    print("M =", M, "N =", N, "K =", K, "BS_R =", BS_R, "BS_C = ", BS_C)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not tvm.testing.device_enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        # te schedule
        with tvm.target.Target(device):
            Y = topi.cuda.sparse_dense(X, W_data, W_indices, W_indptr)
            s = topi.cuda.schedule_sparse_dense([Y])
            func = tvm.build(s, [X, W_data, W_indices, W_indptr, Y])
            Y_tvm = tvm.nd.array( np.zeros(Y_np.shape, dtype=Y_np.dtype), ctx=ctx)
            func(
                tvm.nd.array(X_np, ctx=ctx),
                tvm.nd.array(W_sp_np.data, ctx=ctx),
                tvm.nd.array(W_sp_np.indices, ctx=ctx),
                tvm.nd.array(W_sp_np.indptr, ctx=ctx),
                Y_tvm,
            )
            tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)
            evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
            print("sparse dense te schedule: %f ms" % (evaluator(tvm.nd.array(X_np, ctx=ctx),
                                                                 tvm.nd.array(W_sp_np.data, ctx=ctx),
                                                                 tvm.nd.array(W_sp_np.indices, ctx=ctx),
                                                                 tvm.nd.array(W_sp_np.indptr, ctx=ctx),
                                                                 Y_tvm).mean * 1e3))

        # tir auto tune schedule
        with tvm.target.Target(device):
            func = sparse_dense_bsr
            x, data, _, _, _, N_blocks = func.params
            func = func.specialize(x, tir.decl_buffer([M, K]))
            func = func.specialize(data, tir.decl_buffer(W_data.shape))
            func = func.specialize(N_blocks, N // BS_R).remove_const_param(N_blocks)

            def f_create_args(ctx):
                X = tvm.nd.array(X_np, ctx=ctx)
                W_data = tvm.nd.array(W_sp_np.data, ctx=ctx)
                W_indices = tvm.nd.array(W_sp_np.indices, ctx=ctx)
                W_indptr = tvm.nd.array(W_sp_np.indptr, ctx=ctx)
                Y = tvm.nd.array(Y_np, ctx=ctx)
                return [X, W_data, W_indices, W_indptr, Y]

            s = meta_schedule_sparse_dense_cuda(func, f_create_args)
            func = tvm.build(s.sch.func)
            Y_tvm = tvm.nd.array( np.zeros(Y_np.shape, dtype=Y_np.dtype), ctx=ctx)
            func(
                tvm.nd.array(X_np, ctx=ctx),
                tvm.nd.array(W_sp_np.data, ctx=ctx),
                tvm.nd.array(W_sp_np.indices, ctx=ctx),
                tvm.nd.array(W_sp_np.indptr, ctx=ctx),
                Y_tvm
            )
            tvm.testing.assert_allclose(Y_tvm.asnumpy(), Y_np, atol=1e-4, rtol=1e-4)
            evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
            print("sparse dense tir schedule: %f ms" % (evaluator(tvm.nd.array(X_np, ctx=ctx),
                                                                 tvm.nd.array(W_sp_np.data, ctx=ctx),
                                                                 tvm.nd.array(W_sp_np.indices, ctx=ctx),
                                                                 tvm.nd.array(W_sp_np.indptr, ctx=ctx),
                                                                 Y_tvm).mean * 1e3))


    for device in ["cuda"]:
        check_device(device)


test_sparse_dense()