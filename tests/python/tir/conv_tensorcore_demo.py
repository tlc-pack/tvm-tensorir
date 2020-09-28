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

import tvm
from tvm import tir
from tvm.contrib import nvcc

import numpy as np
from tvm.hybrid import ty


@tvm.hybrid.script
def conv(a: ty.handle, w: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (16, 14, 14, 32, 16, 16), "float32")
    A = tir.match_buffer(a, (16, 14, 14, 16, 16, 16), "float16")
    W = tir.match_buffer(w, (14, 14, 16, 32, 16, 16), "float16")
    reducer = tir.comm_reducer(lambda x, y: x + y, tir.float32(0))

    Apad = tir.buffer_allocate([16, 16, 16, 16, 16, 16], "float16")
    with tir.block([16, 16, 16, 16, 16, 16], "A_pad") as [n, h, w, i, nn, ii]:
        Apad[n, h, w, i, nn, ii] = tir.if_then_else(1 <= h < 15 and 1 <= w < 15,
                                                    A[n, h - 1, w - 1, i, nn, ii], tir.float16(0),
                                                    dtype="float16")
    with tir.block([16, 14, 14, 32, tir.reduce_axis(0, 16), tir.reduce_axis(0, 3),
                    tir.reduce_axis(0, 3), 16, 16, tir.reduce_axis(0, 16)], "Conv") as \
            [n, h, w, o, ic, kh, kw, nn, oo, ii]:
        reducer.step(C[n, h, w, o, nn, oo],
                     tir.cast(Apad[n, h + kh, w + kw, ic, nn, ii], "float32") *
                     tir.cast(W[kh, kw, ic, o, ii, oo], "float32"))


@tvm.hybrid.script
def gemm_desc(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_a")
    B = tir.match_buffer(b, (16, 16), "float16", align=128, offset_factor=1, scope="wmma.matrix_b")
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=1,
                         scope="wmma.accumulator")

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        for i, j, k in tir.grid(16, 16, 16):
            with tir.block([16, 16, tir.reduce_axis(0, 16)], "update") as [vii, vjj, vkk]:
                tir.bind(vii, i)
                tir.bind(vjj, j)
                tir.bind(vkk, k)
                C[vii, vjj] = C[vii, vjj] + tir.cast(A[vii, vkk], "float32") * tir.cast(B[vkk, vjj],
                                                                                        "float32")


@tvm.hybrid.script
def gemm_intrin(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=256, scope="wmma.matrix_a")
    B = tir.match_buffer(b, (16, 16), "float16", align=128, offset_factor=256, scope="wmma.matrix_b")
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=256,
                         scope="wmma.accumulator")

    with tir.block([16, 16, tir.reduce_axis(0, 16)], "root") as [vi, vj, vk]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.bind(vk, 0)
        tir.reads([C[0: 16, 0: 16], A[0: 16, 0: 16], B[0: 16, 0: 16]])
        tir.writes(C[0: 16, 0: 16])
        tir.evaluate(tir.tvm_mma_sync(C.data, C.elem_offset // 256,
                                      A.data, A.elem_offset // 256,
                                      B.data, B.elem_offset // 256,
                                      C.data, C.elem_offset // 256,
                                      dtype="handle"))


@tvm.hybrid.script
def fill_desc(c: ty.handle) -> None:
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=256,
                         scope="wmma.accumulator")

    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        for i, j in tir.grid(16, 16):
            with tir.block([16, 16], "init") as [vii, vjj]:
                tir.bind(vii, i)
                tir.bind(vjj, j)
                C[vii, vjj] = tir.float32(0)


@tvm.hybrid.script
def fill_intrin(c: ty.handle) -> None:
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=256,
                         scope="wmma.accumulator")

    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads([])
        tir.writes(C[0: 16, 0: 16])
        tir.evaluate(tir.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // 256, tir.float32(0),
                                           dtype="handle"))


@tvm.hybrid.script
def store_desc(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32", align=128, offset_factor=256,
                         scope="wmma.accumulator")
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=256,
                         scope="global")

    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        for i, j in tir.grid(16, 16):
            with tir.block([16, 16], "store") as [vii, vjj]:
                tir.bind(vii, i)
                tir.bind(vjj, j)
                C[vii, vjj] = A[vii, vjj]


@tvm.hybrid.script
def store_intrin(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32", align=128, offset_factor=256,
                         scope="wmma.accumulator")
    C = tir.match_buffer(c, (16, 16), "float32", align=128, offset_factor=256,
                         scope="global")

    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads(A[0: 16, 0: 16])
        tir.writes(C[0: 16, 0: 16])
        tir.evaluate(tir.tvm_store_matrix_sync(
            A.data, 16, 16, 16, A.elem_offset // 256, C.access_ptr("w"), 16, "row_major",
            dtype="handle"))


@tvm.hybrid.script
def load_a_desc(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=256,
                         scope="shared")
    C = tir.match_buffer(c, (16, 16), "float16", align=128, offset_factor=256,
                         scope="wmma.matrix_a")

    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        for i, j in tir.grid(16, 16):
            with tir.block([16, 16], "load") as [vii, vjj]:
                tir.bind(vii, i)
                tir.bind(vjj, j)
                C[vii, vjj] = A[vii, vjj]


@tvm.hybrid.script
def load_a_intrin(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=256,
                         scope="shared")
    C = tir.match_buffer(c, (16, 16), "float16", align=128, offset_factor=256,
                         scope="wmma.matrix_a")

    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads(A[0: 16, 0: 16])
        tir.writes(C[0: 16, 0: 16])
        tir.evaluate(tir.tvm_load_matrix_sync(
            C.data, 16, 16, 16, C.elem_offset // 256, A.access_ptr("r"), 16, "row_major",
            dtype="handle"))


@tvm.hybrid.script
def load_b_desc(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=256,
                         scope="shared")
    C = tir.match_buffer(c, (16, 16), "float16", align=128, offset_factor=256,
                         scope="wmma.matrix_b")

    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        for i, j in tir.grid(16, 16):
            with tir.block([16, 16], "load") as [vii, vjj]:
                tir.bind(vii, i)
                tir.bind(vjj, j)
                C[vii, vjj] = A[vii, vjj]


@tvm.hybrid.script
def load_b_intrin(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float16", align=128, offset_factor=256,
                         scope="shared")
    C = tir.match_buffer(c, (16, 16), "float16", align=128, offset_factor=256,
                         scope="wmma.matrix_b")

    with tir.block([16, 16], "root") as [vi, vj]:
        tir.bind(vi, 0)
        tir.bind(vj, 0)
        tir.reads(A[0: 16, 0: 16])
        tir.writes(C[0: 16, 0: 16])
        tir.evaluate(tir.tvm_load_matrix_sync(
            C.data, 16, 16, 16, C.elem_offset // 256, A.access_ptr("r"), 16, "row_major",
            dtype="handle"))


def build_and_test(func, device='cuda', num_runs=10):
    ctx = tvm.context(device, 0)
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return

    ctx = tvm.gpu(0)
    if nvcc.have_tensorcore(ctx.compute_version):
        with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
            func = tvm.build(func, target=device)
    a_np = np.random.uniform(size=(16, 14, 14, 16, 16, 16)).astype("float16")
    w_np = np.random.uniform(size=(14, 14, 16, 32, 16, 16)).astype("float16")
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    c = tvm.nd.array(np.zeros((16, 14, 14, 32, 16, 16), dtype="float32"), ctx)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=num_runs)
    print("conv2d with tensor core: %f ms" % (evaluator(a, w, c).mean * 1e3))


def test_tensorcore():
    mod = tvm.hybrid.create_module({"conv": conv})
    original_func = mod["conv"]

    A = original_func.buffer_map[original_func.params[0]]
    W = original_func.buffer_map[original_func.params[1]]
    C = original_func.buffer_map[original_func.params[2]]
    s = tir.create_schedule(original_func)

    Conv = s.get_block("Conv")
    APad = s.func.body.block.allocations[0].buffer

    AS = s.cache_read(APad, 'shared')
    WS = s.cache_read(W, 'shared')
    Block_AS = tir.schedule.get_stmt(AS)
    Block_WS = tir.schedule.get_stmt(WS)
    AF = s.cache_read(Block_AS.writes[0].buffer, "wmma.matrix_a")
    WF = s.cache_read(Block_WS.writes[0].buffer, "wmma.matrix_b")
    ConvF = s.cache_write(C, "wmma.accumulator")

    block_row_warps = 4
    block_col_warps = 2
    warp_row_tiles = 2
    warp_col_tiles = 4
    warp_size = 32
    chunk = 2

    block_x = tir.thread_axis('blockIdx.x')
    block_y = tir.thread_axis('blockIdx.y')
    block_z = tir.thread_axis('blockIdx.z')
    thread_x = tir.thread_axis('threadIdx.x')
    thread_y = tir.thread_axis('threadIdx.y')
    thread_z = tir.thread_axis('threadIdx.z')

    nc, hc, wc, oc, nnc, ooc = s.get_axes(Conv)
    block_k = s.fuse(hc, wc)
    s.bind(block_k, block_z)
    nc, nci = s.split(nc, factor=warp_row_tiles)
    block_i, nc = s.split(nc, factor=block_row_warps)
    oc, oci = s.split(oc, factor=warp_col_tiles)
    block_j, oc = s.split(oc, factor=block_col_warps)
    s.reorder(block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
    s.bind(block_i, block_x)
    s.bind(block_j, block_y)
    s.bind(nc, thread_y)
    s.bind(oc, thread_z)

    # Schedule local computation
    s.compute_at(ConvF, oc)
    n, o, ic, kh, kw, nnf, oof, ii = s.get_axes(ConvF)[-8:]
    ko, ki = s.split(ic, factor=chunk)
    s.reorder(ko, kh, ki, kw, n, o, nnf, oof, ii)

    # Move intermediate computation into each output compute tile
    s.compute_at(AF, kw)
    s.compute_at(WF, kw)

    # Schedule for A's share memory
    s.compute_at(AS, kh)
    n, w, i, nn, ii = s.get_axes(AS)[-5:]
    tx, xo = s.split(n, nparts=block_row_warps)
    ty, yo = s.split(xo, nparts=block_col_warps)
    t = s.fuse(nn, ii)
    to, ti = s.split(t, factor=warp_size)
    s.bind(tx, thread_y)
    s.bind(ty, thread_z)
    s.bind(ti, thread_x)

    # Schedule for W's share memory
    s.compute_at(WS, kh)
    kw, ic, o, ii, oo = s.get_axes(WS)[-5:]
    tx, xo = s.split(o, nparts=block_row_warps)
    ty, yo = s.split(xo, nparts=block_col_warps)
    t = s.fuse(ii, oo)
    to, ti = s.split(t, nparts=warp_size)
    s.bind(tx, thread_y)
    s.bind(ty, thread_z)
    s.bind(to, thread_x)
    s.vectorize(ti)

    s.compute_inline(s.get_block("A_pad"))

    init = s.decompose_reduction(ConvF, ko)

    s.tensorize(nnf, tir.TensorIntrin(gemm_desc, gemm_intrin))
    s.tensorize(s.get_axes(init)[-2], tir.TensorIntrin(fill_desc, fill_intrin))
    s.tensorize(nnc, tir.TensorIntrin(store_desc, store_intrin))
    s.tensorize(s.get_axes(AF)[-2], tir.TensorIntrin(load_a_desc, load_a_intrin))
    s.tensorize(s.get_axes(WF)[-2], tir.TensorIntrin(load_b_desc, load_b_intrin))

    print(tvm.hybrid.ashybrid(s.func))
    build_and_test(s.func)


if __name__ == "__main__":
    test_tensorcore()
