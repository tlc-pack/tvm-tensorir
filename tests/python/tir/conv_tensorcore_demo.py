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
"""TensorCore demo"""
# pylint: disable=missing-function-docstring
import numpy as np
import tvm
import tvm.testing
from tvm import tir
from tvm.meta_schedule.measure import ProgramTester
from tvm.script import ty

VERIFY = True

RPC_RUN = ProgramTester(
    target=tvm.target.Target("nvidia/jetson-agx-xavier"),
    target_host=tvm.target.Target("llvm -mcpu=carmel -mtriple=aarch64-linux-gnu"),
    build_func="tar",
    rpc_key="jetson-agx-xavier",
    rpc_host=None,
    rpc_port=None,
)

LOCAL_RUN = ProgramTester(
    target=tvm.target.Target("llvm"),
    target_host=None,
    build_func="tar",
    rpc_key="local",
    rpc_host=None,
    rpc_port=None,
)

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unexpected-keyword-arg,chained-comparison,misplaced-comparison-constant
# fmt: off

@tvm.script.tir
def conv(a: ty.handle, w: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (2, 14, 14, 4, 16, 16), "float32")
    A = tir.match_buffer(a, (2, 14, 14, 2, 16, 16), "float16")
    W = tir.match_buffer(w, (3, 3, 2, 4, 16, 16), "float16")

    Apad = tir.alloc_buffer([2, 16, 16, 2, 16, 16], "float16")
    with tir.block([2, 16, 16, 2, 16, 16], "A_pad") as [n, h, w, i, nn, ii]:
        Apad[n, h, w, i, nn, ii] = tir.if_then_else(1 <= h and h < 15 and 1 <= w and w < 15,
                                                    A[n, h - 1, w - 1, i, nn, ii], tir.float16(0),
                                                    dtype="float16")
    with tir.block([2, 14, 14, 4, tir.reduce_axis(0, 2), tir.reduce_axis(0, 3),
                    tir.reduce_axis(0, 3), 16, 16, tir.reduce_axis(0, 16)], "Conv") as \
            [n, h, w, o, ic, kh, kw, nn, oo, ii]:
        with tir.init():
            C[n, h, w, o, nn, oo] = tir.float32(0)
        C[n, h, w, o, nn, oo] = C[n, h, w, o, nn, oo] \
                                + tir.cast(Apad[n, h + kh, w + kw, ic, nn, ii], "float32") \
                                * tir.cast(W[kh, kw, ic, o, ii, oo], "float32")


@tvm.script.tir
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


@tvm.script.tir
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


@tvm.script.tir
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


@tvm.script.tir
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


@tvm.script.tir
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


@tvm.script.tir
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


@tvm.script.tir
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


@tvm.script.tir
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


@tvm.script.tir
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


@tvm.script.tir
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


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,unexpected-keyword-arg,chained-comparison,misplaced-comparison-constant
# pylint: disable=invalid-name


def build_and_test(local_func, rpc_func):
    a = np.random.uniform(size=(2, 14, 14, 2, 16, 16)).astype("float16")
    w = np.random.uniform(size=(3, 3, 2, 4, 16, 16)).astype("float16")
    c = np.zeros((2, 14, 14, 4, 16, 16), dtype="float32")
    refs = LOCAL_RUN(local_func, [a, w, c])
    runs = RPC_RUN(rpc_func, [a, w, c])
    assert len(refs) == len(runs)
    for ref, run in zip(refs, runs):
        print(run.shape, ref.shape)
        np.testing.assert_allclose(actual=run, desired=ref, rtol=1e-4, atol=1e-4)


def test_tensorcore():
    mod = tvm.script.create_module({"conv": conv})
    original_func = mod["conv"]

    s = tir.Schedule(original_func, debug_mode=True)

    Conv = s.get_block("Conv")

    AS = s.cache_read(Conv, 1, "shared")
    WS = s.cache_read(Conv, 2, "shared")
    AF = s.cache_read(Conv, 1, "wmma.matrix_a")
    WF = s.cache_read(Conv, 2, "wmma.matrix_b")
    ConvF = s.cache_write(Conv, 0, "wmma.accumulator")

    block_row_warps = 1
    block_col_warps = 1
    warp_row_tiles = 1
    warp_col_tiles = 1
    warp_size = 32
    chunk = 2

    nc, hc, wc, oc, nnc, ooc = s.get_loops(Conv)
    block_k = s.fuse(hc, wc)
    s.bind(block_k, "blockIdx.z")
    nc, nci = s.split(nc, [None, warp_row_tiles])
    block_i, nc = s.split(nc, [None, block_row_warps])
    oc, oci = s.split(oc, [None, warp_col_tiles])
    block_j, oc = s.split(oc, [None, block_col_warps])
    s.reorder(block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
    s.bind(block_i, "blockIdx.x")
    s.bind(block_j, "blockIdx.y")
    s.bind(nc, "threadIdx.y")
    s.bind(oc, "threadIdx.z")

    # Schedule local computation
    s.compute_at(ConvF, oc)
    ic, kh, kw, _nnf, _oof, ii = s.get_loops(ConvF)[-6:]
    ko, ki = s.split(ic, [None, chunk])
    s.reorder(ko, kh, ki)

    # Move intermediate computation into each output compute tile
    s.compute_at(AF, kw)
    s.compute_at(WF, kw)

    # Schedule for A's share memory
    s.compute_at(AS, kh)
    _, _, nn, ii = s.get_loops(AS)[-4:]
    t = s.fuse(nn, ii)
    _, ti = s.split(t, [None, warp_size])
    s.bind(ti, "threadIdx.x")

    # Schedule for W's share memory
    s.compute_at(WS, kh)
    kw, ic, o, ii, oo = s.get_loops(WS)[-5:]
    tx, xo = s.split(o, [block_row_warps, None])
    ty, _ = s.split(xo, [block_col_warps, None])  # pylint: disable=redefined-outer-name
    t = s.fuse(ii, oo)
    to, ti = s.split(t, [warp_size, None])
    s.bind(tx, "threadIdx.y")
    s.bind(ty, "threadIdx.z")
    s.bind(to, "threadIdx.x")
    s.vectorize(ti)

    s.compute_inline(s.get_block("A_pad"))
    init = s.decompose_reduction(ConvF, ko)
    s.tensorize(s.get_loops(ConvF)[-3], tir.TensorIntrin(gemm_desc, gemm_intrin))
    s.tensorize(s.get_loops(init)[-2], tir.TensorIntrin(fill_desc, fill_intrin))
    s.tensorize(s.get_loops(Conv)[-2], tir.TensorIntrin(store_desc, store_intrin))
    s.tensorize(s.get_loops(AF)[-2], tir.TensorIntrin(load_a_desc, load_a_intrin))
    s.tensorize(s.get_loops(WF)[-2], tir.TensorIntrin(load_b_desc, load_b_intrin))

    print(tvm.script.asscript(s.mod["main"]))
    print(tvm.lower(s.mod["main"], None, simple_mode=True))
    build_and_test(conv, s.mod["main"])


if __name__ == "__main__":
    test_tensorcore()
