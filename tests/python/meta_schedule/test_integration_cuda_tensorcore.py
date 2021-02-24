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
"""Integration test for CUDA with Tensor Core"""
# pylint: disable=missing-function-docstring
import te_workload
import tvm
import tir_tensor_intrin  # pylint: disable=unused-import
from tvm import meta_schedule as ms
from tvm import te

TARGET = tvm.target.Target("nvidia/geforce-rtx-2080-ti")


def test_integration_matmul():
    workload = te_workload.matmul_fp16(n=512, m=512, k=512)
    workload = te.create_func(workload)

    def schedule(sch):
        block = sch.get_block("C")
        i, j, k = sch.get_axes(block)
        # Step 1. Rule-Auto-Tensorize
        # pylint: disable=invalid-name
        i, i_tc = sch.split(i, factors=[None, 16])
        j, j_tc = sch.split(j, factors=[None, 16])
        k, k_tc = sch.split(k, factors=[None, 16])
        sch.reorder(
            after_axes=[
                # fmt: off
                i, j, k,
                # tensor core
                i_tc, j_tc, k_tc,
                # fmt: on
            ]
        )
        block_inner = sch.blockize(i_tc)
        block_outer, block_inner = block_inner, block
        del block
        # Step 2. Rule-Multi-Level-Tiling
        i0, i1, i2, i3, i4 = sch.split(i, sch.sample_perfect_tile(5, i))
        j0, j1, j2, j3, j4 = sch.split(j, sch.sample_perfect_tile(5, j))
        k0, k1, k2 = sch.split(k, sch.sample_perfect_tile(3, k))
        # pylint: enable=invalid-name
        sch.reorder(
            after_axes=[
                # fmt: off
                i0, j0,   # S => blockIdx.x
                i1, j1,   # S => vthread
                i2, j2,   # S => threadIdx.x
                # cache_write here
                k0,       # R
                # vectorized cooperative fetching here
                k1,       # R
                i3, j3,   # S
                k2,       # R
                i4, j4,
                # S
                # fmt: on
            ]
        )
        block_idx = sch.fuse([i0, j0])
        vthread = sch.fuse([i1, j1])
        thread_idx = sch.fuse([i2, j2])
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(vthread, "vthread")
        sch.bind(thread_idx, "threadIdx.x")

        block_write_c = sch.cache_write(block_outer, 0, "local")
        block_outer, block_write_c = block_write_c, block_outer
        sch.reverse_compute_at(block_write_c, thread_idx)

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(sch.get_axes(block_read)[-ndim:])
            fused_0, fused_1 = sch.split(fused, [None, 4])
            sch.mark_loop(fused_0, "loop_type", "lazy_cooperative_fetch")
            sch.vectorize(fused_1)

        fetch_to_shared(block_outer, 1, 2)
        fetch_to_shared(block_outer, 2, 2)

        # Step 3. Postproc-Rewrite-Tensorize
        # Step 3.1. Cache read
        loop = sch.get_axes(block_outer)[-1]
        block_read_a = sch.cache_read(block_inner, 1, "wmma.matrix_a")
        block_read_b = sch.cache_read(block_inner, 2, "wmma.matrix_b")
        sch.compute_at(block_read_a, loop)
        sch.compute_at(block_read_b, loop)
        # Step 3.2. Cache write
        block_write_c = sch.cache_write(block_outer, 0, "wmma.accumulator")
        block_outer, block_write_c = block_write_c, block_outer
        sch.reverse_compute_at(block_write_c, loop)
        # Step 3.3. Decompose
        loop = sch.get_axes(block_outer)[3]
        block_init_c = sch.decompose_reduction(block_outer, loop)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]
        # Step 3.4. Tensorize
        loop = sch.get_axes(block_inner)[-3]
        sch.tensorize(loop, "wmma_sync")
        loop = sch.get_axes(block_read_a)[-2]
        sch.tensorize(loop, "wmma_load_a")
        loop = sch.get_axes(block_read_b)[-2]
        sch.tensorize(loop, "wmma_load_b")
        loop = sch.get_axes(block_init_c_inner)[-2]
        sch.tensorize(loop, "wmma_fill")
        loop = sch.get_axes(block_write_c)[-2]
        sch.tensorize(loop, "wmma_store")

    sch = ms.Schedule(func=workload, seed=1024)
    schedule(sch)
    print(tvm.script.asscript(sch.sch.func))


def test_integration_conv2d_nchwc():
    # Input shape:
    #   image: [N=1, C=6, H=98, W=98, c=16]
    #   kernel: [O=12, I=6, H=3, W=3, o=16, i=16]
    # Output shape:
    #   image: [N=1, C=12, H=96, W=96, c=16]
    workload = te_workload.conv2d_nchwc(
        n=1,
        h=98,
        w=98,
        ci=96,
        co=192,
        kh=3,
        kw=3,
        stride=1,
        in_type="float16",
        out_type="float32",
    )
    assert list(workload.shape) == [1, 12, 96, 96, 16]
    workload = te.create_func(workload)

    def schedule(sch):
        block = sch.get_block("conv2d_nchwc")
        # pylint: disable=invalid-name
        n, c0, h, w, c1, rc, rh, rw = sch.get_axes(block)
        w, i_tc = sch.split(w, factors=[6, 16])
        c1, j_tc = sch.split(c1, factors=[1, 16])
        rc, k_tc = sch.split(rc, factors=[6, 16])
        # pylint: enable=invalid-name
        sch.reorder(
            after_axes=[
                n,  # 1
                c0,  # 12
                h,  # 96
                w,  # 6
                c1,  # 1
                rc,  # 6
                rh,  # 3
                rw,  # 3
                # for tensor core
                i_tc,
                j_tc,
                k_tc,
            ]
        )
        # Multi-level tiling: `SSSRRSRS`
        # pylint: disable=invalid-name
        c00, c01, c02, c03, c04 = sch.split(c0, sch.sample_perfect_tile(5, c0))
        h0, h1, h2, h3, h4 = sch.split(h, sch.sample_perfect_tile(5, h))
        w0, w1, w2, w3, w4 = sch.split(w, sch.sample_perfect_tile(5, w))
        c10, c11, c12, c13, c14 = sch.split(c1, sch.sample_perfect_tile(5, c1))
        rc0, rc1, rc2 = sch.split(rc, sch.sample_perfect_tile(3, rc))
        rh0, rh1, rh2 = sch.split(rh, sch.sample_perfect_tile(3, rh))
        rw0, rw1, rw2 = sch.split(rw, sch.sample_perfect_tile(3, rw))
        # pylint: enable=invalid-name
        sch.reorder(
            after_axes=[
                # fmt: off
                c00, h0, w0, c10,   # S => blockIdx.x
                c01, h1, w1, c11,   # S => vthread
                c02, h2, w2, c12,   # S => threadIdx.x
                # cache_write here
                rc0, rh0, rw0,      # R
                # vectorized cooperative fetching here
                rc1, rh1, rw1,      # R
                c03, h3, w3, c13,   # S
                rc2, rh2, rw2,      # R
                c04, h4, w4, c14,   # S
                # tensor core
                i_tc, j_tc, k_tc,
                # fmt: on
            ]
        )
        block_idx = sch.fuse([c00, h0, w0, c10])
        vthread = sch.fuse([c01, h1, w1, c11])
        thread_idx = sch.fuse([c02, h2, w2, c12])
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(vthread, "vthread")
        sch.bind(thread_idx, "threadIdx.x")
        # Y: cache_write
        y_write = sch.cache_write(block, 0, "local")
        block, y_write = y_write, block
        sch.reverse_compute_at(y_write, thread_idx)
        # W: vectorized cooperative fetching
        w_read = sch.cache_read(block, 2, "shared")
        sch.compute_at(w_read, rw0)
        fused = sch.fuse(sch.get_axes(w_read)[-6:])
        fused_0, fused_1 = sch.split(fused, [None, 4])
        sch.mark_loop(fused_0, "loop_type", "lazy_cooperative_fetch")
        sch.vectorize(fused_1)
        # X: vectorized cooperative fetching
        x_read = sch.cache_read(block, 1, "shared")
        sch.compute_at(x_read, rw0)
        fused = sch.fuse(sch.get_axes(x_read)[-5:])
        fused_0, fused_1 = sch.split(fused, [None, 4])
        sch.mark_loop(fused_0, "loop_type", "lazy_cooperative_fetch")
        sch.vectorize(fused_1)
        # Decompose reduction
        sch.decompose_reduction(block, thread_idx)
        # sch.tensorize(i_tc, "test.tensorcore.wmma")
        print(tvm.script.asscript(sch.sch.func))

    sch = ms.Schedule(func=workload)
    schedule(sch)


if __name__ == "__main__":
    test_integration_matmul()
    test_integration_conv2d_nchwc()