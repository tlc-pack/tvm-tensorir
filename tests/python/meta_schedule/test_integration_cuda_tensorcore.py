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
from tir_tensor_intrin import TENSORCORE_WMMA
from tvm import meta_schedule as ms
from tvm import te

TARGET = tvm.target.Target("nvidia/rtx2080ti")


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
        # sch.sch.tensorize(sch.evaluate(i_tc), TENSORCORE_WMMA)
        print(tvm.script.asscript(sch.sch.func))

    sch = ms.Schedule(func=workload)
    schedule(sch)


if __name__ == "__main__":
    test_integration_conv2d_nchwc()
