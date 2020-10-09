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

from tvm.script import ty


@tvm.script.tir
def gpu_gemm(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    A = tir.match_buffer(a, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    C = tir.match_buffer(c, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        for ax0_outer in range(0, 32, annotation={"loop_type": "blockIdx.y"}):
            for ax1_outer in range(0, 32, annotation={"loop_type": "blockIdx.x"}):
                with tir.block([32, 32], "GPU_Block", exec_scope="gpu_block") as [bx, by]:
                    A_shared = tir.buffer_allocate([2048, 2048], scope="shared", align=128)
                    B_shared = tir.buffer_allocate([2048, 2048], scope="shared", align=128)
                    tir.bind(bx, ax1_outer)
                    tir.bind(by, ax0_outer)
                    for ax0_inner_outer in range(0, 2, annotation={"loop_type": "vthread"}):
                        for ax1_inner_outer in range(0, 2, annotation={"loop_type": "vthread"}):
                            for ax0_inner_inner_outer in range(0, 8,
                                                               annotation={"loop_type": "threadIdx.y"}):
                                for ax1_inner_inner_outer in range(0, 8,
                                                                   annotation={"loop_type": "threadIdx.x"}):
                                    with tir.block([16, 16], "GPU_Thread") as [tx, ty]:
                                        A_shared_local = tir.buffer_allocate([2048, 2048],
                                                                             scope="local",
                                                                             align=128)
                                        B_shared_local = tir.buffer_allocate([2048, 2048],
                                                                             scope="local",
                                                                             align=128)
                                        C_local = tir.buffer_allocate([2048, 2048],
                                                                      exec_scope="local", align=128)
                                        tir.bind(tx, ax1_inner_outer * 8 + ax1_inner_inner_outer)
                                        tir.bind(ty, ax0_inner_outer * 8 + ax0_inner_inner_outer)
                                        for ax0_init in range(0, 4):
                                            for ax1_init in range(0, 4):
                                                with tir.block([2048, 2048], "C_init") as [vi_init,
                                                                                           vj_init]:
                                                    tir.bind(vi_init,
                                                             ((by * 64) + (ty * 4) + ax0_init))
                                                    tir.bind(vj_init, ((ax1_outer * 64) + (
                                                                tx * 4) + ax1_init))
                                                    tir.reads([])
                                                    tir.writes([C_local[vi_init:(vi_init + 1),
                                                                vj_init:(vj_init + 1)]])
                                                    C_local[vi_init, vj_init] = tir.float32(0)
                                        for ax2_outer in range(0, 256):
                                            for ax0_outer_1 in range(0, 8,
                                                                     annotation={"loop_type": "threadIdx.y"}):
                                                for ax0_inner in range(0, 1):
                                                    for ax1_outer_1 in range(0, 2):
                                                        for ax1_inner_outer_1 in range(0, 8,
                                                                                       annotation={"loop_type": "threadIdx.x"}):
                                                            for ax1_inner_inner in range(0, 4,
                                                                                         annotation={"loop_type": "vectorize"}):
                                                                with tir.block([2048, 2048],
                                                                               "") as [v0, v1]:
                                                                    tir.bind(v0, (
                                                                                (ax2_outer * 8) + (
                                                                                    ax0_outer_1 + ax0_inner)))
                                                                    tir.bind(v1, ((bx * 64) + ((
                                                                                                           ax1_outer_1 * 32) + (
                                                                                                           (
                                                                                                                       ax1_inner_outer_1 * 4) + ax1_inner_inner))))
                                                                    tir.reads([B[v0:(v0 + 1),
                                                                               v1:(v1 + 1)]])
                                                                    tir.writes([B_shared[
                                                                                v0:(v0 + 1),
                                                                                v1:(v1 + 1)]])
                                                                    B_shared[v0, v1] = B[v0, v1]
                                            for ax0_outer_2 in range(0, 8,
                                                                     annotation={"loop_type": "threadIdx.y"}):
                                                for ax0_inner_1 in range(0, 1):
                                                    for ax1_outer_2 in range(0, 2):
                                                        for ax1_inner_outer_2 in range(0, 8,
                                                                                       annotation={"loop_type": "threadIdx.x"}):
                                                            for ax1_inner_inner_1 in range(0, 4,
                                                                                           annotation={"loop_type": "vectorize"}):
                                                                with tir.block([2048, 2048],
                                                                               "") as [v0_1, v1_1]:
                                                                    tir.bind(v0_1, (
                                                                                (ax2_outer * 8) + (
                                                                                    ax0_outer_2 + ax0_inner_1)))
                                                                    tir.bind(v1_1, ((by * 64) + ((
                                                                                                             ax1_outer_2 * 32) + (
                                                                                                             (
                                                                                                                         ax1_inner_outer_2 * 4) + ax1_inner_inner_1))))
                                                                    tir.reads([A[v0_1:(v0_1 + 1),
                                                                               v1_1:(v1_1 + 1)]])
                                                                    tir.writes([A_shared[
                                                                                v0_1:(v0_1 + 1),
                                                                                v1_1:(v1_1 + 1)]])
                                                                    A_shared[v0_1, v1_1] = A[
                                                                        v0_1, v1_1]
                                            for ax2_inner_outer in range(0, 8):
                                                for ax1 in range(0, 4):
                                                    with tir.block([2048, 2048], "") as [v0_2,
                                                                                         v1_2]:
                                                        tir.bind(v0_2, ((
                                                                                    ax2_outer * 8) + ax2_inner_outer))
                                                        tir.bind(v1_2, ((bx * 64) + (tx * 4) + ax1))
                                                        tir.reads([B_shared[v0_2:(v0_2 + 1),
                                                                   v1_2:(v1_2 + 1)]])
                                                        tir.writes([B_shared_local[v0_2:(v0_2 + 1),
                                                                    v1_2:(v1_2 + 1)]])
                                                        B_shared_local[v0_2, v1_2] = B_shared[
                                                            v0_2, v1_2]
                                                for ax1_1 in range(0, 4):
                                                    with tir.block([2048, 2048], "") as [v0_3,
                                                                                         v1_3]:
                                                        tir.bind(v0_3, ((
                                                                                    ax2_outer * 8) + ax2_inner_outer))
                                                        tir.bind(v1_3,
                                                                 ((by * 64) + (ty * 4) + ax1_1))
                                                        tir.reads([A_shared[v0_3:(v0_3 + 1),
                                                                   v1_3:(v1_3 + 1)]])
                                                        tir.writes([A_shared_local[v0_3:(v0_3 + 1),
                                                                    v1_3:(v1_3 + 1)]])
                                                        A_shared_local[v0_3, v1_3] = A_shared[
                                                            v0_3, v1_3]
                                                for ax2_inner_inner in range(0, 1):
                                                    for ax0 in range(0, 4):
                                                        for ax1_2 in range(0, 4):
                                                            with tir.block([2048, 2048,
                                                                            tir.reduce_axis(0,
                                                                                            2048)],
                                                                           "C_update") as [vi, vj,
                                                                                           vk]:
                                                                tir.bind(vi, ((by * 64) + (
                                                                            ty * 4) + ax0))
                                                                tir.bind(vj, ((bx * 64) + (
                                                                            tx * 4) + ax1_2))
                                                                tir.bind(vk, ((ax2_outer * 8) + (
                                                                            ax2_inner_outer + ax2_inner_inner)))
                                                                tir.reads([C_local[vi:(vi + 1),
                                                                           vj:(vj + 1)],
                                                                           A_shared_local[
                                                                           vk:(vk + 1),
                                                                           vi:(vi + 1)],
                                                                           B_shared_local[
                                                                           vk:(vk + 1),
                                                                           vj:(vj + 1)]])
                                                                tir.writes([C_local[vi:(vi + 1),
                                                                            vj:(vj + 1)]])
                                                                C_local[vi, vj] = (
                                                                            C_local[vi, vj] + (
                                                                                A_shared_local[
                                                                                    vk, vi] *
                                                                                B_shared_local[
                                                                                    vk, vj]))
                                        for ax0_inner_inner_inner in range(0, 4):
                                            for ax1_inner_inner_inner in range(0, 4):
                                                with tir.block([2048, 2048], "") as [v0_4, v1_4]:
                                                    tir.bind(v0_4, ((by * 64) + (
                                                                ty * 4) + ax0_inner_inner_inner))
                                                    tir.bind(v1_4, ((bx * 64) + (
                                                                tx * 4) + ax1_inner_inner_inner))
                                                    tir.reads(
                                                        [C_local[v0_4:(v0_4 + 1), v1_4:(v1_4 + 1)]])
                                                    tir.writes(
                                                        [C[v0_4:(v0_4 + 1), v1_4:(v1_4 + 1)]])
                                                    C[v0_4, v1_4] = C_local[v0_4, v1_4]


@tvm.script.tir
def warp_testcase(a: ty.handle, b: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    A = tir.match_buffer(a, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        for ax0_outer in range(0, 64, annotation={"loop_type": "blockIdx.y"}):
            for ax1_outer in range(0, 64, annotation={"loop_type": "blockIdx.x"}):
                with tir.block([32, 32], "GPU_Block", exec_scope="gpu_block") as [bx, by]:
                    tir.bind(bx, ax1_outer)
                    tir.bind(by, ax0_outer)
                    for ty in range(0, 32, annotation={"loop_type": "threadIdx.y"}):
                        with tir.block([32, 2048], "", exec_scope="gpu_warp") as [wx, wy]:
                            tir.bind(wx, bx)
                            tir.bind(wy, by * 32 + ty)
                            for tx in range(0, 32, annotation={"loop_type": "threadIdx.x"}):
                                with tir.block([2048, 2048], "", exec_scope="gpu_thread") as [tx,
                                                                                              ty]:
                                    tir.bind(tx, wx * 32 + tx)
                                    tir.bind(ty, wy)
                                    B[tx, ty] = A[tx, ty] + 1.0


@tvm.script.tir
def warp_fail_case_1(a: ty.handle, b: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    A = tir.match_buffer(a, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        for ax0_outer in range(0, 32, annotation={"loop_type": "blockIdx.y"}):
            for ax1_outer in range(0, 32, annotation={"loop_type": "blockIdx.x"}):
                with tir.block([32, 32], "GPU_Block", exec_scope="gpu_block") as [bx, by]:
                    tir.bind(bx, ax1_outer)
                    tir.bind(by, ax0_outer)
                    for ty in range(0, 64, annotation={"loop_type": "threadIdx.y"}):
                        with tir.block([32, 2048], "", exec_scope="gpu_warp") as [wx, wy]:
                            tir.bind(wx, bx)
                            tir.bind(wy, by * 32 + ty)
                            for tx in range(0, 64, annotation={"loop_type": "threadIdx.x"}):
                                with tir.block([2048, 2048], "", exec_scope="gpu_thread") as [tx,
                                                                                              ty]:
                                    tir.bind(tx, wx * 32 + tx)
                                    tir.bind(ty, wy)
                                    B[tx, ty] = A[tx, ty] + 1.0


@tvm.script.tir
def warp_fail_case_2(a: ty.handle, b: ty.handle) -> None:
    # function attr dict
    tir.func_attr({})
    A = tir.match_buffer(a, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    B = tir.match_buffer(b, [2048, 2048], elem_offset=0, align=128, offset_factor=1)
    # body
    with tir.block([], "root") as []:
        for ax0_outer in range(0, 64, annotation={"loop_type": "blockIdx.y"}):
            for ax1_outer in range(0, 64, annotation={"loop_type": "blockIdx.x"}):
                with tir.block([32, 32], "GPU_Block", exec_scope="gpu_block") as [bx, by]:
                    tir.bind(bx, ax1_outer)
                    tir.bind(by, ax0_outer)
                    for ty in range(0, 32, annotation={"loop_type": "threadIdx.y"}):
                        for tx in range(0, 32, annotation={"loop_type": "threadIdx.x"}):
                            with tir.block([2048, 2048], "", exec_scope="gpu_warp") as [wx, wy]:
                                tir.bind(wx, bx * 32 + tx)
                                tir.bind(wy, by * 32 + ty)
                                with tir.block([2048, 2048], "", exec_scope="gpu_thread") as [tx, ty]:
                                    tir.bind(tx, wx)
                                    tir.bind(ty, wy)
                                    B[tx, ty] = A[tx, ty] + 1.0


def test_gpu_hierarchy():
    tir.validate_hierarchy(gpu_gemm)
    tir.validate_hierarchy(warp_testcase)
    try:
        tir.validate_hierarchy(warp_fail_case_1)
        assert False
    except:
        pass

    try:
        tir.validate_hierarchy(warp_fail_case_2)
        assert False
    except:
        pass


if __name__ == "__main__":
    test_gpu_hierarchy()
