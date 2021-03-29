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
import logging
import os

import te_workload
import tvm
import tir_tensor_intrin  # pylint: disable=unused-import
from tvm import meta_schedule as ms
from tvm import te
from tvm import tir
from tvm.contrib import nvcc
import numpy as np

from tvm.meta_schedule import postproc
from tvm.te import tensor_intrin

RPC_KEY = "rtx-3080"
TARGET = tvm.target.Target("nvidia/geforce-rtx-3080")
TARGET_HOST = tvm.target.Target("llvm")


def test_integration_matmul():
    os.environ["TVM_TRACKER_KEY"] = RPC_KEY
    os.environ["TVM_TRACKER_HOST"] = "172.16.2.241"
    os.environ["TVM_TRACKER_PORT"] = "4445"

    block_num = 16

    task = ms.SearchTask(
        workload=te_workload.matmul_fp16_packed.specialize(
            {
                te_workload.matmul_fp16_packed.params[0]: tir.decl_buffer(
                    (block_num, block_num, 16, 16)
                )
            }
        ),
        task_name="matmul_tensorize",
        target=TARGET,
        target_host=TARGET_HOST,
    )
    postprocs = [
        ms.postproc.rewrite_cooperative_fetch_tensorcore(),
        ms.postproc.rewrite_parallel_vectorize_unroll(),
        ms.postproc.verify_gpu_code(),
    ]
    space = ms.space.PostOrderApply(
        stages=[
            ms.rule.multi_level_tiling_with_tensor_core(
                structure="SSSRRSRS",
                must_cache_read=True,
                cache_read_scope="shared",
                can_cache_write=True,
                must_cache_write=False,
                cache_write_scope="local",
                consumer_inline_strict=False,
                fusion_levels=[3],
                compute_intrin=tir_tensor_intrin.WMMA_SYNC,
                load_intrin_A=tir_tensor_intrin.WMMA_LOAD_A,
                load_intrin_B=tir_tensor_intrin.WMMA_LOAD_B,
                store_intrin=tir_tensor_intrin.WMMA_STORE,
                vector_load_max_len=4,
                tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
            ),
            ms.rule.inline_pure_spatial(strict_mode=False),
            ms.rule.parallelize_vectorize_unroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
        ],
        # postprocs=postprocs,
        # postprocs=[]
    )
    schs = space.get_support(task=task)
    for sch in schs:
        # for postproc in postprocs:
        #     postproc.apply(task, sch)
        print(tvm.script.asscript(sch.mod))

    # ctx = tvm.gpu(0)
    # if nvcc.have_tensorcore(ctx.compute_version):
    #     with tvm.transform.PassContext():
    #         func = tvm.build(schedule.state.mod["main"], [], "cuda")
    #         print(tvm.script.asscript(schedule.state.mod["main"]))
    #         print(func.imported_modules[0].get_source())
    #         for inst in schedule.trace.as_python():
    #             print(inst)
    #     a_np = np.random.uniform(size=(block_num, block_num, 16, 16)).astype("float16")
    #     b_np = np.random.uniform(size=(block_num, block_num, 16, 16)).astype("float16")
    #     a = tvm.nd.array(a_np, ctx)
    #     b = tvm.nd.array(b_np, ctx)
    #     c = tvm.nd.array(np.zeros((block_num, block_num, 16, 16), dtype="float32"), ctx)
    #     evaluator = func.time_evaluator(func.entry_name, ctx, number = 3, repeat = 1, min_repeat_ms = 40)
    #     print("matmul with tensor core: %f ms" % (evaluator(a, b, c).mean * 1e3))

    #     c_np = c.asnumpy()
    #     a_non_packed = np.array(
    #         [
    #             [a_np[i // 16][j // 16][i % 16][j % 16] for j in range(block_num * 16)]
    #             for i in range(block_num * 16)
    #         ]
    #     )
    #     b_non_packed = np.array(
    #         [
    #             [b_np[i // 16][j // 16][i % 16][j % 16] for j in range(block_num * 16)]
    #             for i in range(block_num * 16)
    #         ]
    #     )
    #     c_non_packed = np.array(
    #         [
    #             [c_np[i // 16][j // 16][i % 16][j % 16] for j in range(block_num * 16)]
    #             for i in range(block_num * 16)
    #         ]
    #     )

    #     np.testing.assert_allclose(
    #         c_non_packed,
    #         np.matmul(a_non_packed.astype("float32"), b_non_packed.astype("float32")),
    #         rtol=1e-4,
    #         atol=1e-4,
    #     )


if __name__ == "__main__":
    test_integration_matmul()
