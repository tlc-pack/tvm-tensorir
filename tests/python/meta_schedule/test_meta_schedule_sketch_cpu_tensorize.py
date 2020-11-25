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
"""Test Ansor-like sketch generation in subgraphs in meta schedule"""
# pylint: disable=missing-function-docstring
import tvm
from tir_tensor_intrin import (
    dot_product_desc,
    dot_product_impl,
    tensorcore_desc,
    tensorcore_impl,
)
from tir_workload import batch_matmul
from tvm import meta_schedule as ms

TARGET = tvm.target.Target("llvm")
DOT_PROD = tvm.tir.TensorIntrin(dot_product_desc, dot_product_impl)
TENSOR_CORE = tvm.tir.TensorIntrin(tensorcore_desc, tensorcore_impl)


def _make_space(tensor_intrin: tvm.tir.TensorIntrin) -> ms.SearchSpace:
    return ms.space.PostOrderApply(
        stages=[
            ms.rule.mark_tensorize(tensor_intrins=[tensor_intrin]),
            ms.rule.inline_pure_spatial(strict_mode=True),
            ms.rule.multi_level_tiling_and_fusion(
                structure="SSRSRS",
                must_cache_read=False,
                cache_read_scope="global",
                can_cache_write=True,
                must_cache_write=False,
                cache_write_scope="global",
                fusion_levels=[1, 2],
            ),
        ],
        postprocs=[
            ms.postproc.rewrite_tensorize(tensor_intrins=[tensor_intrin]),
        ],
    )


def test_meta_schedule_sketch_cpu_matmul_dot():
    space = _make_space(DOT_PROD)
    task = ms.SearchTask(workload=batch_matmul, task_name="matmul")
    schs = space.get_support(task=task)
    for sch in schs:
        space.postprocess(task, sch)
        print(tvm.script.asscript(sch.sch.func))


if __name__ == "__main__":
    test_meta_schedule_sketch_cpu_matmul_dot()
