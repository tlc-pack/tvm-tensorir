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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

from tvm.meta_schedule.space_generator.post_order_apply import PostOrderApply
from tvm.meta_schedule.testing.schedule_rule import multi_level_tiling
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.te import create_prim_func
from tvm.meta_schedule.testing import te_workload
from tvm.target import Target


def _create_context(mod, target, rule) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[rule],
        task_name="test",
    )
    ctx.space_generator.initialize_with_tune_context(ctx)
    for sch_rule in ctx.sch_rules:
        sch_rule.initialize_with_tune_context(ctx)
    return ctx


def test_cpu_matmul():
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16])",
            "v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64)",
            "l23, l24 = sch.split(loop=l4, factors=[v21, v22])",
            "sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)",
            "sch.reverse_compute_at(block=b1, loop=l18, preserve_unit_loops=1)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16])",
            "v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64)",
            "l23, l24 = sch.split(loop=l4, factors=[v21, v22])",
            "sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)",
            "sch.reverse_compute_at(block=b1, loop=l17, preserve_unit_loops=1)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7])",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15])",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21])",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
        ],
    ]
    target = Target("llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_cpu_matmul_relu():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "b1, = sch.get_consumers(block=b0)",
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16])",
            "v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64)",
            "l23, l24 = sch.split(loop=l4, factors=[v21, v22])",
            "sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)",
            "sch.reverse_compute_at(block=b1, loop=l18, preserve_unit_loops=1)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "b1, = sch.get_consumers(block=b0)",
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16])",
            "v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64)",
            "l23, l24 = sch.split(loop=l4, factors=[v21, v22])",
            "sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)",
            "sch.reverse_compute_at(block=b1, loop=l17, preserve_unit_loops=1)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "l1, l2, l3 = sch.get_loops(block=b0)",
            "v4, v5, v6, v7 = sch.sample_perfect_tile(loop=l1, n=4, max_innermost_factor=64)",
            "l8, l9, l10, l11 = sch.split(loop=l1, factors=[v4, v5, v6, v7])",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l2, factors=[v12, v13, v14, v15])",
            "v20, v21 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l22, l23 = sch.split(loop=l3, factors=[v20, v21])",
            "sch.reorder(l8, l16, l9, l17, l22, l10, l18, l23, l11, l19)",
        ],
    ]
    # pylint: enable=line-too-long
    target = Target("llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_relu(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_cuda_matmul():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64)",
            "l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[v5, v6, v7, v8, v9])",
            "v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64)",
            "l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[v15, v16, v17, v18, v19])",
            "v25, v26, v27 = sch.sample_perfect_tile(loop=l4, n=3, max_innermost_factor=64)",
            "l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27])",
            "sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)",
            "l31 = sch.fuse(l10, l20)",
            'sch.bind(loop=l31, thread_axis="blockIdx.x")',
            "l32 = sch.fuse(l11, l21)",
            'sch.bind(loop=l32, thread_axis="vthread.x")',
            "l33 = sch.fuse(l12, l22)",
            'sch.bind(loop=l33, thread_axis="threadIdx.x")',
            'b34 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b34, loop=l28, preserve_unit_loops=1)",
            "l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b34)",
            "l41 = sch.fuse(l39, l40)",
            "v42, v43 = sch.sample_perfect_tile(loop=l41, n=2, max_innermost_factor=4)",
            "l44, l45 = sch.split(loop=l41, factors=[v42, v43])",
            "sch.vectorize(loop=l45)",
            'sch.annotate(block_or_loop=l44, ann_key="meta_schedule.lazy_cooperative_fetch", ann_val=1)',
            'b46 = sch.cache_read(block=b0, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b46, loop=l28, preserve_unit_loops=1)",
            "l47, l48, l49, l50, l51, l52 = sch.get_loops(block=b46)",
            "l53 = sch.fuse(l51, l52)",
            "v54, v55 = sch.sample_perfect_tile(loop=l53, n=2, max_innermost_factor=4)",
            "l56, l57 = sch.split(loop=l53, factors=[v54, v55])",
            "sch.vectorize(loop=l57)",
            'sch.annotate(block_or_loop=l56, ann_key="meta_schedule.lazy_cooperative_fetch", ann_val=1)',
            "sch.reverse_compute_at(block=b1, loop=l33, preserve_unit_loops=1)",
        ]
    ]
    # pylint: enable=line-too-long
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_cuda_matmul_relu():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64)",
            "l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[v5, v6, v7, v8, v9])",
            "v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64)",
            "l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[v15, v16, v17, v18, v19])",
            "v25, v26, v27 = sch.sample_perfect_tile(loop=l4, n=3, max_innermost_factor=64)",
            "l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27])",
            "sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)",
            "l31 = sch.fuse(l10, l20)",
            'sch.bind(loop=l31, thread_axis="blockIdx.x")',
            "l32 = sch.fuse(l11, l21)",
            'sch.bind(loop=l32, thread_axis="vthread.x")',
            "l33 = sch.fuse(l12, l22)",
            'sch.bind(loop=l33, thread_axis="threadIdx.x")',
            'b34 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b34, loop=l28, preserve_unit_loops=1)",
            "l35, l36, l37, l38, l39, l40 = sch.get_loops(block=b34)",
            "l41 = sch.fuse(l39, l40)",
            "v42, v43 = sch.sample_perfect_tile(loop=l41, n=2, max_innermost_factor=4)",
            "l44, l45 = sch.split(loop=l41, factors=[v42, v43])",
            "sch.vectorize(loop=l45)",
            'sch.annotate(block_or_loop=l44, ann_key="meta_schedule.lazy_cooperative_fetch", ann_val=1)',
            'b46 = sch.cache_read(block=b0, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b46, loop=l28, preserve_unit_loops=1)",
            "l47, l48, l49, l50, l51, l52 = sch.get_loops(block=b46)",
            "l53 = sch.fuse(l51, l52)",
            "v54, v55 = sch.sample_perfect_tile(loop=l53, n=2, max_innermost_factor=4)",
            "l56, l57 = sch.split(loop=l53, factors=[v54, v55])",
            "sch.vectorize(loop=l57)",
            'sch.annotate(block_or_loop=l56, ann_key="meta_schedule.lazy_cooperative_fetch", ann_val=1)',
            "sch.reverse_compute_at(block=b1, loop=l33, preserve_unit_loops=1)",
        ]
    ]
    # pylint: enable=line-too-long
    target = Target("cuda", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.matmul_relu(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=target,
        rule=multi_level_tiling(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_cpu_matmul()
    test_cpu_matmul_relu()
    test_cuda_matmul()
    test_cuda_matmul_relu()
