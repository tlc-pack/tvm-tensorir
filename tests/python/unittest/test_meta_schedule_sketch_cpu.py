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
from typing import List

from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.space_generation import check_trace, create_context
from tvm.target import Target
from tvm.te import create_prim_func


def _target() -> Target:
    return Target("llvm --num-cores=16")


def test_meta_schedule_cpu_sketch_matmul():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16])",
            "v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64)",
            "l23, l24 = sch.split(loop=l4, factors=[v21, v22])",
            "sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v25 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v25)',
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'b2 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "l3, l4, l5 = sch.get_loops(block=b0)",
            "v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9])",
            "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17])",
            "v22, v23 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l24, l25 = sch.split(loop=l5, factors=[v22, v23])",
            "sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)",
            "sch.reverse_compute_at(block=b2, loop=l18, preserve_unit_loops=1)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v26 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)',
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'b2 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "l3, l4, l5 = sch.get_loops(block=b0)",
            "v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9])",
            "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17])",
            "v22, v23 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l24, l25 = sch.split(loop=l5, factors=[v22, v23])",
            "sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)",
            "sch.reverse_compute_at(block=b2, loop=l19, preserve_unit_loops=1)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v26 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)',
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.matmul(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_meta_schedule_cpu_sketch_matmul_relu():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16])",
            "v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64)",
            "l23, l24 = sch.split(loop=l4, factors=[v21, v22])",
            "sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v25 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v25)',
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            "b2, = sch.get_consumers(block=b0)",
            "l3, l4, l5 = sch.get_loops(block=b0)",
            "v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9])",
            "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17])",
            "v22, v23 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l24, l25 = sch.split(loop=l5, factors=[v22, v23])",
            "sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)",
            "sch.reverse_compute_at(block=b2, loop=l18, preserve_unit_loops=1)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v26 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)',
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            "b2, = sch.get_consumers(block=b0)",
            "l3, l4, l5 = sch.get_loops(block=b0)",
            "v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9])",
            "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17])",
            "v22, v23 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l24, l25 = sch.split(loop=l5, factors=[v22, v23])",
            "sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)",
            "sch.reverse_compute_at(block=b2, loop=l19, preserve_unit_loops=1)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v26 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)',
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.matmul_relu(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_meta_schedule_cpu_sketch_conv2d_nchw():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="compute", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            "l2, l3, l4, l5, l6, l7, l8 = sch.get_loops(block=b0)",
            "v9, v10, v11, v12 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l13, l14, l15, l16 = sch.split(loop=l2, factors=[v9, v10, v11, v12])",
            "v17, v18, v19, v20 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l21, l22, l23, l24 = sch.split(loop=l3, factors=[v17, v18, v19, v20])",
            "v25, v26, v27, v28 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l29, l30, l31, l32 = sch.split(loop=l4, factors=[v25, v26, v27, v28])",
            "v33, v34, v35, v36 = sch.sample_perfect_tile(loop=l5, n=4, max_innermost_factor=64)",
            "l37, l38, l39, l40 = sch.split(loop=l5, factors=[v33, v34, v35, v36])",
            "v41, v42 = sch.sample_perfect_tile(loop=l6, n=2, max_innermost_factor=64)",
            "l43, l44 = sch.split(loop=l6, factors=[v41, v42])",
            "v45, v46 = sch.sample_perfect_tile(loop=l7, n=2, max_innermost_factor=64)",
            "l47, l48 = sch.split(loop=l7, factors=[v45, v46])",
            "v49, v50 = sch.sample_perfect_tile(loop=l8, n=2, max_innermost_factor=64)",
            "l51, l52 = sch.split(loop=l8, factors=[v49, v50])",
            "sch.reorder(l13, l21, l29, l37, l14, l22, l30, l38, l43, l47, l51, l15, l23, l31, l39, l44, l48, l52, l16, l24, l32, l40)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v53 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v53)',
        ],
        [
            'b0 = sch.get_block(name="compute", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'b2 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "l3, l4, l5, l6, l7, l8, l9 = sch.get_loops(block=b0)",
            "v10, v11, v12, v13 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l14, l15, l16, l17 = sch.split(loop=l3, factors=[v10, v11, v12, v13])",
            "v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l22, l23, l24, l25 = sch.split(loop=l4, factors=[v18, v19, v20, v21])",
            "v26, v27, v28, v29 = sch.sample_perfect_tile(loop=l5, n=4, max_innermost_factor=64)",
            "l30, l31, l32, l33 = sch.split(loop=l5, factors=[v26, v27, v28, v29])",
            "v34, v35, v36, v37 = sch.sample_perfect_tile(loop=l6, n=4, max_innermost_factor=64)",
            "l38, l39, l40, l41 = sch.split(loop=l6, factors=[v34, v35, v36, v37])",
            "v42, v43 = sch.sample_perfect_tile(loop=l7, n=2, max_innermost_factor=64)",
            "l44, l45 = sch.split(loop=l7, factors=[v42, v43])",
            "v46, v47 = sch.sample_perfect_tile(loop=l8, n=2, max_innermost_factor=64)",
            "l48, l49 = sch.split(loop=l8, factors=[v46, v47])",
            "v50, v51 = sch.sample_perfect_tile(loop=l9, n=2, max_innermost_factor=64)",
            "l52, l53 = sch.split(loop=l9, factors=[v50, v51])",
            "sch.reorder(l14, l22, l30, l38, l15, l23, l31, l39, l44, l48, l52, l16, l24, l32, l40, l45, l49, l53, l17, l25, l33, l41)",
            "sch.reverse_compute_at(block=b2, loop=l38, preserve_unit_loops=1)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v54 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v54)',
        ],
        [
            'b0 = sch.get_block(name="compute", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'b2 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "l3, l4, l5, l6, l7, l8, l9 = sch.get_loops(block=b0)",
            "v10, v11, v12, v13 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l14, l15, l16, l17 = sch.split(loop=l3, factors=[v10, v11, v12, v13])",
            "v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l22, l23, l24, l25 = sch.split(loop=l4, factors=[v18, v19, v20, v21])",
            "v26, v27, v28, v29 = sch.sample_perfect_tile(loop=l5, n=4, max_innermost_factor=64)",
            "l30, l31, l32, l33 = sch.split(loop=l5, factors=[v26, v27, v28, v29])",
            "v34, v35, v36, v37 = sch.sample_perfect_tile(loop=l6, n=4, max_innermost_factor=64)",
            "l38, l39, l40, l41 = sch.split(loop=l6, factors=[v34, v35, v36, v37])",
            "v42, v43 = sch.sample_perfect_tile(loop=l7, n=2, max_innermost_factor=64)",
            "l44, l45 = sch.split(loop=l7, factors=[v42, v43])",
            "v46, v47 = sch.sample_perfect_tile(loop=l8, n=2, max_innermost_factor=64)",
            "l48, l49 = sch.split(loop=l8, factors=[v46, v47])",
            "v50, v51 = sch.sample_perfect_tile(loop=l9, n=2, max_innermost_factor=64)",
            "l52, l53 = sch.split(loop=l9, factors=[v50, v51])",
            "sch.reorder(l14, l22, l30, l38, l15, l23, l31, l39, l44, l48, l52, l16, l24, l32, l40, l45, l49, l53, l17, l25, l33, l41)",
            "sch.reverse_compute_at(block=b2, loop=l39, preserve_unit_loops=1)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v54 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v54)',
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.conv2d_nchw(
                n=1,
                h=56,
                w=56,
                ci=512,
                co=512,
                kh=3,
                kw=3,
                stride=1,
                padding=1,
            )
        ),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_meta_schedule_cpu_sketch_conv2d_nchw_bias_bn_relu():  # pylint: disable=invalid-name
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="compute", func_name="main")',
            'b1 = sch.get_block(name="bias_add", func_name="main")',
            'b2 = sch.get_block(name="bn_mul", func_name="main")',
            'b3 = sch.get_block(name="bn_add", func_name="main")',
            'b4 = sch.get_block(name="root", func_name="main")',
            "sch.compute_inline(block=b3)",
            "sch.compute_inline(block=b2)",
            "sch.compute_inline(block=b1)",
            "l5, l6, l7, l8, l9, l10, l11 = sch.get_loops(block=b0)",
            "v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l5, n=4, max_innermost_factor=64)",
            "l16, l17, l18, l19 = sch.split(loop=l5, factors=[v12, v13, v14, v15])",
            "v20, v21, v22, v23 = sch.sample_perfect_tile(loop=l6, n=4, max_innermost_factor=64)",
            "l24, l25, l26, l27 = sch.split(loop=l6, factors=[v20, v21, v22, v23])",
            "v28, v29, v30, v31 = sch.sample_perfect_tile(loop=l7, n=4, max_innermost_factor=64)",
            "l32, l33, l34, l35 = sch.split(loop=l7, factors=[v28, v29, v30, v31])",
            "v36, v37, v38, v39 = sch.sample_perfect_tile(loop=l8, n=4, max_innermost_factor=64)",
            "l40, l41, l42, l43 = sch.split(loop=l8, factors=[v36, v37, v38, v39])",
            "v44, v45 = sch.sample_perfect_tile(loop=l9, n=2, max_innermost_factor=64)",
            "l46, l47 = sch.split(loop=l9, factors=[v44, v45])",
            "v48, v49 = sch.sample_perfect_tile(loop=l10, n=2, max_innermost_factor=64)",
            "l50, l51 = sch.split(loop=l10, factors=[v48, v49])",
            "v52, v53 = sch.sample_perfect_tile(loop=l11, n=2, max_innermost_factor=64)",
            "l54, l55 = sch.split(loop=l11, factors=[v52, v53])",
            "sch.reorder(l16, l24, l32, l40, l17, l25, l33, l41, l46, l50, l54, l18, l26, l34, l42, l47, l51, l55, l19, l27, l35, l43)",
            'sch.annotate(block_or_loop=b4, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b4, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v56 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b4, ann_key="meta_schedule.unroll_explicit", ann_val=v56)',
        ],
        [
            'b0 = sch.get_block(name="compute", func_name="main")',
            'b1 = sch.get_block(name="bias_add", func_name="main")',
            'b2 = sch.get_block(name="bn_mul", func_name="main")',
            'b3 = sch.get_block(name="bn_add", func_name="main")',
            'b4 = sch.get_block(name="root", func_name="main")',
            "sch.compute_inline(block=b3)",
            "sch.compute_inline(block=b2)",
            "sch.compute_inline(block=b1)",
            "b5, = sch.get_consumers(block=b0)",
            "l6, l7, l8, l9, l10, l11, l12 = sch.get_loops(block=b0)",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l6, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l6, factors=[v13, v14, v15, v16])",
            "v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l7, n=4, max_innermost_factor=64)",
            "l25, l26, l27, l28 = sch.split(loop=l7, factors=[v21, v22, v23, v24])",
            "v29, v30, v31, v32 = sch.sample_perfect_tile(loop=l8, n=4, max_innermost_factor=64)",
            "l33, l34, l35, l36 = sch.split(loop=l8, factors=[v29, v30, v31, v32])",
            "v37, v38, v39, v40 = sch.sample_perfect_tile(loop=l9, n=4, max_innermost_factor=64)",
            "l41, l42, l43, l44 = sch.split(loop=l9, factors=[v37, v38, v39, v40])",
            "v45, v46 = sch.sample_perfect_tile(loop=l10, n=2, max_innermost_factor=64)",
            "l47, l48 = sch.split(loop=l10, factors=[v45, v46])",
            "v49, v50 = sch.sample_perfect_tile(loop=l11, n=2, max_innermost_factor=64)",
            "l51, l52 = sch.split(loop=l11, factors=[v49, v50])",
            "v53, v54 = sch.sample_perfect_tile(loop=l12, n=2, max_innermost_factor=64)",
            "l55, l56 = sch.split(loop=l12, factors=[v53, v54])",
            "sch.reorder(l17, l25, l33, l41, l18, l26, l34, l42, l47, l51, l55, l19, l27, l35, l43, l48, l52, l56, l20, l28, l36, l44)",
            "sch.reverse_compute_at(block=b5, loop=l41, preserve_unit_loops=1)",
            'sch.annotate(block_or_loop=b4, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b4, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v57 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b4, ann_key="meta_schedule.unroll_explicit", ann_val=v57)',
        ],
        [
            'b0 = sch.get_block(name="compute", func_name="main")',
            'b1 = sch.get_block(name="bias_add", func_name="main")',
            'b2 = sch.get_block(name="bn_mul", func_name="main")',
            'b3 = sch.get_block(name="bn_add", func_name="main")',
            'b4 = sch.get_block(name="root", func_name="main")',
            "sch.compute_inline(block=b3)",
            "sch.compute_inline(block=b2)",
            "sch.compute_inline(block=b1)",
            "b5, = sch.get_consumers(block=b0)",
            "l6, l7, l8, l9, l10, l11, l12 = sch.get_loops(block=b0)",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l6, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l6, factors=[v13, v14, v15, v16])",
            "v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l7, n=4, max_innermost_factor=64)",
            "l25, l26, l27, l28 = sch.split(loop=l7, factors=[v21, v22, v23, v24])",
            "v29, v30, v31, v32 = sch.sample_perfect_tile(loop=l8, n=4, max_innermost_factor=64)",
            "l33, l34, l35, l36 = sch.split(loop=l8, factors=[v29, v30, v31, v32])",
            "v37, v38, v39, v40 = sch.sample_perfect_tile(loop=l9, n=4, max_innermost_factor=64)",
            "l41, l42, l43, l44 = sch.split(loop=l9, factors=[v37, v38, v39, v40])",
            "v45, v46 = sch.sample_perfect_tile(loop=l10, n=2, max_innermost_factor=64)",
            "l47, l48 = sch.split(loop=l10, factors=[v45, v46])",
            "v49, v50 = sch.sample_perfect_tile(loop=l11, n=2, max_innermost_factor=64)",
            "l51, l52 = sch.split(loop=l11, factors=[v49, v50])",
            "v53, v54 = sch.sample_perfect_tile(loop=l12, n=2, max_innermost_factor=64)",
            "l55, l56 = sch.split(loop=l12, factors=[v53, v54])",
            "sch.reorder(l17, l25, l33, l41, l18, l26, l34, l42, l47, l51, l55, l19, l27, l35, l43, l48, l52, l56, l20, l28, l36, l44)",
            "sch.reverse_compute_at(block=b5, loop=l42, preserve_unit_loops=1)",
            'sch.annotate(block_or_loop=b4, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b4, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v57 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b4, ann_key="meta_schedule.unroll_explicit", ann_val=v57)',
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.conv2d_nchw_bias_bn_relu(
                n=1,
                h=56,
                w=56,
                ci=512,
                co=512,
                kh=3,
                kw=3,
                stride=1,
                padding=1,
            )
        ),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_meta_schedule_sketch_cpu_max_pool2d_nchw():
    # pylint: disable=line-too-long
    expected: List[List[str]] = [
        [
            'b0 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v1 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_explicit", ann_val=v1)',
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.max_pool2d_nchw(
                n=1,
                h=56,
                w=56,
                ci=512,
                padding=1,
            )
        ),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_meta_schedule_cpu_sketch_matmul()
    test_meta_schedule_cpu_sketch_matmul_relu()
    test_meta_schedule_cpu_sketch_conv2d_nchw()
    test_meta_schedule_cpu_sketch_conv2d_nchw_bias_bn_relu()
    test_meta_schedule_sketch_cpu_max_pool2d_nchw()
