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

from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.space_generation import check_trace, create_context
from tvm.target import Target
from tvm.te import create_prim_func


def _target() -> Target:
    return Target("cuda", host="llvm")


def test_meta_schedule_cuda_sketch_matmul():
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
            'sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)',
            'b44 = sch.cache_read(block=b0, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b44, loop=l28, preserve_unit_loops=1)",
            "l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b44)",
            "l51 = sch.fuse(l49, l50)",
            "v52, v53 = sch.sample_perfect_tile(loop=l51, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v53)',
            "sch.reverse_compute_at(block=b1, loop=l33, preserve_unit_loops=1)",
        ]
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
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_meta_schedule_cuda_sketch_matmul_relu():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="compute", func_name="main")',
            'b2 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "l3, l4, l5 = sch.get_loops(block=b0)",
            "v6, v7, v8, v9, v10 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64)",
            "l11, l12, l13, l14, l15 = sch.split(loop=l3, factors=[v6, v7, v8, v9, v10])",
            "v16, v17, v18, v19, v20 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64)",
            "l21, l22, l23, l24, l25 = sch.split(loop=l4, factors=[v16, v17, v18, v19, v20])",
            "v26, v27, v28 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=64)",
            "l29, l30, l31 = sch.split(loop=l5, factors=[v26, v27, v28])",
            "sch.reorder(l11, l21, l12, l22, l13, l23, l29, l30, l14, l24, l31, l15, l25)",
            "l32 = sch.fuse(l11, l21)",
            'sch.bind(loop=l32, thread_axis="blockIdx.x")',
            "l33 = sch.fuse(l12, l22)",
            'sch.bind(loop=l33, thread_axis="vthread.x")',
            "l34 = sch.fuse(l13, l23)",
            'sch.bind(loop=l34, thread_axis="threadIdx.x")',
            'b35 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b35, loop=l29, preserve_unit_loops=1)",
            "l36, l37, l38, l39, l40, l41 = sch.get_loops(block=b35)",
            "l42 = sch.fuse(l40, l41)",
            "v43, v44 = sch.sample_perfect_tile(loop=l42, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch", ann_val=v44)',
            'b45 = sch.cache_read(block=b0, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b45, loop=l29, preserve_unit_loops=1)",
            "l46, l47, l48, l49, l50, l51 = sch.get_loops(block=b45)",
            "l52 = sch.fuse(l50, l51)",
            "v53, v54 = sch.sample_perfect_tile(loop=l52, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b45, ann_key="meta_schedule.cooperative_fetch", ann_val=v54)',
            "sch.reverse_compute_at(block=b2, loop=l34, preserve_unit_loops=1)",
            "sch.reverse_compute_inline(block=b1)",
        ]
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
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_meta_schedule_cuda_sketch_conv2d_nchw():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="compute", func_name="main")',
            'b2 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")',
            "l3, l4, l5, l6, l7, l8, l9 = sch.get_loops(block=b1)",
            "v10, v11, v12, v13, v14 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64)",
            "l15, l16, l17, l18, l19 = sch.split(loop=l3, factors=[v10, v11, v12, v13, v14])",
            "v20, v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64)",
            "l25, l26, l27, l28, l29 = sch.split(loop=l4, factors=[v20, v21, v22, v23, v24])",
            "v30, v31, v32, v33, v34 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64)",
            "l35, l36, l37, l38, l39 = sch.split(loop=l5, factors=[v30, v31, v32, v33, v34])",
            "v40, v41, v42, v43, v44 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64)",
            "l45, l46, l47, l48, l49 = sch.split(loop=l6, factors=[v40, v41, v42, v43, v44])",
            "v50, v51, v52 = sch.sample_perfect_tile(loop=l7, n=3, max_innermost_factor=64)",
            "l53, l54, l55 = sch.split(loop=l7, factors=[v50, v51, v52])",
            "v56, v57, v58 = sch.sample_perfect_tile(loop=l8, n=3, max_innermost_factor=64)",
            "l59, l60, l61 = sch.split(loop=l8, factors=[v56, v57, v58])",
            "v62, v63, v64 = sch.sample_perfect_tile(loop=l9, n=3, max_innermost_factor=64)",
            "l65, l66, l67 = sch.split(loop=l9, factors=[v62, v63, v64])",
            "sch.reorder(l15, l25, l35, l45, l16, l26, l36, l46, l17, l27, l37, l47, l53, l59, l65, l54, l60, l66, l18, l28, l38, l48, l55, l61, l67, l19, l29, l39, l49)",
            "l68 = sch.fuse(l15, l25, l35, l45)",
            'sch.bind(loop=l68, thread_axis="blockIdx.x")',
            "l69 = sch.fuse(l16, l26, l36, l46)",
            'sch.bind(loop=l69, thread_axis="vthread.x")',
            "l70 = sch.fuse(l17, l27, l37, l47)",
            'sch.bind(loop=l70, thread_axis="threadIdx.x")',
            'b71 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b71, loop=l65, preserve_unit_loops=1)",
            "l72, l73, l74, l75, l76, l77, l78, l79, l80, l81 = sch.get_loops(block=b71)",
            "l82 = sch.fuse(l78, l79, l80, l81)",
            "v83, v84 = sch.sample_perfect_tile(loop=l82, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b71, ann_key="meta_schedule.cooperative_fetch", ann_val=v84)',
            'b85 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b85, loop=l65, preserve_unit_loops=1)",
            "l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b85)",
            "l96 = sch.fuse(l92, l93, l94, l95)",
            "v97, v98 = sch.sample_perfect_tile(loop=l96, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch", ann_val=v98)',
            "sch.reverse_compute_at(block=b2, loop=l70, preserve_unit_loops=1)",
            "sch.compute_inline(block=b0)",
        ]
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
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_meta_schedule_cuda_sketch_conv2d_nchw_bias_bn_relu():  # pylint: disable=invalid-name
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="compute", func_name="main")',
            'b2 = sch.get_block(name="bias_add", func_name="main")',
            'b3 = sch.get_block(name="bn_mul", func_name="main")',
            'b4 = sch.get_block(name="bn_add", func_name="main")',
            'b5 = sch.get_block(name="compute_1", func_name="main")',
            "sch.compute_inline(block=b4)",
            "sch.compute_inline(block=b3)",
            "sch.compute_inline(block=b2)",
            'b6 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")',
            "l7, l8, l9, l10, l11, l12, l13 = sch.get_loops(block=b1)",
            "v14, v15, v16, v17, v18 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64)",
            "l19, l20, l21, l22, l23 = sch.split(loop=l7, factors=[v14, v15, v16, v17, v18])",
            "v24, v25, v26, v27, v28 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64)",
            "l29, l30, l31, l32, l33 = sch.split(loop=l8, factors=[v24, v25, v26, v27, v28])",
            "v34, v35, v36, v37, v38 = sch.sample_perfect_tile(loop=l9, n=5, max_innermost_factor=64)",
            "l39, l40, l41, l42, l43 = sch.split(loop=l9, factors=[v34, v35, v36, v37, v38])",
            "v44, v45, v46, v47, v48 = sch.sample_perfect_tile(loop=l10, n=5, max_innermost_factor=64)",
            "l49, l50, l51, l52, l53 = sch.split(loop=l10, factors=[v44, v45, v46, v47, v48])",
            "v54, v55, v56 = sch.sample_perfect_tile(loop=l11, n=3, max_innermost_factor=64)",
            "l57, l58, l59 = sch.split(loop=l11, factors=[v54, v55, v56])",
            "v60, v61, v62 = sch.sample_perfect_tile(loop=l12, n=3, max_innermost_factor=64)",
            "l63, l64, l65 = sch.split(loop=l12, factors=[v60, v61, v62])",
            "v66, v67, v68 = sch.sample_perfect_tile(loop=l13, n=3, max_innermost_factor=64)",
            "l69, l70, l71 = sch.split(loop=l13, factors=[v66, v67, v68])",
            "sch.reorder(l19, l29, l39, l49, l20, l30, l40, l50, l21, l31, l41, l51, l57, l63, l69, l58, l64, l70, l22, l32, l42, l52, l59, l65, l71, l23, l33, l43, l53)",
            "l72 = sch.fuse(l19, l29, l39, l49)",
            'sch.bind(loop=l72, thread_axis="blockIdx.x")',
            "l73 = sch.fuse(l20, l30, l40, l50)",
            'sch.bind(loop=l73, thread_axis="vthread.x")',
            "l74 = sch.fuse(l21, l31, l41, l51)",
            'sch.bind(loop=l74, thread_axis="threadIdx.x")',
            'b75 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b75, loop=l69, preserve_unit_loops=1)",
            "l76, l77, l78, l79, l80, l81, l82, l83, l84, l85 = sch.get_loops(block=b75)",
            "l86 = sch.fuse(l82, l83, l84, l85)",
            "v87, v88 = sch.sample_perfect_tile(loop=l86, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b75, ann_key="meta_schedule.cooperative_fetch", ann_val=v88)',
            'b89 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b89, loop=l69, preserve_unit_loops=1)",
            "l90, l91, l92, l93, l94, l95, l96, l97, l98, l99 = sch.get_loops(block=b89)",
            "l100 = sch.fuse(l96, l97, l98, l99)",
            "v101, v102 = sch.sample_perfect_tile(loop=l100, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b89, ann_key="meta_schedule.cooperative_fetch", ann_val=v102)',
            "sch.reverse_compute_at(block=b6, loop=l74, preserve_unit_loops=1)",
            "sch.reverse_compute_inline(block=b5)",
            "sch.compute_inline(block=b0)",
        ]
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
    assert len(spaces) == 1
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_meta_schedule_cuda_sketch_matmul()
    test_meta_schedule_cuda_sketch_matmul_relu()
    test_meta_schedule_cuda_sketch_conv2d_nchw()
    test_meta_schedule_cuda_sketch_conv2d_nchw_bias_bn_relu()
