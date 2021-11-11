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
            'b54 = sch.get_block(name="compute", func_name="main")',
            "sch.reverse_compute_inline(block=b54)",
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
            'b0 = sch.get_block(name="compute", func_name="main")',
            'b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "l2, l3, l4, l5, l6, l7, l8 = sch.get_loops(block=b0)",
            "v9, v10, v11, v12, v13 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64)",
            "l14, l15, l16, l17, l18 = sch.split(loop=l2, factors=[v9, v10, v11, v12, v13])",
            "v19, v20, v21, v22, v23 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64)",
            "l24, l25, l26, l27, l28 = sch.split(loop=l3, factors=[v19, v20, v21, v22, v23])",
            "v29, v30, v31, v32, v33 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64)",
            "l34, l35, l36, l37, l38 = sch.split(loop=l4, factors=[v29, v30, v31, v32, v33])",
            "v39, v40, v41, v42, v43 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64)",
            "l44, l45, l46, l47, l48 = sch.split(loop=l5, factors=[v39, v40, v41, v42, v43])",
            "v49, v50, v51 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64)",
            "l52, l53, l54 = sch.split(loop=l6, factors=[v49, v50, v51])",
            "v55, v56, v57 = sch.sample_perfect_tile(loop=l7, n=3, max_innermost_factor=64)",
            "l58, l59, l60 = sch.split(loop=l7, factors=[v55, v56, v57])",
            "v61, v62, v63 = sch.sample_perfect_tile(loop=l8, n=3, max_innermost_factor=64)",
            "l64, l65, l66 = sch.split(loop=l8, factors=[v61, v62, v63])",
            "sch.reorder(l14, l24, l34, l44, l15, l25, l35, l45, l16, l26, l36, l46, l52, l58, l64, l53, l59, l65, l17, l27, l37, l47, l54, l60, l66, l18, l28, l38, l48)",
            "l67 = sch.fuse(l14, l24, l34, l44)",
            'sch.bind(loop=l67, thread_axis="blockIdx.x")',
            "l68 = sch.fuse(l15, l25, l35, l45)",
            'sch.bind(loop=l68, thread_axis="vthread.x")',
            "l69 = sch.fuse(l16, l26, l36, l46)",
            'sch.bind(loop=l69, thread_axis="threadIdx.x")',
            'b70 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b70, loop=l64, preserve_unit_loops=1)",
            "l71, l72, l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b70)",
            "l81 = sch.fuse(l77, l78, l79, l80)",
            "v82, v83 = sch.sample_perfect_tile(loop=l81, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b70, ann_key="meta_schedule.cooperative_fetch", ann_val=v83)',
            'b84 = sch.cache_read(block=b0, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b84, loop=l64, preserve_unit_loops=1)",
            "l85, l86, l87, l88, l89, l90, l91, l92, l93, l94 = sch.get_loops(block=b84)",
            "l95 = sch.fuse(l91, l92, l93, l94)",
            "v96, v97 = sch.sample_perfect_tile(loop=l95, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b84, ann_key="meta_schedule.cooperative_fetch", ann_val=v97)',
            "sch.reverse_compute_at(block=b1, loop=l69, preserve_unit_loops=1)",
            'b98 = sch.get_block(name="pad_temp", func_name="main")',
            "sch.compute_inline(block=b98)",
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
            'b0 = sch.get_block(name="bias_add", func_name="main")',
            'b1 = sch.get_block(name="bn_mul", func_name="main")',
            'b2 = sch.get_block(name="bn_add", func_name="main")',
            "sch.compute_inline(block=b2)",
            "sch.compute_inline(block=b1)",
            "sch.compute_inline(block=b0)",
            'b3 = sch.get_block(name="compute", func_name="main")',
            'b4 = sch.cache_write(block=b3, write_buffer_index=0, storage_scope="local")',
            "l5, l6, l7, l8, l9, l10, l11 = sch.get_loops(block=b3)",
            "v12, v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64)",
            "l17, l18, l19, l20, l21 = sch.split(loop=l5, factors=[v12, v13, v14, v15, v16])",
            "v22, v23, v24, v25, v26 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64)",
            "l27, l28, l29, l30, l31 = sch.split(loop=l6, factors=[v22, v23, v24, v25, v26])",
            "v32, v33, v34, v35, v36 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64)",
            "l37, l38, l39, l40, l41 = sch.split(loop=l7, factors=[v32, v33, v34, v35, v36])",
            "v42, v43, v44, v45, v46 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64)",
            "l47, l48, l49, l50, l51 = sch.split(loop=l8, factors=[v42, v43, v44, v45, v46])",
            "v52, v53, v54 = sch.sample_perfect_tile(loop=l9, n=3, max_innermost_factor=64)",
            "l55, l56, l57 = sch.split(loop=l9, factors=[v52, v53, v54])",
            "v58, v59, v60 = sch.sample_perfect_tile(loop=l10, n=3, max_innermost_factor=64)",
            "l61, l62, l63 = sch.split(loop=l10, factors=[v58, v59, v60])",
            "v64, v65, v66 = sch.sample_perfect_tile(loop=l11, n=3, max_innermost_factor=64)",
            "l67, l68, l69 = sch.split(loop=l11, factors=[v64, v65, v66])",
            "sch.reorder(l17, l27, l37, l47, l18, l28, l38, l48, l19, l29, l39, l49, l55, l61, l67, l56, l62, l68, l20, l30, l40, l50, l57, l63, l69, l21, l31, l41, l51)",
            "l70 = sch.fuse(l17, l27, l37, l47)",
            'sch.bind(loop=l70, thread_axis="blockIdx.x")',
            "l71 = sch.fuse(l18, l28, l38, l48)",
            'sch.bind(loop=l71, thread_axis="vthread.x")',
            "l72 = sch.fuse(l19, l29, l39, l49)",
            'sch.bind(loop=l72, thread_axis="threadIdx.x")',
            'b73 = sch.cache_read(block=b3, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b73, loop=l67, preserve_unit_loops=1)",
            "l74, l75, l76, l77, l78, l79, l80, l81, l82, l83 = sch.get_loops(block=b73)",
            "l84 = sch.fuse(l80, l81, l82, l83)",
            "v85, v86 = sch.sample_perfect_tile(loop=l84, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b73, ann_key="meta_schedule.cooperative_fetch", ann_val=v86)',
            'b87 = sch.cache_read(block=b3, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b87, loop=l67, preserve_unit_loops=1)",
            "l88, l89, l90, l91, l92, l93, l94, l95, l96, l97 = sch.get_loops(block=b87)",
            "l98 = sch.fuse(l94, l95, l96, l97)",
            "v99, v100 = sch.sample_perfect_tile(loop=l98, n=2, max_innermost_factor=4)",
            'sch.annotate(block_or_loop=b87, ann_key="meta_schedule.cooperative_fetch", ann_val=v100)',
            "sch.reverse_compute_at(block=b4, loop=l72, preserve_unit_loops=1)",
            'b101 = sch.get_block(name="pad_temp", func_name="main")',
            'b102 = sch.get_block(name="compute_1", func_name="main")',
            "sch.reverse_compute_inline(block=b102)",
            "sch.compute_inline(block=b101)",
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
