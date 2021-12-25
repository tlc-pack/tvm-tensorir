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
from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.schedule_rule import cross_thread_reduction
from tvm.meta_schedule.testing.space_generation import check_trace
from tvm.meta_schedule.tune_context import TuneContext
from tvm.target import Target
from tvm.te.operation import create_prim_func


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


def test_gpu_softmax_mn():
    expected = [
        [],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            "b1, = sch.get_consumers(block=b0)",
            "l2, l3 = sch.get_loops(block=b1)",
            "v4, v5 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l6, l7 = sch.split(loop=l3, factors=[v4, v5])",
            'sch.bind(loop=l7, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l2, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l8, l9, l10 = sch.get_loops(block=b0)",
            "l11, l12 = sch.split(loop=l10, factors=[None, v5])",
            'sch.bind(loop=l12, thread_axis="threadIdx.x")',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            "b1, = sch.get_consumers(block=b0)",
            "l2, l3 = sch.get_loops(block=b1)",
            "v4, v5 = sch.sample_perfect_tile(loop=l3, n=2, max_innermost_factor=64)",
            "l6, l7 = sch.split(loop=l3, factors=[v4, v5])",
            'sch.bind(loop=l7, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l2, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l8, l9, l10 = sch.get_loops(block=b0)",
            "l11, l12 = sch.split(loop=l10, factors=[None, v5])",
            'sch.bind(loop=l12, thread_axis="threadIdx.x")',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            "b2, = sch.get_consumers(block=b1)",
            "l3, l4 = sch.get_loops(block=b2)",
            "v5, v6 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64)",
            "l7, l8 = sch.split(loop=l4, factors=[v5, v6])",
            'sch.bind(loop=l8, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b1, loop=l3, preserve_unit_loops=True)",
            'sch.set_scope(block=b1, buffer_index=0, storage_scope="shared")',
            "l9, l10, l11 = sch.get_loops(block=b1)",
            "l12, l13 = sch.split(loop=l11, factors=[None, v6])",
            'sch.bind(loop=l13, thread_axis="threadIdx.x")',
            "b14, = sch.get_consumers(block=b0)",
            "l15, l16 = sch.get_loops(block=b14)",
            "v17, v18 = sch.sample_perfect_tile(loop=l16, n=2, max_innermost_factor=64)",
            "l19, l20 = sch.split(loop=l16, factors=[v17, v18])",
            'sch.bind(loop=l20, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l15, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l21, l22, l23 = sch.get_loops(block=b0)",
            "l24, l25 = sch.split(loop=l23, factors=[None, v18])",
            'sch.bind(loop=l25, thread_axis="threadIdx.x")',
        ],
    ]
    target = Target("nvidia/geforce-rtx-3090", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.softmax_mn(
                n=256,
                m=256,
            )
        ),
        target=target,
        rule=cross_thread_reduction(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 4
    check_trace(spaces, expected)


def test_gpu_batch_norm_bmn():
    expected = [
        [],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            "b1, = sch.get_consumers(block=b0)",
            "l2, = sch.get_loops(block=b1)",
            "v3, v4 = sch.sample_perfect_tile(loop=l2, n=2, max_innermost_factor=64)",
            "l5, l6 = sch.split(loop=l2, factors=[v3, v4])",
            'sch.bind(loop=l6, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l5, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l7, l8, l9, l10 = sch.get_loops(block=b0)",
            "l11 = sch.fuse(l9, l10)",
            "l12, l13 = sch.split(loop=l11, factors=[None, v4])",
            'sch.bind(loop=l13, thread_axis="threadIdx.x")',
        ],
    ]
    target = Target("nvidia/geforce-rtx-3090", host="llvm")
    ctx = _create_context(
        create_prim_func(
            te_workload.norm_bmn(
                B=1,
                M=512,
                N=512,
            )
        ),
        target=target,
        rule=cross_thread_reduction(target=target),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 2
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_gpu_softmax_mn()
    test_gpu_batch_norm_bmn()
