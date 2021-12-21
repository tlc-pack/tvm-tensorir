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
            "l4, l5 = sch.split(loop=l3, factors=[None, 32])",
            'sch.bind(loop=l5, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l2, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l6, l7, l8 = sch.get_loops(block=b0)",
            "l9, l10 = sch.split(loop=l8, factors=[None, 32])",
            'sch.bind(loop=l10, thread_axis="threadIdx.x")',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            "b1, = sch.get_consumers(block=b0)",
            "l2, l3 = sch.get_loops(block=b1)",
            "l4, l5 = sch.split(loop=l3, factors=[None, 32])",
            'sch.bind(loop=l5, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l2, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l6, l7, l8 = sch.get_loops(block=b0)",
            "l9, l10 = sch.split(loop=l8, factors=[None, 32])",
            'sch.bind(loop=l10, thread_axis="threadIdx.x")',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            "b2, = sch.get_consumers(block=b1)",
            "l3, l4 = sch.get_loops(block=b2)",
            "l5, l6 = sch.split(loop=l4, factors=[None, 32])",
            'sch.bind(loop=l6, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b1, loop=l3, preserve_unit_loops=True)",
            'sch.set_scope(block=b1, buffer_index=0, storage_scope="shared")',
            "l7, l8, l9 = sch.get_loops(block=b1)",
            "l10, l11 = sch.split(loop=l9, factors=[None, 32])",
            'sch.bind(loop=l11, thread_axis="threadIdx.x")',
            "b12, = sch.get_consumers(block=b0)",
            "l13, l14 = sch.get_loops(block=b12)",
            "l15, l16 = sch.split(loop=l14, factors=[None, 32])",
            'sch.bind(loop=l16, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l13, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l17, l18, l19 = sch.get_loops(block=b0)",
            "l20, l21 = sch.split(loop=l19, factors=[None, 32])",
            'sch.bind(loop=l21, thread_axis="threadIdx.x")',
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
            "l3, l4 = sch.split(loop=l2, factors=[None, 32])",
            'sch.bind(loop=l4, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l3, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l5, l6, l7, l8 = sch.get_loops(block=b0)",
            "l9 = sch.fuse(l7, l8)",
            "l10, l11 = sch.split(loop=l9, factors=[None, 32])",
            'sch.bind(loop=l11, thread_axis="threadIdx.x")',
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
