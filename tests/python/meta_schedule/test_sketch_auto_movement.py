import tvm
from tvm import te, tir
from tvm.script import tir as T
import numpy as np
from tvm.target import Target
import tir_tensor_intrin
import te_workload
from tvm.tir.schedule import Trace
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.space_generator import PostOrderApply

from tvm.meta_schedule.schedule_rule import (
    AutoInline,
    MultiLevelTiling,
    MultiLevelTilingAutoMovement,
    ParallelizeVectorizeUnroll,
    RandomComputeLocation,
    ReuseType,
    ScheduleRule,
    AddConstraintsAutoMovement
)

from tvm.meta_schedule.postproc import (
    RewriteParallelVectorizeUnroll,
    RewriteReductionBlockAutoMovement,
    RewriteTensorize,
    DisallowDynamicLoop,
    VerifyGPUCode
)



def _target() -> Target:
    return Target("nvidia/geforce-rtx-3080")

postprocs=[
    RewriteParallelVectorizeUnroll(),
    RewriteReductionBlockAutoMovement(),
    RewriteTensorize(
        compute_intrin="wmma_sync",
        load_intrin_A="wmma_load_a",
        load_intrin_B="wmma_load_b",
        store_intrin="wmma_store",
        init_intrin="wmma_fill",
    ),
    DisallowDynamicLoop(),
    VerifyGPUCode()
]

def _create_context(mod, target) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=[
            AutoInline(
                into_producer=False,
                into_consumer=True,
                into_cache_only=False,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            MultiLevelTilingAutoMovement(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
                max_innermost_factor=64,
                vector_load_max_len=4,
                reuse_read=ReuseType(
                    req="must",
                    levels=[[4,]],
                    scope=["shared",],
                ),
                reuse_write=ReuseType(
                    req="must",
                    levels=[[3,]],
                    scope=["local"],
                ),
                compute_intrin="wmma_sync"
            ),
            AutoInline(
                into_producer=True,
                into_consumer=True,
                into_cache_only=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            AddConstraintsAutoMovement(),
        ],
        postprocs=postprocs,
        task_name="test",
    )
    ctx.space_generator.initialize_with_tune_context(ctx)
    for rule in ctx.sch_rules:
        rule.initialize_with_tune_context(ctx)
    for rule in ctx.postprocs:
        rule.initialize_with_tune_context(ctx)
    return ctx


workload = te_workload.matmul_fp16(n=1024, m=1024, k=1024)
workload = te.create_prim_func(workload)
ctx = _create_context(
    workload,
    target=_target(),
)
spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
for space in spaces:
    for p in ctx.postprocs:
        p.apply(space)
    trace = Trace(space.trace.insts, {})
    # trace = trace.simplified(remove_postproc=False)
    str_trace = "\n".join(str(trace).strip().splitlines())
    print(str_trace)
    print()