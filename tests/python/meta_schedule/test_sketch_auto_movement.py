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
from tvm.meta_schedule import tune_tir
from tvm.meta_schedule.search_strategy import ReplayTraceConfig
import tvm.testing
import logging

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


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

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

def _create_context( mod,
                     target,
                     config,
                     task_name
                     ) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        search_strategy=config.create_strategy(),
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
                    scope=["shared.dyn",],
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
        task_name=task_name,
        rand_state=-1,
        num_threads=24,
    )
    ctx.space_generator.initialize_with_tune_context(ctx)
    for rule in ctx.sch_rules:
        rule.initialize_with_tune_context(ctx)
    for rule in ctx.postprocs:
        rule.initialize_with_tune_context(ctx)
    return ctx


workload = te_workload.matmul_fp16(n=1024, m=1024, k=1024)
workload = te.create_prim_func(workload)
sch = tune_tir(
    mod=workload,
    target=Target("nvidia/geforce-rtx-3080"),
    config=ReplayTraceConfig(
        num_trials_per_iter=32,
        num_trials_total=1024,
    ),
    f_tune_context=_create_context
)
if sch is None:
    print("No valid schedule found!")
else:
    print(sch.mod.script())
    print(sch.trace)

# ctx = _create_context(
#     workload,
#     target=_target(),
#     config=ReplayTraceConfig(
#                 num_trials_per_iter=32,
#                 num_trials_total=1024,
#             ),
#     task_name="matmul"
# )
# spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
# for space in spaces:
#     for p in ctx.postprocs:
#         p.apply(space)
#     # print(space.mod.script())
#     trace = Trace(space.trace.insts, {})
#     trace = trace.simplified(remove_postproc=False)
#     str_trace = "\n".join(str(trace).strip().splitlines())
#     print(str_trace)
#     print()

# mod= sch.mod['main']
# M=N=K=1024
# dev = tvm.device("cuda", 0)
# a_np = np.random.uniform(size=(N, K)).astype("float16")
# b_np = np.random.uniform(size=(K, M)).astype("float16")
# c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
# a = tvm.nd.array(a_np, dev)
# b = tvm.nd.array(b_np, dev)
# c = tvm.nd.array(np.zeros((N, M), dtype="float32"), dev)
# f = tvm.build(mod, target="cuda", name="dense")
# # print(f.imported_modules[0].get_source())
# f(a, b, c)
# tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)
#
# evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
# gflops = (N*M*K) * 2 / 1e9
# time_ms = evaluator(a, b, c).mean * 1e3
# print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))