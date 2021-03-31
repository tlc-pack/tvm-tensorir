import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.search_rule import SearchRule
from tvm.script import ty
import os

@tvm.script.tir
def dyn_mm(a: ty.handle, b: ty.handle, c: ty.handle, M: ty.int32, N: ty.int32) -> None:
    A = tir.match_buffer(a, (M, 1024), "float32")
    B = tir.match_buffer(b, (1024, N), "float32")
    C = tir.match_buffer(c, (M, N), "float32")
    with tir.block([M, N, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@ms.rule.register_rule("do_mk")
def micro_kernel(_task: ms.SearchTask, sch: ms.Schedule, block: ms.BlockRV):
    axes = sch.get_axes(block=block)
    # iter_types = ms.analysis.get_block_var_types(sch.sch, sch.evaluate(block))
    # assert len(axes) == len(iter_types)
    # print('num of axes')
    outer, inner = [], []
    for i, axis in enumerate(axes):
        tiles = sch.sample_tile_factor(loop=axis, n=2, where=[16, 32, 64, 128])
        oaxis, iaxis = sch.split(loop=axis, factors=tiles)
        outer.append(oaxis)
        inner.append(iaxis)
    new_axes = outer + inner
    sch.reorder(*new_axes)
    return sch


def create_measurer():
    return ms.ProgramMeasurer(
        builder=ms.LocalBuilder(),
        runner=ms.RPCRunner(
            key="local",
            host="0.0.0.0",
            port=9190,
        ),
        measure_callbacks=[
            ms.RecordToFile(),
        ],
    )


def test_dynamic_matmul_schedule():
    task=ms.SearchTask(workload=dyn_mm)
    space = ms.space.PostOrderApply(stages=[micro_kernel])
    sch = space.sample_schedule(task)
    print(tvm.script.asscript(sch.mod))



def test_dynamic_matmul_autotune():
    os.environ["TVM_TRACKER_KEY"] = "local"
    shape_vars = ('M', 'N')
    shape_variants = [
        (256, 256),
        (512, 512)
    ]
    shape_freq = (0.5, 0.5)
    task=ms.SearchTask(workload=dyn_mm,
                       log_file='matmul_dynamic.log',
                       shape_vars=shape_vars, 
                       shape_variants=shape_variants, 
                       shape_freq=shape_freq, 
                       )
    print(task)
    print(task.shape_vars)
    print(task.shape_variants)
    print(task.shape_freq)
    space = ms.space.PostOrderApply(
        stages=[
            micro_kernel,
            ms.rule.inline_pure_spatial(strict_mode=True),
            ms.rule.parallelize_vectorize_unroll(
                max_jobs_per_core=16,
                max_vectorize_extent=32,
                unroll_max_steps=[0, 16, 64, 512],
                unroll_explicit=True,
            ),
            ms.rule.random_compute_location(),
        ],
        postprocs=[
            ms.postproc.rewrite_parallel_vectorize_unroll(),
            ms.postproc.rewrite_reduction_block(),
        ],
    )
    sch = ms.autotune(
        task=task,
        space=space,
        strategy="replay",
        measurer=create_measurer(),
    )

# test_dynamic_matmul_schedule()
test_dynamic_matmul_autotune()
