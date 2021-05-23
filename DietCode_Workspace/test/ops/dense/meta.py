import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.script import ty

from ..shared import meta



@tvm.script.tir
def Dense_static(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 768), "float32")
    B = tir.match_buffer(b, (768,  768), "float32")
    C = tir.match_buffer(c, (1024, 768), "float32")
    with tir.block([1024, 768, tir.reduce_axis(0, 768)], "matmul") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
def Dense_dynamic(a: ty.handle, b: ty.handle, c: ty.handle, M: ty.int32, N: ty.int32) -> None:
    A = tir.match_buffer(a, (M, 768), "float32")
    B = tir.match_buffer(b, (768, N), "float32")
    C = tir.match_buffer(c, (M, N), "float32")
    with tir.block([M, N, tir.reduce_axis(0, 768)], "matmul") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def test_tune_static():
    log_file='matmul_static.log'
    sched = ms.autotune(
            task=ms.SearchTask(workload=Dense_static, log_file=log_file),
            space=meta.cpu_space(),
            strategy=ms.strategy.Replay(num_trials=200),
            measurer=meta.measurer())
    if sched is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sched.mod))
        print("Schedule:")
        print("\n".join(sched.trace.as_python()))


@ms.rule.register_rule("do_micro_kernel")
def micro_kernel(_task: ms.SearchTask, sch: ms.Schedule, block: ms.BlockRV):
    axes = sch.get_axes(block=block)
    outer, inner = [], []
    for i, axis in enumerate(axes):
        tiles = sch.sample_tile_factor(loop=axis, n=2, where=[16, 32, 64, 128])
        outer_axis, inner_axis = sch.split(loop=axis, factors=tiles)
        outer.append(outer_axis)
        inner.append(inner_axis)
    new_axes = outer + inner
    sch.reorder(*new_axes)
    return sch
