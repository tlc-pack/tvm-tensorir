import tvm
from tvm import meta_schedule as ms
from tvm import te, tir, topi
from tvm.meta_schedule.search_rule import SearchRule
from tvm.script import ty
from typing import Tuple


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 1024), "float32")
    B = tir.match_buffer(b, (1024, 1024), "float32")
    C = tir.match_buffer(c, (1024, 1024), "float32")
    with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

print(matmul)
# def matmul(n: int, m: int, k: int) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
#     a = te.placeholder((n, k), name="A")
#     b = te.placeholder((k, m), name="B")
#     k = te.reduce_axis((0, k), name="k")
#     c = te.compute(
#         (n, m),
#         lambda i, j: te.sum(a[i, k] * b[k, j], axis=[k]),
#         name="C",
#     )
#     return (a, b, c)

def cpu_space():
    return ms.space.PostOrderApply(
        stages=[
            ms.rule.inline_pure_spatial(strict_mode=True),
            ms.rule.multi_level_tiling(
                structure="SSRSRS",
                must_cache_read=False,
                cache_read_scope="global",
                can_cache_write=True,
                must_cache_write=False,
                cache_write_scope="global",
                consumer_inline_strict=True,
                fusion_levels=[1, 2],
            ),
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
            ms.postproc.disallow_dynamic_loops(),
        ],
    )

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


def main():
    task = ms.SearchTask(workload=matmul,
                         log_file='matmul_static.log')
    space = cpu_space()
    measurer = create_measurer()
    sch = ms.autotune(
        task=task,
        space=space,
        strategy="replay",
        measurer=measurer,
    )
    if sch is None:
        print("No valid schedule found")
    else:
        print(tvm.script.asscript(sch.mod))
        print("Schedule:")
        print("\n".join(sch.trace.as_python()))


if __name__ == "__main__":
    main()