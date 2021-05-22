import tvm
from tvm import meta_schedule as ms


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


def measurer():
    return ms.ProgramMeasurer(
            builder=ms.LocalBuilder(),
            runner =ms.RPCRunner(
                key="local",
                host="0.0.0.0",
                port=9190,
            ),
            measure_callbacks=[
                ms.RecordToFile(),
            ],
            )
