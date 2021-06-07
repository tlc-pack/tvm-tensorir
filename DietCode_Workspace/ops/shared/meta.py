import tvm
from tvm import meta_schedule as ms
from tvm.rpc.server  import Server
from tvm.rpc.tracker import Tracker

import time


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


def cuda_space():
    return ms.space.PostOrderApply(
            stages=[
                ms.rule.multi_level_tiling(
                    structure="SSSRRSRS",
                    must_cache_read=True,
                    cache_read_scope="shared",
                    can_cache_write=True,
                    must_cache_write=True,
                    cache_write_scope="local",
                    consumer_inline_strict=False,
                    fusion_levels=[3],
                    vector_load_max_len=4,
                    tile_binds=["blockIdx.x", "vthread", "threadIdx.x"],
                ),
                ms.rule.inline_pure_spatial(strict_mode=False),
                ms.rule.parallelize_vectorize_unroll(
                    max_jobs_per_core=-1,  # disable parallelize
                    max_vectorize_extent=-1,  # disable vectorize
                    unroll_max_steps=[0, 16, 64, 512, 1024],
                    unroll_explicit=True,
                ),
            ],
            postprocs=[
                ms.postproc.rewrite_cooperative_fetch(),
                ms.postproc.rewrite_unbound_blocks(),
                ms.postproc.rewrite_parallel_vectorize_unroll(),
                ms.postproc.rewrite_reduction_block(),
                ms.postproc.disallow_dynamic_loops(),
                ms.postproc.verify_gpu_code(),
            ],
            )


def measurer():
    host = '0.0.0.0'
    tracker = Tracker(host, port=9000, port_end=10000, silent=True)
    device_key = '$local$device$%d' % tracker.port
    server = Server(
            host,
            port=tracker.port,
            port_end=10000,
            key=device_key,
            use_popen=True,
            silent=True,
            tracker_addr=(tracker.host, tracker.port),
            )
    runner = ms.RPCRunner(
            key=device_key,
            host=host,
            port=tracker.port,
            )
    time.sleep(0.5)
    return ms.ProgramMeasurer(
            builder=ms.LocalBuilder(), runner=runner,
            measure_callbacks=[ms.RecordToFile()],
            )
