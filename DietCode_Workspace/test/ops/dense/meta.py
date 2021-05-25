import tvm
from tvm import meta_schedule as ms

import logging
import numpy as np
import os
logger = logging.getLogger(__name__)

from ..shared import get_time_evaluator_results, meta, utils, get_log_filename, \
                     CPUContext, CPUTarget
from .fixture import numpyDenseFixture
from .wkl_def import Dense_static, Dense_dynamic, Dense_dynamic_BTIH
from .meta_saved_schedules import *


def test_tune_static():
    log_file = get_log_filename('meta', 'dense')
    sch = ms.autotune(
            task=ms.SearchTask(workload=Dense_static, log_file=log_file),
            space=meta.cpu_space(),
            strategy=ms.strategy.Replay(num_trials=200),
            measurer=meta.measurer())
    if sch is None:
        logger.info("No valid schedule found")
    else:
        logger.info(tvm.script.asscript(sch.mod))
        logger.info("Schedule:")
        logger.info("\n".join(sch.trace.as_python()))


@ms.rule.register_rule("do_micro_kernel")
def micro_kernel(task: ms.SearchTask, sch: ms.Schedule, block: ms.BlockRV):
    axes = sch.get_axes(block=block)
    outer, inner = [], []
    for axis in axes:
        tiles = sch.sample_tile_factor(loop=axis, n=2, where=[16, 32, 64, 128])
        outer_axis, inner_axis = sch.split(loop=axis, factors=tiles)
        outer.append(outer_axis)
        inner.append(inner_axis)
    new_axes = outer + inner
    sch.reorder(*new_axes)
    return sch


def test_sched_dynamic():
    task = ms.SearchTask(workload=Dense_dynamic)
    space = ms.space.PostOrderApply(stages=[micro_kernel])
    sch = space.sample_schedule(task)
    logger.info(tvm.script.asscript(sch.mod))


def test_sched_dynamic_experimental():
    task = ms.SearchTask(workload=Dense_dynamic_BTIH,
                         log_file=get_log_filename('meta', 'dense'),
                         shape_vars=('B', 'T', 'I', 'H'),
                         shape_freq={(16, 64, 768, 2304) : 1.0})


def test_tune_dynamic():
    os.environ["TVM_TRACKER_KEY"] = "local"

    task = ms.SearchTask(workload=Dense_dynamic,
                         log_file=get_log_filename('meta', 'dense'),
                         shape_vars=('M', 'N'),
                         shape_freq={(1024, 768) : 1.0})
    logger.info(task)
    sch = ms.autotune(
            task=task, space=meta.cpu_space(),
            strategy=ms.strategy.Replay(num_trials=300),
            measurer=meta.measurer()
            )
    if sch is None:
        logger.info("No valid schedule found")
    else:
        logger.info(tvm.script.asscript(sch.mod))
        logger.info("Schedule:")
        logger.info("\n".join(sch.trace.as_python()))


def build_and_test(mod, M, N):
    np_fixture = numpyDenseFixture(M, 768, N)
    build_func = tvm.build(mod["main"], target=CPUTarget)
    mod_data = np_fixture.module_data() + [M, N]
    build_func(*mod_data)
    # correctness checking
    np.testing.assert_allclose(mod_data[-1].asnumpy(),
                               np_fixture.Y_np_expected,
                               rtol=1e-5)
    # schedule logging
    logger.info(tvm.script.asscript(mod))
    # performance evaluation
    return np.average(get_time_evaluator_results(build_func, mod_data, CPUContext))


def test_perf():
    M, N = 1024, 768
    FLOPs = 2.0 * M * 768 * N

    sch = ms.Schedule(Dense_dynamic, debug_mode=True)
    sch = dense_1024x768x768(sch)
    manual_schedule_avg = build_and_test(sch.mod)
    
    mod = Dense_Mx768xN_Module()
    tvm.script.from_source(tvm.script.asscript(mod, True))
    manual_kernel_avg = build_and_test(mod)
    
    logger.info("Average Throughput: {} and {} GFLOPS"
                .format(FLOPs * 1e-9 / manual_schedule_avg,
                        FLOPs * 1e-9 / manual_kernel_avg))
