import tvm
from tvm import meta_schedule as ms

import logging
import os
logger = logging.getLogger(__name__)

from ..shared import meta, get_log_filename
from .wkl_def import Dense_static, Dense_dynamic


def test_tune_static():
    log_file = get_log_filename('meta', 'dense')
    sched = ms.autotune(
            task=ms.SearchTask(workload=Dense_static, log_file=log_file),
            space=meta.cpu_space(),
            strategy=ms.strategy.Replay(num_trials=200),
            measurer=meta.measurer())
    if sched is None:
        logger.info("No valid schedule found")
    else:
        logger.info(tvm.script.asscript(sched.mod))
        logger.info("Schedule:")
        logger.info("\n".join(sched.trace.as_python()))


@ms.rule.register_rule("do_micro_kernel")
def micro_kernel(task: ms.SearchTask, sched: ms.Schedule, block: ms.BlockRV):
    axes = sched.get_axes(block=block)
    outer, inner = [], []
    for axis in axes:
        tiles = sched.sample_tile_factor(loop=axis, n=2, where=[16, 32, 64, 128])
        outer_axis, inner_axis = sched.split(loop=axis, factors=tiles)
        outer.append(outer_axis)
        inner.append(inner_axis)
    new_axes = outer + inner
    sched.reorder(*new_axes)
    return sched


def test_sched_dynamic():
    task = ms.SearchTask(workload=Dense_dynamic)
    space = ms.space.PostOrderApply(stages=[micro_kernel])
    sched = space.sample_schedule(task)
    logger.info(tvm.script.asscript(sched.mod))


def test_tune_dynamic():
    os.environ["TVM_TRACKER_KEY"] = "local"

    task = ms.SearchTask(workload=Dense_dynamic,
                         log_file=get_log_filename('meta', 'dense'),
                         shape_vars=('M', 'N'),
                         shape_variants=[
                             (1024, 768),
                         ],
                         shape_freq=(1.0, ),
                         )
    logger.info(task)
    sched = ms.autotune(
            task=task, space=meta.cpu_space(),
            strategy=ms.strategy.Replay(num_trials=300),
            measurer=meta.measurer()
            )
    if sched is None:
        logger.info("No valid schedule found")
    else:
        logger.info(tvm.script.asscript(sched.mod))
        logger.info("Schedule:")
        logger.info("\n".join(sched.trace.as_python()))
