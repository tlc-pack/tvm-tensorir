import tvm
from tvm import auto_scheduler

import logging
import os

logger = logging.getLogger(__name__)

from . import rand_seed, CUDATarget, get_log_filename

ansor_ntrials = int(os.getenv('ANSOR_NTRIALS', '20'))
logger.info("Ansor is doing {} trials".format(ansor_ntrials))


def auto_schedule(func, args, shape_vars=None, shape_freq=None):
    task = auto_scheduler.SearchTask(func=func, args=args,
                                     shape_vars=shape_vars,
                                     shape_freq=shape_freq,
                                     target=CUDATarget)

    if isinstance(args, list):
        logger.info("DietCode dynamic auto-scheduler")
        exit()

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=3, min_repeat_ms=100, timeout=10)
    
    log_filename = get_log_filename('ansor', func.__name__.lower())
    
    tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=ansor_ntrials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_filename)])

    cost_model = tvm.auto_scheduler.XGBModel(seed=rand_seed)
    search_policy = tvm.auto_scheduler.SketchPolicy(task, cost_model, seed=rand_seed)
    task.tune(tune_option, search_policy)

    return task.apply_best(log_filename), task.print_best(log_filename)
