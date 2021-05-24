import tvm
from tvm import auto_scheduler

import logging
import os

logger = logging.getLogger(__name__)

from . import rand_seed, CUDATarget, get_log_filename

ansor_ntrials = int(os.getenv('ANSOR_NTRIALS', '20'))
logger.info("Ansor is doing {} trials".format(ansor_ntrials))


def auto_schedule(func, args, distrib=None):
    task = auto_scheduler.SearchTask(func=func, args=args, distrib=distrib,
                                     target=CUDATarget)

    if isinstance(args, list):
        if distrib is None:
            distrib = [1. for _ in args]
        logger.info("DietCode dynamic auto-scheduler")
        exit()

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=3, min_repeat_ms=100, timeout=10)
    
    log_filename = get_log_filename('ansor', func.str().lower())
    
    tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=ansor_ntrials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_filename)])

    cost_model = tvm.auto_scheduler.XGBModel(seed=rand_seed)
    search_policy = tvm.auto_scheduler.SketchPolicy(task, cost_model, seed=rand_seed)
    exit()
    task.tune(tune_option, search_policy)

    return task.apply_best(log_filename), task.print_best(log_filename)
