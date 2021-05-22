import tvm
from tvm import auto_scheduler

import logging
import os
import random

logger = logging.getLogger(__name__)

rand_seed = 0
random.seed(rand_seed)

CUTarget = tvm.target.Target(os.getenv('CUTARGET', 'cuda'))
CUDevice = tvm.cuda()


def _log_file(file):
    try:
        os.remove("ansor_autosched_{}.json".format(
                os.path.splitext(
                    os.path.basename(file))[0]))
    except OSError:
        pass
    return "autosched_{}.json".format(os.path.splitext(os.path.basename(file))[0])


ansor_ntrials = int(os.getenv('ANSOR_NTRIALS', '20'))
logger.info("Ansor is doing {} trials".format(ansor_ntrials))


def auto_schedule(func, args):
    task = auto_scheduler.SearchTask(func=func, args=args, target=CUTarget)

    if isinstance(args, list):
        logger.info("DietCode dynamic auto-scheduler")
        exit()

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=3, min_repeat_ms=100, timeout=10)
    
    log_file = _log_file(func.str().lower())
    
    tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=ansor_ntrials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)])

    cost_model = tvm.auto_scheduler.XGBModel(seed=rand_seed)
    search_policy = tvm.auto_scheduler.SketchPolicy(task, cost_model, seed=rand_seed)
    task.tune(tune_option, search_policy)

    return task.apply_best(log_file), task.print_best(log_file)
