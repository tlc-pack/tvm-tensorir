import tvm

import os

CPUTarget = "llvm"
CPUContext = tvm.context(CPUTarget, 0)
CUDATarget = tvm.target.Target(os.getenv('CUDA_TARGET', 'cuda'))
CUDAContext = tvm.gpu()

import logging
import numpy as np
import random

logger = logging.getLogger(__name__)

rand_seed = 0
random.seed(rand_seed)
np.random.seed(rand_seed)

auto_sched_ntrials = int(os.getenv('AUTO_SCHED_NTRIALS', '20'))
logger.info("Auto-scheduler is doing {} trials".format(auto_sched_ntrials))


def get_time_evaluator_results(kernel, module_data, ctx, number=100, repeat=10,
                               min_repeat_ms=100):
    warmup_evaluator = kernel.time_evaluator(kernel.entry_name, ctx,
                                             number=3, repeat=1,
                                             min_repeat_ms=300)
    warmup_evaluator(*module_data)
    time_evaluator = kernel.time_evaluator(kernel.entry_name, ctx,
                                           number=number, repeat=repeat,
                                           min_repeat_ms=min_repeat_ms)
    return time_evaluator(*module_data).results


def get_log_filename(auto_scheduler_name, wkl_name):
    log_filename = "{}_autosched_{}.json".format(auto_scheduler_name, wkl_name)
    try:
        os.remove(log_filename)
    except OSError:
        pass
    return log_filename