import tvm

import os

CPUTarget = "llvm"
CPUContext = tvm.context(CPUTarget, 0)
CUDATarget = tvm.target.Target(os.getenv('CUDA_TARGET', 'cuda'))
CUDAContext = tvm.gpu()


import numpy as np
import random

rand_seed = 0
random.seed(rand_seed)
np.random.seed(rand_seed)


def get_time_evaluator_results(kernel, module_data, context, number=100, repeat=10,
                               min_repeat_ms=100):
    warmup_evaluator = kernel.time_evaluator(kernel.entry_name, context,
                                             number=3, repeat=1,
                                             min_repeat_ms=300)
    warmup_evaluator(*module_data)
    time_evaluator = kernel.time_evaluator(kernel.entry_name, context,
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
