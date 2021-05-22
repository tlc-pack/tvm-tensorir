from .ansor import rand_seed, CUDevice, CUTarget

import numpy as np

np.random.seed(rand_seed)


def get_time_evaluator_results(kernel, module_data, number=100, repeat=10,
                               min_repeat_ms=100):
    warmup_evaluator = kernel.time_evaluator(kernel.entry_name, CUDevice,
                                             number=3, repeat=1,
                                             min_repeat_ms=300)
    warmup_evaluator(*module_data)
    time_evaluator = kernel.time_evaluator(kernel.entry_name, CUDevice,
                                           number=number, repeat=repeat,
                                           min_repeat_ms=min_repeat_ms)
    return time_evaluator(*module_data).results
