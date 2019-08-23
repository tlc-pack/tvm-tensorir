import logging
import sys
import numpy as np
import tvm
from tvm import autotvm, tensorir

@autotvm.template
def matmul(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)
        C_init, C = s.blocks()
        y, x, k = s.axis(C)

        cfg = autotvm.get_config()
        cfg.define_split("tile_y", y, num_outputs=2)
        cfg.define_split("tile_x", x, num_outputs=2)
        yo, yi = cfg["tile_y"].apply(s, C, y, use_tensorir=True)
        xo, xi = cfg["tile_x"].apply(s, C, x, use_tensorir=True)

        s.reorder(yo, xo, k, yi, xi)
        stmt = s.to_halide()
        return stmt

    return s, [A, B, C], _schedule_pass

N, L, M = 512, 512, 512
task = autotvm.task.create(matmul, args=(N, L, M, 'float32'), target='llvm')
print(task.config_space)

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

# begin tuning, log records to file `matmul.log`
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(n_trial=10,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul.log')])

# apply history best from log file
with autotvm.apply_history_best('matmul.log'):
    with tvm.target.create("llvm"):
        s, args, stmt = matmul(N, L, M, 'float32')
        func = tvm.build(stmt)

# check correctness
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
