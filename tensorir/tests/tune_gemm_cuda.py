import logging
import sys
import numpy as np
import tvm
from tvm import autotvm, tensorir

@autotvm.template
def matmul(N, L, M, dtype):
    A = tvm.placeholder((L, N), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[k, i] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        thread_y = tvm.thread_axis("threadIdx.y")
        thread_xz = tvm.thread_axis("vthread", name="vx")
        thread_yz = tvm.thread_axis("vthread", name="vy")

        CL_init, CL_update = s.blocks()
        AS = s.cache_read(CL_update.inputs[1].data, "shared")
        BS = s.cache_read(CL_update.inputs[2].data, "shared")
        AL = s.cache_read(CL_update.inputs[1].data, "local")
        BL = s.cache_read(CL_update.inputs[2].data, "local")
        C = s.cache_write(CL_update.outputs[0].data, "local")
        cfg = autotvm.get_config()

        cfg.define_knob("scale", [2, 4, 8])
        cfg.define_knob("num_thread", [2, 4, 8])
        cfg.define_knob("shared", [1, 2, 4])
        scale = cfg["scale"].val
        num_thread = cfg["num_thread"].val
        shared = cfg["shared"].val
        block_factor = scale * num_thread

        def split_calc(block):
            by, yi = s.split(s.axis(block)[-2], factor=block_factor)
            bx, xi = s.split(s.axis(block)[-1], factor=block_factor)
            s.bind(by, block_y)
            s.bind(bx, block_x)
            bx, yi = s.reorder(bx, yi)

            tyz, yi = s.split(yi, nparts=2)
            ty, yi = s.split(yi, nparts=num_thread)
            txz, xi = s.split(xi, nparts=2)
            tx, xi = s.split(xi, nparts=num_thread)
            s.bind(tyz, thread_yz)
            s.bind(txz, thread_xz)
            s.bind(ty, thread_y)
            s.bind(tx, thread_x)
            tyz, txz, ty, tx, yi, xi = s.reorder(tyz, txz, ty, tx, yi, xi)

            return tx, yi
        tx, yi = split_calc(C)

        s.compute_at(CL_update, tx)
        s.compute_at(CL_init, s.axis(CL_update)[-2])

        k = s.axis(CL_update)[-1]
        yo, xo = s.axis(CL_init)[-2:]
        ko, ki = s.split(k, factor=num_thread)
        kt, ki = s.split(ki, factor=1)
        ko, kt, ki, yo, xo = s.reorder(ko, kt, ki, yo, xo)

        s.compute_at(AL, kt)
        s.compute_at(BL, kt)
        s.compute_at(AS, ko)
        s.compute_at(BS, ko)
        s.annotate(kt, "unroll")

        # Schedule for A's shared memory load

        ty, tx = s.axis(AS)[-2:]
        _, xi = s.split(tx, factor=num_thread * shared)
        tx, xi = s.split(xi, nparts=num_thread)

        s.bind(ty, thread_y)
        s.bind(tx, thread_x)
        s.annotate(xi, "vectorize")

        # Schedule for B's shared memory load
        ty, tx = s.axis(BS)[-2:]
        _, xi = s.split(tx, factor=num_thread * shared)
        tx, xi = s.split(xi, nparts=num_thread)
        s.bind(ty, thread_y)
        s.bind(tx, thread_x)
        s.annotate(xi, "vectorize")

        s.double_buffer_scope(AS.outputs[0].data)
        s.double_buffer_scope(BS.outputs[0].data)
        stmt = s.to_halide()
        return stmt

    return s, [A, B, C], _schedule_pass

N, L, M = 1024, 1024, 1024
task = autotvm.task.create(matmul, args=(N, L, M, 'float32'), target='cuda')
print(task.config_space)

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

measure_option = autotvm.measure_option(
    builder='local',
    runner=autotvm.LocalRunner(number=5))

# begin tuning, log records to file `matmul.log`
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(n_trial=12,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('matmul.log')])

# apply history best from log file
with autotvm.apply_history_best('matmul.log'):
    with tvm.target.create("cuda"):
        s, args, stmt = matmul(N, L, M, 'float32')
        func = tvm.build(stmt)

# check correctness
ctx = tvm.gpu()
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.transpose().dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
func(tvm.nd.array(a_np, ctx=ctx), tvm.nd.array(b_np, ctx=ctx), c_tvm)
tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
