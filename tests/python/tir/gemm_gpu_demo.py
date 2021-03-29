import numpy as np
import tvm
import tvm.testing
from tvm import tir
from tvm.script import ty


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    C = tir.match_buffer(c, (2048, 2048), "float32")
    A = tir.match_buffer(a, (2048, 2048), "float32")
    B = tir.match_buffer(b, (2048, 2048), "float32")

    with tir.block([2048, 2048, tir.reduce_axis(0, 2048)], "C") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = tir.float32(0)
        C[vi, vj] = C[vi, vj] + A[vk, vi] * B[vk, vj]


n = 2048
device = "cuda"
ctx = tvm.device(device, 0)
mod = tvm.script.create_module({"matmul": matmul})

original_func = mod["matmul"]

a_np = np.random.uniform(size=(n, n)).astype("float32")
b_np = np.random.uniform(size=(n, n)).astype("float32")
a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(b_np, ctx)
c = tvm.nd.array(np.zeros((n, n)).astype("float32"), ctx)


def build_and_test(mod):
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return
    f = tvm.build(mod["main"], target=device)
    print(tvm.script.asscript(mod))
    f(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np.dot(a_np.T, b_np), rtol=1e-5)

    num_flops = 2 * n * n * n
    num_runs = 100
    timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("Device %s" % device)
    print("average time cost of %d runs = %g ms, %g GFLOPS." % (num_runs, t * 1e3, GFLOPS))


scale = 8
num_thread = 8
block_factor = scale * num_thread

block_x = tir.thread_axis("blockIdx.x")
thread_x = tir.thread_axis((0, num_thread), "threadIdx.x")
block_y = tir.thread_axis("blockIdx.y")
thread_y = tir.thread_axis((0, num_thread), "threadIdx.y")
thread_xz = tir.thread_axis((0, 2), "vthread", name="vx")
thread_yz = tir.thread_axis((0, 2), "vthread", name="vy")

s = tir.Schedule(original_func, debug_mode=True)

C = s.get_block("C")

AA = s.cache_read(C, 1, "shared")
BB = s.cache_read(C, 2, "shared")
AL = s.cache_read(C, 1, "local")
BL = s.cache_read(C, 2, "local")
CC = s.cache_write(C, 0, "local")

y, x = s.get_loops(C)
by, yi = s.split(y, factor=block_factor)
bx, xi = s.split(x, factor=block_factor)
s.reorder(by, bx, yi, xi)
s.bind(by, block_y)
s.bind(bx, block_x)

tyz, yi = s.split(yi, nparts=2)
ty, yi = s.split(yi, nparts=num_thread)
txz, xi = s.split(xi, nparts=2)
tx, xi = s.split(xi, nparts=num_thread)
s.reorder(tyz, txz, ty, tx, yi, xi)
s.bind(tyz, thread_yz)
s.bind(txz, thread_xz)
s.bind(ty, thread_y)
s.bind(tx, thread_x)

s.compute_at(CC, tx)
y, x, k = s.get_loops(CC)[-3:]
ko, ki = s.split(k, factor=8)
kt, ki = s.split(ki, factor=1)
s.reorder(ko, kt, ki, y, x)
decompose_pos = ko
s.unroll(kt)

s.compute_at(AL, kt)
s.compute_at(BL, kt)
s.compute_at(AA, ko)
s.compute_at(BB, ko)

x, y = s.get_loops(AA)[-2:]
ty, xi = s.split(x, nparts=num_thread)
_, xi = s.split(y, factor=num_thread * 4)
tx, xi = s.split(xi, nparts=num_thread)
s.bind(ty, thread_y)
s.bind(tx, thread_x)
s.vectorize(xi)
s.double_buffer(AA)

x, y = s.get_loops(BB)[-2:]
ty, xi = s.split(x, nparts=num_thread)
_, xi = s.split(y, factor=num_thread * 4)
tx, xi = s.split(xi, nparts=num_thread)
s.bind(ty, thread_y)
s.bind(tx, thread_x)
s.vectorize(xi)
s.double_buffer(BB)

s.decompose_reduction(CC, decompose_pos)

with tvm.transform.PassContext(
    config={
        "tir.UnrollLoop": {
            "auto_max_step": 128,
            "explicit_unroll": device != "cuda",
        },
    }
):
    build_and_test(s.mod)
