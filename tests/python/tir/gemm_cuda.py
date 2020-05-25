import numpy as np
import tvm
from tvm import tir
from tvm import te


@tvm.tir.hybrid.script
def matmul(a, b, c):
    C = buffer_bind(c, (2048, 2048), "float32")
    A = buffer_bind(a, (2048, 2048), "float32")
    B = buffer_bind(b, (2048, 2048), "float32")
    reducer = comm_reducer(lambda x, y: x + y, float32(0))

    with block({}, writes=[C[0:2048, 0:2048]], reads=[A[0:2048, 0:2048], B[0:2048, 0:2048]],
               name="root"):
        for i in range(0, 2048):
            for j in range(0, 2048):
                for k in range(0, 2048):
                    with block({vi(0, 2048): i, vj(0, 2048): j, vk(0, 2048, iter_type="reduce"): k},
                               writes=[C[vi:(vi + 1), vj:(vj + 1)]],
                               reads=[C[vi:(vi + 1), vj:(vj + 1)], A[vk:(vk + 1), vi:(vi + 1)],
                                      B[vk:(vk + 1), vj:(vj + 1)]], name="C"):
                        reducer.step(C[vi, vj], A[vk, vi] * B[vk, vj])


n = 2048
device = 'cuda'
ctx = tvm.context(device, 0)
mod = tir.hybrid.create_module({"matmul": matmul})

original_func = mod["matmul"]

a_np = np.random.uniform(size=(n, n)).astype("float32")
b_np = np.random.uniform(size=(n, n)).astype("float32")
a = tvm.nd.array(a_np, ctx)
b = tvm.nd.array(b_np, ctx)
c = tvm.nd.array(np.zeros((n, n)).astype("float32"), ctx)


def build_and_test(func):
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return
    f = tvm.build(s.func, target=device)
    print(tvm.lower(func))
    print(f.imported_modules[0].get_source())
    f(a, b, c)
    tvm.testing.assert_allclose(c.asnumpy(), np.dot(a_np.T, b_np), rtol=1e-5)

    num_flops = 2 * n * n * n
    num_runs = 10
    timer_f = f.time_evaluator(f.entry_name, ctx, number=num_runs)
    t = timer_f(a, b, c).mean
    GFLOPS = num_flops / (t * 1e3) / 1e6
    print("Device %s" % device)
    print("average time cost of %d runs = %g ms, %g GFLOPS." % (num_runs, t * 1e3, GFLOPS))


scale = 8
num_thread = 8
block_factor = scale * num_thread

block_x = te.thread_axis("blockIdx.x")
thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
thread_xz = te.thread_axis((0, 2), "vthread", name="vx")
thread_yz = te.thread_axis((0, 2), "vthread", name="vy")

s = tir.create_schedule(original_func)
A = original_func.buffer_map[original_func.params[0]]
B = original_func.buffer_map[original_func.params[1]]
C = original_func.buffer_map[original_func.params[2]]

C_block = s.get_block("C")

AA = s.cache_read(A, "shared")
BB = s.cache_read(B, "shared")
AL = s.cache_read(AA.writes[0].buffer, "local")
BL = s.cache_read(BB.writes[0].buffer, "local")
CC = s.cache_write(C, "local")

y, x = s.get_axes(C_block)
by, yi = s.split(y, block_factor)
bx, xi = s.split(x, block_factor)
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

y, x, k = s.get_axes(CC)[-3:]
ko, ki = s.split(k, factor=8)
kt, ki = s.split(ki, factor=1)
s.reorder(ko, kt, ki, y, x)

s.compute_at(AL, kt)
s.compute_at(BL, kt)
s.compute_at(AA, ko)
s.compute_at(BB, ko)

print(s.func)

x, y = s.get_axes(AA)[-2:]
s.bind(x, thread_x)
x, y = s.get_axes(BB)[-2:]
s.bind(x, thread_y)

s.decompose_reduction(CC, tx)
build_and_test(s.func)
