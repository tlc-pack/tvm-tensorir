"""
Workload Definition
"""
import tvm
from tvm import te, topi, tir, auto_scheduler
from tvm.script import ty

import logging
logger = logging.getLogger(__name__)


@auto_scheduler.register_workload
def Dense(B, I, H):
    logger.info("Dense search task created with (B={B}, I={I}, H={H})"
                .format(B=B, I=I, H=H))
    X = te.placeholder((B, I), name='X')
    W = te.placeholder((H, I), name='W')
    Y = topi.nn.dense(X, W)
    return [X, W, Y]


@tvm.script.tir
def Dense_static(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (1024, 768), "float32")
    B = tir.match_buffer(b, (768,  768), "float32")
    C = tir.match_buffer(c, (1024, 768), "float32")
    with tir.block([1024, 768, tir.reduce_axis(0, 768)], "matmul") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
def Dense_dynamic(a: ty.handle, b: ty.handle, c: ty.handle, M: ty.int32, N: ty.int32) -> None:
    A = tir.match_buffer(a, (M, 768), "float32")
    B = tir.match_buffer(b, (768, N), "float32")
    C = tir.match_buffer(c, (M, N), "float32")
    with tir.block([M, N, tir.reduce_axis(0, 768)], "matmul") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
def Dense_dynamic_BTIH(X: ty.handle, W: ty.handle, Y: ty.handle,
                       B: ty.int32, T: ty.int32,
                       I: ty.int32, H: ty.int32) -> None:
    X_buf = tir.match_buffer(X, (B * T, I), "float32")
    W_buf = tir.match_buffer(W, (H, I), "float32")
    Y_buf = tir.match_buffer(Y, (B * T, I), "float32")
    with tir.block([B * T, H, tir.reduce_axis(0, I)], "matmul") as [vi, vj, vk]:
        with tir.init():
            Y_buf[vi, vj] = 0.0
        Y_buf[vi, vj] = Y_buf[vi, vj] + X_buf[vi, vk] * W_buf[vk, vj]