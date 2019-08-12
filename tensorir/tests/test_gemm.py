# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Example code to do square matrix multiplication."""
import tvm
from tvm import tensorir
from test_common import check_correctness

def origin_schedule():
    # graph
    nn = 2048
    n = tvm.var('n')
    n = tvm.convert(nn)
    m, l = n, n
    A = tvm.placeholder((l, n), name='A')
    B = tvm.placeholder((l, m), name='B')
    k = tvm.reduce_axis((0, l), name='k')
    C = tvm.compute(
        (m, n),
        lambda ii, jj: tvm.sum(A[k, jj] * B[k, ii], axis=k),
        name='C')

    # schedule
    s = tvm.create_schedule(C.op)
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AS, "local", [C])
    BL = s.cache_read(BS, "local", [C])
    CC = s.cache_write(C, "local")
    #
    scale = 8
    num_thread = 8
    block_factor = scale * num_thread
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, 2), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, 2), "vthread", name="vy")
    #
    by, yi = s[C].split(C.op.axis[0], factor=block_factor)
    bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].reorder(by, bx, yi, xi)
    #
    tyz, yi = s[C].split(yi, nparts=2)
    ty, yi = s[C].split(yi, nparts=num_thread)
    txz, xi = s[C].split(xi, nparts=2)
    tx, xi = s[C].split(xi, nparts=num_thread)
    s[C].bind(tyz, thread_yz)
    s[C].bind(txz, thread_xz)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(tyz, txz, ty, tx, yi, xi)
    s[CC].compute_at(s[C], tx)

    yo, xo = CC.op.axis
    ko, ki = s[CC].split(k, factor=8)
    kt, ki = s[CC].split(ki, factor=1)
    s[CC].reorder(ko, kt, ki, yo, xo)

    s[AS].compute_at(s[CC], ko)
    s[BS].compute_at(s[CC], ko)

    s[CC].unroll(kt)
    s[AL].compute_at(s[CC], kt)
    s[BL].compute_at(s[CC], kt)

    # Schedule for A's shared memory load
    ty, xi = s[AS].split(s[AS].op.axis[0], nparts=num_thread)
    _, xi = s[AS].split(s[AS].op.axis[1], factor=num_thread * 4)
    tx, xi = s[AS].split(xi, nparts=num_thread)
    s[AS].bind(ty, thread_y)
    s[AS].bind(tx, thread_x)
    s[AS].vectorize(xi)
    # Schedule for B' shared memory load
    ty, xi = s[BS].split(s[BS].op.axis[0], nparts=num_thread)
    _, xi = s[BS].split(s[BS].op.axis[1], factor=num_thread * 4)
    tx, xi = s[BS].split(xi, nparts=num_thread)
    s[BS].bind(ty, thread_y)
    s[BS].bind(tx, thread_x)
    s[BS].vectorize(xi)
    s[AS].double_buffer()
    s[BS].double_buffer()
    return s, [A, B, C]

def test_gemm():
    s, args = origin_schedule()

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)

        stmt = s.to_halide()
        return stmt

    check_correctness(s, args, _schedule_pass, 'cuda')

def test_gemm_schedule():
    # graph
    n, m, l = 2048, 2048, 2048
    A = tvm.placeholder((l, n), name='A')
    B = tvm.placeholder((l, m), name='B')
    k = tvm.reduce_axis((0, l), name='k')
    C = tvm.compute(
        (m, n),
        lambda ii, jj: tvm.sum(A[k, jj] * B[k, ii], axis=k),
        name='C')

    # schedule
    s = tvm.create_schedule(C.op)

    def _schedule_pass(stmt):
        s = tensorir.create_schedule(stmt)
        scale = 8
        num_thread = 8
        block_factor = scale * num_thread

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
        ko, ki = s.split(k, factor=8)
        kt, ki = s.split(ki, factor=1)
        ko, kt, ki, yo, xo = s.reorder(ko, kt, ki, yo, xo)

        s.compute_at(AL, kt)
        s.compute_at(BL, kt)

        s.compute_at(AS, ko)
        s.compute_at(BS, ko)
        s.annotate(kt, "unroll")

        # Schedule for A's shared memory load
        ty, tx = s.axis(AS)[-2:]
        _, xi = s.split(tx, factor=num_thread * 4)
        tx, xi = s.split(xi, nparts=num_thread)
        s.bind(ty, thread_y)
        s.bind(tx, thread_x)
        s.annotate(xi, "vectorize")

        # Schedule for B's shared memory load
        ty, tx = s.axis(BS)[-2:]
        _, xi = s.split(tx, factor=num_thread * 4)
        tx, xi = s.split(xi, nparts=num_thread)
        s.bind(ty, thread_y)
        s.bind(tx, thread_x)
        s.annotate(xi, "vectorize")
        stmt = s.to_halide()
        return stmt

    check_correctness(s, [A, B, C], _schedule_pass, 'cuda', tvm.build(*origin_schedule(), 'cuda'))

if __name__ == "__main__":
    test_gemm()
    test_gemm_schedule()

