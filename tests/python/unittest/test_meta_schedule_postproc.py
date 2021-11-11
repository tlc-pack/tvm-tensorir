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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import math
import re

import tvm
from tvm.script import tir as T

from tvm.meta_schedule.postproc import PyPostproc, VerifyGPUCode, DisallowDynamicLoop
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.utils import _get_hex_address
from tvm.target.target import Target

from tvm.tir.schedule import Schedule

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class Conv_cuda0:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        blockIdx_y = T.env_thread("blockIdx.y")
        blockIdx_z = T.env_thread("blockIdx.z")
        A = T.match_buffer(a, [14, 14, 256, 256], dtype="float32")
        B = T.match_buffer(b, [14, 14, 512, 256], dtype="float32")
        # body
        T.launch_thread(blockIdx_z, 196)
        B_local = T.allocate([64], "float32", "local")
        Apad_shared = T.allocate([512], "float32", "shared")
        Apad_shared_local = T.allocate([8], "float32", "local")
        T.launch_thread(blockIdx_y, 8)
        T.launch_thread(blockIdx_x, 4)
        T.launch_thread(threadIdx_y, 8)
        T.launch_thread(threadIdx_x, 8)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            T.store(B_local, ff_c_init * 8 + nn_c_init, T.float32(0), True)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                T.store(Apad_shared, T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4), T.if_then_else(1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15, T.load("float32x4", A.data, T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4), T.broadcast(True, 4)), T.broadcast(T.float32(0), 4), dtype="float32x4"), T.broadcast(True, 4))
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    T.store(Apad_shared_local, ax3, T.load("float32", Apad_shared, rc_inner * 64 + threadIdx_x * 8 + ax3), True)
                for ff_c, nn_c in T.grid(8, 8):
                    T.store(B_local, ff_c * 8 + nn_c, T.load("float32", B_local, ff_c * 8 + nn_c) + T.load("float32", Apad_shared_local, nn_c), True)
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            T.store(B.data, blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner, T.load("float32", B_local, ff_inner_inner_inner * 8 + nn_inner_inner_inner), True)# fmt: on


@tvm.script.ir_module
class Conv_cuda1:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        blockIdx_y = T.env_thread("blockIdx.y")
        blockIdx_z = T.env_thread("blockIdx.z")
        A = T.match_buffer(a, [14, 14, 256, 256], dtype="float32")
        B = T.match_buffer(b, [14, 14, 512, 256], dtype="float32")
        # body
        T.launch_thread(blockIdx_z, 196)
        B_local = T.allocate([6400000], "float32", "local")
        Apad_shared = T.allocate([512], "float32", "shared")
        Apad_shared_local = T.allocate([8], "float32", "local")
        T.launch_thread(blockIdx_y, 8)
        T.launch_thread(blockIdx_x, 4)
        T.launch_thread(threadIdx_y, 8)
        T.launch_thread(threadIdx_x, 8)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            T.store(B_local, ff_c_init * 8 + nn_c_init, T.float32(0), True)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                T.store(Apad_shared, T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4), T.if_then_else(1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15, T.load("float32x4", A.data, T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4), T.broadcast(True, 4)), T.broadcast(T.float32(0), 4), dtype="float32x4"), T.broadcast(True, 4))
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    T.store(Apad_shared_local, ax3, T.load("float32", Apad_shared, rc_inner * 64 + threadIdx_x * 8 + ax3), True)
                for ff_c, nn_c in T.grid(8, 8):
                    T.store(B_local, ff_c * 8 + nn_c, T.load("float32", B_local, ff_c * 8 + nn_c) + T.load("float32", Apad_shared_local, nn_c), True)
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            T.store(B.data, blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner, T.load("float32", B_local, ff_inner_inner_inner * 8 + nn_inner_inner_inner), True)# fmt: on


@tvm.script.ir_module
class Conv_cuda2:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        blockIdx_y = T.env_thread("blockIdx.y")
        blockIdx_z = T.env_thread("blockIdx.z")
        A = T.match_buffer(a, [14, 14, 256, 256], dtype="float32")
        B = T.match_buffer(b, [14, 14, 512, 256], dtype="float32")
        # body
        T.launch_thread(blockIdx_z, 196)
        B_local = T.allocate([64], "float32", "local")
        Apad_shared = T.allocate([512000], "float32", "shared")
        Apad_shared_local = T.allocate([8], "float32", "local")
        T.launch_thread(blockIdx_y, 8)
        T.launch_thread(blockIdx_x, 4)
        T.launch_thread(threadIdx_y, 8)
        T.launch_thread(threadIdx_x, 8)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            T.store(B_local, ff_c_init * 8 + nn_c_init, T.float32(0), True)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                T.store(Apad_shared, T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4), T.if_then_else(1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15, T.load("float32x4", A.data, T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4), T.broadcast(True, 4)), T.broadcast(T.float32(0), 4), dtype="float32x4"), T.broadcast(True, 4))
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    T.store(Apad_shared_local, ax3, T.load("float32", Apad_shared, rc_inner * 64 + threadIdx_x * 8 + ax3), True)
                for ff_c, nn_c in T.grid(8, 8):
                    T.store(B_local, ff_c * 8 + nn_c, T.load("float32", B_local, ff_c * 8 + nn_c) + T.load("float32", Apad_shared_local, nn_c), True)
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            T.store(B.data, blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner, T.load("float32", B_local, ff_inner_inner_inner * 8 + nn_inner_inner_inner), True)# fmt: on


@tvm.script.ir_module
class Conv_cuda3:
    @T.prim_func
    def main(a: T.handle, b: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "T.noalias": True})
        # var definition
        threadIdx_x = T.env_thread("threadIdx.x")
        threadIdx_y = T.env_thread("threadIdx.y")
        blockIdx_x = T.env_thread("blockIdx.x")
        blockIdx_y = T.env_thread("blockIdx.y")
        blockIdx_z = T.env_thread("blockIdx.z")
        A = T.match_buffer(a, [14, 14, 256, 256], dtype="float32")
        B = T.match_buffer(b, [14, 14, 512, 256], dtype="float32")
        # body
        T.launch_thread(blockIdx_z, 196)
        B_local = T.allocate([64], "float32", "local")
        Apad_shared = T.allocate([512], "float32", "shared")
        Apad_shared_local = T.allocate([8], "float32", "local")
        T.launch_thread(blockIdx_y, 8)
        T.launch_thread(blockIdx_x, 4)
        T.launch_thread(threadIdx_y, 8)
        T.launch_thread(threadIdx_x, 800000)
        for ff_c_init, nn_c_init in T.grid(8, 8):
            T.store(B_local, ff_c_init * 8 + nn_c_init, T.float32(0), True)
        for rc_outer, ry, rx in T.grid(32, 3, 3):
            for ax3_inner_outer in T.serial(0, 2):
                T.store(Apad_shared, T.ramp(threadIdx_y * 64 + threadIdx_x * 8 + ax3_inner_outer * 4, 1, 4), T.if_then_else(1 <= blockIdx_z // 14 + ry and blockIdx_z // 14 + ry < 15 and 1 <= rx + blockIdx_z % 14 and rx + blockIdx_z % 14 < 15, T.load("float32x4", A.data, T.ramp(ry * 917504 + blockIdx_z * 65536 + rx * 65536 + rc_outer * 2048 + threadIdx_y * 256 + blockIdx_x * 64 + threadIdx_x * 8 + ax3_inner_outer * 4 - 983040, 1, 4), T.broadcast(True, 4)), T.broadcast(T.float32(0), 4), dtype="float32x4"), T.broadcast(True, 4))
            for rc_inner in T.serial(0, 8):
                for ax3 in T.serial(0, 8):
                    T.store(Apad_shared_local, ax3, T.load("float32", Apad_shared, rc_inner * 64 + threadIdx_x * 8 + ax3), True)
                for ff_c, nn_c in T.grid(8, 8):
                    T.store(B_local, ff_c * 8 + nn_c, T.load("float32", B_local, ff_c * 8 + nn_c) + T.load("float32", Apad_shared_local, nn_c), True)
        for ff_inner_inner_inner, nn_inner_inner_inner in T.grid(8, 8):
            T.store(B.data, blockIdx_z * 131072 + blockIdx_y * 16384 + threadIdx_y * 2048 + ff_inner_inner_inner * 256 + blockIdx_x * 64 + threadIdx_x * 8 + nn_inner_inner_inner, T.load("float32", B_local, ff_inner_inner_inner * 8 + nn_inner_inner_inner), True)# fmt: on


@tvm.script.ir_module
class DynamicLoop:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j in T.grid(1024, 1024):
            for k in T.serial(0, i):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _check_correct(schedule: Schedule):
    trace = schedule.trace
    for inst in trace.decisions:
        assert math.prod(trace.decisions[inst]) == 1024


def schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def test_meta_schedule_postproc():
    class FancyPostproc(PyPostproc):
        def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
            pass

        def apply(self, sch: Schedule) -> bool:
            schedule_matmul(sch)
            return True

    postproc = FancyPostproc()
    mod = Matmul
    sch = Schedule(mod)
    assert postproc.apply(sch)
    try:
        tvm.ir.assert_structural_equal(sch.mod, mod)
        raise Exception("The postprocessors did not change the schedule.")
    except (ValueError):
        _check_correct(sch)


def test_meta_schedule_postproc_fail():
    class FailingPostproc(PyPostproc):
        def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
            pass

        def apply(self, sch: Schedule) -> bool:
            return False

    postproc = FailingPostproc()
    sch = Schedule(Matmul)
    assert not postproc.apply(sch)


def test_meta_schedule_postproc_as_string():
    class NotSoFancyPostproc(PyPostproc):
        def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
            pass

        def apply(self, sch: Schedule) -> bool:
            pass

        def __str__(self) -> str:
            return f"NotSoFancyPostproc({_get_hex_address(self.handle)})"

    postproc = NotSoFancyPostproc()
    pattern = re.compile(r"NotSoFancyPostproc\(0x[a-f|0-9]*\)")
    assert pattern.match(str(postproc))


def test_meta_schedule_postproc_verify_gpu_code0():
    postproc = VerifyGPUCode()
    sch = Schedule(Conv_cuda0)
    postproc.initialize_with_tune_context(TuneContext(target=Target("nvidia/geforce-rtx-3080")))
    assert postproc.apply(sch)


def test_meta_schedule_postproc_verify_gpu_code1():
    # local mem exceeded
    postproc = VerifyGPUCode()
    sch = Schedule(Conv_cuda1)
    postproc.initialize_with_tune_context(TuneContext(target=Target("nvidia/geforce-rtx-3080")))
    assert not postproc.apply(sch)


def test_meta_schedule_postproc_verify_gpu_code2():
    # shared mem exceeded
    postproc = VerifyGPUCode()
    sch = Schedule(Conv_cuda2)
    postproc.initialize_with_tune_context(TuneContext(target=Target("nvidia/geforce-rtx-3080")))
    assert not postproc.apply(sch)


def test_meta_schedule_postproc_verify_gpu_code3():
    # number of threads exceeded
    postproc = VerifyGPUCode()
    sch = Schedule(Conv_cuda3)
    postproc.initialize_with_tune_context(TuneContext(target=Target("nvidia/geforce-rtx-3080")))
    assert not postproc.apply(sch)


def test_meta_schedule_postproc_disallow_dynamic_loops():
    postproc = DisallowDynamicLoop()
    sch = Schedule(Matmul)
    assert postproc.apply(sch)


def test_meta_schedule_postproc_disallow_dynamic_loops_fail():
    postproc = DisallowDynamicLoop()
    sch = Schedule(DynamicLoop)
    assert not postproc.apply(sch)


if __name__ == "__main__":
    test_meta_schedule_postproc()
    test_meta_schedule_postproc_fail()
    test_meta_schedule_postproc_as_string()
    test_meta_schedule_postproc_verify_gpu_code0()
    test_meta_schedule_postproc_verify_gpu_code1()
    test_meta_schedule_postproc_verify_gpu_code2()
    test_meta_schedule_postproc_verify_gpu_code3()
    test_meta_schedule_postproc_disallow_dynamic_loops()
    test_meta_schedule_postproc_disallow_dynamic_loops_fail()
