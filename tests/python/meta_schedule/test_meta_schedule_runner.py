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
""" Test Meta Schedule Runner """

import sys  # pylint: disable=unused-import
import time
import itertools
import numpy as np
from typing import List

import pytest
import tvm
from tvm import tir
from tvm._ffi import register_func
from tvm.meta_schedule import (
    BuilderInput,
    EvaluatorConfig,
    LocalBuilder,
    PyRunner,
    RPCConfig,
    RPCRunner,
    RunnerFuture,
    RunnerInput,
)
from tvm.meta_schedule.arg_info import TensorArgInfo, PyArgsInfo, Args, ArgInfo
from tvm.meta_schedule.testing import Server, Tracker
from tvm.meta_schedule.utils import get_global_func_with_default_on_worker
from tvm.rpc import RPCSession
from tvm.runtime import Device, Module, NDArray
from tvm.script import ty
from tvm.target import Target
from tvm.tir import FloatImm

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring,unbalanced-tuple-unpacking


@tvm.script.tir
class SmallMatmulModule:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, (2, 2), "float32")
        B = tir.match_buffer(b, (2, 2), "float32")
        C = tir.match_buffer(c, (2, 2), "float32")
        with tir.block([2, 2, tir.reduce_axis(0, 2)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
class MatmulModule:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, (1024, 1024), "float32")
        B = tir.match_buffer(b, (1024, 1024), "float32")
        C = tir.match_buffer(c, (1024, 1024), "float32")
        with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.tir
class MatmulReluModule:
    def main(a: ty.handle, b: ty.handle, d: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, (1024, 1024), "float32")
        B = tir.match_buffer(b, (1024, 1024), "float32")
        D = tir.match_buffer(d, (1024, 1024), "float32")
        C = tir.alloc_buffer((1024, 1024), "float32")
        with tir.block([1024, 1024, tir.reduce_axis(0, 1024)], "matmul") as [vi, vj, vk]:
            with tir.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        with tir.block([1024, 1024], "relu") as [vi, vj]:
            D[vi, vj] = tir.max(C[vi, vj], 0.0)


@tvm.script.tir
class BatchMatmulModule:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, [16, 128, 128])
        B = tir.match_buffer(b, [16, 128, 128])
        C = tir.match_buffer(c, [16, 128, 128])
        with tir.block([16, 128, 128, tir.reduce_axis(0, 128)], "update") as [vn, vi, vj, vk]:
            with tir.init():
                C[vn, vi, vj] = 0.0
            C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


@tvm.script.tir
class AddModule:
    def main(a: ty.handle, b: ty.handle, c: ty.handle) -> None:  # pylint: disable=no-self-argument
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = tir.match_buffer(a, [128], "float32")
        B = tir.match_buffer(b, [128], "float32")
        C = tir.match_buffer(c, [128], "float32")
        with tir.block([128], "add") as [vi]:
            C[vi] = A[vi] + B[vi]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


def _clean_build(artifact_path: str) -> None:
    f_clean_build = get_global_func_with_default_on_worker("meta_schedule.remove_build_dir", None)
    if f_clean_build is not None:
        f_clean_build(artifact_path)
    else:
        raise RuntimeError("Unable to find remove_build_dir function.")


def test_meta_schedule_single_run():
    """Test meta schedule runner for a single run"""
    # Build the module
    mod = MatmulModule()
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
        ],
    )

    with Tracker(silent=True) as tracker:
        with Server(tracker, silent=True) as server:
            rpc_config = RPCConfig(
                tracker_host=tracker.host,
                tracker_port=tracker.port,
                tracker_key=server.key,
                session_priority=1,
                session_timeout_sec=100,
            )
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )
            runner = RPCRunner(rpc_config, evaluator_config)
            # Run the module
            (runner_future,) = runner.run([runner_input])
            runner_result = runner_future.result()
    assert runner_result.error_msg is None
    for result in runner_result.run_sec:
        if isinstance(result, FloatImm):
            result = result.value
        assert isinstance(result, float)
        assert result >= 0.0
    _clean_build(builder_result.artifact_path)


def test_meta_schedule_multiple_runs():
    """Test meta schedule runner for multiple runs"""
    # Build the module
    mods = [
        MatmulModule(),
        MatmulReluModule(),
        BatchMatmulModule(),
    ]
    builder = LocalBuilder()
    builder_inputs = [BuilderInput(mod, Target("llvm")) for mod in mods]
    builder_results = builder.build(builder_inputs)
    for builder_result in builder_results:
        assert builder_result.artifact_path is not None
        assert builder_result.error_msg is None

    args_infos = [
        [
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
        ],
        [
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
        ],
        [
            TensorArgInfo("float32", [16, 128, 128]),
            TensorArgInfo("float32", [16, 128, 128]),
            TensorArgInfo("float32", [16, 128, 128]),
        ],
    ]

    runner_inputs = [
        RunnerInput(builder_results[i].artifact_path, "llvm", args_infos[i])
        for i in range(len(mods))
    ]

    with Tracker(silent=True) as tracker:
        with Server(tracker, silent=True) as server:
            rpc_config = RPCConfig(
                tracker_host=tracker.host,
                tracker_port=tracker.port,
                tracker_key=server.key,
                session_priority=1,
                session_timeout_sec=100,
            )
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )

            runner = RPCRunner(rpc_config, evaluator_config)

            # Run the module
            runner_futures = runner.run(runner_inputs)
            runner_results = [runner_future.result() for runner_future in runner_futures]

    for runner_result in runner_results:
        assert runner_result.error_msg is None
        for result in runner_result.run_sec:
            if isinstance(result, FloatImm):
                result = result.value
            assert isinstance(result, float)
            assert result >= 0.0

    for builder_result in builder_results:
        _clean_build(builder_result.artifact_path)


def test_meta_schedule_py_runner():
    """Test meta schedule PyRunner"""

    class TestRunner(PyRunner):
        def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
            raise ValueError("TestRunner")

    runner = TestRunner()
    with pytest.raises(ValueError, match="TestRunner"):
        runner.run([])


def test_meta_schedule_rpc_runner_time_out():
    """Test meta schedule RPC Runner time out"""

    def initializer():
        @register_func("meta_schedule.runner.test_time_out")
        def timeout_session_creator(  # pylint: disable=unused-variable
            rpc_config: RPCConfig,  # pylint: disable=unused-argument
        ) -> RPCSession:
            time.sleep(2)

    runner_input = RunnerInput(
        "test",
        "llvm",
        [
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
        ],
    )

    with Tracker(silent=True) as tracker:
        with Server(tracker, silent=True) as server:
            rpc_config = RPCConfig(
                tracker_host=tracker.host,
                tracker_port=tracker.port,
                tracker_key=server.key,
                session_priority=1,
                session_timeout_sec=1,
            )
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )

            runner = RPCRunner(
                rpc_config,
                evaluator_config,
                initializer=initializer,
                f_create_session="meta_schedule.runner.test_time_out",
            )

            # Run the module
            (runner_future,) = runner.run([runner_input])
            runner_result = runner_future.result()

    assert runner_result.error_msg is not None and runner_result.error_msg.startswith(
        "RPCRunner: Timeout, killed after"
    )
    assert runner_result.run_sec is None


def test_meta_schedule_rpc_runner_exception():
    """Test meta schedule RPC Runner exception"""

    def initializer():
        @register_func("meta_schedule.runner.test_exception")
        def exception_session_creator(  # pylint: disable=unused-variable
            rpc_config: RPCConfig,  # pylint: disable=unused-argument
        ) -> RPCSession:
            raise Exception("Test")

    runner_input = RunnerInput(
        "test",
        "llvm",
        [
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
        ],
    )

    with Tracker(silent=True) as tracker:
        with Server(tracker, silent=True) as server:
            rpc_config = RPCConfig(
                tracker_host=tracker.host,
                tracker_port=tracker.port,
                tracker_key=server.key,
                session_priority=1,
                session_timeout_sec=100,
            )
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )

            runner = RPCRunner(
                rpc_config,
                evaluator_config,
                initializer=initializer,
                f_create_session="meta_schedule.runner.test_exception",
            )
            (runner_future,) = runner.run([runner_input])
            runner_result = runner_future.result()

    assert runner_result.error_msg is not None and runner_result.error_msg.startswith(
        "RPCRunner: An exception occurred\n"
    )
    assert runner_result.run_sec is None


def test_meta_schedule_runner_add_test():
    """Test meta schedule runner with add module"""

    repeated_args_local = []
    repeated_args_remote = []

    def test_alloc_argument(
        session: RPCSession,
        device: Device,
        alloc_repeat: int,
        args_info: PyArgsInfo,
    ) -> List[Args]:
        try:
            f_random_fill = session.get_function("tvm.contrib.random.random_fill")
        except AttributeError as error:
            raise AttributeError(
                'Unable to find function "tvm.contrib.random.random_fill" on remote RPC server. '
                "Please make sure USE_RANDOM is turned ON in the config.cmake on the RPC server."
            ) from error
        repeated_args: List[Args] = []
        for _ in range(alloc_repeat):
            args: Args = []
            for arg_info in args_info:
                arg = ArgInfo.alloc(arg_info, device)
                if isinstance(arg, NDArray):
                    f_random_fill(arg)
                args.append(arg)
            repeated_args_local.append([arg.asnumpy() for arg in args])
            repeated_args.append(args)
        return repeated_args

    def _check_correct_add(args_local: List[np.array], args_remote: List[np.array]) -> None:
        assert (args_local[0] == args_remote[0]).all()
        assert (args_local[1] == args_remote[1]).all()
        assert (args_local[0] + args_local[1] == args_remote[2]).all()

    def test_run_evaluator(
        session: RPCSession,  # pylint: disable=unused-argument
        rt_mod: Module,
        device: Device,
        evaluator_config: EvaluatorConfig,
        repeated_args: List[Args],
    ) -> List[float]:
        evaluator = rt_mod.time_evaluator(
            func_name=rt_mod.entry_name,
            dev=device,
            number=evaluator_config.number,
            repeat=evaluator_config.repeat,
            min_repeat_ms=evaluator_config.min_repeat_ms,
            f_preproc="cache_flush_cpu_non_first_arg"
            if evaluator_config.enable_cpu_cache_flush
            else "",
        )
        repeated_costs: List[List[float]] = []
        for args in repeated_args:
            device.sync()
            profile_result = evaluator(*args)
            repeated_costs.append(profile_result.results)
            repeated_args_remote.append([arg.asnumpy() for arg in args])
        for args_local, args_remote in zip(repeated_args_local, repeated_args_remote):
            _check_correct_add(args_local, args_remote)
        costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
        return costs

    # Build the module
    mod = AddModule()
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorArgInfo("float32", [128]),
            TensorArgInfo("float32", [128]),
            TensorArgInfo("float32", [128]),
        ],
    )

    with Tracker(silent=True) as tracker:
        with Server(tracker, silent=True) as server:
            rpc_config = RPCConfig(
                tracker_host=tracker.host,
                tracker_port=tracker.port,
                tracker_key=server.key,
                session_priority=1,
                session_timeout_sec=100,
            )
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )
            runner = RPCRunner(
                rpc_config,
                evaluator_config,
                f_alloc_argument=test_alloc_argument,
                f_run_evaluator=test_run_evaluator,
            )
            # Run the module
            (runner_future,) = runner.run([runner_input])
            runner_result = runner_future.result()
    assert runner_result.error_msg is None
    for result in runner_result.run_sec:
        if isinstance(result, FloatImm):
            result = result.value
        assert isinstance(result, float)
        assert result >= 0.0
    _clean_build(builder_result.artifact_path)


def test_meta_schedule_runner_matmul_test():
    """Test meta schedule runner with add module"""

    repeated_args_local = []
    repeated_args_remote = []

    def test_alloc_argument(
        session: RPCSession,
        device: Device,
        alloc_repeat: int,
        args_info: PyArgsInfo,
    ) -> List[Args]:
        try:
            f_random_fill = session.get_function("tvm.contrib.random.random_fill")
        except AttributeError as error:
            raise AttributeError(
                'Unable to find function "tvm.contrib.random.random_fill" on remote RPC server. '
                "Please make sure USE_RANDOM is turned ON in the config.cmake on the RPC server."
            ) from error
        repeated_args: List[Args] = []
        for _ in range(alloc_repeat):
            args: Args = []
            for arg_info in args_info:
                arg = ArgInfo.alloc(arg_info, device)
                if isinstance(arg, NDArray):
                    f_random_fill(arg)
                args.append(arg)
            repeated_args_local.append([arg.asnumpy() for arg in args])
            repeated_args.append(args)
        return repeated_args

    def _check_correct_matmul(args_local: List[np.array], args_remote: List[np.array]) -> None:
        assert (args_local[0] == args_remote[0]).all()
        assert (args_local[1] == args_remote[1]).all()
        assert np.isclose(np.matmul(args_local[0], args_local[1]), args_remote[2], rtol=1e-5).all()

    def test_run_evaluator(
        session: RPCSession,  # pylint: disable=unused-argument
        rt_mod: Module,
        device: Device,
        evaluator_config: EvaluatorConfig,
        repeated_args: List[Args],
    ) -> List[float]:
        evaluator = rt_mod.time_evaluator(
            func_name=rt_mod.entry_name,
            dev=device,
            number=evaluator_config.number,
            repeat=evaluator_config.repeat,
            min_repeat_ms=evaluator_config.min_repeat_ms,
            f_preproc="cache_flush_cpu_non_first_arg"
            if evaluator_config.enable_cpu_cache_flush
            else "",
        )
        repeated_costs: List[List[float]] = []
        for args in repeated_args:
            device.sync()
            profile_result = evaluator(*args)
            repeated_costs.append(profile_result.results)
            repeated_args_remote.append([arg.asnumpy() for arg in args])
        for args_local, args_remote in zip(repeated_args_local, repeated_args_remote):
            _check_correct_matmul(args_local, args_remote)
        costs = [float(cost) for cost in itertools.chain.from_iterable(repeated_costs)]
        return costs

    # Build the module
    mod = MatmulModule()
    builder = LocalBuilder()
    (builder_result,) = builder.build([BuilderInput(mod, Target("llvm"))])
    assert builder_result.artifact_path is not None
    assert builder_result.error_msg is None

    runner_input = RunnerInput(
        builder_result.artifact_path,
        "llvm",
        [
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
            TensorArgInfo("float32", (1024, 1024)),
        ],
    )

    with Tracker(silent=True) as tracker:
        with Server(tracker, silent=True) as server:
            rpc_config = RPCConfig(
                tracker_host=tracker.host,
                tracker_port=tracker.port,
                tracker_key=server.key,
                session_priority=1,
                session_timeout_sec=100,
            )
            evaluator_config = EvaluatorConfig(
                number=1,
                repeat=1,
                min_repeat_ms=0,
                enable_cpu_cache_flush=False,
            )
            runner = RPCRunner(
                rpc_config,
                evaluator_config,
                f_alloc_argument=test_alloc_argument,
                f_run_evaluator=test_run_evaluator,
            )
            # Run the module
            (runner_future,) = runner.run([runner_input])
            runner_result = runner_future.result()
    assert runner_result.error_msg is None
    for result in runner_result.run_sec:
        if isinstance(result, FloatImm):
            result = result.value
        assert isinstance(result, float)
        assert result >= 0.0
    _clean_build(builder_result.artifact_path)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
