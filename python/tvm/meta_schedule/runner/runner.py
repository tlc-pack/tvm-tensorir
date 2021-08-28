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
"""Runners"""
from typing import List, NamedTuple, Optional

from tvm._ffi import register_object
from tvm.runtime import Object

from .. import _ffi_api
from ..arg_info import ArgInfo


@register_object("meta_schedule.RunnerInput")
class RunnerInput(Object):

    artifact_path: str
    device_type: str
    args_info: List[ArgInfo]

    def __init__(
        self,
        artifact_path: str,
        device_type: str,
        args_info: List[ArgInfo],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.RunnerInput,  # type: ignore # pylint: disable=no-member
            artifact_path,
            device_type,
            args_info,
        )


@register_object("meta_schedule.RunnerResult")
class RunnerResult(Object):

    run_sec: Optional[List[float]]
    error_msg: Optional[str]

    def __init__(
        self,
        run_sec: Optional[List[float]],
        error_msg: Optional[str],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.RunnerResult,  # type: ignore # pylint: disable=no-member
            run_sec,
            error_msg,
        )


@register_object("meta_schedule.RunnerFuture")
class RunnerFuture(Object):
    def __init__(self) -> None:
        def f_done():
            return self.done()

        def f_result():
            return self.result()

        self.__init_handle_by_constructor__(
            _ffi_api.RunnerFuture,  # type: ignore # pylint: disable=no-member
            f_done,
            f_result,
        )

    def done(self) -> bool:
        raise NotImplementedError

    def result(self) -> RunnerResult:
        raise NotImplementedError


@register_object("meta_schedule.Runner")
class Runner(Object):
    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        return _ffi_api.RunnerRun(runner_inputs)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.PyRunner")
class PyRunner(Runner):
    def __init__(self) -> None:
        def f_run(runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
            return self.run(runner_inputs)

        self.__init_handle_by_constructor__(
            _ffi_api.PyRunner,  # type: ignore # pylint: disable=no-member
            f_run,
        )

    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        raise NotImplementedError


class EvaluatorConfig(NamedTuple):
    number: int = 3
    repeat: int = 1
    min_repeat_ms: int = 40
    enable_cpu_cache_flush: bool = False

    @staticmethod
    def _parse(config: Optional["EvaluatorConfig"]) -> "EvaluatorConfig":
        if config is None:
            return EvaluatorConfig()
        return config
