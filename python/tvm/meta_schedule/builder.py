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
"""Builders"""
import os
import tempfile
from typing import Callable, List, Optional

from tvm._ffi import register_func, register_object, get_global_func
from tvm.ir import IRModule
from tvm.runtime import Module, Object
from tvm.target import Target

from ..contrib.popen_pool import PopenPoolExecutor, StatusKind, MapResult
from . import _ffi_api
from .utils import cpu_count


@register_object("meta_schedule.BuildInput")
class BuildInput(Object):

    mod: IRModule
    target: Target

    def __init__(self, mod: IRModule, target: Target) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BuildInput,  # type: ignore # pylint: disable=no-member
            mod,
            target,
        )


@register_object("meta_schedule.BuildResult")
class BuildResult(Object):

    artifact_path: Optional[str]
    error_msg: Optional[str]

    def __init__(
        self,
        artifact_path: Optional[str],
        error_msg: Optional[str],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BuildResult,  # type: ignore # pylint: disable=no-member
            artifact_path,
            error_msg,
        )


@register_object("meta_schedule.Builder")
class Builder(Object):
    def build(self, build_inputs: List[BuildInput]) -> List[BuildResult]:
        raise NotImplementedError


@register_object("meta_schedule.PyBuilder")
class PyBuilder(Builder):
    def __init__(self):
        def build_func(build_inputs: List[BuildInput]) -> List[BuildResult]:
            return self.build(build_inputs)

        self.__init_handle_by_constructor__(
            _ffi_api.PyBuilder,  # type: ignore # pylint: disable=no-member
            build_func,
        )

    def build(self, build_inputs: List[BuildInput]) -> List[BuildResult]:
        raise NotImplementedError


class LocalBuilder(PyBuilder):

    pool: PopenPoolExecutor
    timeout_sec: float
    build_func: Optional[str]
    export_func: Optional[str]

    def __init__(
        self,
        *,
        max_workers: Optional[int] = None,
        timeout_sec: float = 30.0,
        build_func: str = None,
        export_func: str = None,
    ) -> None:
        super().__init__()

        if max_workers is None:
            max_workers = cpu_count()

        self.pool = PopenPoolExecutor(
            max_workers=max_workers,
            timeout=timeout_sec,
        )
        self.timeout_sec = timeout_sec
        self.build_func = build_func
        self.export_func = export_func

    def build(self, build_inputs: List[BuildInput]) -> List[BuildResult]:
        results: List[BuildResult] = []
        map_result: MapResult

        for map_result in self.pool.map_with_error_catching(
            lambda x: LocalBuilder._worker_func(*x),
            [
                (
                    self.build_func,
                    self.export_func,
                    build_input.mod,
                    build_input.target,
                )
                for build_input in build_inputs
            ],
        ):
            if map_result.status == StatusKind.COMPLETE:
                results.append(map_result.value)
            elif map_result.status == StatusKind.EXCEPTION:
                results.append(
                    BuildResult(
                        None,
                        "LocalBuilder: An exception occurred\n" + repr(map_result.value),
                    )
                )
            elif map_result.status == StatusKind.TIMEOUT:
                results.append(
                    BuildResult(
                        None,
                        f"LocalBuilder: Timeout, killed after {self.timeout_sec} seconds",
                    )
                )
            else:
                raise NotImplementedError(map_result.status)
        return results

    @staticmethod
    def _worker_func(
        build_func: Optional[str],
        export_func: Optional[str],
        mod: IRModule,
        target: Target,
    ) -> BuildResult:
        # Step 1.1. Get the build function
        f_build: Callable[[IRModule, Target], Module]
        if build_func is None:
            f_build = default_build
        else:
            f_build = get_global_func(build_func)

        # Step 1.2. Get the export function
        f_export: Callable[[Module], str]
        if export_func is None:
            f_export = export_tar
        else:
            f_export = get_global_func(export_func)

        # Step 2.1. Build the IRModule
        try:
            rt_mod: Module = f_build(mod, target)
        except Exception as err:  # pylint: disable=broad-except
            return BuildResult(None, "LocalBuilder: Error building the IRModule\n" + repr(err))

        # Step 2.2. Export the Module
        try:
            artifact_path: str = f_export(rt_mod)
        except Exception as err:  # pylint: disable=broad-except
            return BuildResult(None, "LocalBuilder: Error exporting the Module\n" + repr(err))
        return BuildResult(artifact_path, None)


@register_func("meta_schedule.builder.default_build")
def default_build(mod: IRModule, target: Target) -> Module:
    # pylint: disable=import-outside-toplevel
    from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
    from tvm.driver import build as tvm_build

    # pylint: enable=import-outside-toplevel

    if target.kind.name == "cuda":
        set_cuda_target_arch(target.attrs["arch"])

    return tvm_build(mod, target=target)


@register_func("meta_schedule.builder.export_tar")
def export_tar(mod: Module) -> str:
    from tvm.contrib.tar import tar  # pylint: disable=import-outside-toplevel

    artifact_path = os.path.join(tempfile.mkdtemp(), "tvm_tmp_mod." + tar.output_format)
    mod.export_library(artifact_path, tar)
    return artifact_path
