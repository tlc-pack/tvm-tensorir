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

from tvm._ffi import register_func, register_object
from tvm.ir import IRModule
from tvm.runtime import Module, Object
from tvm.target import Target

from ..contrib.popen_pool import PopenPoolExecutor, StatusKind, MapResult
from . import _ffi_api
from .utils import cpu_count, get_global_func_with_default_on_worker


@register_object("meta_schedule.BuildInput")
class BuildInput(Object):
    """The builder's input.

    Parameters
    ----------
    mod : IRModule
        The IRModule to be built.
    target : Target
        The target to be built for.
    """

    mod: IRModule
    target: Target

    def __init__(self, mod: IRModule, target: Target) -> None:
        """Constructor.

        Parameters
        ----------
        mod : IRModule
            The IRModule to be built.
        target : Target
            The target to be built for.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.BuildInput,  # type: ignore # pylint: disable=no-member
            mod,
            target,
        )


@register_object("meta_schedule.BuildResult")
class BuildResult(Object):
    """The builder's result.

    Parameters
    ----------
    artifact_path : Optional[str]
        The path to the artifact.
    error_msg : Optional[str]
        The error message.
    """

    artifact_path: Optional[str]
    error_msg: Optional[str]

    def __init__(
        self,
        artifact_path: Optional[str],
        error_msg: Optional[str],
    ) -> None:
        """Constructor.

        Parameters
        ----------
        artifact_path : Optional[str]
            The path to the artifact.
        error_msg : Optional[str]
            The error message.
        """
        self.__init_handle_by_constructor__(
            _ffi_api.BuildResult,  # type: ignore # pylint: disable=no-member
            artifact_path,
            error_msg,
        )


@register_object("meta_schedule.Builder")
class Builder(Object):
    """The abstract builder interface."""

    def build(self, build_inputs: List[BuildInput]) -> List[BuildResult]:
        """Build the given inputs.

        Parameters
        ----------
        build_inputs : List[BuildInput]
            The inputs to be built.

        Returns
        -------
        build_results : List[BuildResult]
            The results of building the given inputs.
        """
        raise NotImplementedError


@register_object("meta_schedule.PyBuilder")
class PyBuilder(Builder):
    """An abstract builder with customized build method on the python-side."""

    def __init__(self):
        """Constructor."""

        def build_func(build_inputs: List[BuildInput]) -> List[BuildResult]:
            return self.build(build_inputs)

        self.__init_handle_by_constructor__(
            _ffi_api.PyBuilder,  # type: ignore # pylint: disable=no-member
            build_func,
        )

    def build(self, build_inputs: List[BuildInput]) -> List[BuildResult]:
        raise NotImplementedError


class LocalBuilder(PyBuilder):
    """A builder that builds the given input on local host.

    Parameters
    ----------
    pool : PopenPoolExecutor
        The process pool to run the build.
    timeout_sec : float
        The timeout in seconds for the build.
    build_func : Optional[str]
        Name of the build function to be used.
        Defaults to `meta_schedule.builder.default_build`.
        The signature is Callable[[IRModule, Target], Module].
    export_func : Optional[str]
        Name of the export function to be used.
        Defaults to `meta_schedule.builder.default_export`.
        The signature is Callable[[Module], str].

    Note
    ----
    The build function and export function should be registered in the worker process.
    The worker process is only aware of functions registered in TVM package,
    if there are extra functions to be registered,
    please send the registration logic via initializer.
    """

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
        initializer: Optional[Callable[[], None]] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        max_workers : Optional[int]
            The maximum number of worker processes to be used.
            Defaults to number of CPUs.
        timeout_sec : float
            The timeout in seconds for the build.
        build_func : Optional[str]
            Name of the build function to be used.
            Defaults to `meta_schedule.builder.default_build`.
            The signature is Callable[[IRModule, Target], Module].
        export_func : Optional[str]
            Name of the export function to be used.
            Defaults to `meta_schedule.builder.default_export`.
            The signature is Callable[[Module], str].
        initializer : Optional[Callable[[], None]]
            The initializer to be used for the worker processes.
        """
        super().__init__()

        if max_workers is None:
            max_workers = cpu_count()

        self.pool = PopenPoolExecutor(
            max_workers=max_workers,
            timeout=timeout_sec,
            initializer=initializer,
        )
        self.timeout_sec = timeout_sec
        self.build_func = build_func
        self.export_func = export_func
        self._pre_check()

    def build(self, build_inputs: List[BuildInput]) -> List[BuildResult]:
        results: List[BuildResult] = []
        map_result: MapResult

        # Dispatch the build inputs to the worker processes.
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
                results.append(BuildResult(map_result.value, None))
            elif map_result.status == StatusKind.EXCEPTION:
                results.append(
                    BuildResult(
                        None,
                        "LocalBuilder: An exception occurred\n" + str(map_result.value),
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

    def _pre_check(self) -> None:
        value = self.pool.submit(LocalBuilder._worker_pre_check, self.build_func, self.export_func)
        value.result()

    @staticmethod
    def _worker_pre_check(
        build_func: Optional[str],
        export_func: Optional[str],
    ) -> None:
        get_global_func_with_default_on_worker(name=build_func, default=None)
        get_global_func_with_default_on_worker(name=export_func, default=None)

    @staticmethod
    def _worker_func(
        build_func: Optional[str],
        export_func: Optional[str],
        mod: IRModule,
        target: Target,
    ) -> str:
        # Step 1.1. Get the build function
        f_build: Callable[[IRModule, Target], Module] = get_global_func_with_default_on_worker(
            name=build_func,
            default=default_build,
        )
        # Step 1.2. Get the export function
        f_export: Callable[[Module], str] = get_global_func_with_default_on_worker(
            name=export_func,
            default=export_tar,
        )
        # Step 2.1. Build the IRModule
        rt_mod: Module = f_build(mod, target)
        # Step 2.2. Export the Module
        artifact_path: str = f_export(rt_mod)
        return artifact_path


@register_func("meta_schedule.builder.default_build")
def default_build(mod: IRModule, target: Target) -> Module:
    """Default build function.

    Parameters
    ----------
    mod : IRModule
        The IRModule to be built.
    target : Target
        The target to be built.

    Returns
    -------
    rt_mod : Module
        The built Module.
    """
    # pylint: disable=import-outside-toplevel
    from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
    from tvm.driver import build as tvm_build

    # pylint: enable=import-outside-toplevel

    if target.kind.name == "cuda":
        set_cuda_target_arch(target.attrs["arch"])

    return tvm_build(mod, target=target)


@register_func("meta_schedule.builder.export_tar")
def export_tar(mod: Module) -> str:
    """Default export function.

    Parameters
    ----------
    mod : Module
        The Module to be exported.

    Returns
    -------
    artifact_path : str
        The path to the exported Module.
    """
    from tvm.contrib.tar import tar  # pylint: disable=import-outside-toplevel

    artifact_path = os.path.join(tempfile.mkdtemp(), "tvm_tmp_mod." + tar.output_format)
    mod.export_library(artifact_path, tar)
    return artifact_path
