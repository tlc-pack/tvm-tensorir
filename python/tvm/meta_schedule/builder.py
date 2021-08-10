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
from typing import List, Optional

from tvm._ffi import register_object as _register_object
from tvm.ir import IRModule
from tvm.runtime import Object
from tvm.target import Target

from . import _ffi_api


@_register_object("meta_schedule.BuildInput")
class BuildInput(Object):

    mod: IRModule
    target: Target

    def __init__(self, mod: IRModule, target: Target) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BuildInput,  # type: ignore # pylint: disable=no-member
            mod,
            target,
        )


@_register_object("meta_schedule.BuildResult")
class BuildResult(Object):

    artifact_path: Optional[str]
    error_msg: Optional[str]
    build_secs: float

    def __init__(
        self,
        artifact_path: Optional[str],
        error_msg: Optional[str],
        build_secs: float,
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.BuildResult,  # type: ignore # pylint: disable=no-member
            artifact_path,
            error_msg,
            build_secs,
        )


@_register_object("meta_schedule.Builder")
class Builder(Object):
    def build(self, build_inputs: List[BuildInput]) -> List[BuildResult]:
        raise NotImplementedError


@_register_object("meta_schedule.PyBuilder")
class PyBuilder(Builder):
    def __init__(self):
        def f_build(build_inputs: List[BuildInput]) -> List[BuildResult]:
            return self.build(build_inputs)

        self.__init_handle_by_constructor__(
            _ffi_api.PyBuilder,  # type: ignore # pylint: disable=no-member
            f_build,
        )

    def build(self, build_inputs: List[BuildInput]) -> List[BuildResult]:
        raise NotImplementedError
