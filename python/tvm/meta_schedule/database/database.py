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
"""Tuning record database"""
from typing import Any, List, TYPE_CHECKING

from tvm._ffi import register_object
from tvm.ir.module import IRModule
from tvm.runtime import Object
from tvm.target import Target
from tvm.tir.schedule import Trace

from .. import _ffi_api
from ..arg_info import ArgInfo
from ..workload_registry import WorkloadRegistry, WorkloadToken

if TYPE_CHECKING:
    from ..tune_context import TuneContext


@register_object("meta_schedule.TuningRecord")
class TuningRecord(Object):

    trace: Trace
    run_secs: List[float]
    workload: WorkloadToken
    target: Target
    args_info: List[ArgInfo]

    def __init__(
        self,
        trace: Trace,
        run_secs: List[float],
        workload: WorkloadToken,
        target: Target,
        args_info: List[ArgInfo],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api.TuningRecord,  # type: ignore # pylint: disable=no-member
            trace,
            run_secs,
            workload,
            target,
            args_info,
        )

    def as_json(self) -> Any:
        return _ffi_api.TuningRecordAsJSON(self)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_json(json_obj: Any, reg: WorkloadRegistry) -> "TuningRecord":
        return _ffi_api.TuningRecordFromJSON(json_obj, reg)  # type: ignore # pylint: disable=no-member


@register_object("meta_schedule.Database")
class Database(Object):
    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        _ffi_api.DatabaseInitializeWithTuneContext(self, tune_context)  # type: ignore # pylint: disable=no-member

    def add(self, record: TuningRecord) -> None:
        _ffi_api.DatabaseAdd(self, record)  # type: ignore # pylint: disable=no-member

    def get_top_k(self, workload: WorkloadToken, top_k: int) -> List[TuningRecord]:
        return _ffi_api.DatabaseGetTopK(self, workload, top_k)  # type: ignore # pylint: disable=no-member

    def lookup_or_add(self, mod: IRModule) -> WorkloadToken:
        return _ffi_api.DatabaseLookupOrAdd(self, mod)  # type: ignore # pylint: disable=no-member

    def __len__(self) -> int:
        return _ffi_api.DatabaseSize(self)  # type: ignore # pylint: disable=no-member
