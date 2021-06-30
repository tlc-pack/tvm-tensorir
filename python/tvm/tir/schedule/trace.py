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
# pylint: disable=unused-import
"""A sequence of instructions and optionally the sampling decisions"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from tvm._ffi import register_object as _register_object
from tvm.runtime import Object
from .inst import Inst
from . import _ffi_api_schedule

if TYPE_CHECKING:
    from .schedule import Schedule


@_register_object("tir.Trace")
class Trace(Object):
    """A sequence of instructions and optionally the sampling decisions"""

    insts: List[Inst]
    decisions: Dict[Inst, Any]

    def __init__(
        self,
        insts: List[Inst],
        decisions: Dict[Inst, Any],
    ) -> None:
        self.__init_handle_by_constructor__(
            _ffi_api_schedule.Trace,  # pylint: disable=no-member
            insts,
            decisions,
        )

    def append(
        self,
        inst: Inst,
        decision: Any = None,
    ) -> None:
        _ffi_api_schedule.TraceAppend(self, inst, decision)  # pylint: disable=no-member

    def pop(self) -> Optional[Inst]:
        return _ffi_api_schedule.TracePop(self)  # pylint: disable=no-member

    def apply_to_schedule(self, sch: "Schedule", remove_postproc: bool = True) -> None:
        _ffi_api_schedule.TraceApplyToSchedule(  # pylint: disable=no-member
            self, sch, remove_postproc
        )

    def as_json(self) -> Any:
        return _ffi_api_schedule.TraceAsJSON(self)  # pylint: disable=no-member

    def as_python(self) -> List[str]:
        return _ffi_api_schedule.TraceAsPython(self)  # pylint: disable=no-member

    def with_decision(self, inst: Inst, decision: Any) -> None:
        _ffi_api_schedule.TraceWithDecision(self, inst, decision)  # pylint: disable=no-member

    def simplified(self, remove_postproc: bool = True) -> "Trace":
        return _ffi_api_schedule.TraceSimplified(self, remove_postproc)  # pylint: disable=no-member

    @staticmethod
    def apply_json_to_schedule(json: Any, sch: "Schedule") -> None:
        _ffi_api_schedule.TraceApplyJSONToSchedule(json, sch)  # pylint: disable=no-member
