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
"""Trace class of the program execution"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api
from .instruction import Instruction

if TYPE_CHECKING:
    from .schedule import Schedule


@register_object("meta_schedule.Trace")
class Trace(Object):
    """The trace of program execution."""

    insts: List[Instruction]
    decisions: Dict[Instruction, Object]

    def __init__(
        self,
        insts: Optional[List[Instruction]] = None,
        decisions: Optional[Dict[Instruction, Object]] = None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Trace, insts, decisions  # pylint: disable=no-member
        )

    def append(
        self,
        inst: Instruction,
        decision: Optional[Object] = None,
    ) -> None:
        _ffi_api.TraceAppend(self, inst, decision)  # pylint: disable=no-member

    def pop(self) -> Optional[Instruction]:
        return _ffi_api.TracePop(self)  # pylint: disable=no-member

    def apply(self, sch: "Schedule") -> None:
        _ffi_api.TraceApply(self, sch)  # pylint: disable=no-member

    def serialize(self) -> Any:
        return _ffi_api.TraceSerialize(self)  # pylint: disable=no-member

    @staticmethod
    def deserialize(json: Any, sch: "Schedule") -> None:
        _ffi_api.TraceDeserialize(json, sch)  # pylint: disable=no-member

    def as_python(self) -> List[str]:
        return _ffi_api.TraceAsPython(self)  # pylint: disable=no-member