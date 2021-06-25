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
"""Instructions each corresponds to a schedule primitive"""
from typing import List, Union, Optional, TYPE_CHECKING
from tvm._ffi import register_object as _register_object
from tvm.runtime import Object

from . import _ffi_api_schedule

if TYPE_CHECKING:
    from .schedule import RAND_VAR_TYPE


@_register_object("tir.InstKind")
class InstKind(Object):
    name: str
    is_pure: bool

    @staticmethod
    def get(name: str) -> "InstKind":
        return _ffi_api_schedule.InstKindGet(name)  # pylint: disable=no-member


@_register_object("tir.Inst")
class Inst(Object):
    kind: InstKind
    inputs: List[Optional[Union["RAND_VAR_TYPE", str, int, float]]]
    attrs: List[Object]
    outputs: List["RAND_VAR_TYPE"]
