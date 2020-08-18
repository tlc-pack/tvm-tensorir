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
"""Instructions for meta scheduling"""

import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object("auto_scheduler.Instruction")
class Instruction(Object):
    """Base class for all instructions"""


@tvm._ffi.register_object("auto_scheduler.DeclIntVar")
class DeclIntVar(Instruction):
    """Declare an integer variable with categorical prior"""

    def __init__(self, var, choices):
        self.__init_handle_by_constructor__(_ffi_api.DeclIntVar, var, choices)


@tvm._ffi.register_object("auto_scheduler.SplitInnerToOuter")
class SplitInnerToOuter(Instruction):
    """Perform a split on a loop of the cursor, from inner to outer"""

    def __init__(self, loop_id, inferred_factor_id, factors):
        self.__init_handle_by_constructor__(
            _ffi_api.SplitInnerToOuter, loop_id, inferred_factor_id, factors)


@tvm._ffi.register_object("auto_scheduler.Reorder")
class Reorder(Instruction):
    """Perform reorder on the cursor"""

    def __init__(self, after_ids):
        self.__init_handle_by_constructor__(_ffi_api.Reorder, after_ids)


@tvm._ffi.register_object("auto_scheduler.ComputeAtOffset")
class ComputeAtOffset(Instruction):
    """Perform compute the cursor to a block at a sibling with a specific offset"""

    def __init__(self, offset, loop_id):
        self.__init_handle_by_constructor__(
            _ffi_api.ComputeAtOffset, offset, loop_id)


@tvm._ffi.register_object("auto_scheduler.CursorMoveOffset")
class CursorMoveOffset(Instruction):
    """Move the cursor to its sibling with a specific offset"""

    def __init__(self, offset):
        self.__init_handle_by_constructor__(_ffi_api.CursorMoveOffset, offset)
