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
"""Namespace for meta schedule"""

import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api


@tvm._ffi.register_object("auto_scheduler.MetaSchedule")
class MetaSchedule(Object):
    """Class for meta schedule"""

    def __init__(self, meta_ir):
        self.__init_handle_by_constructor__(FromMetaIR, meta_ir)

    def decl_int_var(self, choices, name_hint=""):
        return DeclIntVarNode(self, choices, name_hint)

    def split_inner_to_outer(self, loop_id, factors, name_hint=""):
        return SplitInnerToOuter(self, loop_id, factors, name_hint)

    def reorder(self, after_ids):
        Reorder(self, after_ids)

    def compute_at_offset(self, offset, loop_id):
        ComputeAtOffset(self, offset, loop_id)

    def cursor_move_offset(self, offset):
        return CursorMoveOffset(self, offset)

    def apply_to_schedule(self, schedule, sampled_vars):
        return ApplyToSchedule(self, schedule, sampled_vars)


tvm._ffi._init_api("auto_scheduler.meta_schedule", __name__)
