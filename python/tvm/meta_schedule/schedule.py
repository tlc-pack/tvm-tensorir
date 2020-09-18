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
""" Main class of meta schedule """
from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api


@register_object("meta_schedule.Schedule")
class Schedule(Object):
    """ defined in src/meta_schedule/schedule.h """

    def __init__(self, prim_func):
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleCreate, prim_func  # pylint: disable=no-member
        )

    def evaluate(self, random_variable):
        return _ffi_api.ScheduleEval(self, random_variable)  # pylint: disable=no-member

    def sample_tile_factor(self, n, loop, where):
        return _ffi_api.ScheduleSampleTileFactor(  # pylint: disable=no-member
            self, n, loop, where
        )

    def get_block(self, name):
        return _ffi_api.ScheduleGetBlock(self, name)  # pylint: disable=no-member

    def get_axes(self, block):
        return _ffi_api.ScheduleGetAxes(self, block)  # pylint: disable=no-member

    def split(self, loop, factors):
        return _ffi_api.ScheduleSplit(self, loop, factors)  # pylint: disable=no-member

    def reorder(self, after_axes):
        return _ffi_api.ScheduleReorder(self, after_axes)  # pylint: disable=no-member

    def decompose_reduction(self, block, loop):
        return _ffi_api.ScheduleDecomposeReduction(  # pylint: disable=no-member
            self, block, loop
        )

    def replay_once(self):
        return _ffi_api.ScheduleReplayOnce(self)  # pylint: disable=no-member
