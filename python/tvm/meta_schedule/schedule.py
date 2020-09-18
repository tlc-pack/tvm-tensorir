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
from tvm._ffi import _init_api
from tvm._ffi import register_object as _register_object
from tvm.runtime import Object as _Object


@_register_object("meta_schedule.Schedule")
class Schedule(_Object):
    """ defined in src/meta_schedule/schedule.h """

    def __init__(self, prim_func):
        self.__init_handle_by_constructor__(
            Create, prim_func  # pylint: disable=undefined-variable
        )

    def evaluate(self, random_variable):
        return Eval(self, random_variable)  # pylint: disable=undefined-variable

    def sample_tile_factor(self, n, loop, where):
        return SampleTileFactor(  # pylint: disable=undefined-variable
            self, n, loop, where
        )

    def get_block(self, name):
        return GetBlock(self, name)  # pylint: disable=undefined-variable

    def get_axes(self, block):
        return GetAxes(self, block)  # pylint: disable=undefined-variable

    def split(self, loop, factors):
        return Split(self, loop, factors)  # pylint: disable=undefined-variable

    def reorder(self, after_axes):
        return Reorder(self, after_axes)  # pylint: disable=undefined-variable

    def decompose_reduction(self, block, loop):
        return DecomposeReduction(  # pylint: disable=undefined-variable
            self, block, loop
        )

    def replay_once(self):
        return ReplayOnce(self)  # pylint: disable=undefined-variable


_init_api("meta_schedule.schedule", __name__)
