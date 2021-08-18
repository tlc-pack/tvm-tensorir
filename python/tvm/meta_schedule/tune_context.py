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
"""Tune Context"""

from typing import List, Optional

from tvm._ffi import register_object
from tvm import IRModule
from tvm.runtime import Object
from tvm.target import Target

from . import _ffi_api
from .space_generator import SpaceGenerator
from .search_strategy import SearchStrategy


class Database:
    pass


class CostModel:
    pass


class PostProc:
    pass


class MeasureCallback:
    pass


@register_object("meta_schedule.TuneContext")
class TuneContext(Object):
    """Description and abstraction of a tune context class."""

    def __init__(
        self,
        workload: Optional[IRModule],
        space_generator: Optional[SpaceGenerator],
        search_strategy: Optional[SearchStrategy],
        database: Optional[Database],
        cost_model: Optional[CostModel],
        target: Optional[Target],
        post_procs: Optional[List[PostProc]],
        measure_callbacks: Optional[List[MeasureCallback]],
        name: str,
        seed: int,
        num_threads: int,
        verbose: int,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.TuneContext,  # pylint: disable=no-member
            workload,
            space_generator,
            search_strategy,
            database,
            cost_model,
            target,
            post_procs,
            measure_callbacks,
            name,
            seed,
            num_threads,
            verbose,
        )

    def post_process(self) -> None:
        return _ffi_api.TuneContextPostProcess()  # pylint: disable=no-member

    def measure_callback(self) -> None:
        return _ffi_api.TuneContextMeasureCallback()  # pylint: disable=no-member
