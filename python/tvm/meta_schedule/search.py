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
""" Search API """
from __future__ import annotations

from typing import Callable, List, Optional, Union

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.tir import PrimFunc

from . import _ffi_api
from .measure import MeasureCallback, ProgramBuilder, ProgramRunner
from .schedule import Schedule
from .search_task import SearchTask

########## SearchSpace ##########


@register_object("meta_schedule.SearchSpace")
class SearchSpace(Object):
    """ defined in src/meta_schedule/search.h """


ScheduleFnType = Callable[[Schedule], None]


@register_object("meta_schedule.ScheduleFn")
class ScheduleFn(SearchSpace):
    """ defined in src/meta_schedule/search.h """

    def __init__(self, func: ScheduleFnType):
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleFn,  # pylint: disable=no-member
            func,
        )


########## SearchStrategy ##########


@register_object("meta_schedule.SearchStrategy")
class SearchStrategy(Object):
    """ defined in src/meta_schedule/search.h """

    @staticmethod
    def create(strategy: str) -> SearchStrategy:
        if strategy == "replay":
            return Replay()
        raise ValueError("Cannot create search strategy from: " + strategy)


@register_object("meta_schedule.Replay")
class Replay(SearchStrategy):
    """ defined in src/meta_schedule/search.h """

    def __init__(
        self,
        batch_size: int = 16,
        num_iterations: int = 128,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Replay,  # pylint: disable=no-member
            batch_size,
            num_iterations,
        )


########## Search API ##########


def autotune(
    task: Union[PrimFunc, SearchTask],
    space: Union[ScheduleFnType, SearchSpace],
    strategy: Union[str, SearchStrategy],
    builder: Union[str, ProgramBuilder] = "local",
    runner: Union[str, ProgramRunner] = "rpc",
    measure_callbacks: Optional[List[MeasureCallback]] = None,
    verbose: int = 1,
) -> Schedule:
    """ Search API """
    if isinstance(task, PrimFunc):
        task = SearchTask(task)
    if callable(space):
        space = ScheduleFn(space)
    if isinstance(strategy, str):
        strategy = SearchStrategy.create(strategy)
    if isinstance(builder, str):
        builder = ProgramBuilder.create(builder)
    if isinstance(runner, str):
        runner = ProgramRunner.create(runner)
    if measure_callbacks is None:
        measure_callbacks = []
    return _ffi_api.AutoTune(  # pylint: disable=no-member
        task, space, strategy, builder, runner, measure_callbacks, verbose
    )
