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
from typing import Any, Dict, List, Optional, Union

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.target import Target
from tvm.target import create as create_target
from tvm.tir import PrimFunc

from . import _ffi_api
from .schedule import Schedule

########## SearchTask ##########

TargetType = Union[Target, str, Dict[str, Any]]


@register_object("meta_schedule.SearchTask")
class SearchTask(Object):
    """Descrption of a search task

    Parameters
    ----------
    func: PrimFunc
        The function to be optimized
    task_name: str
        Name of this search task
    target: Target
        The target to be built at
    target_host: Target
        The target host to be built at
    """

    func: PrimFunc
    task_name: str
    target: Target
    target_host: Target

    def __init__(
        self,
        func: PrimFunc,
        task_name: Optional[str] = None,
        target: TargetType = "llvm",
        target_host: TargetType = "llvm",
    ):
        if task_name is None:
            task_name = func.__qualname__
        if not isinstance(target, Target):
            target = create_target(target)
        if not isinstance(target_host, Target):
            target_host = create_target(target_host)
        self.__init_handle_by_constructor__(
            _ffi_api.SearchTask,  # pylint: disable=no-member
            func,
            task_name,
            target,
            target_host,
        )


########## SearchSpace ##########


@register_object("meta_schedule.SearchSpace")
class SearchSpace(Object):
    """Description and abstraction of a search space.
    The search space could be specified by manually written schedule function,
    generated via loop analysis, ansor-like rules that apply to each block, etc."""

    @staticmethod
    def create(arg) -> "SearchSpace":
        from . import space  # pylint: disable=import-outside-toplevel

        if callable(arg):
            return space.ScheduleFn(arg)

        raise ValueError("Cannot create search space from: " + space)

    def sample_schedule(self, task: SearchTask) -> Schedule:
        return _ffi_api.SearchSpaceSampleSchedule(  # pylint: disable=no-member
            self, task
        )

    def get_support(self, task: SearchTask) -> List[Schedule]:
        return _ffi_api.SearchSpaceGetSupport(self, task)  # pylint: disable=no-member


########## SearchStrategy ##########


@register_object("meta_schedule.SearchStrategy")
class SearchStrategy(Object):
    """The search strategy for exploring the search space.
    It could be always replay the sampling function, or saving several traces
    from the sample function and then do lightweight-metropolis-hastings, or integrate those with
    evolutionary search, etc.
    """

    @staticmethod
    def create(arg) -> "SearchStrategy":
        from . import strategy  # pylint: disable=import-outside-toplevel

        if arg == "replay":
            return strategy.Replay()
        raise ValueError("Cannot create search strategy from: " + strategy)

    def search(
        self,
        task: SearchTask,
        space: SearchSpace,
        measurer: "ProgramMeasurer",
        verbose: int,
    ) -> Optional[Schedule]:
        """Explore the search space and find the best schedule

        Parameters
        ----------
        task : SearchTask
            The search task
        space : SearchSpace
            The search space
        measurer : ProgramMeasurer
            The measurer that builds, runs and profiles sampled programs
        verbose : int
            Whether or not in verbose mode

        Returns
        -------
        schedule : Optional[Schedule]
            The best schedule found, None if no valid schedule is found
        """
        return _ffi_api.SearchStrategySearch(  # pylint: disable=no-member
            task, space, measurer, verbose
        )
