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
    workload: PrimFunc
        The function to be optimized
    task_name: str
        Name of this search task
    target: Target
        The target to be built at
    target_host: Target
        The target host to be built at
    """

    workload: PrimFunc
    task_name: str
    target: Target
    target_host: Target
    log_file: Optional[str]

    def __init__(
        self,
        workload: PrimFunc,
        task_name: Optional[str] = None,
        target: TargetType = "llvm",
        target_host: TargetType = "llvm",
        log_file: Optional[str] = None,
    ):
        if task_name is None:
            if hasattr(workload, "__qualname__"):
                task_name = workload.__qualname__
            else:
                raise ValueError(
                    "Unable to extract task_name from the PrimFunc, please specific it explicitly"
                )
        if not isinstance(target, Target):
            target = Target(target)
        if not isinstance(target_host, Target):
            target_host = Target(target_host)
        self.__init_handle_by_constructor__(
            _ffi_api.SearchTask,  # pylint: disable=no-member
            workload,
            task_name,
            target,
            target_host,
            log_file,
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

    def postprocess(
        self,
        task: SearchTask,
        sch: Schedule,
        seed: Optional[int] = None,
    ) -> bool:
        return _ffi_api.SearchSpacePostprocess(self, task, sch, seed)  # pylint: disable=no-member

    def sample_schedule(
        self,
        task: SearchTask,
        seed: Optional[int] = None,
    ) -> Schedule:
        return _ffi_api.SearchSpaceSampleSchedule(self, task, seed)  # pylint: disable=no-member

    def get_support(
        self,
        task: SearchTask,
        seed: Optional[int] = None,
    ) -> List[Schedule]:
        return _ffi_api.SearchSpaceGetSupport(self, task, seed)  # pylint: disable=no-member


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
        seed: Optional[int] = None,
        verbose: int = 1,
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
            task, space, measurer, seed, verbose
        )
