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
""" The Auto-Tuning API """
from typing import Callable, Optional, Union

from tvm.tir import PrimFunc

from . import _ffi_api
from .measure import ProgramMeasurer
from .schedule import Schedule
from .search import SearchSpace, SearchStrategy, SearchTask


def autotune(
    task: Union[PrimFunc, SearchTask],
    space: Union[Callable[[Schedule], None], SearchSpace],
    strategy: Union[str, SearchStrategy],
    measurer: Optional[ProgramMeasurer] = None,
    verbose: int = 1,
) -> Optional[Schedule]:
    """The entry function for auto tuning.

    Parameters
    ----------
    task: Union[PrimFunc, SearchTask]
        The search task
    space: Union[Callable[[Schedule], None], SearchSpace]
        The search space
    strategy: Union[str, SearchStrategy]
        The search strategy
    measurer: Optional[ProgramMeasurer]
        The measurer that builds, runs and profiles sampled programs
    verbose: int
        Flag for the verbose mode

    Returns
    -------
    best_schedule : Optional[Schedule]
        The best schedule found, None if no valid schedule is found in the search space
    """
    if not isinstance(task, SearchTask):
        task = SearchTask(task)
    if not isinstance(space, SearchSpace):
        space = SearchSpace.create(space)
    if not isinstance(strategy, SearchStrategy):
        strategy = SearchStrategy.create(strategy)
    if measurer is None:
        measurer = ProgramMeasurer()
    return _ffi_api.AutoTune(  # pylint: disable=no-member
        task,
        space,
        strategy,
        measurer,
        verbose,
    )
