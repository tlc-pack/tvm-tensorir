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

from typing import TYPE_CHECKING, List, Optional

from tvm import IRModule
from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.target import Target
from tvm.meta_schedule.utils import cpu_count

from . import _ffi_api

if TYPE_CHECKING:
    from .space_generator import SpaceGenerator


class Database:
    pass


class CostModel:
    pass


class Postproc:
    pass


class MeasureCallback:
    pass


class SearchStrategy:
    pass


@register_object("meta_schedule.TuneContext")
class TuneContext(Object):
    """
    The tune context class is designed to contain all resources for a tuning task.

    All tuning tasks are separated in different tune contexts, but classes can access other class in
    the same tune context through this class.

    Most classes have a function to initialize with a tune context.
    """

    mod: IRModule
    target: Target
    space_generator: "SpaceGenerator"
    search_strategy: SearchStrategy
    database: Database
    cost_model: CostModel
    postprocs: List[Postproc]
    measure_callbacks: List[MeasureCallback]
    task_name: str
    seed: int
    num_threads: int
    verbose: int

    def __init__(
        self,
        mod: Optional[IRModule] = None,
        target: Optional[Target] = None,
        space_generator: Optional["SpaceGenerator"] = None,
        search_strategy: Optional[SearchStrategy] = None,
        database: Optional[Database] = None,
        cost_model: Optional[CostModel] = None,
        postprocs: Optional[List[Postproc]] = None,
        measure_callbacks: Optional[List[MeasureCallback]] = None,
        task_name: Optional[str] = None,
        seed: int = -1,
        num_threads: Optional[int] = -1,
        verbose: Optional[int] = 0,
    ):
        """Construct a TuneContext.

        Parameters
        ----------
        mod : Optional[IRModule] (default is None)
            The workload to be optimized.
        target : Optional[Target] (default is None)
            The target to be optimized for.
        space_generator : Optional[SpaceGenerator] (default is None)
            The design space generator.
        search_strategy : Optional[SearchStrategy] (default is None)
           The search strategy to be used.
        database : Optional[Database] (default is None)
            The database for querying and storage.
            Provides interface to query and store the results.
        cost_model : Optional[CostModel] (default is None)
             The cost model for estimation.
             Provides interface to update and query the cost model for estimation.
        postprocs : Optional[List[Postproc]] (default is None)
            The post processing functions.
            Each post processor is a single callable function.
        measure_callbacks : Optional[List[MeasureCallback]] (default is None)
            The measure callback functions.
            Each measure callback is a single callable function.
        task_name : Optional[str] (default is None)
            The name of the tuning task.
        seed : int (default is -1)
            The seed value of random state.
            Need to be in integer in [1, 2^31-1], -1 means use a random seed.
        num_threads : int (default is -1)
            The number of threads to be used, -1 means use the logic cpu count.
        verbose : int (default is 0)
            The verbosity level.
        """
        if num_threads == -1:
            num_threads = cpu_count()

        self.__init_handle_by_constructor__(
            _ffi_api.TuneContext,  # pylint: disable=no-member
            mod,
            target,
            space_generator,
            search_strategy,
            database,
            cost_model,
            postprocs,
            measure_callbacks,
            task_name,
            seed,
            num_threads,
            verbose,
        )
