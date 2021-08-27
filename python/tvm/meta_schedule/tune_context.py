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

    def __init__(
        self,
        mod: Optional[IRModule],
        target: Optional[Target],
        space_generator: Optional["SpaceGenerator"],
        search_strategy: Optional[SearchStrategy],
        database: Optional[Database],
        cost_model: Optional[CostModel],
        postprocs: Optional[List[Postproc]],
        measure_callbacks: Optional[List[MeasureCallback]],
        task_name: str,
        seed: int,
        num_threads: int,
        verbose: int,
    ):
        """Construct a TuneContext.

        Parameters
        ----------
        mod : Optional[IRModule],
            The workload to be optimized.
        target : Optional[Target],
            The target to be optimized for.
        space_generator : Optional[SpaceGenerator],
            The design space generator.
        search_strategy : Optional[SearchStrategy],
           The search strategy to be used.
        database : Optional[Database],
            The database for querying and storage.
            Provides interface to query and store the results.
        cost_model : Optional[CostModel],
             The cost model for estimation.
             Provides interface to update and query the cost model for estimation.
        postprocs : Optional[List[Postproc]],
            The post processing functions.
            Each post processor is a single callable function.
        measure_callbacks : Optional[List[MeasureCallback]],
            The measure callback functions.
            Each measure callback is a single callable function.
        task_name : str,
            The name of the tuning task.
        seed : int,
            The seed value of random state.
            Need to be in integer in [1, 2^31-1].
        num_threads : int,
            The number of threads to be used.
        verbose : int,
            The verbosity level.
        """
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
