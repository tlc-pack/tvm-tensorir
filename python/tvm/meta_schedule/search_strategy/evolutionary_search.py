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
"""Evolutionary Search Strategy"""

from typing import TYPE_CHECKING, Dict

from tvm._ffi import register_object
from ...tir import FloatImm

from .search_strategy import SearchStrategy
from ..mutator import Mutator
from ..database import Database

from .. import _ffi_api

if TYPE_CHECKING:
    from ..cost_model import CostModel


@register_object("meta_schedule.EvolutionarySearch")
class EvolutionarySearch(SearchStrategy):
    """
    Replay Trace Search Strategy is a search strategy that always replays the trace by removing its
    decisions so that the decisions would be randomly re-generated.

    Parameters
    ----------
    num_trials_per_iter : int
        Number of trials per iteration.
    num_trials_total : int
        Total number of trials.
    population : int
        The initial population of traces from measured samples and randomly generated samples.
    init_measured_ratio : int
        The ratio of measured samples in the initial population.
    genetic_algo_iters : int
        The number of iterations for genetic algorithm.
    p_mutate : float
        The probability of mutation.
    eps_greedy : float
        The ratio of greedy selected samples in the final picks.
    mutator_probs: Dict[Mutator, FloatImm]
        The probability contribution of all mutators.
    database : Database
        The database used in the search.
    cost_model : CostModel
        The cost model used in the search.
    """

    num_trials_per_iter: int
    num_trials_total: int
    population: int
    init_measured_ratio: int
    genetic_algo_iters: int
    p_mutate: float
    eps_greedy: float
    mutator_probs: Dict[Mutator, FloatImm]
    database: Database
    cost_model: "CostModel"

    def __init__(
        self,
        num_trials_per_iter: int,
        num_trials_total: int,
        population: int,
        init_measured_ratio: float,
        genetic_algo_iters: int,
        p_mutate: float,
        eps_greedy: float,
        mutator_probs: Dict[Mutator, FloatImm],
        database: Database,
        cost_model: "CostModel",
    ):
        """Constructor"""
        self.__init_handle_by_constructor__(
            _ffi_api.SearchStrategyEvolutionarySearch,  # pylint: disable=no-member
            num_trials_per_iter,
            num_trials_total,
            population,
            init_measured_ratio,
            genetic_algo_iters,
            p_mutate,
            eps_greedy,
            mutator_probs,
            database,
            cost_model,
        )
