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
"""Search strategy"""
from typing import Dict, List, Optional

from tvm._ffi import register_object

from ..tir.schedule import Trace
from . import _ffi_api
from .cost_model import CostModel
from .measure import ProgramMeasurer
from .measure_record import MeasureResult
from .mutator import Mutator
from .search import SearchSpace, SearchStrategy, SearchTask
from .utils import cpu_count


@register_object("meta_schedule.Replay")
class Replay(SearchStrategy):
    """A search strategy that just repeatedly replay the sampling process,
    do random sampling, and picks the best from the results

    Parameters
    ----------
    num_trials : int
        Number of iterations of replaying
    batch_size : int
        Size of a batch for measurement
    """

    num_trials: int
    batch_size: int

    def __init__(
        self,
        num_trials: int = 32,
        batch_size: Optional[int] = None,
    ):
        if batch_size is None:
            batch_size = cpu_count()
        self.__init_handle_by_constructor__(
            _ffi_api.Replay,  # pylint: disable=no-member
            batch_size,
            num_trials,
        )


@register_object("meta_schedule.Evolutionary")
class Evolutionary(SearchStrategy):
    """Evolutionary Search.

    The algorithm:

    Loop until #measured >= total_measures:
      init =
            pick top `k = population *      init_measured_ratio ` from measured
            pick     `k = population * (1 - init_measured_ratio)` from random support
      best = generate `population` states with the cost model,
            starting from `init`,
            using mutators
            and return the top-n states during the search,
            where `n = num_measures_per_iter`
      chosen = pick top `k = num_measures_per_iter * (1 - eps_greedy)` from `best`
               pick     `k = num_measures_per_iter *      eps_greedy ` from `init`
    do the measurement on `chosen` & update the cost model


    Parameters
    ----------
    total_measures : int
        The maximum number of measurements performed by genetic algorithm
    num_measures_per_iter : int
        The number of measures to be performed in each iteration
    population : int
        The population size in the evolutionary search
    database : Database
        A table storing all states that have been measured
    init_measured_ratio : float
        The ratio of measured states used in the initial population
    genetic_algo_iters : int
        The number of iterations performed by generic algorithm
    p_mutate : float
        The probability to perform mutation
    mutator_probs : Dict[Mutator, float]
        Mutators and their probability mass
    cost_model : CostModel
        A cost model helping to explore the search space
    eps_greedy: float
        The ratio of measurements to use randomly sampled states
    """

    # Configuration: global
    total_measures: int
    population: int
    database: "Database"
    # Configuration: the initial population
    init_measured_ratio: float
    # Configuration: evolution
    genetic_algo_iters: int
    p_mutate: float
    mutator_probs: Dict[Mutator, float]
    cost_model: CostModel
    # Configuration: pick for measurement
    eps_greedy: float

    def __init__(
        self,
        total_measures: int,
        num_measures_per_iter: int,
        population: int,
        init_measured_ratio: float,
        genetic_algo_iters: int,
        p_mutate: float,
        mutator_probs: Dict[Mutator, float],
        cost_model: CostModel,
        eps_greedy: float,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Evolutionary,  # pylint: disable=no-member
            total_measures,
            num_measures_per_iter,
            population,
            init_measured_ratio,
            genetic_algo_iters,
            p_mutate,
            mutator_probs,
            cost_model,
            eps_greedy,
        )

    def sample_init_population(
        self,
        support: List[Trace],
        task: SearchTask,
        space: SearchSpace,
        seed: Optional[int] = None,
    ) -> List[Trace]:
        """Sample the initial population from the support

        Parameters
        ----------
        support : List[Trace]
            The search task
        num_samples : SearchSpace
            The number of samples to be drawn

        Returns
        -------
        samples : List[Trace]
            The initial population sampled from support
        """
        return _ffi_api.EvolutionarySampleInitPopulation(  # pylint: disable=no-member
            self, support, task, space, seed
        )

    def evolve_with_cost_model(
        self,
        inits: List[Trace],
        task: SearchTask,
        space: SearchSpace,
        seed: Optional[int] = None,
    ) -> List[Trace]:
        """Perform evolutionary search using genetic algorithm with the cost model

        Parameters
        ----------
        task : SearchTask
            The search task
        inits : List[Trace]
            The initial population
        num_samples : int
            The number of samples to be drawn

        Returns
        -------
        samples : List[Trace]
            The best samples in terms of the cost model's scores
        """
        return _ffi_api.EvolutionaryEvolveWithCostModel(  # pylint: disable=no-member
            self, inits, task, space, seed
        )

    def pick_with_eps_greedy(
        self,
        inits: List[Trace],
        bests: List[Trace],
        task: SearchTask,
        space: SearchSpace,
        seed: Optional[int] = None,
    ) -> List[Trace]:
        """Pick a batch of samples for measurement with epsilon greedy

        Parameters
        ----------
        task : SearchTask
            The search task
        inits : List[Trace]
            The initial population
        bests : List[Trace]
            The best populations according to the cost model when picking top states

        Returns
        -------
        samples : List[Trace]
            A list of traces, result of epsilon-greedy sampling
        """
        return _ffi_api.EvolutionaryPickWithEpsGreedy(  # pylint: disable=no-member
            self, inits, bests, task, space, seed
        )

    def measure_and_update_cost_model(
        self,
        task: SearchTask,
        picks: List[Trace],
        measurer: ProgramMeasurer,
        verbose: int,
    ) -> List[MeasureResult]:
        """Make measurements and update the cost model

        Parameters
        ----------
        task : SearchTask
            The search task
        picks : List[Trace]
            The picked traces to be measured
        measurer : ProgramMeasurer
            The measurer
        verbose : int
            A boolean flag for verbosity

        Returns
        -------
        samples : List[MeasureResult]
            A list of MeasureResult for measurements
        """
        return _ffi_api.EvolutionaryMeasureAndUpdateCostModel(  # pylint: disable=no-member
            self, task, picks, measurer, verbose
        )
