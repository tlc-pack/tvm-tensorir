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
from typing import List, Dict

from tvm._ffi import register_object

from . import _ffi_api
from .cost_model import CostModel
from .mutator import Mutator
from .schedule import Schedule
from .search import SearchStrategy, SearchTask


@register_object("meta_schedule.Replay")
class Replay(SearchStrategy):
    """A search strategy that just repeatedly replay the sampling process,
    do random sampling, and picks the best from the results

    Parameters
    ----------
    batch_size : int
        Size of a batch for measurement
    num_iterations : int
        Number of iterations of replaying
    """

    batch_size: int
    num_iterations: int

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


@register_object("meta_schedule.Evolutionary")
class Evolutionary(SearchStrategy):
    """Evolutionary Search.

    Parameters
    ----------
    num_measure_trials : int
        The number of iterations of measurements performed by genetic algorithm
    num_measure_per_batch : int
        The number of measurements in each batch
    num_iters_in_genetic_algo : int
        The number of iterations performed by generic algorithm.*/
    eps_greedy : float
        The percentage of measurements to use randomly sampled states.
    use_measured_ratio : float
        The percentage of previously measured states used in the initial population
    population : int
        The population size for evolutionary search
    p_mutate : float
        The probability to perform mutation
    mutator_probs : Dict[Mutator, float]
        Mutators and their probability mass
    cost_model : CostModel
        A cost model helping to explore the search space
    """

    num_measure_trials: int
    num_measure_per_batch: int
    num_iters_in_genetic_algo: int
    eps_greedy: float
    use_measured_ratio: float
    population: int
    p_mutate: float
    mutator_probs: Dict[Mutator, float]
    cost_model: CostModel

    def __init__(
        self,
        num_measure_trials: int,
        num_measure_per_batch: int,
        num_iters_in_genetic_algo: int,
        eps_greedy: float,
        use_measured_ratio: float,
        population: int,
        p_mutate: float,
        mutator_probs: Dict[Mutator, float],
        cost_model: CostModel,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.Evolutionary,  # pylint: disable=no-member
            num_measure_trials,
            num_measure_per_batch,
            num_iters_in_genetic_algo,
            eps_greedy,
            use_measured_ratio,
            population,
            p_mutate,
            mutator_probs,
            cost_model,
        )

    def sample_init_population(
        self,
        support: List[Schedule],
        num_samples: int,
    ) -> List[Schedule]:
        """Sample the initial population from the support

        Parameters
        ----------
        support : List[Schedule]
            The search task
        num_samples : SearchSpace
            The number of samples to be drawn

        Returns
        -------
        samples : List[Schedule]
            The initial population sampled from support
        """
        return _ffi_api.EvolutionarySampleInitPopulation(  # pylint: disable=no-member
            self, support, num_samples
        )

    def evolve_with_cost_model(
        self,
        task: SearchTask,
        inits: List[Schedule],
        num_samples: int,
    ) -> List[Schedule]:
        """Perform evolutionary search using genetic algorithm with the cost model

        Parameters
        ----------
        task : SearchTask
            The search task
        inits : List[Schedule]
            The initial population
        num_samples : int
            The number of samples to be drawn

        Returns
        -------
        samples : List[Schedule]
            The best samples in terms of the cost model's scores
        """
        return _ffi_api.EvolutionaryEvolveWithCostModel(  # pylint: disable=no-member
            self, task, inits, num_samples
        )
