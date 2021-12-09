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

from typing import NamedTuple

from tvm._ffi import register_object

from .. import _ffi_api
from .search_strategy import SearchStrategy


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
    max_replay_fail_cnt : int
        The maximum number to fail trace replaying.
    init_measured_ratio : int
        The ratio of measured samples in the initial population.
    genetic_algo_iters : int
        The number of iterations for genetic algorithm.
    max_evolve_fail_cnt : int
        The maximum number to retry mutation.
    p_mutate : float
        The probability of mutation.
    eps_greedy : float
        The ratio of greedy selected samples in the final picks.
    """

    num_trials_per_iter: int
    num_trials_total: int
    population: int
    init_measured_ratio: int
    genetic_algo_iters: int
    max_replay_fail_cnt: int
    max_evolve_fail_cnt: int
    p_mutate: float
    eps_greedy: float

    def __init__(
        self,
        *,
        num_trials_per_iter: int,
        num_trials_total: int,
        population: int = 2048,
        max_replay_fail_cnt: int = 64,
        init_measured_ratio: float = 0.2,
        genetic_algo_iters: int = 10,
        max_evolve_fail_cnt: int = 10,
        p_mutate: float = 0.85,
        eps_greedy: float = 0.25,
    ):
        """Constructor"""
        self.__init_handle_by_constructor__(
            _ffi_api.SearchStrategyEvolutionarySearch,  # type: ignore # pylint: disable=no-member
            num_trials_per_iter,
            num_trials_total,
            population,
            max_replay_fail_cnt,
            init_measured_ratio,
            genetic_algo_iters,
            max_evolve_fail_cnt,
            p_mutate,
            eps_greedy,
        )


class EvolutionarySearchConfig(NamedTuple):
    """Configuration for EvolutionarySearch"""

    num_trials_per_iter: int
    num_trials_total: int
    population: int = 2048
    max_replay_fail_cnt: int = 64
    init_measured_ratio: float = 0.2
    genetic_algo_iters: int = 10
    max_evolve_fail_cnt: int = 10
    p_mutate: float = 0.85
    eps_greedy: float = 0.25

    def create_strategy(self) -> EvolutionarySearch:
        return EvolutionarySearch(
            num_trials_per_iter=self.num_trials_per_iter,
            num_trials_total=self.num_trials_total,
            population=self.population,
            max_replay_fail_cnt=self.max_replay_fail_cnt,
            init_measured_ratio=self.init_measured_ratio,
            genetic_algo_iters=self.genetic_algo_iters,
            max_evolve_fail_cnt=self.max_evolve_fail_cnt,
            p_mutate=self.p_mutate,
            eps_greedy=self.eps_greedy,
        )
