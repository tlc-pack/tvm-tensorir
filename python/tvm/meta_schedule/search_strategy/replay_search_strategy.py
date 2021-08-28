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
"""Replay Search Strategy"""

import random
from typing import List, TYPE_CHECKING
from tvm.tir.schedule import Trace

from .search_strategy import PySearchStrategy
from ..builder import BuildInput

if TYPE_CHECKING:
    from ..tune_context import TuneContext


class ReplaySearchStrategy(PySearchStrategy):
    """
    Replay Search Strategy is a search strategy that always replays the trace by removing its
    decisions so that the decisions would be randomly re-generated.
    """

    def __init__(self, trials, batch_size):
        """Constructor.

        Parameters
        ----------
        trials : int
            The number of total trials to be replayed.
        batch_size : int
            The batch size of the replayed trials.
        """
        super().__init__()  # use abstract class's __init__ method
        self.trials = trials
        self.batch_size = batch_size

    def initialize_with_tune_context(self, tune_context: "TuneContext") -> None:
        pass

    def generate_measure_candidates(self) -> List[BuildInput]:
        if self.count >= self.trials:
            return None
        candidates = []
        for _ in range(self.count, min(self.count + self.batch_size, self.trials)):
            # Randomly select a design space to replay
            trace = Trace(random.choice(self.design_spaces).insts, None)
            candidates.append(trace)
        return candidates

    def notify_runner_results(self, results: List["RunnerResult"]) -> None:
        self.count += len(results)

    def pre_tuning(self, design_spaces: List[Trace]) -> None:
        self.design_spaces = design_spaces  # assign the design spaces to the class
        self.count = 0  # reset the count

    def post_tuning(self) -> None:
        pass
