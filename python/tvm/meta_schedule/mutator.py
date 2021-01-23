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
"""Mutators that helps explores the space.
It mutates a schedule to an adjacent one by doing small modification"""
from typing import Optional

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api_mutator
from .search import SearchTask
from .trace import Trace


@register_object("meta_schedule.Mutator")
class Mutator(Object):
    """A mutation rule for the genetic algorithm"""

    name: str

    def apply(
        self,
        task: SearchTask,
        trace: Trace,
        seed: Optional[int] = None,
    ) -> Optional[Trace]:
        """Mutate the schedule by applying the mutation.

        Parameters
        ----------
        task : SearchTask
            The search task
        trace: Trace
            The trace to be mutated

        Returns
        -------
        new_trace : Optional[Trace]
            The new trace after mutation, or None if cannot find a viable solution
        """
        return _ffi_api_mutator.Apply(self, task, trace, seed)  # pylint: disable=no-member


def mutate_tile_size() -> Mutator:
    """Create a mutator that randomly mutate the tile size

    Returns
    ----------
    mutator: Mutator
        The mutator created
    """
    return _ffi_api_mutator.MutateTileSize()  # pylint: disable=no-member
