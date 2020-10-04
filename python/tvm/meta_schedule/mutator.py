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
from .schedule import Schedule
from .search_task import SearchTask


@register_object("meta_schedule.Mutator")
class Mutator(Object):
    """A mutation rule for the genetic algorithm

    Parameters
    ----------
    p: float
        The probability mass of choosing this mutator
    """

    p: float

    def apply(self, task: SearchTask, sch: Schedule) -> Optional[Schedule]:
        """Mutate the schedule by applying the mutation.

        Parameters
        ----------
        task : SearchTask
            The search task
        sch: Schedule
            The schedule to be mutated

        Returns
        -------
        new_sch : Optional[Schedule]
            The new schedule after mutation, or None if cannot find a viable solution
        """
        return _ffi_api_mutator.MutatorApply(  # pylint: disable=no-member
            self, task, sch, None
        )


@register_object("meta_schedule.MutateTileSize")
class MutateTileSize(Mutator):
    """Mutate the sampled tile size by re-factorized two axes"""

    def __init__(self, p: float):
        self.__init_handle_by_constructor__(
            _ffi_api_mutator.MutateTileSize, p  # pylint: disable=no-member
        )
