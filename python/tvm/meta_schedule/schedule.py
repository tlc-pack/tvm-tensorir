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
""" Main class of meta schedule """
from typing import List, Optional, Union

from tvm import tir
from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api
from .instruction import Instruction
from .random_variable import RAND_VAR_TYPE, BlockRV, ExprRV, LoopRV


@register_object("meta_schedule.Schedule")
class Schedule(Object):
    """The meta schedule class.

    Parameters
    ----------
    orig_func : PrimFunc
        The original TIR PrimFunc to be scheduled
    sch: tir.Schedule
        The TIR schedule in the current stage
    trace: List[Instruction]
        The trace of instructions used
    """

    orig_func: tir.PrimFunc
    sch: tir.Schedule
    trace: List[Instruction]

    def __init__(self, func: tir.PrimFunc):
        self.__init_handle_by_constructor__(
            _ffi_api.Schedule,  # pylint: disable=no-member
            func,
        )

    def copy(self) -> "Schedule":
        """Copy the schedule into a new one.
        Operation on the new schedule won't affect the original schedule, and vice versa.

        Returns
        -------
        new_schedule : Schedule
            A new schedule
        """
        return _ffi_api.ScheduleCopy(self)  # pylint: disable=no-member

    def evaluate(
        self,
        random_variable: RAND_VAR_TYPE,
    ) -> Union[tir.schedule.StmtSRef, int]:
        """Evaluates a random variable

        Parameters
        ----------
        random_variable : Union[BlockRV, LoopRV, ExprRV]
            The random variable to be evaluated

        Returns
        -------
        concrete_value : Union[tir.schedule.StmtSRef, int]
            The concrete value that is evaluated to
        """
        return _ffi_api.ScheduleEval(self, random_variable)  # pylint: disable=no-member

    ######### Sampling #########

    def sample_perfect_tile(
        self,
        n_splits: int,
        loop: LoopRV,
        max_innermost_factor: int = 16,
    ) -> List[ExprRV]:
        """Split a loop by the given tiling factors

        Parameters
        ----------
        n_splits: int
            The number of loops after tiling
        loop: LoopRV
            The loop to be tiled
        max_innermost_factor: int
            The maximum factor in the innermost loop

        Returns
        -------
        factors : List[ExprRV]
            The result of sampling
        """
        return _ffi_api.ScheduleSamplePerfectTile(  # pylint: disable=no-member
            self, n_splits, loop, max_innermost_factor
        )

    def sample_tile_factor(
        self,
        n_splits: int,
        loop: LoopRV,
        where: List[int],
    ) -> List[ExprRV]:
        """Split a loop by the given tiling factors

        Parameters
        ----------
        n_splits: int
            The number of loops after tiling
        loop: LoopRV
            The loop to be tiled
        where: List[int]
            The distribution of tile size to be sampled

        Returns
        -------
        factors : List[ExprRV]
            The result of sampling
        """
        return _ffi_api.ScheduleSampleTileFactor(  # pylint: disable=no-member
            self, n_splits, loop, where
        )

    ######### Block/Loop Relationship #########

    def get_only_consumer(self, block: BlockRV) -> Optional[BlockRV]:
        """Apply the instruction GetBlock, get a block by its name

        Parameters
        ----------
        block: BlockRV
            The block to be queried

        Returns
        -------
        only_consumer : Optional[BlockRV]
            A block, its only consumer; or None if it does not exist
        """
        return _ffi_api.ScheduleGetOnlyConsumer(  # pylint: disable=no-member
            self, block
        )

    def get_block(self, name: str) -> BlockRV:
        """Apply the instruction GetBlock, get a block by its name

        Parameters
        ----------
        name: str
            Name of the block

        Returns
        -------
        block : BlockRV
            The block retrieved
        """
        return _ffi_api.ScheduleGetBlock(self, name)  # pylint: disable=no-member

    def get_axes(self, block: BlockRV) -> List[LoopRV]:
        """Get loop nests above a block

        Parameters
        ----------
        block: BlockRV
            The block to be queried

        Returns
        -------
        axes : List[LoopRV]
            The loop nests above the block
        """
        return _ffi_api.ScheduleGetAxes(self, block)  # pylint: disable=no-member

    ########## Scheduling Primitives ##########

    def split(
        self,
        loop: LoopRV,
        factors: List[ExprRV],
    ) -> List[LoopRV]:
        """Split the given loop with the specific factors

        Parameters
        ----------
        loop: LoopRV
            The loop to be split
        factors: List[ExprRV]
            The factors used for split

        Returns
        -------
        axes : List[LoopRV]
            The loop axes after split
        """
        return _ffi_api.ScheduleSplit(self, loop, factors)  # pylint: disable=no-member

    def reorder(self, after_axes: List[LoopRV]) -> None:
        """Reorder the loops into the order given

        Parameters
        ----------
        after_axes: List[LoopRV]
            The order of the loop after reordering
        """
        _ffi_api.ScheduleReorder(self, after_axes)  # pylint: disable=no-member

    def compute_inline(self, block: BlockRV) -> None:
        """Apply the instruction compute_inline

        Parameters
        ----------
        block: BlockRV
            The block to be computed inline
        """
        _ffi_api.ScheduleComputeInline(self, block)  # pylint: disable=no-member

    def cache_write(self, block: BlockRV, storage_scope: str) -> BlockRV:
        """Apply the instruction cache_write

        Parameters
        ----------
        block: BlockRV
            The block to be buffered
        storage_scope: str
            The storage scope

        Returns
        -------
        block : BlockRV
            The cache write stage
        """
        return _ffi_api.ScheduleCacheWrite(  # pylint: disable=no-member
            self, block, storage_scope
        )

    def decompose_reduction(
        self,
        block: BlockRV,
        loop: LoopRV,
    ) -> BlockRV:
        """Decompose the reduction in the specific block under the specific loop

        Parameters
        ----------
        block: BlockRV
            The block that contains the reduction
        loop: LoopRV
            The loop that the initialization should be under

        Returns
        -------
        block : BlockRV
            The result of the decomposition
        """
        return _ffi_api.ScheduleDecomposeReduction(  # pylint: disable=no-member
            self, block, loop
        )

    ########## Replay ##########

    def replay_once(self) -> None:
        """ Replay the trace to generate a new state of scheduling """
        return _ffi_api.ScheduleReplayOnce(self)  # pylint: disable=no-member
