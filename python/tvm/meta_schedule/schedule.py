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
import json
from typing import List, Optional, Union, Any

from tvm import ir, tir
from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api
from .instruction import RAND_VAR_TYPE, BlockRV, BufferRV, ExprRV, Instruction, LoopRV


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
    decisions: List[List[Object]]

    def __init__(self, func: tir.PrimFunc, seed: Optional[int] = None):
        self.__init_handle_by_constructor__(
            _ffi_api.Schedule, func, seed  # pylint: disable=no-member
        )

    ######### Utility #########

    def seed(self, new_seed: int) -> None:
        """Seed the randomness

        Parameters
        -------
        new_seed : int
            The new seed used
        """
        return _ffi_api.ScheduleSeed(self, new_seed)  # pylint: disable=no-member

    def copy(self, seed: int) -> "Schedule":
        """Copy the schedule into a new one.
        Operation on the new schedule won't affect the original schedule, and vice versa.

        Returns
        -------
        new_schedule : Schedule
            A new schedule
        """
        return _ffi_api.ScheduleCopy(self, seed)  # pylint: disable=no-member

    ######### Serialization #########

    @staticmethod
    def import_(
        record: str,
        func: tir.PrimFunc,
        seed: Optional[int] = None,
    ) -> "Schedule":
        """Import from the records

        Parameters
        ----------
        record : str
            The serialized trace of scheduling
        func : tir.PrimFunc
            The TIR function to be scheduled
        seed : Optional[int]
            The random seed

        Returns
        -------
        schedule : Schedule
            The schedule imported
        """
        record = json.loads(record)
        return _ffi_api.ScheduleImport(record, func, seed)  # pylint: disable=no-member

    def export(self) -> str:
        """Export as records

        Returns
        -------
        records : Any
            The record exported
        """
        def to_native_py(obj):
            if isinstance(obj, ir.Array):
                return list(to_native_py(item) for item in obj)
            if isinstance(obj, ir.Map):
                return {to_native_py(k): to_native_py(v) for k, v in obj.items()}  # pylint: disable=unnecessary-comprehension)
            if isinstance(obj, tir.IntImm):
                return int(obj)
            return obj
        records = _ffi_api.ScheduleExport(self)  # pylint: disable=no-member
        records = to_native_py(records)
        return json.dumps(records)

    ######### Evaluation of random variables #########

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

    def sample_fusible_loops(
        self,
        loops: List[LoopRV],
        loop_types: List[int],
        max_extent: int,
        include_overflow_loop: bool = True,
        order: str = "outer_to_inner",
        mode: str = "max",
    ) -> ExprRV:
        """Sample fusible loops, in the specific order (inner-to-outer or outer-to-inner),
        where their product of extent is limited. The sampling could have two modes: max or rand.
        If it is using mode "max", the sampling deterministically choose the maximum number of
        loops to fuse; Otherwise, if choose mode "rand", it samples the number of viable choices
        uniformly and return a randomly selected number of loops to fuse.

        Parameters
        ----------
        loops : List[LoopRV]
            The loops to be fused
        loop_types : List[int]
            Type of the loop
        max_extent : int
            The maximum extent of loops
        include_overflow_loop : bool
            Whether to include the last loop that makes the extent larger then `max_extent`
        order : str
            The order of fusion, can be `inner_to_outer` or `outer_to_inner`
        mode : str
            The mode of the fusion, can be `max` or `rand`

        Returns
        -------
        n_fusible : ExprRV
            A ExprRV, a random variable indicates the number of loops that can be potentially fused
        """
        order = {"outer_to_inner": 0, "inner_to_outer": 1}.get(order, None)
        mode = {"max": 0, "rand": 1}.get(mode, None)
        if order is None:
            raise ValueError('"order" needs to be one of: "outer_to_inner", "inner_to_order"')
        if mode is None:
            raise ValueError('"mode" needs to be one of: "max", "rand"')
        return _ffi_api.SampleFusibleLoops(  # pylint: disable=no-member
            self, loops, loop_types, max_extent, include_overflow_loop, order, mode
        )

    ######### Block/Loop Relationship #########

    def get_only_consumer(self, block: BlockRV) -> Optional[BlockRV]:
        """Apply the instruction GetBlock, get a block by its name

        Parameters
        ----------
        block : BlockRV
            The block to be queried

        Returns
        -------
        only_consumer : Optional[BlockRV]
            A block, its only consumer; or None if it does not exist
        """
        return _ffi_api.ScheduleGetOnlyConsumer(self, block)  # pylint: disable=no-member

    def get_block(self, name: str) -> BlockRV:
        """Apply the instruction GetBlock, get a block by its name

        Parameters
        ----------
        name : str
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
        block : BlockRV
            The block to be queried

        Returns
        -------
        axes : List[LoopRV]
            The loop nests above the block
        """
        return _ffi_api.ScheduleGetAxes(self, block)  # pylint: disable=no-member

    def get_read_buffers(self, block: BlockRV) -> List[BufferRV]:
        """Get the buffers the block reads

        Parameters
        ----------
        block : BlockRV
            The block to be queried

        Returns
        -------
        buffers : List[BufferRV]
            A list of buffers the block reads
        """
        return _ffi_api.ScheduleGetReadBuffers(self, block)  # pylint: disable=no-member

    def get_write_buffers(self, block: BlockRV) -> List[BufferRV]:
        """Get the buffers the block writes

        Parameters
        ----------
        block : BlockRV
            The block to be queried

        Returns
        -------
        buffers : List[BufferRV]
            A list of buffers the block writes
        """
        return _ffi_api.ScheduleGetWriteBuffers(self, block)  # pylint: disable=no-member

    def get_root_blocks(self) -> List[BlockRV]:
        """Get the root blocks which are direct children of the root node

        Returns
        ----------
        blocks : List[BlockRV]
            The direct childs block of the root node
        """
        return _ffi_api.ScheduleGetRootBlocks(self)  # pylint: disable=no-member

    def get_leaf_blocks(self) -> List[BlockRV]:
        """Get the leaf blocks who do not have any child block

        Returns
        ----------
        blocks : List[BlockRV]
            The direct childs block of the root node
        """
        return _ffi_api.ScheduleGetLeafBlocks(self)  # pylint: disable=no-member

    ########## Scheduling Primitives ##########

    def mark_loop_type(self, loops: List[LoopRV], mark: str, mark_range: ir.Range) -> None:
        """Mark a range of loops with the specific mark

        Parameters
        ----------
        loops: List[LoopRV]
            The loops to be marked
        mark : str
            The annotation
        mark_range : Range
            The range to be marked
        """
        _ffi_api.ScheduleMarkLoopType(self, loops, mark, mark_range)  # pylint: disable=no-member

    def mark_block_type(self, block: BlockRV, mark: str) -> None:
        """Mark a range of loops with the specific mark

        Parameters
        ----------
        block : BlockRV
            The block to be marked
        mark : str
            The annotation
        """
        _ffi_api.ScheduleMarkBlockType(self, block, mark)  # pylint: disable=no-member

    def fuse(self, loops: List[LoopRV]):
        """Fuse the loops

        Parameters
        ----------
        loops : List[LoopRV]
            The block to be queried

        Returns
        -------
        fused : LoopRV
            The fused loop
        """
        return _ffi_api.ScheduleFuse(self, loops)  # pylint: disable=no-member

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

    def compute_at(self, block: BlockRV, loop: LoopRV) -> None:
        """Move the block under the loop and regenerate the loops to cover the producing region.

        Parameters
        ----------
        block : BlockRV
            block The block to be moved
        loop : LoopRV
            loop The loop to be moved to
        """
        _ffi_api.ScheduleComputeAt(self, block, loop)  # pylint: disable=no-member

    def reverse_compute_at(self, block: BlockRV, loop: LoopRV) -> None:
        """Move the block under the loop and regenerate the loops to cover the producing region.

        Parameters
        ----------
        block : BlockRV
            block The block to be moved
        loop : LoopRV
            loop The loop to be moved to
        """
        _ffi_api.ScheduleReverseComputeAt(self, block, loop)  # pylint: disable=no-member

    def compute_inline(self, block: BlockRV) -> None:
        """Apply the instruction compute_inline

        Parameters
        ----------
        block: BlockRV
            The block to be computed inline
        """
        _ffi_api.ScheduleComputeInline(self, block)  # pylint: disable=no-member

    def reverse_compute_inline(self, block: BlockRV) -> None:
        """Apply the instruction reverse_compute_inline

        Parameters
        ----------
        block: BlockRV
            The block to be reverse computed inline
        """
        _ffi_api.ScheduleReverseComputeInline(self, block)  # pylint: disable=no-member

    def cache_read(self, buffer: BufferRV, storage_scope: str) -> BlockRV:
        """Apply the instruction cache_read

        Parameters
        ----------
        buffer: BufferRV
            The buffer to be cached
        storage_scope: str
            The storage scope

        Returns
        -------
        block : BlockRV
            The cache write stage
        """
        return _ffi_api.ScheduleCacheRead(self, buffer, storage_scope)  # pylint: disable=no-member

    def cache_write(self, buffer: BufferRV, storage_scope: str) -> BlockRV:
        """Apply the instruction cache_write

        Parameters
        ----------
        buffer: BufferRV
            The buffer to be cached
        storage_scope: str
            The storage scope

        Returns
        -------
        block : BlockRV
            The cache write stage
        """
        return _ffi_api.ScheduleCacheWrite(self, buffer, storage_scope)  # pylint: disable=no-member

    def blockize(self, loop: LoopRV, exec_scope: str = "") -> BlockRV:
        """Apply the instruction blockize

        Parameters
        ----------
        loop : LoopRV
            The loop to be blockized
        exec_scope: str
            The execution scope

        Returns
        -------
        block : BlockRV
            The new block
        """
        return _ffi_api.ScheduleBlockize(self, loop, exec_scope)  # pylint: disable=no-member

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
        return _ffi_api.ScheduleDecomposeReduction(self, block, loop)  # pylint: disable=no-member

    ########## Trace-related ##########

    def mutate_decision(
        self,
        inst: Instruction,
        decision: Optional[List[Object]],
    ) -> None:
        """Mutate the decision on the specific instruction

        Parameters
        ----------
        inst: Instruction
            The instruction whose decision is mutated
        decision: Optional[List[Object]]
            The decision to be mutated to. If it is None, then remove it from decisions
        """
        return _ffi_api.ScheduleMutateDecision(self, inst, decision)  # pylint: disable=no-member

    def resample(self) -> None:
        """Re-sample along the trace to generatea new sequence of
        scheduling instructions and program states"""
        return _ffi_api.ScheduleReSample(self)  # pylint: disable=no-member

    def replay_decision(self) -> None:
        """Replay the trace with the decision stored in the schedule class.
        If a decision has been changed using MutateDecision, then it will generate
        different schedule. This process is theoretically deterministic if all sampling
        instructions have decision made."""
        return _ffi_api.ScheduleReplayDecision(self)  # pylint: disable=no-member
