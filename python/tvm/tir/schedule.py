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
"""Schedule nodes and APIs in TIR schedule"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from tvm._ffi import register_object as _register_object
from tvm.ir import PrimExpr
from tvm.runtime import Object, String

from . import _ffi_api_schedule
from .expr import IntImm, Var

if TYPE_CHECKING:
    from tvm.tir import Block, For, IterVar, PrimFunc, Stmt, TensorIntrin


@_register_object("tir.StmtSRef")
class StmtSRef(Object):
    """The schedulable reference node for TIR"""

    @property
    def stmt(self) -> Optional[Union[Block, For]]:
        return _ffi_api_schedule.StmtSRefStmt(self)  # pylint: disable=no-member

    @staticmethod
    def root() -> StmtSRef:
        return _ffi_api_schedule.StmtSRefRootMark()  # pylint: disable=no-member

    @staticmethod
    def inline() -> StmtSRef:
        return _ffi_api_schedule.StmtSRefInlineMark()  # pylint: disable=no-member


@_register_object("tir.DepEdge")
class DepEdge(Object):
    """An edge in the dependency graph"""

    kRAW = 0
    kWAW = 1
    kWAR = 2
    kOpaque = 3

    dst: StmtSRef
    type: int


@_register_object("tir.BlockScope")
class BlockScope(Object):
    """Dependency Graph that stores read/write dependency between Blocks"""

    def get_predecessors(self, block: StmtSRef) -> List[DepEdge]:
        """Get the dependency predecessors of the block

        Parameters
        ----------
        block: StmtSRef
            The queried block

        Returns
        -------
        blocks: List of DepEdge
            The predecessors of the block
        """
        return _ffi_api_schedule.BlockScopeGetPredecessors(self, block)  # pylint: disable=no-member

    def get_successor(self, block: StmtSRef) -> List[DepEdge]:
        """Get the dependency successor of the block

        Parameters
        ----------
        block: StmtSRef
            The queried block

        Returns
        -------
        blocks: List of DepEdge
            The predecessors of the block
        """
        return _ffi_api_schedule.BlockScopeGetSuccessors(self, block)  # pylint: disable=no-member


@_register_object("tir.ScheduleState")
class ScheduleState(Object):
    """The state of scheduling"""

    func: PrimFunc
    root: StmtSRef
    debug_mode: bool

    def __init__(self, func: PrimFunc, debug_mode: bool):
        self.__init_handle_by_constructor__(
            _ffi_api_schedule.ScheduleState,  # pylint: disable=no-member
            func,
            debug_mode,
        )

    def get_sref(self, stmt: Stmt) -> Optional[StmtSRef]:
        return _ffi_api_schedule.ScheduleStateGetSRef(self, stmt)  # pylint: disable=no-member

    def scope(self, block: StmtSRef) -> BlockScope:
        return _ffi_api_schedule.ScheduleStateGetScope(self, block)  # pylint: disable=no-member

    def replace(
        self,
        src_sref: StmtSRef,
        tgt_stmt: Stmt,
        block_sref_reuse: Optional[Dict[Block, Block]] = None,
    ) -> None:
        if block_sref_reuse is None:
            block_sref_reuse = {}
        _ffi_api_schedule.ScheduleStateReplace(  # pylint: disable=no-member
            self,
            src_sref,
            tgt_stmt,
            block_sref_reuse,
        )


@_register_object("tir.LoopRV")
class LoopRV(Object):
    """A random variable that refers to a loop"""


@_register_object("tir.BlockRV")
class BlockRV(Object):
    """A random variable that refers to a block"""


VarRV = Var

ExprRV = PrimExpr

RAND_VAR_TYPE = Union[ExprRV, BlockRV, LoopRV]  # pylint: disable=invalid-name


@_register_object("tir.Schedule")
class Schedule(Object):
    """The schedule node for TIR"""

    state: ScheduleState

    def __init__(self, func: PrimFunc, debug_mode: bool = False):
        self.__init_handle_by_constructor__(
            _ffi_api_schedule.Schedule,  # pylint: disable=no-member
            func,
            -1,  # seed
            debug_mode,
        )

    @property
    def module(self) -> PrimFunc:
        return self.state.func

    def show(self, rand_var: Union[LoopRV, BlockRV, ExprRV]) -> str:
        # TODO(@junrushao1994): complete it
        return str(self.get(rand_var))

    ########## Utilities ##########

    def copy(self) -> Schedule:
        return _ffi_api_schedule.ScheduleCopy(self)  # pylint: disable=no-member

    def seed(self, seed: int) -> Schedule:
        return _ffi_api_schedule.ScheduleSeed(self, seed)  # pylint: disable=no-member

    ########## Lookup ##########

    def get(self, rand_var: Union[LoopRV, BlockRV, ExprRV]) -> Optional[Union[int, Block, For]]:
        if isinstance(rand_var, StmtSRef):
            return rand_var.stmt
        result = _ffi_api_schedule.ScheduleGet(self, rand_var)  # pylint: disable=no-member
        if isinstance(result, IntImm):
            result = result.value
        return result

    def get_sref(self, rand_var_or_stmt: Union[BlockRV, LoopRV, Stmt]) -> Optional[StmtSRef]:
        return _ffi_api_schedule.ScheduleGetSRef(  # pylint: disable=no-member
            self, rand_var_or_stmt
        )

    ########## Sampling ##########

    def sample_perfect_tile(
        self,
        loop: LoopRV,
        n: int,
        max_innermost_factor: int = 16,
        decision: Optional[List[int]] = None,
    ) -> List[VarRV]:
        return _ffi_api_schedule.ScheduleSamplePerfectTile(  # pylint: disable=no-member
            self,
            loop,
            n,
            max_innermost_factor,
            decision,
        )

    def sample_categorical(
        self,
        candidates: List[int],
        probs: List[float],
        decision: Optional[int] = None,
    ) -> VarRV:
        return _ffi_api_schedule.ScheduleSampleCategorical(  # pylint: disable=no-member
            self,
            candidates,
            probs,
            decision,
        )

    def sample_compute_location(
        self,
        block: BlockRV,
        decision: Optional[int] = None,
    ) -> LoopRV:
        return _ffi_api_schedule.ScheduleSampleComputeLocation(  # pylint: disable=no-member
            self,
            block,
            decision,
        )

    ########## Block/Loop relation ##########

    def get_block(self, name: str) -> BlockRV:
        return _ffi_api_schedule.ScheduleGetBlock(self, name)  # pylint: disable=no-member

    def get_axes(self, block: BlockRV) -> List[LoopRV]:
        return _ffi_api_schedule.ScheduleGetAxes(self, block)  # pylint: disable=no-member

    def get_child_blocks(self, block_or_loop: Union[BlockRV, LoopRV]) -> List[BlockRV]:
        return _ffi_api_schedule.ScheduleGetChildBlocks(  # pylint: disable=no-member
            self, block_or_loop
        )

    def get_producers(self, block: BlockRV) -> List[BlockRV]:
        return _ffi_api_schedule.ScheduleGetProducers(self, block)  # pylint: disable=no-member

    def get_consumers(self, block: BlockRV) -> List[BlockRV]:
        return _ffi_api_schedule.ScheduleGetConsumers(self, block)  # pylint: disable=no-member

    ########## Schedule: loops ##########

    def fuse(self, *loops: List[LoopRV]) -> LoopRV:
        return _ffi_api_schedule.ScheduleFuse(self, loops)  # pylint: disable=no-member

    def split(
        self,
        loop: LoopRV,
        *,
        nparts: Optional[ExprRV] = None,
        factor: Optional[ExprRV] = None,
        factors: Optional[List[ExprRV]] = None,
    ) -> Tuple[LoopRV, LoopRV]:
        if factors is not None:
            if (nparts is not None) or (factor is not None):
                raise ValueError("`nparts`/`factor` are not allowed when `factors` is specified")
        elif (nparts is None) and (factor is None):
            raise ValueError("None of the `nparts`, `factor` and `factors` are specified")
        elif (nparts is not None) and (factor is not None):
            raise ValueError("Only one of the `nparts`, `factor` are allowed to be specified")
        else:
            factors = [nparts, factor]
        return _ffi_api_schedule.ScheduleSplit(self, loop, factors)  # pylint: disable=no-member

    def reorder(self, *loops: List[LoopRV]) -> None:
        _ffi_api_schedule.ScheduleReorder(self, loops)  # pylint: disable=no-member

    ########## Schedule: compute location ##########

    def compute_at(
        self,
        block: BlockRV,
        loop: LoopRV,
        preserve_unit_loop: bool = False,
    ) -> None:
        _ffi_api_schedule.ScheduleComputeAt(  # pylint: disable=no-member
            self, block, loop, preserve_unit_loop
        )

    def reverse_compute_at(
        self,
        block: BlockRV,
        loop: LoopRV,
        preserve_unit_loop: bool = False,
    ) -> None:
        _ffi_api_schedule.ScheduleReverseComputeAt(  # pylint: disable=no-member
            self, block, loop, preserve_unit_loop
        )

    def compute_inline(self, block: BlockRV) -> None:
        _ffi_api_schedule.ScheduleComputeInline(self, block)  # pylint: disable=no-member

    def reverse_compute_inline(self, block: BlockRV) -> None:
        _ffi_api_schedule.ScheduleReverseComputeInline(self, block)  # pylint: disable=no-member

    ########## Schedule: parallelize / annotate ##########

    def vectorize(self, loop: LoopRV) -> None:
        _ffi_api_schedule.ScheduleVectorize(self, loop)  # pylint: disable=no-member

    def parallel(self, loop: LoopRV) -> None:
        _ffi_api_schedule.ScheduleParallel(self, loop)  # pylint: disable=no-member

    def unroll(self, loop: LoopRV) -> None:
        _ffi_api_schedule.ScheduleUnroll(self, loop)  # pylint: disable=no-member

    def bind(self, loop: LoopRV, thread: Union[str, IterVar]) -> None:
        if isinstance(thread, str):
            thread = String(thread)
        _ffi_api_schedule.ScheduleBind(self, loop, thread)  # pylint: disable=no-member

    def double_buffer(self, block: BlockRV) -> None:
        _ffi_api_schedule.ScheduleDoubleBuffer(self, block)  # pylint: disable=no-member

    def pragma(self, loop: LoopRV, pragma_type: str, pragma_value: ExprRV) -> None:
        if isinstance(pragma_value, bool):
            pragma_value = IntImm("bool", pragma_value)
        _ffi_api_schedule.SchedulePragma(  # pylint: disable=no-member
            self, loop, pragma_type, pragma_value
        )

    ########## Schedule: cache read/write ##########

    def cache_read(self, block: BlockRV, i: int, storage_scope: str) -> BlockRV:
        return _ffi_api_schedule.ScheduleCacheRead(  # pylint: disable=no-member
            self, block, i, storage_scope
        )

    def cache_write(self, block: BlockRV, i: int, storage_scope: str) -> BlockRV:
        return _ffi_api_schedule.ScheduleCacheWrite(  # pylint: disable=no-member
            self, block, i, storage_scope
        )

    ########## Schedule: reduction ##########

    def rfactor(self, loop: LoopRV, factor: int) -> LoopRV:
        return _ffi_api_schedule.ScheduleRFactor(self, loop, factor)  # pylint: disable=no-member

    def decompose_reduction(self, block: BlockRV, loop: Optional[LoopRV]) -> BlockRV:
        return _ffi_api_schedule.ScheduleDecomposeReduction(  # pylint: disable=no-member
            self, block, loop
        )

    def merge_reduction(self, init: BlockRV, update: BlockRV) -> None:
        _ffi_api_schedule.ScheduleMergeReduction(self, init, update)  # pylint: disable=no-member

    ########## Schedule: blockize / tensorize ##########

    def blockize(self, loop: LoopRV, exec_scope: str = "") -> BlockRV:
        return _ffi_api_schedule.ScheduleBlockize(  # pylint: disable=no-member
            self, loop, exec_scope
        )

    def tensorize(self, loop: LoopRV, intrin: Union[str, TensorIntrin]) -> None:
        if isinstance(intrin, str):
            intrin = String(intrin)
        _ffi_api_schedule.ScheduleTensorize(self, loop, intrin)  # pylint: disable=no-member
