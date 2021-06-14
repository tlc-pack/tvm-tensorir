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
# pylint: disable=unused-import
"""The schedule class"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from tvm._ffi import register_object as _register_object
from tvm.ir import IRModule, PrimExpr
from tvm.runtime import Object, String
from tvm.tir import Block, For, IntImm, IterVar, PrimFunc, Stmt, TensorIntrin, Var

from . import _ffi_api_schedule
from .state import ScheduleState, StmtSRef


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

    def __init__(
        self,
        func_or_mod: Union[PrimFunc, IRModule],
        debug_mode: Union[bool, int] = False,
    ):
        if isinstance(debug_mode, bool):
            if debug_mode:
                debug_mode = -1
            else:
                debug_mode = 0
        assert isinstance(debug_mode, int)
        self.__init_handle_by_constructor__(
            _ffi_api_schedule.Schedule,  # pylint: disable=no-member
            func_or_mod,
            -1,  # seed
            debug_mode,
        )

    @property
    def mod(self) -> IRModule:
        return _ffi_api_schedule.ScheduleModule(self)  # pylint: disable=no-member

    @property
    def state(self) -> ScheduleState:
        return _ffi_api_schedule.ScheduleGetState(self)  # pylint: disable=no-member

    def show(self, rand_var: RAND_VAR_TYPE) -> str:
        # TODO(@junrushao1994): complete it
        return str(self.get(rand_var))

    ########## Utilities ##########

    def copy(self) -> Schedule:
        return _ffi_api_schedule.ScheduleCopy(self)  # pylint: disable=no-member

    def seed(self, seed: int) -> Schedule:
        return _ffi_api_schedule.ScheduleSeed(self, seed)  # pylint: disable=no-member

    ########## Lookup ##########

    def get(self, rand_var: RAND_VAR_TYPE) -> Optional[Union[int, Block, For]]:
        if isinstance(rand_var, StmtSRef):
            return rand_var.stmt
        result = _ffi_api_schedule.ScheduleGet(self, rand_var)  # pylint: disable=no-member
        if isinstance(result, IntImm):
            result = result.value
        return result

    def get_sref(self, rand_var_or_stmt: Union[RAND_VAR_TYPE, Stmt]) -> Optional[StmtSRef]:
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

    def set_scope(self, block: BlockRV, i: int, storage_scope: str) -> None:
        _ffi_api_schedule.ScheduleSetScope(  # pylint: disable=no-member
            self, block, i, storage_scope
        )

    def pragma(self, loop: LoopRV, pragma_type: str, pragma_value: ExprRV) -> None:
        if isinstance(pragma_value, bool):
            pragma_value = IntImm("bool", pragma_value)
        _ffi_api_schedule.SchedulePragma(  # pylint: disable=no-member
            self, loop, pragma_type, pragma_value
        )

    def storage_align(
        self,
        block: BlockRV,
        buffer_index: int,
        axis: int,
        factor: int,
        offset: int,
    ) -> None:
        _ffi_api_schedule.ScheduleStorageAlign(  # pylint: disable=no-member
            self, block, buffer_index, axis, factor, offset
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

    def blockize(self, loop: LoopRV) -> BlockRV:
        return _ffi_api_schedule.ScheduleBlockize(self, loop)  # pylint: disable=no-member

    def tensorize(self, loop: LoopRV, intrin: Union[str, TensorIntrin]) -> None:
        if isinstance(intrin, str):
            intrin = String(intrin)
        _ffi_api_schedule.ScheduleTensorize(self, loop, intrin)  # pylint: disable=no-member

    ########## Schedule: Marks and NO-OPs ##########

    def mark_loop(
        self,
        loop: LoopRV,
        ann_key: str,
        ann_val: str,
    ) -> None:
        """Mark a range of loops with the specific mark

        Parameters
        ----------
        loop: LoopRV
            The loops to be marked
        ann_key : str
            The annotation key
        ann_val : str
            The annotation value
        """
        if isinstance(ann_val, str):
            ann_val = String(ann_val)
        elif isinstance(ann_val, int):
            ann_val = IntImm("int64", ann_val)
        _ffi_api_schedule.ScheduleMarkLoop(  # pylint: disable=no-member
            self, loop, ann_key, ann_val
        )

    def mark_block(
        self,
        block: BlockRV,
        ann_key: str,
        ann_val: ExprRV,
    ) -> None:
        """Mark a block

        Parameters
        ----------
        block : BlockRV
            The block to be marked
        ann_key : str
            The annotation key
        ann_val : ExprRV
            The annotation value
        """
        _ffi_api_schedule.ScheduleMarkBlock(  # pylint: disable=no-member
            self, block, ann_key, ann_val
        )

    ########## Schedule: Misc ##########

    def inline_argument(self, i: int, func_name: str = "main"):
        _ffi_api_schedule.ScheduleInlineArgument(self, i, func_name)  # pylint: disable=no-member
