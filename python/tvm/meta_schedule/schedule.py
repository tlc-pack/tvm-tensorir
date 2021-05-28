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
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from tvm import tir
from tvm._ffi import register_object
from tvm.runtime.container import String
from tvm.tir import IntImm
from tvm.tir import Schedule as TIRSchedule
from tvm.tir.schedule import BlockRV, ExprRV, LoopRV

from . import _ffi_api
from .trace import Trace

if TYPE_CHECKING:
    from tvm.tir.schedule import ScheduleState, StmtSRef


@register_object("meta_schedule.Schedule")
class Schedule(TIRSchedule):
    """The meta schedule class.
    Parameters
    ----------
    orig_func : PrimFunc
        The original TIR PrimFunc to be scheduled
    """

    state: ScheduleState
    trace: Trace

    def __init__(  # pylint: disable=super-init-not-called
        self,
        func: tir.PrimFunc,
        seed: Optional[int] = None,
        debug_mode: Union[bool, int] = False,
    ):
        if isinstance(debug_mode, bool):
            if debug_mode:
                debug_mode = -1
            else:
                debug_mode = 0
        assert isinstance(debug_mode, int)
        if seed is None:
            seed = -1
        self.__init_handle_by_constructor__(
            _ffi_api.Schedule, func, seed, debug_mode  # pylint: disable=no-member
        )

    ######### Utility #########

    def copy(self, seed: int) -> Schedule:  # pylint: disable=arguments-differ
        """Copy the schedule into a new one.
        Operation on the new schedule won't affect the original schedule, and vice versa.
        Returns
        -------
        new_schedule : Schedule
            A new schedule
        """
        return _ffi_api.ScheduleCopy(self, seed)  # pylint: disable=no-member

    ########## Schedule: compute location ##########

    def compute_at(
        self,
        block: BlockRV,
        loop: LoopRV,
        preserve_unit_loop: bool = True,
    ) -> None:
        return super().compute_at(block, loop, preserve_unit_loop)

    def reverse_compute_at(
        self,
        block: BlockRV,
        loop: LoopRV,
        preserve_unit_loop: bool = True,
    ) -> None:
        return super().reverse_compute_at(block, loop, preserve_unit_loop)

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
        _ffi_api.ScheduleMarkLoop(self, loop, ann_key, ann_val)  # pylint: disable=no-member

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
        _ffi_api.ScheduleMarkBlock(self, block, ann_key, ann_val)  # pylint: disable=no-member
