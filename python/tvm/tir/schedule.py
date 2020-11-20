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
import tvm
from . import _ffi_api_schedule
from tvm.runtime import Object


@tvm._ffi.register_object("tir.Schedule")
class Schedule(Object):
    """The schedule node for TIR"""

    # Utils
    def blocks(self, scope=None):
        """Return all blocks in the schedule
        Returns
        -------
        blocks : List of Block or Block
            The blocks in the schedule
        scope: StmtSRef, optional
            The scope block stmt sref
        """
        blocks = _ffi_api_schedule.ScheduleBlocks(self, scope)
        if not blocks:
            return None
        if len(blocks) == 1:
            blocks = blocks[0]
        return blocks

    def get_block(self, arg, scope=None):
        """Return blocks with queried patten
        Parameters
        ----------
        arg: str or Buffer
            The queried arguments
        scope: StmtSRef, optional
            The scope block stmt sref
        Returns
        -------
        blocks : List of StmtSRef or StmtSRef
            The blocks sref that match the arguments
        """
        if isinstance(arg, str):
            blocks = _ffi_api_schedule.GetBlocksFromTag(self, arg)
        else:
            blocks = _ffi_api_schedule.GetBlocksFromBuffer(self, arg, scope)
        if not blocks:
            return None
        if len(blocks) == 1:
            blocks = blocks[0]
        return blocks

    def get_axes(self, block):
        """Return all axes of the specific block
        Parameters
        ----------
        block: StmtSRef
            The queried block sref
        Returns
        -------
        axes: List of StmtSRef or StmtSRef
            The axes of the block
        """
        axes = _ffi_api_schedule.ScheduleGetLoopsInScope(self, block)
        if len(axes) == 1:
            axes = axes[0]
        return axes

    def get_sref(self, stmt):
        """Get the stmt schedulable reference of the specific stmt
        Parameters
        ----------
        stmt: Stmt
            The Stmt to be queried
        Returns
        -------
        sref : StmtSRef
            The stmt schedulable reference
        """
        return _ffi_api_schedule.GetStmtSRef(self, stmt)

    def replace(self, sref, target_stmt, block_sref_map=None):
        """Replace a subtree of AST with new stmt
        and auto maintain the schedulable reference tree
        Parameters
        ----------
        sref: StmtSRef
            The stmt schedulable reference of the Stmt to be replaced
        target_stmt: Stmt
            The target stmt
        block_sref_map: Map
            The remap of block_sref
        """
        return _ffi_api_schedule.Replace(self, sref, target_stmt, block_sref_map)

    def validate_sref(self):
        return _ffi_api_schedule.ValidateSRef(self)

    # Dependency
    def get_successors(self, block, scope=None):
        """Get the dependency successors of the block
        Parameters
        ----------
        block: StmtSRef
            The queried block
        scope: StmtSRef
            The scope
        Returns
        -------
        blocks: List of StmtSRef or StmtSRef
            The successors of the block
        """

        if scope is None:
            scope = self.root
        return _ffi_api_schedule.GetSuccessors(self, scope, block)

    def get_predecessors(self, block, scope=None):
        """Get the dependency predecessors of the block
        Parameters
        ----------
        block: StmtSRef
            The queried block
        scope: StmtSRef
            The scope
        Returns
        -------
        blocks: List of StmtSRef or StmtSRef
            The predecessors of the block
        """

        if scope is None:
            scope = self.root
        return _ffi_api_schedule.GetPredecessors(self, scope, block)

    def reorder(self, *args):
        """reorder the arguments in the specified order
        Parameters
        ----------
        args: list of Loop
            The order to be ordered
        """

        _ffi_api_schedule.ScheduleReorder(self, args)

    def fuse(self, outer_loop, inner_loop):
        """Return all axes of the specific block
        Parameters
        ----------
        outer_loop: Loop
            The outer loop

        inner_loop: Loop
            The inner loop

        Returns
        -------
        loop: Loop
            The fused loop
        """
        return _ffi_api_schedule.ScheduleFuse(self, outer_loop, inner_loop)

    def split(self, loop, factor=None, nparts=None):
        """split a specified loop into two loops by factor or nparts

        Parameters
        ----------
        loop: Loop
            The loop to be split

        factor : Expr, optional
             The splitting factor
        nparts : Expr, optional
             The number of outer parts.
        Returns
        -------
        outer : Loop
            The outer loop.
        inner : Loop
            The inner loop.
        """
        if nparts is not None:
            if factor is not None:
                raise ValueError("Do not need to provide both outer and nparts")
            outer, inner = _ffi_api_schedule.ScheduleSplitByNParts(self, loop, nparts)
        else:
            if factor is None:
                raise ValueError("Either nparts or factor need to be provided")
            outer, inner = _ffi_api_schedule.ScheduleSplitByFactor(self, loop, factor)
        return outer, inner

    def vectorize(self, loop):
        """vectorize a loop
        Parameters
        ----------
        loop : Loop
            The loop to be vectorized
        """
        _ffi_api_schedule.ScheduleVectorize(self, loop)

    def parallel(self, loop):
        """parallel a loop
        Parameters
        ----------
        loop : Loop
            The loop to be paralleled
        """
        _ffi_api_schedule.ScheduleParallel(self, loop)

    def unroll(self, loop):
        """unroll a loop
        Parameters
        ----------
        loop : Loop
            The loop to be unrolled
        """
        _ffi_api_schedule.ScheduleUnroll(self, loop)

    def cache_read(self, block, index, scope):
        """Create a cache read of original tensor for readers.
        Parameters
        ----------
        block : Block
            The consumer of the buffer
        index : int
            The index of the buffer in block's read region
        scope : str
            The storage scope
        """
        return _ffi_api_schedule.ScheduleCacheRead(self, block, index, scope)

    def cache_write(self, block, index, scope):
        """Create a cache write of original tensor, before storing into tensor.
        Parameters
        ----------
        block : Block
            The write block of the buffer to be cache_writte
        index : int
            The index of the buffer in block's write region
        scope : str
            The storage scope
        """
        return _ffi_api_schedule.ScheduleCacheWrite(self, block, index, scope)

    def compute_inline(self, block):
        """Mark one block as inline, then the body of computation will be expanded and
        inserted at the address where the tensor is required.
        Parameters
        ----------
        block: Block
            The Block to be inlined
        """
        return _ffi_api_schedule.ScheduleComputeInline(self, block)

    def reverse_compute_inline(self, block):
        """Mark one block as inline, then the body of computation will be expanded and
        inserted at the address where the tensor is required.
        Parameters
        ----------
        block: Block
            The Block to be inlined
        """
        return _ffi_api_schedule.ScheduleReverseComputeInline(self, block)

    def compute_at(self, block, loop):
        """Attach one block under specific loop and cover the required region.
        Node that only complete block can do compute_at

        Parameters
        ----------
        block: Block
            The Block to be compute_at

        loop: Loop
            The target loop

        """
        _ffi_api_schedule.ScheduleComputeAt(self, block, loop)

    def reverse_compute_at(self, block, loop):
        """Attach one block under specific loop and cover the required region.
        Node that only complete block can do reverse_compute_at

        Parameters
        ----------
        block: Block
            The Block to be reverse_compute_at

        loop: Loop
            The target loop

        Example
        -------
        .. code-block:: python

            for i0_outer, i1_outer, i0_inner, i1_inner in tir.grid(8, 8, 16, 16):
                with tir.block([128, 128], "B") as [vi, vj]:
                    tir.bind(vi, ((i0_outer*16) + i0_inner))
                    tir.bind(vj, ((i1_outer*16) + i1_inner))
                    B[vi, vj] = A[vi, vj] * 2 .0
            with tir.block([128, 128], "C") as [vi, vj]:
                C[vi, vj] = B[vi, vj] + 1.0

        After reverse_compute_at(C, i0_inner)
        .. code-block:: python

            for i0_outer, i1_outer, i1_inner in tir.grid(8, 8, 16):
                for i1_inner in range(0, 16):
                    with tir.block([128, 128], "B") as [vi, vj]:
                        tir.bind(vi, ((i0_outer*16) + i0_inner))
                        tir.bind(vj, ((i1_outer*16) + i1_inner))
                        B[vi, vj] = A[vi, vj] * 2.0
                for ax1 in range(0, 16):
                    with tir.block([128, 128], "C") as [vi, vj]:
                        tir.bind(vi, ((i0_outer*16) + i0_inner))
                        tir.bind(vj, ((i1_outer*16) + ax1))
                        C[vi, vj] = B[vi, vj] + 1.0
        """
        _ffi_api_schedule.ScheduleReverseComputeAt(self, block, loop)

    def bind(self, loop, thread_ivar):
        """Bind ivar to thread index thread_ivar
        Parameters
        ----------
        loop : Loop
            The loop to be binded to thread.
        thread_ivar : IterVar
            The thread to be binded.
        """
        _ffi_api_schedule.ScheduleBind(self, loop, thread_ivar)

    def blockize(self, loop):
        """make subtree rooted by sref into a block
        Parameters
        ----------
        loop: Loop
            the subtree root
        Returns
        -------
        block: Block
            The new block
        """
        return _ffi_api_schedule.ScheduleBlockize(self, loop)

    def tensorize(self, loop, intrinsic):
        """Tensorize the computation enclosed by loop with tensor_intrin
        Parameters
        ----------
        loop: Loop
            the subtree root
        intrinsic: Intrinsic
            the tensor intrinsic
        """
        _ffi_api_schedule.ScheduleTensorize(self, loop, intrinsic)

    def decompose_reduction(self, block, loop):
        """Decompose reduction block into init&update blocks

        Parameters
        ----------
        block: Block
            The reduction block
        loop: Loop
            The position where init block will be
        Returns
        -------
        init: Block
            The init block
        """
        return _ffi_api_schedule.ScheduleDecomposeReduction(self, block, loop)

    def merge_reduction(self, init, update):
        """Merge init&update block into reduction block

        Parameters
        ----------
        init: Block
            The init Block
        update: Block
            The update Block
        """
        _ffi_api_schedule.ScheduleMergeReduction(self, init, update)

    def register_reducer(self, fcombine, identity):
        """Register a reducer into schedule

        Parameters
        ----------
        fcombine : lambda expression
            the combiner function of reducer
        identity : PrimExpr
            the identity of reducer
        """
        code = fcombine.__code__
        lvar = tvm.te.var(code.co_varnames[0])
        rvar = tvm.te.var(code.co_varnames[1])
        lhs = tvm.runtime.convert([lvar])
        rhs = tvm.runtime.convert([rvar])
        result = tvm.runtime.convert([fcombine(lvar, rvar)])
        id_elem = tvm.runtime.convert([identity])
        combiner = tvm.tir.CommReducer(lhs, rhs, result, id_elem)
        _ffi_api_schedule.ScheduleRegisterReducer(self, combiner)

    def rfactor(self, loop, factor_axis):
        """rfactor a reduction block using loop

        Parameters
        ----------
        loop : StmtSRef
            the loop outside block we want to do rfactor

        factor_axis : int
            the position where the new axis is placed

        Returns
        -------
        new_block : StmtSRef
             the sref of new block
        """
        return _ffi_api_schedule.ScheduleRfactor(self, loop, factor_axis)


def create_schedule(func):
    """Create a schedule for a function
    Parameters
    ----------
    func: TIRFunction
    Returns
    ------
    schedule: tir.Schedule
    """
    return _ffi_api_schedule.CreateSchedule(func)


def validate_hierarchy(func):
    """Validate whether a func satisfies the hierarchy constraints
    Parameters
    ----------
    func: TIRFunction
    """
    _ffi_api_schedule.ValidateHierarchy(func)


def get_stmt(sref):
    """Get Stmt from sref
    Parameters
    ----------
    sref: StmtSRef
    Returns
    ------
    stmt: stmt
    """
    return _ffi_api_schedule.GetStmt(sref)


@tvm._ffi.register_object
class StmtSRef(Object):
    """The schedulable reference node for TIR"""

    @property
    def stmt(self):
        return _ffi_api_schedule.GetStmt(self)


tvm._ffi._init_api("tir.schedule", __name__)
