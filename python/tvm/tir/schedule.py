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
import tvm._ffi
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
        blocks = ScheduleBlocks(self, scope)
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
            blocks = GetBlocksFromTag(self, arg, scope)
        else:
            blocks = GetBlocksFromBuffer(self, arg, scope)
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
        axes = ScheduleGetLoopsInScope(self, block)
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
        return GetStmtSRef(self, stmt)

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
        return Replace(self, sref, target_stmt, block_sref_map)

    def validate_sref(self):
        return ValidateSRef(self)

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
        return GetSuccessors(self, scope, block)

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
        return GetPredecessors(self, scope, block)

    def reorder(self, *args):
        """reorder the arguments in the specified order
        Parameters
        ----------
        args: list of Loop
            The order to be ordered
        """

        ScheduleReorder(self, args)

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
        return ScheduleFuse(self, outer_loop, inner_loop)

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
            outer, inner = ScheduleSplitByNParts(self, loop, nparts)
        else:
            if factor is None:
                raise ValueError("Either nparts or factor need to be provided")
            outer, inner = ScheduleSplitByFactor(self, loop, factor)
        return outer, inner

    def bind(self, loop, thread_ivar):
        """Bind ivar to thread index thread_ivar

        Parameters
        ----------
        loop : Loop
            The loop to be binded to thread.

        thread_ivar : IterVar
            The thread to be binded.
        """
        ScheduleBind(self, loop, thread_ivar)

    def vectorize(self, loop):
        """vectorize a loop
        Parameters
        ----------
        loop : Loop
            The loop to be vectorized
        """
        ScheduleVectorize(self, loop)

    def parallel(self, loop):
        """parallel a loop
        Parameters
        ----------
        loop : Loop
            The loop to be paralleled
        """
        ScheduleParallel(self, loop)

    def unroll(self, loop):
        """unroll a loop
        Parameters
        ----------
        loop : Loop
            The loop to be unrolled
        """
        ScheduleUnroll(self, loop)

    def cache_read(self, buffer, scope):
        """Create a cache read of original tensor for readers.
        Parameters
        ----------
        buffer : Buffer
            The buffer to be cache_read
        scope : str
            The storage scope
        """
        return ScheduleCacheRead(self, buffer, scope)

    def cache_write(self, buffer, scope):
        """Create a cache write of original tensor, before storing into tensor.
        Parameters
        ----------
        buffer : Buffer
            The buffer to be cache_written
        scope : str
            The storage scope
        """
        return ScheduleCacheWrite(self, buffer, scope)

    def compute_inline(self, block):
        """Mark one stage as inline, then the body of computation will be expanded and
        inserted at the address where the tensor is required.
        Parameters
        ----------
        block: Block
            The Block to be inlined
        """
        return ScheduleComputeInline(self, block)

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
        ScheduleComputeAt(self, block, loop)

    def decompose_reduction(self, block, loop):
        """ Decompose reduction block into init&update blocks

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
        return ScheduleDecomposeReduction(self, block, loop)

    def merge_reduction(self, init, update):
        """ Merge init&update block into reduction block

        Parameters
        ----------
        init: Block
            The init Block
        update: Block
            The update Block
        """
        ScheduleMergeReduction(self, init, update)

    def register_reducer(self, fcombine, identity):
        """ Register a reducer into schedule

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
        ScheduleRegisterReducer(self, combiner)


def create_schedule(func):
    """Create a schedule for a function
    Parameters
    ----------
    func: TIRFunction
    Returns
    ------
    schedule: tir.Schedule
    """
    return CreateSchedule(func)


def get_stmt(sref):
    """Get Stmt from sref
    Parameters
    ----------
    sref: StmtSRef
    Returns
    ------
    stmt: stmt
    """
    return GetStmt(sref)


@tvm._ffi.register_object
class StmtSRef(Object):
    """The schedulable reference node for TIR"""


@tvm._ffi.register_object
class BlockSRef(Object):
    """The schedulable reference node for TIR blocks"""


@tvm._ffi.register_object
class LoopSRef(Object):
    """The schedulable reference node for TIR loops"""


tvm._ffi._init_api("tir.schedule", __name__)
