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
"""Schedule nodes and APIs in TE IR"""
from .. import NodeBase
from .util import register_te_node
from ..api import _init_api


@register_te_node
class Schedule(NodeBase):
    """The schedule node for TE IR"""

    # Utils
    def blocks(self, scope=None):
        """Return all blocks in the schedule

        Returns
        -------
        blocks : List of TeBlock or TeBlock
            The blocks in the schedule
        scope: StmtSRef, optional
            The scope block stmt sref
        """
        blocks = ScheduleBlocks(self, scope)
        if len(blocks) == 0:
            return None
        elif len(blocks) == 1:
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
        if len(blocks) == 0:
            return None
        elif len(blocks) == 1:
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
        axes = ScheduleGetAxes(self, block)
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

    def replace(self, sref, target_stmt):
        """Replace a subtree of AST with new stmt
        and auto maintain the schedulable reference tree

        Parameters
        ----------
        sref: StmtSRef
            The stmt schedulable reference of the Stmt to be replaced

        target_stmt: Stmt
            The target stmt

        """
        return Replace(self, sref, target_stmt)

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

    def fuse(self, outer_axis, inner_axis):
        """Return all axes of the specific block

        Parameters
        ----------
        outer_axis: Loop
            The outer axis

        inner_axis: Loop
            The inner axis

        Returns
        -------
        axis: Loop
            The fused axis
        """
        return ScheduleFuse(self, outer_axis, inner_axis)

    def split(self, axis, factor=None, nparts=None):
        """split a specified axis into two axises by factor or nparts

        Parameters
        ----------
        axis: Loop
            The axis to be split

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
            outer, inner = ScheduleSplitByNParts(self, axis, nparts)
        else:
            if factor is None:
                raise ValueError("Either nparts or factor need to be provided")
            outer, inner = ScheduleSplitByFactor(self, axis, factor)
        return outer, inner

    def compute_inline(self, block):
        """Mark one stage as inline, then the body of computation will be expanded and
        inserted at the address where the tensor is required.

        Parameters
        ----------
        block: Block
            The Block to be inlined

        """
        return ScheduleComputeInline(self, block)

def create_schedule(func):
    """Create a schedule for a function

    Parameters
    ----------
    func: TeFunction

    Returns
    ------
    schedule: te.Schedule
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


@register_te_node
class StmtSRef(NodeBase):
    """The schedulable reference node for TE IR"""


_init_api('tvm.te.schedule')
