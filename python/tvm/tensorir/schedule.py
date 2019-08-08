from ..expr import Expr
from ..api import _init_api
from .._ffi.node import NodeBase, register_node

from .tree_node import register_tensorir_node

@register_tensorir_node
class Schedule(NodeBase):

    ##### Getters #####
    def blocks(self):
        """Return all blocks in the schedule
        
        Returns
        -------
        blocks : List of BlockTreeNode
        """
        return ScheduleBlocks(self)

    def output_blocks(self):
        """Return all output blocks in the schedule
        Definition of output block: The last blocks that write to output tensors

        Returns
        -------
        blocks : List of BlockTreeNode
        """
        return ScheduleOutputBlocks(self)

    def reduction_blocks(self):
        """Get all blocks that perform reduction computation

        Returns
        -------
        blocks : List of BlockTreeNode
            All reduction blocks
        """
        return ScheduleReductionBlocks(self)

    def axis(self, node):
        """Return all axes in the path which starts from this block to the root node

        Returns
        -------
        axes : List of AxisTreeNode
            All axes in the path to the root node
        """
        return ScheduleAxis(self, node)

    def predecessor(self, block):
        """Return all predecessor blocks of a block in the dependency graph

        Returns
        -------
        block : List of BlockTreeNode
            Successor blocks
        """
        return SchedulePredecessor(self, block)

    def successor(self, block):
        """Return all successor blocks of a block in the dependency graph

        Returns
        -------
        block : List of BlockTreeNode
            Successor blocks
        """
        return ScheduleSuccessorBlocks(self, block)

    ##### Schedule Primitives #####
    def split(self, node, factor=None, nparts=None):
        if factor is not None:
            if nparts is not None:
                raise ValueError("Do not need to provide both outer and nparts")
            return ScheduleSplit(self, node, factor)
        else:
            if nparts is None:
                raise ValueError("Either nparts or factor need to be provided")
            return ScheduleSplitNParts(self, node, nparts)

    def fuse(self, outer_axis, inner_axis):
        return ScheduleFuse(self, outer_axis, inner_axis)

    def unroll(self, axis):
        return ScheduleUnroll(self, axis)

    def reorder(self, *args):
        return ScheduleReorder(self, args)

    def compute_inline(self, stmt):
        return ScheduleComputeInline(self, stmt)

    def compute_at(self, stmt, axis):
        return ScheduleComputeAt(self, stmt, axis)

    def compute_after(self, stmt, axis):
        return ScheduleComputeAfter(self, stmt, axis)

    def compute_root(self, stmt):
        return ScheduleComputeRoot(self, stmt)

    def blockize(self, axis):
        return ScheduleBlockize(self, axis)

    def unblockize(self, block):
        return ScheduleUnblockize(self, block)

    def tensorize(self, block, intrin):
        return ScheduleTensorize(self, block, intrin)

    def untensorize(self, block):
        return ScheduleUntensorize(self, block)

    def annotate(self, axis, type_name):
        return ScheduleAnnotate(self, axis, type_name)

    def bind(self, axis, thread_var):
        """Bind an axis to a thread index"""
        return ScheduleBind(self, axis, thread_var)

    ##### Schedule Helper #####
    def inline_all_injective(self):
        """Inline all elemwise and broadcast blocks"""
        return ScheduleInlineAllInjective(self)

    ##### Output #####
    def to_halide(self):
        """Lower TensorIR to halide statement

        Returns
        ----------
        stmt: Stmt
        """
        return ScheduleToHalide(self)

    ##### Debug Tools #####
    def check_father_link(self):
        """ Check the correctness of double-link between every father-child pair in the
        schedule tree. Print error message if finding erros.
        This is used for debug.
        """
        return ScheduleCheckFatherLink(self)


def create_schedule(stmt):
    """ Create a schedule for a statement

    Parameters
    ----------
    stmt: Stmt

    Returns
    ------
    schedule: Schedule
    """
    return CreateSchedule(stmt)


_init_api('tvm.tensorir.schedule')
