from ..expr import Expr
from ..api import _init_api
from .._ffi.node import NodeBase, register_node

from .tree_node import register_tensorir_node

@register_tensorir_node
class Schedule(NodeBase):

    def blocks(self):
        return ScheduleBlocks(self)

    def axis(self, stmt):
        return ScheduleAxis(self, stmt)

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

    def vectorize(self, axis):
        return ScheduleVectorize(self, axis)

    def to_halide(self):
        return ScheduleToHalide(self)

    def check_father_link(self):
        return ScheduleCheckFatherLink(self)

    def bind(self, axis, iter_var):
        return ScheduleBind(self, axis, iter_var)

def create_schedule(stmt):
    return CreateSchedule(stmt)


_init_api('tvm.tensorir.schedule')
