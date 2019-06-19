
from ..expr import Expr
from ..api import _init_api
from .._ffi.node import NodeBase, register_node


def register_tensorir_node(type_key=None):
    """Register a Relay node type.

    Parameters
    ----------
    type_key : str or cls
        The type key of the node.
    """
    if not isinstance(type_key, str):
        return register_node(
            "tensorir." + type_key.__name__)(type_key)
    return register_node(type_key)

@register_node("tensorir.Schedule")
class Schedule(NodeBase):

    def split(self, node, factor):
        return ScheduleSplit(self, node, factor)

    def blocks(self):
        return ScheduleBlocks(self)

    def axis(self, stmt):
        return ScheduleAxis(self, stmt)

    def fuse(self, outer_axis, inner_axis):
        return ScheduleFuse(self, outer_axis, inner_axis)

    def reorder(self, outer_axis, inner_axis):
        return ScheduleReorder(self, outer_axis, inner_axis)

    def compute_root(self, stmt):
        return ScheduleComputeRoot(self, stmt)

    def compute_at(self, stmt, axis):
        return ScheduleComputeAt(self, stmt, axis)

    def compute_inline(self, stmt):
        return ScheduleComputeInline(self, stmt)

    def shrink_layout(self, stmt):
        return ScheduleShrinkLayout(self, stmt)


@register_tensorir_node
class ScheduleTreeNode(NodeBase):

    def __str__(self):
        return PrintTreeNode(self)

@register_tensorir_node
class AxisTreeNode(ScheduleTreeNode):
    pass

@register_tensorir_node
class BlockTreeNode(ScheduleTreeNode):
    pass

def create_schedule(stmt):
    return CreateSchedule(stmt)

_init_api('tvm.tensorir.schedule')
