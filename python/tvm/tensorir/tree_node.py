from .. import make as _make, expr as _expr
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


@register_tensorir_node
class ScheduleTreeNode(NodeBase):
    def __str__(self):
        return PrintTreeNode(self)


@register_tensorir_node
class AxisTreeNode(ScheduleTreeNode):
    def __init__(self, loop_var, min, extent, axis_type, children):
        self.__init_handle_by_constructor__(_make.AxisTreeNode,
                                            loop_var,
                                            min,
                                            extent,
                                            axis_type,
                                            children)


@register_tensorir_node
class BlockTreeNode(ScheduleTreeNode):
    def __init__(self, args, vars, inputs, outputs, stmt, children):
        stmt = _make.Evaluate(stmt) if isinstance(stmt, _expr.Expr) else stmt
        self.__init_handle_by_constructor__(_make.BlockTreeNode,
                                            args,
                                            vars,
                                            inputs,
                                            outputs,
                                            stmt,
                                            children)

_init_api('tvm.tensorir.tree_node')
