
from .._ffi.node import NodeBase
from .. import make as _make, api as _api, _api_internal
from .tree_node import register_tensorir_node

@register_tensorir_node
class TensorIntrinsic(NodeBase):
    def __init__(self, op, intrin_func, name):
        self.__init_handle_by_constructor__(_make._TensorIntrinsic, op, intrin_func, name)

    def __call__(self, inputs, outputs):
        return _api_internal._TensorIntrinsic_Instantiate(self, inputs, outputs)

def decl_tensor_intrin(op, intrin_func, name):
    """decleare a new tensor intrinsic

    Parameters
    ----------
    op: Operation
    intrin_func: (List[TensorRegion], List[TensorRegion]) -> Stmt or BlockTreeNode
    name: str
    """
    return _make._TensorIntrinsic(op, intrin_func, name)
