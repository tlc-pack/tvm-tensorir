from .._ffi.node import NodeBase
from .. import make as _make, api as _api, intrin as _intrin, expr as _expr, ir_pass as _ir_pass, \
    _api_internal
from ..build_module import current_build_config

from .tree_node import register_tensorir_node

@register_tensorir_node
class TensorRegion(NodeBase):
    def __init__(self, tensor_slice):
        ranges = []
        for x in tensor_slice.indices:
            assert x.step is None or x.step == 1, "Only support step = 1"
            ranges.append(_api.Range(x.start, x.stop))
        self.__init_handle_by_constructor__(_make.TensorRegion,
                                            tensor_slice.tensor,
                                            ranges)

    def __getitem__(self, item):
        assert len(item) == self.data.ndim, "The dimension of index is wrong"

        mins = []
        extents = []
        for x in item:
            mins.append(x.start)
            extents.append((x.stop - x.start))
            assert x.step is None or x.step == 1, "Only support step == 1"

        return _api_internal._TensorRegion_MakeView(self, mins, extents)

    def emit_buffer_bind(self, ib, **kwargs):
        """Emit buffer_bind_scope Attr Stmt to an IRBuilder"""
        data = self.data

        # skip ones   todo(lmzheng) : fix this to match inputs placeholder
        shape = [_ir_pass.Simplify(x.extent) for x in self.ranges]
        while isinstance(shape[0], _expr.IntImm) and shape[0].value == 1 and len(shape) > 1:
            shape = shape[1:]

        shape_dtype = shape[0].dtype if hasattr(shape[0], "dtype") else "int32"
        tmp = _api.const(1, shape_dtype)

        cfg = current_build_config()
        B = _api.decl_buffer(shape, dtype=data.dtype, name="B" + data.name,
                             elem_offset=_api.var(data.name + "_offset", dtype=shape_dtype),
                             **kwargs)
        ranges = []
        for x in self.ranges:
            ranges.append(x.min)
            ranges.append(x.extent)
        ib.scope_attr([B, data], "buffer_bind_scope", _intrin.call_intrin("handle", "tvm_tuple", *ranges))

        return B
