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
"""Function data types."""

import inspect
from typing import Callable, List, Mapping, Union

from tvm._ffi import get_global_func, register_object
from tvm.ir import BaseFunc
from tvm.runtime import Object, convert

from . import _ffi_api
from .buffer import Buffer
from .expr import PrimExpr, Var


@register_object("tir.PrimFunc")
class PrimFunc(BaseFunc):
    """A function declaration expression.

    Parameters
    ----------
    params: List[Union[tvm.tir.Var, tvm.tir.Buffer]]
        List of input parameters to the function.

    body: tvm.tir.Stmt
        The body of the function.

    ret_type: tvm.ir.Type
        The return type annotation of the function.

    buffer_map : Map[tvm.tir.Var, tvm.tir.Buffer]
        The buffer binding map.

    attrs: Optional[tvm.Attrs]
        Attributes of the function, can be None

    span : Optional[Span]
        The location of this itervar in the source code.
    """

    def __init__(self, params, body, ret_type=None, buffer_map=None, attrs=None, span=None):
        param_list = []
        buffer_map = {} if buffer_map is None else buffer_map
        for x in params:
            x = convert(x) if not isinstance(x, Object) else x
            if isinstance(x, Buffer):
                var = Var(x.name, dtype="handle")
                param_list.append(var)
                buffer_map[var] = x
            elif isinstance(x, Var):
                param_list.append(x)
            else:
                raise TypeError("params can only contain Var or Buffer")

        self.__init_handle_by_constructor__(
            _ffi_api.PrimFunc,  # type: ignore # pylint: disable=no-member
            param_list,
            body,
            ret_type,
            buffer_map,
            attrs,
            span,
        )

    def with_body(self, new_body, span=None):
        """Create a new PrimFunc with the same set signatures but a new body.

        Parameters
        ----------
        new_body : Stmt
            The new body.

        span : Optional[Span]
            The location of this itervar in the source code.

        Returns
        -------
        new_func : PrimFunc
            The created new function.
        """
        return PrimFunc(self.params, new_body, self.ret_type, self.buffer_map, self.attrs, span)

    def specialize(self, param_map: Mapping[Var, Union[PrimExpr, Buffer]]):
        """Specialize parameters of PrimFunc

        Parameters
        ----------

        param_map : Mapping[Var, Union[PrimExpr, Buffer]]
            The mapping from function params to the instance

        Examples
        --------
        We can define a Meta TIR function with symbolic shape:

        .. code-block:: python

            @T.prim_func
            def mem_copy(a: T.handle, b: T.handle, m: T.int32, n: T.int32) -> None:
                A = T.match_buffer(a, (m, n), "float32")
                B = T.match_buffer(b, (m, n), "float32")

                for i, j in T.grid(m, n):
                    with T.block():
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj]

        Then we can make it specialized with given shapes or buffers.

        .. code-block:: python

            a, _, m, n = mem_copy.params
            func = mem_copy.specialize({a: tir.decl_buffer((16, 16))})
            # or
            func = mem_copy.specialize({n: 16, m: 16})

        The specialized function:

        .. code-block:: python

            @T.prim_func
            def mem_copy_16_16(a: T.handle, b: T.handle) -> None:
                A = T.match_buffer(a, (16, 16), "float32")
                B = T.match_buffer(b, (16, 16), "float32")

                for i, j in T.grid(16, 16):
                    with T.block():
                        vi, vj = T.axis.remap("SS", [i, j])
                        B[vi, vj] = A[vi, vj]

        Returns
        -------
        func : PrimFunc
            The new function with parameter specialized
        """
        return _ffi_api.Specialize(self, param_map)  # type: ignore # pylint: disable=no-member

    def script(self, tir_prefix: str = "tir", show_meta: bool = False) -> str:
        """Print IRModule into TVMScript

        Parameters
        ----------
        tir_prefix : str
            The tir namespace prefix

        show_meta : bool
            Whether to show meta information

        Returns
        -------
        script : str
            The TVM Script of the PrimFunc
        """
        return get_global_func("script.AsTVMScript")(self, tir_prefix, show_meta)  # type: ignore


@register_object("tir.IndexMap")
class IndexMap(Object):
    """A mapping from multi-dimensional indices to another set of multi-dimensional indices

    Parameters
    ----------
    src_iters : list of Var
        The source indices
    tgt_iters : list of PrimExpr
        The target indices
    """

    src_iters: List[Var]
    """The source indices"""

    tgt_iters: List[PrimExpr]
    """The target indices"""

    def __init__(self, src_iters: List[Var], tgt_iters: List[PrimExpr]):
        self._init_handle_by_constructor(
            _ffi_api.IndexMap,  # type: ignore # pylint: disable=no-member
            src_iters,
            tgt_iters,
        )

    def apply(self, indices: List[PrimExpr]) -> List[PrimExpr]:
        """Apply the index map to a set of indices

        Parameters
        ----------
        indices : List[PriExpr]
            The indices to be mapped

        Returns
        -------
        result : List[PrimExpr]
            The mapped indices
        """
        return _ffi_api.IndexMapApply(self, indices)  # type: ignore # pylint: disable=no-member

    @staticmethod
    def from_func(func: Callable) -> "IndexMap":
        """Create an index map from a function

        Parameters
        ----------
        func : Callable
            The function to map from source indices to target indices
        """

        def wrap(args: List[Var]) -> List[PrimExpr]:
            result = func(*args)
            if isinstance(result, tuple):
                return list(result)
            if not isinstance(result, list):
                result = [result]
            return result

        ndim = len(inspect.signature(func).parameters)
        return _ffi_api.IndexMapFromFunc(ndim, wrap)  # type: ignore # pylint: disable=no-member


@register_object("tir.TensorIntrin")
class TensorIntrin(Object):
    """A function declaration expression.

    Parameters
    ----------
    desc_func: PrimFunc
        The function to describe the computation

    intrin_func: PrimFunc
        The function for execution
    """

    def __init__(self, desc_func, intrin_func):
        self.__init_handle_by_constructor__(
            _ffi_api.TensorIntrin, desc_func, intrin_func  # type: ignore  # pylint: disable=no-member
        )

    @staticmethod
    def register(name: str, desc_func: PrimFunc, intrin_func: PrimFunc):
        return _ffi_api.TensorIntrinRegister(  # pylint: disable=no-member
            name, desc_func, intrin_func
        )

    @staticmethod
    def get(name: str):
        return _ffi_api.TensorIntrinGet(name)  # pylint: disable=no-member
