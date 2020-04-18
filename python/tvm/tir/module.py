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
"""A global module storing everything needed to compile a hybrid script TIR program."""

from tvm.ir import module as _ir_module
from tvm.ir import BaseExpr, GlobalVar

from .util import register_tir_object
from . import _ffi_api


@register_tir_object
class Function(BaseExpr):
    """Function node in TIR.

    Parameters
    ----------
    params : List[te.Var]
        The function params.

    buffer_map : Map[te.Var, te.Buffer]
        Buffer binding information.

    reducers : List[tir.CommReducer]
        User defined reducers

    name : Str
        The function name.

    body : Stmt
        The function body.

    """
    def __init__(self, params, buffer_map, reducers, name, body):
        self.__init_handle_by_constructor__(
            _ffi_api.Function, params, buffer_map, reducers, name, body)


class Module:
    """The global module containing collection of functions, which is a helper class
    in hybrid script that actually operates on IRModule in backend.

    Parameters
    ----------
    functions : Optional[list].
        List of Function
    """

    def __init__(self, functions=None):
        if functions is None:
            functions = {}
        elif isinstance(functions, list):
            mapped_funcs = {}
            for function in functions:
                if not isinstance(function, Function):
                    raise TypeError("Expect functions to be TirFunction")
                mapped_funcs[GlobalVar(function.name)] = function
            functions = mapped_funcs
        self.module = _ir_module.IRModule(functions=functions)

    def __getitem__(self, name):
        """Look up a Function by name

        Parameters
        ----------
        name : str
            The name of Function to be looked up

        Returns
        -------
        function : Function
            The Function to be looked up
        """

        if not isinstance(name, str):
            raise TypeError("Expect the name to be an str")
        return self.module[name]


def create_module(funcs=None):
    """Construct a module from list of functions.

    Parameters
    -----------
    funcs : Optional[List[Function]]
        List of Function

    Returns
    -------
    mod : Module
        A module containing the passed functions
    """

    funcs = funcs if funcs is not None else []
    return Module(funcs)
