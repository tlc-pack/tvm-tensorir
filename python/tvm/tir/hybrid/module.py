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
from tvm import Object
from tvm import container as _container
from tvm import make as _make
from tvm.api import _init_api

from ..util import register_tir_object


@register_tir_object
class GlobalVar(Object):
    """A global variable in TIR

    GlobalVar is used to refer to the global functions stored in the module.

    Parameters
    ----------
    name_hint : str
        The name of the variable.
    """

    def __init__(self, name_hint):
        self.__init_handle_by_constructor__(_make.TirGlobalVar, name_hint)


@register_tir_object
class Module(Object):
    """The global module containing collection of functions

    Each global function is identified by an unique tvm.tir.GlobalVar.

    Parameters
    ----------
    functions : Optional[list].
        List of functions
    """

    def __init__(self, name, functions=None):
        if functions is None:
            functions = {}
        elif isinstance(functions, list):
            mapped_funcs = {}
            for function in functions:
                if not isinstance(function, _container.Function):
                    raise TypeError("Expect functions to be Function")
                mapped_funcs[GlobalVar(function.name)] = function
            functions = mapped_funcs
        self.__init_handle_by_constructor__(_make.TirModule, name, functions)

    def append(self, function):
        if not isinstance(function, _container.Function):
            raise TypeError("Expect a Function to be appended")
        Append(self, GlobalVar(function.name), function)


def create_module(name="MyModule", functions=None):
    """Construct a module from list of functions.

    Parameters
    -----------
    name : Optional[str]
        Name of module

    functions : Optional[list]
        List of functions

    Returns
    -------
    mod : Module
        A module containing the passed definitions
    """

    funcs = functions if functions is not None else []
    return Module(name, funcs)


_init_api("tvm.tir.hybrid.module")
