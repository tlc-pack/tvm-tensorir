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
"""Utility"""
import json
import os
import shutil
from typing import Any, Callable, List, Union

import psutil

import tvm
from tvm._ffi import get_global_func, register_func
from tvm.error import TVMError
from tvm.ir import Array, Map, IRModule
from tvm.runtime import String
from tvm.tir import FloatImm, IntImm


@register_func("meta_schedule.cpu_count")
def cpu_count(logical: bool = True) -> int:
    """Return the number of logical or physical CPUs in the system

    Parameters
    ----------
    logical : bool = True
        If True, return the number of logical CPUs, otherwise return the number of physical CPUs

    Returns
    -------
    cpu_count : int
        The number of logical or physical CPUs in the system
    """
    return psutil.cpu_count(logical=logical) or 1


def get_global_func_with_default_on_worker(
    name: Union[None, str, Callable],
    default: Callable,
) -> Callable:
    """Get the registered global function on the worker process.

    Parameters
    ----------
    name : Union[None, str, Callable]
        If given a string, retrieve the function in TVM's global registry;
        If given a python function, return it as it is;
        Otherwise, return `default`.

    default : Callable
        The function to be returned if `name` is None.

    Returns
    -------
    result : Callable
        The retrieved global function or `default` if `name` is None
    """
    if name is None:
        return default
    if callable(name):
        return name
    try:
        return get_global_func(name)
    except TVMError as error:
        raise ValueError(
            "Function '{name}' is not registered on the worker process. "
            "The build function and export function should be registered in the worker process. "
            "Note that the worker process is only aware of functions registered in TVM package, "
            "if there are extra functions to be registered, "
            "please send the registration logic via initializer."
        ) from error


@register_func("meta_schedule.remove_build_dir")
def remove_build_dir(artifact_path: str):
    """Clean up the build directory"""
    shutil.rmtree(os.path.dirname(artifact_path))


@register_func("meta_schedule.batch_json_str2obj")
def batch_json_str2obj(json_strs: List[str]) -> List[Any]:
    """Covert a list of JSON strings to a list of json objects.

    Parameters
    ----------
    json_strs : List[str]
        The list of JSON strings

    Returns
    -------
    result : List[Any]
        The list of json objects
    """
    return [
        json.loads(json_str)
        for json_str in map(str.strip, json_strs)
        if json_str and (not json_str.startswith("#")) and (not json_str.startswith("//"))
    ]


def _json_de_tvm(obj: Any) -> Any:
    """Unpack a json object.

    Parameters
    ----------
    obj : Any
        The json object

    Returns
    -------
    result : Any
        The unpacked json object.
    """
    if obj is None:
        return None
    if isinstance(obj, Array):
        return [_json_de_tvm(i) for i in obj]
    if isinstance(obj, Map):
        return {_json_de_tvm(k): _json_de_tvm(v) for k, v in obj.items()}
    if isinstance(obj, String):
        return str(obj)
    if isinstance(obj, (IntImm, FloatImm)):
        return obj.value
    raise TypeError("Not supported type: " + str(type(obj)))


@register_func("meta_schedule.json_obj2str")
def json_obj2str(json_obj: Any) -> str:
    json_obj = _json_de_tvm(json_obj)
    return json.dumps(json_obj)


def structural_hash(mod: IRModule) -> str:
    shash = tvm.ir.structural_hash(mod)
    if shash < 0:
        # Workaround because `structural_hash` returns a size_t, i.e., unsigned integer
        # but ffi can't handle unsigned integers properly so it's parsed into a negative number
        shash += 1 << 64
    return str(shash)
