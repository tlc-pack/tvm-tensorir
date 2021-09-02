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
from typing import Callable, Optional

import os
import shutil
import psutil
from tvm._ffi import get_global_func, register_func
from tvm.error import TVMError


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


def get_global_func_with_default_on_worker(name: Optional[str], default: Callable) -> Callable:
    """Get the registered global function on the worker process.

    Parameters
    ----------
    name : Optional[str]
        If given, retrieve the function in TVM's global registry;
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


@register_func("meta_schedule.clean_up_build")
def clean_up_build(artifact_path: Optional[str]):
    """Clean up the build directory"""
    if artifact_path is not None:
        shutil.rmtree(os.path.dirname(artifact_path))
