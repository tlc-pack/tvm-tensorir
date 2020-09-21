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
""" Description of a search task """
from typing import Any, Dict, Optional, Union

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.target import Target
from tvm.target import create as create_target
from tvm.tir import PrimFunc

from . import _ffi_api

# TODO(@junrushao1994): rebase and remove target-related util funcs

TargetType = Union[Target, str, Dict[str, Any]]


def normalize_target(target: TargetType) -> Target:
    return create_target(target)


@register_object("meta_schedule.SearchTask")
class SearchTask(Object):
    """ defined in src/meta_schedule/search_task.h """

    func: PrimFunc
    task_name: str
    target: Target
    target_host: Target

    def __init__(
        self,
        func: PrimFunc,
        task_name: Optional[str] = None,
        target: TargetType = "llvm",
        target_host: TargetType = "llvm",
    ):
        if task_name is None:
            task_name = func.__qualname__
        self.__init_handle_by_constructor__(
            _ffi_api.SearchTask,  # pylint: disable=no-member
            func,
            task_name,
            normalize_target(target),
            normalize_target(target_host),
        )
