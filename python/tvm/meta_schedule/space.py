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
"""Search space"""
from typing import Callable, List, Optional

from tvm._ffi import register_object

from ..tir.schedule import Schedule
from . import _ffi_api
from .postproc import Postproc
from .search import SearchSpace
from .search_rule import SearchRule


@register_object("meta_schedule.ScheduleFn")
class ScheduleFn(SearchSpace):
    """Search space that is specified by a schedule function"""

    TYPE = Callable[[Schedule], None]

    postprocs: List[Postproc]

    def __init__(
        self,
        func: TYPE,
        postprocs: Optional[List[Postproc]] = None,
    ):
        if postprocs is None:
            postprocs = []
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleFn,  # pylint: disable=no-member
            func,
            postprocs,
        )


@register_object("meta_schedule.PostOrderApply")
class PostOrderApply(SearchSpace):
    """Search space that is specified by applying rules in post-DFS order"""

    stages: List[SearchRule]
    postprocs: List[Postproc]

    def __init__(
        self,
        stages: List[SearchRule],
        postprocs: Optional[List[Postproc]] = None,
    ):
        if postprocs is None:
            postprocs = []
        self.__init_handle_by_constructor__(
            _ffi_api.PostOrderApply,  # pylint: disable=no-member
            stages,
            postprocs,
        )
