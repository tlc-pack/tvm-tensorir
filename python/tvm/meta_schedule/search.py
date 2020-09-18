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
""" Search API """

from typing import List, Optional, Union

from . import _ffi_api
from .measure import LocalBuilder, MeasureCallback, ProgramBuilder, ProgramRunner
from .schedule import Schedule
from .search_policy import SearchPolicy
from .search_task import SearchTask


def search(
    task: SearchTask,
    policy: SearchPolicy,
    builder: Union[str, ProgramBuilder],
    runner: ProgramRunner,
    measure_callbacks: Optional[List[MeasureCallback]] = None,
    verbose: int = 1,
) -> Schedule:
    """ Search API """
    if isinstance(builder, str):
        if builder == "local":
            builder = LocalBuilder()
        else:
            raise ValueError("Unknown name of program builder: " + builder)
    if measure_callbacks is None:
        measure_callbacks = []
    return _ffi_api.Search(  # pylint: disable=no-member
        task, policy, builder, runner, measure_callbacks, verbose
    )
