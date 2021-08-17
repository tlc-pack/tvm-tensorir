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
"""Task Scheduler"""

from typing import List

from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api
from .tune_context import TuneContext


class Builder:
    pass


class Runner:
    pass


@register_object("meta_schedule.TaskScheduler")
class TaskScheduler(Object):
    """Description and abstraction of a task scheduler class."""

    def sort_all_tasks(self) -> None:
        return _ffi_api.TaskSchedulerSortAllTasks()  # pylint: disable=no-member

    def tune_all_tasks(self) -> None:
        return _ffi_api.TaskSchedulerTuneAllTasks()  # pylint: disable=no-member


@register_object("meta_schedule.PyTaskScheduler")
class PyTaskScheduler(TaskScheduler):
    """Task Scheduler that is implemented in python"""

    def __init__(self, tune_contexts: List[TuneContext], builder: Builder, runner: Runner):
        def sort_all_tasks_func() -> None:
            self.sort_all_tasks()

        def tune_all_tasks_func() -> None:
            self.tune_all_tasks()

        self.__init_handle_by_constructor__(
            _ffi_api.PyTaskScheduler,  # pylint: disable=no-member
            tune_contexts,
            builder,
            runner,
            sort_all_tasks_func,
            tune_all_tasks_func,
        )

    def sort_all_tasks(self) -> None:
        """Sort all tasks with priority during auto tuning."""
        raise NotImplementedError

    def tune_all_tasks(self) -> None:
        """Auto tune all the tasks."""
        raise NotImplementedError
