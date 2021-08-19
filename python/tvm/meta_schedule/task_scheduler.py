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

from typing import List, Optional
from tvm.runtime import Object

from tvm._ffi import register_object

from . import _ffi_api
from .tune_context import TuneContext


class Builder:
    pass


class Runner:
    pass


@register_object("meta_schedule.TaskScheduler")
class TaskScheduler(Object):
    """
    The abstract task scheduler for auto tuning.

    Task scheduler is responsible for scheduling tasks to maximize tuning rewards with given
    computing resources. While the task scheduler is running, it will try to do measure candidates
    generation while waiting for runner to asynchronously return measure results.

    The task scheduler works with two functions:
    1. sort_all_tasks: sort all tasks according to certain priority.
    2. tune_all_tasks: tune all tasks based on given scheduling mechanism.

    The PyTaskScheduler will call the functions defined in the class body, and the
    StandardTaskScheduler (available from the c++ side) works on the task in a round-robin fashion.
    """

    def sort_all_tasks(self) -> None:
        """Sort all the tuning tasks."""
        _ffi_api.TaskSchedulerSortAllTasks()  # pylint: disable=no-member

    def tune_all_tasks(self) -> None:
        """Tune all the tasks."""
        _ffi_api.TaskSchedulerTuneAllTasks()  # pylint: disable=no-member


@register_object("meta_schedule.PyTaskScheduler")
class PyTaskScheduler(TaskScheduler):
    """
    The PyTaskScheduler is defined for cutomizable task scheduling from python side.
    With PyTaskScheduler, you can define your own task sorting and tuning mechanism.
    """

    def __init__(
        self, tune_contexts: List[TuneContext], builder: Optional[Builder], runner: Optional[Runner]
    ):
        """Construct a PyTaskScheduler.
        Parameters
        ----------
        tune_contexts: List[TuneContext]
            The list of tuning contexts.
            The sort_all_tasks function can manipulate the order of tuning contexts.
        builder: Builder
            The builder of the task scheduler. You may want to check whether it's None before usage.
        runner: Runner
            The runner of the task scheduler. You may want to check whether it's None before usage.

        Note
        ----
        The PyTaskScheduler will use the sort_all_tasks and tune_all_tasks functions defined in the
        class body to schedule all tasks.
        """

        def sort_all_tasks_func() -> None:
            """Pass the sort_all_tasks function to constructor."""
            self.sort_all_tasks()

        def tune_all_tasks_func() -> None:
            """Pass the tune_all_tasks function to constructor."""
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
        """Sort all tasks with certain priority."""
        raise NotImplementedError

    def tune_all_tasks(self) -> None:
        """Auto tune all the tasks."""
        raise NotImplementedError
