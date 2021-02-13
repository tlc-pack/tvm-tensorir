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
"""
The global context that dispatches best schedules to workloads.

In auto-scheduler, a state (loop_state.py::StateObject) saves the
schedule configuration by its transform_steps, so a state is used
as a schedule configuration here.
"""
# pylint: disable=invalid-name

import logging
import pathlib
from . import _ffi_api
from .search import SearchTask

logger = logging.getLogger("meta_schedule")


class DispatchContext(object):
    """
    Base class of dispatch context.
    """

    current = None

    def __init__(self):
        self._old_ctx = DispatchContext.current

    def query(self, task):
        """
        Query the context to get the specific config for a workload.
        If cannot find the result inside this context, this function will query it
        from the upper contexts.

        Parameters
        ----------
        target: Target
            The current target
        task : str
            The task name
        Returns
        -------
        state : StateObject
            The state that stores schedule configuration for the workload
        """
        ret = self._query_inside(task)
        if ret is None:
            ret = self._old_ctx.query(task)
        return ret

    def update(self, task, state):
        """
        Update the config for a workload

        Parameters
        ----------
        target: Target
            The current target
        task : str
            The current workload_key.
        state : Object
            The state that stores schedule configuration for the workload
        """
        raise NotImplementedError()

    def _query_inside(self, task):
        """
        Query the context to get the specific config for a workload.
        This function only query config inside this context.

        Parameters
        ----------
        target: Target
            The current target
        task : str
            The current workload_key.

        Returns
        -------
        state : StateObject
            The schedule configuration for the workload
        """
        raise NotImplementedError()

    def __enter__(self):
        self._old_ctx = DispatchContext.current
        DispatchContext.current = self
        return self

    def __exit__(self, ptype, value, trace):
        DispatchContext.current = self._old_ctx


#
class ApplyHistoryBest(DispatchContext):
    """
    Apply the history best config

    Parameters
    ----------
    search_task : SearchTask
        the search task
    n_lines: Optional[int]
        if it is not None, only load the first `n_lines` lines of log.
    include_compatible: bool
        When set to True, compatible records will also be considered.
    """

    def __init__(self, records, space):
        super(ApplyHistoryBest, self).__init__()

        if isinstance(records, pathlib.Path):
            records = str(records)

        self.space = space
        self.database = _ffi_api.GetInMemoryDB(records)

    def _query_inside(self, task):
        # print("task:", task)
        return _ffi_api.GetBest(self.database, task)


class FallbackContext(DispatchContext):
    """
    A fallback dispatch context.
    This is used as the root context.
    """

    def __init__(self):
        super(FallbackContext, self).__init__()
        self.memory = {}

        # Verbose level:
        # 0: Completely silent.
        # 1: Warning the missing configs for querying all tasks.
        self.verbose = 1

        # a set to prevent print duplicated message
        self.messages = set()

    def query(self, task):
        key = task
        if key in self.memory:
            return self.memory[key]

        if self.verbose == 1:
            msg = (
                    "-----------------------------------\n"
                    "Cannot find tuned schedules for task=%s"
                    "A fallback TOPI schedule is used, "
                    "which may bring great performance regression or even compilation failure. "
                    % task
            )
            if msg not in self.messages:
                self.messages.add(msg)
                logger.warning(msg)

        state = None

        # cache this config to avoid duplicated warning message
        self.memory[key] = state
        return state

    def _query_inside(self, task):
        raise RuntimeError("This function should never be called")

    def update(self, task, state):
        key = task
        self.memory[key] = state


DispatchContext.current = FallbackContext()
