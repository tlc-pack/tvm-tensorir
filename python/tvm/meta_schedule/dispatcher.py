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

In meta schedule, a trace saves the schedule configuration by its instructions and decisions,
 so a trace is used as a schedule configuration here.
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
        task : SearchTask
            The task to query
        Returns
        -------
        trace : Trace
            The schedule trace for the workload
        """
        ret = self._query_inside(task)
        if ret is None:
            ret = self._old_ctx.query(task)
        return ret

    def _query_inside(self, task):
        """
        Query the context to get the specific config for a workload.
        This function only query config inside this context.

        Parameters
        ----------
        task : SearchTask
            The current task

        Returns
        -------
        trace : Trace
            The schedule trace for the workload
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
    records : Union[str, pathlib.Path]
        the path to the tune records
    space : SearchSpace
        the search space which contains postprocessor
    """

    def __init__(self, records, space):
        super(ApplyHistoryBest, self).__init__()

        if isinstance(records, pathlib.Path):
            records = str(records)

        self.space = space
        self.database = _ffi_api.GetInMemoryDB(records)

    def _query_inside(self, task):
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
                "which may bring great performance regression or even compilation failure. " % task
            )
            if msg not in self.messages:
                self.messages.add(msg)
                logger.warning(msg)

        trace = None

        # cache this config to avoid duplicated warning message
        self.memory[key] = trace
        return trace

    def _query_inside(self, task):
        raise RuntimeError("This function should never be called")


DispatchContext.current = FallbackContext()
