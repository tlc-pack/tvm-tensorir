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
""" Search Policy """
from tvm._ffi import register_object
from tvm.runtime import Object

from . import _ffi_api


@register_object("meta_schedule.SearchPolicy")
class SearchPolicy(Object):
    """ defined in src/meta_schedule/search_policy.h """


@register_object("meta_schedule.ScheduleFn")
class ScheduleFn(Object):
    """ defined in src/meta_schedule/search_policy/schedule_fn.h """

    sch_fn: str

    def __init__(
        self,
        sch_fn: str,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleFn,  # pylint: disable=no-member
            sch_fn,
        )
