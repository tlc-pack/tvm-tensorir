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
"""TVM Script Parser Function Registry """
# pylint: disable=inconsistent-return-statements
import inspect


class Registry(object):
    """Registration map
    All these maps are static
    """

    registrations = dict()

    @staticmethod
    def lookup(name):
        if name in Registry.registrations:
            return Registry.registrations[name]
        return None


def register(registration):
    if inspect.isclass(registration):
        registration_obj = registration()
        key = registration_obj.signature()[0]
    else:
        raise ValueError()
    print(key, registration_obj)
    Registry.registrations[key] = registration_obj
    return registration
