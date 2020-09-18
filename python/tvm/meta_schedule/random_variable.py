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
""" Random variables in meta schedule """
from tvm._ffi import register_object as _register_object
from tvm.runtime import Object as _Object
from tvm.tir import Var  # pylint: disable=unused-import


@_register_object("meta_schedule.BlockRV")
class BlockRV(_Object):
    """ defined in src/meta_schedule/random_variable.h """


@_register_object("meta_schedule.LoopRV")
class LoopRV(_Object):
    """ defined in src/meta_schedule/random_variable.h """
