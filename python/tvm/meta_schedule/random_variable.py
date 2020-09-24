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
from typing import Union

from tvm._ffi import register_object
from tvm.ir import PrimExpr as ExprRV  # pylint: disable=unused-import
from tvm.runtime import Object


@register_object("meta_schedule.BlockRV")
class BlockRV(Object):
    """ A random variable that evaluates to a TIR block """


@register_object("meta_schedule.LoopRV")
class LoopRV(Object):
    """ A random variable that evaluates to a TIR loop axis """


RAND_VAR_TYPE = Union[ExprRV, BlockRV, LoopRV]  # pylint: disable=invalid-name
