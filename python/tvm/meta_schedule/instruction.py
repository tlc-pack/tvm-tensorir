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
from tvm._ffi import register_object as _register_object
from tvm.runtime import Object as _Object


@_register_object("meta_schedule.Instruction")
class Instruction(_Object):
    """ defined in src/meta_schedule/instruction.h """


@_register_object("meta_schedule.SampleTileFactorInst")
class SampleTileFactorInst(Instruction):
    """ defined in src/meta_schedule/instruction.h """


@_register_object("meta_schedule.GetBlockInst")
class GetBlockInst(Instruction):
    """ defined in src/meta_schedule/instruction.h """


@_register_object("meta_schedule.GetAxesInst")
class GetAxesInst(Instruction):
    """ defined in src/meta_schedule/instruction.h """


@_register_object("meta_schedule.SplitInst")
class SplitInst(Instruction):
    """ defined in src/meta_schedule/instruction.h """


@_register_object("meta_schedule.ReorderInst")
class ReorderInst(Instruction):
    """ defined in src/meta_schedule/instruction.h """


@_register_object("meta_schedule.DecomposeReductionInst")
class DecomposeReductionInst(Instruction):
    """ defined in src/meta_schedule/instruction.h """
