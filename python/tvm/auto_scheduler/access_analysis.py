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
"""Access analysis for loop trees"""
import tvm._ffi
from tvm.runtime import Object


@tvm._ffi.register_object("auto_scheduler.BaseAccessPattern")
class BaseAccessPattern(Object):
    """Base class of the access analysis. """


@tvm._ffi.register_object("auto_scheduler.DummyAccessPattern")
class DummyAccessPattern(Object):
    """Empty class. """


@tvm._ffi.register_object("auto_scheduler.LeafAccessPattern")
class LeafAccessPattern(BaseAccessPattern):
    """Resulf of access analysis of a leaf node in the loop tree. """

    def __str__(self):
        attrs = {
            "num_stmts": self.num_stmts,
            "has_branch": self.has_branch,
            "has_expensive_op": self.has_expensive_op,
            "all_trivial_store": self.all_trivial_store,
            "block_vars_in_trivial_store": self.block_vars_in_trivial_store,
            "lsmap_exists": self.lsmap_exists,
            "lsmap_surjective": self.lsmap_surjective,
            "lsmap_injective": self.lsmap_injective,
            "lsmap_ordered": self.lsmap_ordered,
            "num_axes_reuse": self.num_axes_reuse,
        }
        return str(attrs)


def analyze(loop_tree):
    return AnalyzeAccess(loop_tree)


tvm._ffi._init_api("auto_scheduler.access_analysis", __name__)
