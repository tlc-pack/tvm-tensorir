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
""" Test for search task class """
# pylint: disable=missing-function-docstring
import tvm
from tvm import meta_schedule as ms
from tir_workload import matmul


def test_meta_schedule_search_task_creation():
    task = ms.SearchTask(func=matmul, target="cuda", target_host="llvm")
    assert tvm.ir.structural_equal(task.func, matmul)
    assert str(task.target).startswith("cuda ")
    assert str(task.target_host).startswith("llvm ")


if __name__ == "__main__":
    test_meta_schedule_search_task_creation()
