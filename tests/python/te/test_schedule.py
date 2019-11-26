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

import tvm
from tvm import te
import util


def test_create_schedule():
    func, tensors, tensor_map = util.element_wise_stmt()
    te.create_schedule(func)


def test_block_axis():
    func, tensors, tensor_map = util.matmul_stmt()
    s = te.create_schedule(func)
    init = s.get_block("init")
    update = s.get_block("update")
    assert len(s.get_axes(init)) == 2
    assert len(s.get_axes(update)) == 3


def test_fuse():
    func, tensors, tensor_map = util.element_wise_stmt()
    s = te.create_schedule(func)
    B = s.get_block("B")
    axes = s.get_axes(B)


if __name__ == "__main__":
    test_create_schedule()
    test_block_axis()
    test_fuse()
