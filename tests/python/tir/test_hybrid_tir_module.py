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
import util


def test_module_define():
    func1 = util.matmul_stmt()
    func2 = util.element_wise_stmt()
    func3 = util.predicate_stmt()
    mod1 = tvm.tir.hybrid.create_module({"func1": func1, "func2": func2, "func3": func3})
    mod2 = tvm.tir.hybrid.create_module(
        {"func1": util.matmul, "func2": util.element_wise, "func3": util.predicate})
    tvm.ir.assert_structural_equal(mod1, mod2)


if __name__ == '__main__':
    test_module_define()
