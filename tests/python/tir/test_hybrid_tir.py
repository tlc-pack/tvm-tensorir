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
from tvm import tir
import util


def test_matmul():
    func = util.matmul_stmt()
    print(tvm.tir.hybrid.to_python(func))

    assert isinstance(func.body, tvm.stmt.Block)
    assert isinstance(func.body.body, tvm.stmt.Loop)
    assert isinstance(func.body.body.body, tvm.stmt.Loop)
    assert isinstance(func.body.body.body.body, tvm.stmt.SeqStmt)
    assert isinstance(func.body.body.body.body[0], tvm.stmt.Block)
    assert isinstance(func.body.body.body.body[1], tvm.stmt.Loop)
    assert isinstance(func.body.body.body.body[1].body, tvm.stmt.Block)


def test_element_wise():
    func = util.element_wise_stmt()
    print(tvm.tir.hybrid.to_python(func))

    assert isinstance(func.body, tvm.stmt.Block)
    assert isinstance(func.body.body, tvm.stmt.SeqStmt)
    assert isinstance(func.body.body[0], tvm.stmt.Loop)
    assert isinstance(func.body.body[0].body, tvm.stmt.Loop)
    assert isinstance(func.body.body[0].body.body, tvm.stmt.Block)

    assert isinstance(func.body.body[1], tvm.stmt.Loop)
    assert isinstance(func.body.body[1].body, tvm.stmt.Loop)
    assert isinstance(func.body.body[1].body.body, tvm.stmt.Block)


def test_predicate():
    func = util.predicate_stmt()
    print(tvm.tir.hybrid.to_python(func))

    assert isinstance(func.body, tvm.stmt.Block)
    assert isinstance(func.body.body, tvm.stmt.Loop)
    assert isinstance(func.body.body.body, tvm.stmt.Loop)
    assert isinstance(func.body.body.body.body, tvm.stmt.Loop)
    assert isinstance(func.body.body.body.body.body, tvm.stmt.Block)


if __name__ == '__main__':
    test_matmul()
    test_element_wise()
    test_predicate()
