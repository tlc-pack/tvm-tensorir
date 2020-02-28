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

    assert isinstance(func.body.block, tvm.stmt.Block)
    assert isinstance(func.body.block.body, tvm.stmt.Loop)
    assert isinstance(func.body.block.body.body, tvm.stmt.Loop)
    assert isinstance(func.body.block.body.body.body, tvm.stmt.SeqStmt)
    assert isinstance(func.body.block.body.body.body[0].block, tvm.stmt.Block)
    assert isinstance(func.body.block.body.body.body[1], tvm.stmt.Loop)
    assert isinstance(func.body.block.body.body.body[1].body.block, tvm.stmt.Block)


def test_element_wise():
    func = util.element_wise_stmt()

    assert isinstance(func.body.block, tvm.stmt.Block)
    assert isinstance(func.body.block.body, tvm.stmt.SeqStmt)
    assert isinstance(func.body.block.body[0], tvm.stmt.Loop)
    assert isinstance(func.body.block.body[0].body, tvm.stmt.Loop)
    assert isinstance(func.body.block.body[0].body.body.block, tvm.stmt.Block)

    assert isinstance(func.body.block.body[1], tvm.stmt.Loop)
    assert isinstance(func.body.block.body[1].body, tvm.stmt.Loop)
    assert isinstance(func.body.block.body[1].body.body.block, tvm.stmt.Block)


def test_predicate():
    func = util.predicate_stmt()

    assert isinstance(func.body.block, tvm.stmt.Block)
    assert isinstance(func.body.block.body, tvm.stmt.Loop)
    assert isinstance(func.body.block.body.body, tvm.stmt.Loop)
    assert isinstance(func.body.block.body.body.body, tvm.stmt.Loop)
    assert isinstance(func.body.block.body.body.body.body.block, tvm.stmt.Block)


if __name__ == '__main__':
    test_matmul()
    test_element_wise()
    test_predicate()
