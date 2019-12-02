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
    s = te.create_scheduleX(func)


def test_replace_fuse():
    m, n = 128, 128
    func, tensors, tensor_map = util.element_wise_stmt(m, n)
    s = te.create_scheduleX(func)
    ib = tvm.ir_builder.create()

    # Declare buffer will construct a new Var
    # So this code cannot run but only for print IR
    A = ib.declare_buffer((m, n), "float32", name="A")
    B = ib.declare_buffer((m, n), "float32", name="B")

    with ib.loop_range(0, m * n, name="i0") as fused:
        bv = ib.iter_var((0, m * n), name="vi0")
        v = bv.var
        i = tvm.truncdiv(v, n)
        j = tvm.truncmod(v, n)
        with ib.block([bv], [fused], A[i:i + 1, j:j + 1], B[i:i + 1, j:j + 1],
                name="B"):
            B[i, j] = A[i, j] * 2

    target = ib.get()
    stmt = s.func.body.body.seq[0]
    sref = s.get_s_ref(stmt)
    del stmt
    s.replace(sref, target)
    print(s.func)


if __name__ == "__main__":
    test_create_schedule()
    test_replace_fuse()
