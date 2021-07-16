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
from tvm.script import ty


@tvm.script.tir
def elementwise(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, [4, 4], 'float32')
    B = tir.match_buffer(b, [4, 4], 'float32')

    for i in tir.serial(2, 4):
        for j in tir.serial(1, 4):
            with tir.block([2, 3], 'B') as [ii, jj]:
                tir.bind(ii, i - 2)
                tir.bind(jj, j - 1)
                B[ii, jj] = A[ii, jj]


@tvm.script.tir
def reduction(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [100, ], 'float32')
    B = tir.alloc_buffer([100, ], 'float32')
    C = tir.match_buffer(c, [1, ], 'float32')

    with tir.block([100], 'B') as [i]:
        B[i] = A[i] * 2.

    with tir.block([tir.reduce_axis(1, 100)], 'C') as [i]:
        C[0] = C[0] + B[i]


def test_elementwise():
    sch = tir.Schedule(elementwise, debug_mode=True)
    loops = sch.get_loops(sch.get_block('B'))
    sch.normalize(*loops)
    print(tvm.script.asscript(sch.mod['main']))
    print(tvm.lower(sch.mod['main']))


def test_reduction():
    sch = tir.Schedule(reduction, debug_mode=True)
    blk_B = sch.get_block('B')
    blk_C = sch.get_block('C')
    sch.compute_inline(blk_B)
    sch.normalize(sch.get_loops(blk_C)[0])
    print(tvm.lower(sch.mod['main']))


if __name__ == "__main__":
    test_elementwise()
    test_reduction()
