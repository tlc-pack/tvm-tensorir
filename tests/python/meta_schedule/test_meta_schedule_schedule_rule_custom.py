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
""" Test meta schedule ScheduleRule """
# pylint: disable=missing-function-docstring

from tir_workload import matmul
from tvm import meta_schedule as ms
from tvm.tir.schedule import Schedule, BlockRV


def test_meta_schedule_schedule_rule_do_nothing():
    @ms.as_schedule_rule("do_nothing")
    def do_nothing(sch: Schedule, _block: BlockRV):
        return sch

    sch = Schedule(matmul)
    (ret,) = do_nothing.apply(schedule=sch, block=sch.get_block("matmul"))
    assert sch.same_as(ret)


def test_meta_schedule_print_name():
    def do_nothing(sch: Schedule, _block: BlockRV):
        return sch

    sch_rule = ms.PyScheduleRule("Hanamichi Sakuragi", do_nothing)
    assert sch_rule.name == "Hanamichi Sakuragi"
    sch = Schedule(matmul)
    (ret,) = sch_rule.apply(schedule=sch, block=sch.get_block("matmul"))
    assert sch.same_as(ret)


if __name__ == "__main__":
    test_meta_schedule_print_name()
    test_meta_schedule_schedule_rule_do_nothing()
