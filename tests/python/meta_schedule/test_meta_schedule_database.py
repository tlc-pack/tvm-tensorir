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
"""Test Meta Schedule Database"""
import tempfile
import os.path as osp
from tvm.meta_schedule import DefaultDatabase, TuningRecord


def test_meta_schedule_database_create():
    with tempfile.TemporaryDirectory() as tmpdir:
        record_path = osp.join(tmpdir, "records.json")
        workload_path = osp.join(tmpdir, "workloads.json")
        database = DefaultDatabase(record_path=record_path, workload_path=workload_path)


if __name__ == "__main__":
    test_meta_schedule_database_create()
