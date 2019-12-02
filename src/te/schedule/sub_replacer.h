/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_TE_SCHEDULE_SUB_REPLACER_H_
#define TVM_TE_SCHEDULE_SUB_REPLACER_H_

#include <tvm/te/schedule.h>

namespace tvm {
namespace te {
class SubReplacer {
 public:
  SubReplacer(ScheduleNode* schedule, Stmt old_stmt, Stmt new_stmt)
      : schedule_(schedule), old_stmt_(old_stmt), new_stmt_(new_stmt) {}
  Stmt Mutate_(const LoopNode* op);
  Stmt Mutate_(const SeqStmtNode* op);
  Stmt Mutate_(const BlockNode* op);
  Stmt Mutate(const StmtNode* op);
  bool need_copy{true};
 private:
  ScheduleNode* schedule_;
  Stmt old_stmt_;
  Stmt new_stmt_;
};

}  // namespace te
}  // namespace tvm

#endif  // TVM_TE_SCHEDULE_SUB_REPLACER_H_
