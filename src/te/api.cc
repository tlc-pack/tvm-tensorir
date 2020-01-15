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


/*!
 *  \brief TE API registration
 */

#include <tvm/te/transform.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/schedule.h>

namespace tvm {
namespace te {

TVM_REGISTER_GLOBAL("ir_pass.TeLower")
.set_body_typed(TeLower);

// schedule
TVM_REGISTER_GLOBAL("te.schedule.CreateSchedule")
.set_body_typed(Schedule::Create);

TVM_REGISTER_GLOBAL("te.schedule.Replace")
.set_body_method(&Schedule::Replace);

TVM_REGISTER_GLOBAL("te.schedule.GetStmtSRef")
.set_body_typed<StmtSRef(Schedule, Stmt)>(
    [](Schedule schedule, Stmt stmt) {
      return schedule->stmt2ref.at(stmt.operator->());
    });

TVM_REGISTER_GLOBAL("te.schedule.GetStmt")
.set_body_typed<Stmt(StmtSRef)>(
    [](StmtSRef sref) {
      return GetRef<Stmt>(sref->node);
    });

TVM_REGISTER_GLOBAL("te.schedule.ScheduleBlocks")
.set_body_method(&Schedule::Blocks);

TVM_REGISTER_GLOBAL("te.schedule.GetBlocksFromTag")
.set_body_typed<Array<StmtSRef>(Schedule, std::string, StmtSRef)>(
    [](Schedule schedule, std::string tag, StmtSRef scope) {
      return schedule.GetBlock(tag, scope);
    });

TVM_REGISTER_GLOBAL("te.schedule.GetBlocksFromBuffer")
.set_body_typed<Array<StmtSRef>(Schedule, Buffer, StmtSRef)>(
    [](Schedule schedule, Buffer buffer, StmtSRef scope) {
      return schedule.GetBlock(buffer, scope);
    });

TVM_REGISTER_GLOBAL("te.schedule.ScheduleGetAxes")
.set_body_method(&Schedule::GetAxes);

// schedule primitive
TVM_REGISTER_GLOBAL("te.schedule.ScheduleFuse")
.set_body_method(&Schedule::fuse);

TVM_REGISTER_GLOBAL("te.schedule.ScheduleSplitByFactor")
.set_body_method(& Schedule::split);

TVM_REGISTER_GLOBAL("te.schedule.ScheduleSplitByNParts")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, Expr)>(
    [](Schedule schedule, StmtSRef node, Expr nparts) {
      const auto* loop = GetRef<Stmt>(node->node).as<LoopNode>();
      return schedule.split(node, truncdiv(loop->extent + nparts - 1, nparts));
    });

// dependency graph
TVM_REGISTER_GLOBAL("te.schedule.GetSuccessors")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, StmtSRef)>(
    [](Schedule schedule, StmtSRef scope, StmtSRef block) {
      return schedule->scopes_[scope].GetSuccessors(block);
    });

TVM_REGISTER_GLOBAL("te.schedule.GetPredecessors")
.set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, StmtSRef)>(
    [](Schedule schedule, StmtSRef scope, StmtSRef block) {
      return schedule->scopes_[scope].GetPredecessors(block);
    });

// maker
TVM_REGISTER_GLOBAL("make.TensorRegion")
.set_body_typed<TensorRegion(Buffer, Array<Range>)>(
    [](Buffer buffer, Array<Range> region) {
      return TensorRegion(buffer, region);
    });

TVM_REGISTER_GLOBAL("make.BufferAllocate")
.set_body_typed<BufferAllocate(Buffer, std::string)>(
    [](Buffer buffer, std::string scope) {
      return BufferAllocate(buffer, scope);
    });

TVM_REGISTER_GLOBAL("make.BufferLoad")
.set_body_typed<BufferLoad(DataType, Buffer, Array<Expr>)>(
    [](DataType type, Buffer buffer, Array<Expr> indices) {
      return BufferLoad(type, buffer, indices);
    });

TVM_REGISTER_GLOBAL("make.BufferStore")
.set_body_typed<BufferStore(Buffer, Expr, Array<Expr>)>(
    [](Buffer buffer, Expr value, Array<Expr> indices) {
      return BufferStore(buffer, value, indices);
    });

TVM_REGISTER_GLOBAL("make.Loop")
.set_body_typed<Loop(Var, Expr, Expr, Array<Annotation>, Stmt)>(
    [](Var loop_var, Expr min, Expr extent,
       Array<Annotation> annotations, Stmt body) {
      return Loop(loop_var, min, extent, annotations, body);
    });

TVM_REGISTER_GLOBAL("make.TeBlock")
.set_body_typed<Block(Array<IterVar>,
                      Array<Expr>,
                      Array<TensorRegion>,
                      Array<TensorRegion>,
                      Stmt, Expr,
                      Array<BufferAllocate>,
                      Array<Annotation>,
                      std::string)>(
    [](Array<IterVar> iter_vars,
       Array<Expr> values,
       Array<TensorRegion> reads,
       Array<TensorRegion> writes,
       Stmt body,
       Expr predicate,
       Array<BufferAllocate> allocates,
       Array<Annotation> annotations,
       std::string tag) {
      if (!predicate.dtype().is_bool()) {
        // To support python ir_builder
        CHECK(is_one(predicate));
        predicate = UIntImm::make(DataType::Bool(), 1);
      }
      return Block(iter_vars, values, reads, writes,
                   body, predicate, allocates, annotations, tag);
    });

TVM_REGISTER_GLOBAL("make.TeFunction")
.set_body_typed<Function(Array<Var>, Map<Var, Buffer>,
                         std::string, Stmt)>(
    [](Array<Var> params, Map<Var, Buffer> buffer_map,
       std::string name, Stmt body) {
      return Function(params, buffer_map, name, body);
    });

}  // namespace te
}  // namespace tvm
