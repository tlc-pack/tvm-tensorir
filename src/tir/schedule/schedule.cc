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
#include <tvm/tir/schedule/schedule.h>

namespace tvm {
namespace tir {

/**************** Constructor ****************/

BlockRV::BlockRV() { this->data_ = make_object<BlockRVNode>(); }

LoopRV::LoopRV() { this->data_ = make_object<LoopRVNode>(); }

/**************** GetSRef ****************/

StmtSRef ScheduleNode::GetSRef(const StmtNode* stmt) const {
  ScheduleState state = this->state();
  auto it = state->stmt2ref.find(stmt);
  if (it == state->stmt2ref.end()) {
    LOG(FATAL) << "IndexError: The stmt doesn't exist in the IR";
  }
  return it->second;
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(BlockRVNode);
TVM_REGISTER_NODE_TYPE(LoopRVNode);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCopy")  //
    .set_body_method<Schedule>(&ScheduleNode::Copy);

/**************** (FFI) Constructor ****************/

TVM_REGISTER_GLOBAL("tir.schedule.ConcreteSchedule")
    .set_body_typed([](ObjectRef obj, int debug_mode) -> Schedule {
      IRModule mod{nullptr};
      if (const auto* func = obj.as<PrimFuncNode>()) {
        mod = IRModule({{GlobalVar("main"), GetRef<BaseFunc>(func)}});
      } else if (const auto* p_mod = obj.as<IRModuleNode>()) {
        mod = GetRef<IRModule>(p_mod);
      } else {
        LOG(FATAL) << "TypeError: Expects `IRModule` or `PrimFunc`, but gets: "
                   << obj->GetTypeKey();
      }
      return Schedule::Concrete(mod, debug_mode);
    });

/******** (FFI) Lookup random variables ********/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGet")
    .set_body_typed([](Schedule self, ObjectRef obj) -> ObjectRef {
      if (const auto* loop_rv = obj.as<LoopRVNode>()) {
        return self->Get(GetRef<LoopRV>(loop_rv));
      }
      if (const auto* block_rv = obj.as<BlockRVNode>()) {
        return self->Get(GetRef<BlockRV>(block_rv));
      }
      if (const auto* expr_rv = obj.as<ExprRVNode>()) {
        return self->Get(GetRef<ExprRV>(expr_rv));
      }
      LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << obj->GetTypeKey()
                 << ". Its value is: " << obj;
      throw;
    });
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetSRef")
    .set_body_typed([](Schedule self, ObjectRef obj) -> Optional<ObjectRef> {
      if (const auto* loop_rv = obj.as<LoopRVNode>()) {
        return self->GetSRef(GetRef<LoopRV>(loop_rv));
      }
      if (const auto* block_rv = obj.as<BlockRVNode>()) {
        return self->GetSRef(GetRef<BlockRV>(block_rv));
      }
      if (const auto* stmt = obj.as<StmtNode>()) {
        return self->GetSRef(GetRef<Stmt>(stmt));
      }
      LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
      throw;
    });
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleRemoveRV")
    .set_body_typed([](Schedule self, ObjectRef obj) -> void {
      if (const auto* loop_rv = obj.as<LoopRVNode>()) {
        return self->RemoveRV(GetRef<LoopRV>(loop_rv));
      }
<<<<<<< HEAD
      if (const auto* block_rv = obj.as<BlockRVNode>()) {
        return self->RemoveRV(GetRef<BlockRV>(block_rv));
=======
      failed = true;
      LOG(FATAL) << "Not in stmt2ref:\n" << GetRef<Stmt>(stmt);
    }
    if (failed) {
      LOG(FATAL) << "\n" << Repr(sch->func);
    }
  }
}

void ScheduleNode::Replace(StmtSRef ref, Stmt target, Map<Block, Block> block_sref_map) {
  // Note that old_ref is only a temporary SRef
  StmtSRef old_ref = StmtSRef(ref->stmt, ref->parent);
  auto root_node = root->stmt;
  const Stmt& old_stmt = GetRef<Stmt>(ref->stmt);
  // Collect loop_var to Loop mapping under old stmt
  LoopCollector collector(&stmt2ref);
  collector(old_stmt);
  // Create SRef tree for the incoming target Stmt
  SRefCreator creator(&stmt2ref, std::move(collector.loop_var2sref), &scopes,
                      std::move(block_sref_map), old_ref->parent);
  creator(target);
  // Initialize old SRef remover
  SRefRemover remover(&stmt2ref, std::move(creator.used_border_parent_), &scopes,
                      std::move(creator.reuse_sref_));
  // num_copy_steps: maximum number of hops until we don't need to copy
  int curr_step = 0;
  int num_copy_steps = -1;
  // Find the highest non-unique Stmt
  for (const StmtSRefNode* ptr = old_ref.operator->(); ptr != nullptr;
       ptr = ptr->parent, ++curr_step) {
    if (!ptr->stmt->unique()) {
      num_copy_steps = curr_step;
    }
  }
  if (!func.unique()) num_copy_steps = curr_step;
  // Update the function body
  curr_step = 0;
  for (StmtSRefNode* ptr = old_ref.operator->(); ptr->stmt != root_node;
       ptr = ptr->parent, ++curr_step) {
    StmtSRefNode* parent = ptr->parent;
    // parent_step = current_step + 1
    // if parent_step <= num_copy_step, then it implies
    // that parent is not unique and we need to copy
    bool parent_is_uniquely_referenced = curr_step + 1 > num_copy_steps;
    // replace ptr(son of parent->node) with target and return a new parent Stmt)
    Stmt new_stmt =
        SubReplacer(ptr, target, &stmt2ref)(parent->stmt, parent_is_uniquely_referenced);
    if (curr_step != 0) UpdateSRef(ptr, target);
    if (parent_is_uniquely_referenced) {
      CHECK(new_stmt.get() == parent->stmt);
      // if one node has been direct write, there is no need to
      // update its parent and the function
      remover(old_stmt);
      // ValidateSRefs(GetRef<Schedule>(this));
      return;
    }
    target = new_stmt;
  }
  remover(old_stmt);
  if (old_ref->stmt == root_node) {
    // The replace point is root, we directly use the sref tree created by SRefCreator
    root = stmt2ref[target.operator->()];
  } else {
    // Otherwise we reuse root sref
    UpdateSRef(root.operator->(), target);
  }
  func = UpdateFuncBody(func.operator->(), target);
  // ValidateSRefs(GetRef<Schedule>(this));
}

void ScheduleNode::UpdateSRef(StmtSRefNode* sref, const Stmt& stmt) {
  CHECK(stmt->IsInstance<BlockNode>() || stmt->IsInstance<LoopNode>());
  stmt2ref[stmt.operator->()] = GetRef<StmtSRef>(sref);
  stmt2ref.erase(sref->stmt);
  sref->stmt = stmt.operator->();
}

Array<StmtSRef> ScheduleNode::GetBlock(const std::string& tag) const {
  std::vector<StmtSRef> ret, scope_stack;
  scope_stack.push_back(root);
  while (!scope_stack.empty()) {
    StmtSRef scope = scope_stack.back();
    scope_stack.pop_back();
    CHECK(GetRef<Stmt>(scope->stmt).as<BlockNode>());
    for (const auto& block : Blocks(scope)) {
      if (GetRef<Stmt>(block->stmt).as<BlockNode>()->tag == tag) {
        ret.push_back(block);
>>>>>>> Move methods in Scope to ScopeNode (#292)
      }
      if (const auto* expr_rv = obj.as<ExprRVNode>()) {
        return self->RemoveRV(GetRef<ExprRV>(expr_rv));
      }
      LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
      throw;
    });

/***** (FFI) Block/Loop relation *****/

<<<<<<< HEAD
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetBlock")
    .set_body_method<Schedule>(&ScheduleNode::GetBlock);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetLoops")
    .set_body_method<Schedule>(&ScheduleNode::GetLoops);
=======
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSplitByFactor")
    .set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, PrimExpr)>([](Schedule schedule,
                                                                      StmtSRef node,
                                                                      PrimExpr factor) {
      const auto* loop = GetRef<Stmt>(node->stmt).as<LoopNode>();
      return schedule->split(node, floordiv(loop->extent + factor - 1, factor), factor);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSplitByNParts")
    .set_body_typed<Array<StmtSRef>(Schedule, StmtSRef, PrimExpr)>([](Schedule schedule,
                                                                      StmtSRef node,
                                                                      PrimExpr nparts) {
      const auto* loop = GetRef<Stmt>(node->stmt).as<LoopNode>();
      return schedule->split(node, nparts, floordiv(loop->extent + nparts - 1, nparts));
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReorder")
    .set_body_typed<void(Schedule, Array<StmtSRef>)>([](Schedule schedule, Array<StmtSRef> order) {
      return schedule->reorder(order);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleComputeAt")
    .set_body_typed<void(Schedule, StmtSRef, StmtSRef)>([](Schedule schedule, StmtSRef block_sref,
                                                           StmtSRef loop_sref) {
      return schedule->compute_at(block_sref, loop_sref);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReverseComputeAt")
    .set_body_typed<void(Schedule, StmtSRef, StmtSRef)>([](Schedule schedule, StmtSRef block_sref,
                                                           StmtSRef loop_sref) {
      return schedule->reverse_compute_at(block_sref, loop_sref);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleComputeInline")
    .set_body_typed<void(Schedule, StmtSRef)>([](Schedule schedule, StmtSRef block_sref) {
      return schedule->compute_inline(block_sref);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReverseComputeInline")
    .set_body_typed<void(Schedule, StmtSRef)>([](Schedule schedule, StmtSRef block_sref) {
      return schedule->reverse_compute_inline(block_sref);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleVectorize")
    .set_body_typed<void(Schedule, StmtSRef)>([](Schedule schedule, StmtSRef node) {
      schedule->vectorize(node);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleParallel")
    .set_body_typed<void(Schedule, StmtSRef)>([](Schedule schedule, StmtSRef node) {
      schedule->parallel(node);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleBind")
    .set_body_typed<void(Schedule, StmtSRef, IterVar)>([](Schedule schedule, StmtSRef loop,
                                                          IterVar thread) {
      schedule->bind(loop, thread);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleUnroll")
    .set_body_typed<void(Schedule, StmtSRef)>([](Schedule schedule, StmtSRef node) {
      schedule->unroll(node);
    });

TVM_REGISTER_GLOBAL("tir.schedule.SchedulePragma")
    .set_body_typed<void(Schedule, StmtSRef, String, PrimExpr)>(
        [](Schedule schedule, StmtSRef loop_sref, String pragma_type, PrimExpr pragma_value) {
          schedule->pragma(loop_sref, pragma_type, pragma_value);
        });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleDoubleBuffer")
    .set_body_typed<void(Schedule, StmtSRef)>([](Schedule schedule, StmtSRef block_sref) {
      schedule->double_buffer(block_sref);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleDecomposeReduction")
    .set_body_typed<StmtSRef(Schedule, StmtSRef, Optional<StmtSRef>)>([](Schedule schedule,
                                                                         StmtSRef block,
                                                                         Optional<StmtSRef> loop) {
      return schedule->decompose_reduction(block, loop);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCacheWrite")
    .set_body_typed<StmtSRef(Schedule, StmtSRef, int, std::string)>([](Schedule schedule,
                                                                       StmtSRef block, int i,
                                                                       std::string scope) {
      return schedule->cache_write(block, i, scope);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCacheRead")
    .set_body_typed<StmtSRef(Schedule, StmtSRef, int, std::string)>([](Schedule schedule,
                                                                       StmtSRef block, int i,
                                                                       std::string scope) {
      return schedule->cache_read(block, i, scope);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleMergeReduction")
    .set_body_typed<void(Schedule, StmtSRef, StmtSRef)>([](Schedule schedule, StmtSRef init,
                                                           StmtSRef update) {
      schedule->merge_reduction(init, update);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleRfactor")
    .set_body_typed<StmtSRef(Schedule, StmtSRef, int)>([](Schedule schedule, StmtSRef loop_sref,
                                                          int factor_axis) {
      return schedule->rfactor(loop_sref, factor_axis);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleTensorize")
    .set_body_typed<void(Schedule, StmtSRef, TensorIntrin)>([](Schedule schedule, StmtSRef sref,
                                                               TensorIntrin intrinsic) {
      return schedule->tensorize(sref, intrinsic);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleBlockize")
    .set_body_typed<StmtSRef(Schedule, StmtSRef)>([](Schedule schedule, StmtSRef sref) {
      return schedule->blockize(sref, "");
    });

// dependency graph
TVM_REGISTER_GLOBAL("tir.schedule.GetSuccessors")
    .set_body_typed<Array<DepEdge>(Schedule, StmtSRef, StmtSRef)>([](Schedule schedule,
                                                                     StmtSRef scope,
                                                                     StmtSRef block) {
      return schedule->scopes[scope]->GetSuccessors(block);
    });

TVM_REGISTER_GLOBAL("tir.schedule.GetPredecessors")
    .set_body_typed<Array<DepEdge>(Schedule, StmtSRef, StmtSRef)>([](Schedule schedule,
                                                                     StmtSRef scope,
                                                                     StmtSRef block) {
      return schedule->scopes[scope]->GetPredecessors(block);
    });
>>>>>>> Move methods in Scope to ScopeNode (#292)

}  // namespace tir
}  // namespace tvm
