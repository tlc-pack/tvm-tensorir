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

Schedule::Schedule(ScheduleState state, TSymbolTable symbol_table) {
  ObjectPtr<ScheduleNode> n = make_object<ScheduleNode>();
  n->state = std::move(state);
  n->symbol_table = std::move(symbol_table);
  this->data_ = std::move(n);
}

Schedule::Schedule(PrimFunc func, bool debug_mode) : Schedule(ScheduleState(func, debug_mode)) {}

/******** Lookup random variables ********/

Block ScheduleNode::Get(const BlockRV& block_rv) const {
  auto it = this->symbol_table.find(block_rv);
  if (it == this->symbol_table.end()) {
    LOG(FATAL) << "IndexError: Cannot find corresponding BlockRV: " << block_rv;
  }
  const ObjectRef& obj = (*it).second;
  const auto* sref = obj.as<StmtSRefNode>();
  if (sref == nullptr) {
    LOG(FATAL) << "ValueError: BlockRV's corresponding type is invalid: "
               << (obj.defined() ? obj->GetTypeKey() : "None");
  }
  if (sref->stmt) {
    LOG(FATAL) << "ValueError: The StmtSRef has expired";
  }
  const auto* block = TVM_SREF_TO_BLOCK(block, sref);
  return GetRef<Block>(block);
}

For ScheduleNode::Get(const LoopRV& loop_rv) const {
  auto it = this->symbol_table.find(loop_rv);
  if (it == this->symbol_table.end()) {
    LOG(FATAL) << "IndexError: Cannot find corresponding LoopRV: " << loop_rv;
  }
  const ObjectRef& obj = (*it).second;
  const auto* sref = obj.as<StmtSRefNode>();
  if (sref == nullptr) {
    LOG(FATAL) << "ValueError: LoopRV's corresponding type is invalid: "
               << (obj.defined() ? obj->GetTypeKey() : "None");
  }
  if (sref->stmt) {
    LOG(FATAL) << "ValueError: The StmtSRef has expired";
  }
  const auto* loop = TVM_SREF_TO_FOR(loop, sref);
  return GetRef<For>(loop);
}

int64_t ScheduleNode::Get(const Var& var_rv) const {
  auto it = this->symbol_table.find(var_rv);
  if (it == this->symbol_table.end()) {
    LOG(FATAL) << "IndexError: Cannot find corresponding LoopRV: " << var_rv;
  }
  const ObjectRef& obj = (*it).second;
  const auto* int_imm = obj.as<IntImmNode>();
  if (int_imm == nullptr) {
    LOG(FATAL) << "ValueError: VarRV's corresponding type is invalid: "
               << (obj.defined() ? obj->GetTypeKey() : "None");
  }
  return int_imm->value;
}

int64_t ScheduleNode::Get(const ExprRV& expr_rv) const {
  // Replace all the tir::Var with their corresponding value in the symbol table
  PrimExpr transformed = tir::Substitute(
      expr_rv,
      [this](const tir::Var& var) -> Optional<PrimExpr> { return Integer(this->Get(var)); });
  PrimExpr result = arith::Analyzer().Simplify(transformed);
  const auto* int_imm = result.as<IntImmNode>();
  ICHECK(int_imm) << "ValueError: Expects Integer, but gets type: " << result->GetTypeKey()
                  << ", value = " << result;
  return int_imm->value;
}
/******** Block/Loop relation ********/

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(BlockRVNode);
TVM_REGISTER_NODE_TYPE(LoopRVNode);
TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCopy")  //
    .set_body_method<Schedule>(&ScheduleNode::Copy);

/**************** (FFI) Constructor ****************/

TVM_REGISTER_GLOBAL("tir.schedule.ConcreteSchedule")
    .set_body_typed([](ObjectRef obj, int debug_mode, int error_render_level) -> Schedule {
      IRModule mod{nullptr};
      if (const auto* func = obj.as<PrimFuncNode>()) {
        mod = IRModule({{GlobalVar("main"), GetRef<BaseFunc>(func)}});
      } else if (const auto* p_mod = obj.as<IRModuleNode>()) {
        mod = GetRef<IRModule>(p_mod);
      } else {
        LOG(FATAL) << "TypeError: Expects `IRModule` or `PrimFunc`, but gets: "
                   << obj->GetTypeKey();
Array<StmtSRef> ScheduleNode::GetBlock(const String& name) const {
  return schedule::GetBlock(this->state, name);
}

Array<StmtSRef> ScheduleNode::GetAxes(const StmtSRef& block) const {
  return schedule::GetAxes(this->state, block);
}

Array<StmtSRef> ScheduleNode::GetChildBlocks(const StmtSRef& parent_sref) const {
  return schedule::GetChildBlocks(this->state, parent_sref);
}

/******** Schedule: loops ********/

StmtSRef ScheduleNode::Fuse(const StmtSRef& outer_sref, const StmtSRef& inner_sref) {
  return schedule::Fuse(this->state, outer_sref, inner_sref);
}

Array<StmtSRef> ScheduleNode::Split(const StmtSRef& loop_sref, const PrimExpr& nparts,
                                    const PrimExpr& factor) {
  return schedule::Split(this->state, loop_sref, nparts, factor);
}

void ScheduleNode::Reorder(const Array<StmtSRef>& order) { schedule::Reorder(this->state, order); }

/******** Schedule: compute location ********/

void ScheduleNode::ComputeAt(const StmtSRef& block_sref, const StmtSRef& loop_sref,
                             bool preserve_trivial_loop) {
  schedule::ComputeAt(this->state, block_sref, loop_sref, preserve_trivial_loop);
}

void ScheduleNode::ReverseComputeAt(const StmtSRef& block_sref, const StmtSRef& loop_sref,
                                    bool preserve_trivial_loop) {
  schedule::ReverseComputeAt(this->state, block_sref, loop_sref, preserve_trivial_loop);
}

void ScheduleNode::ComputeInline(const StmtSRef& block_sref) {
  schedule::ComputeInline(this->state, block_sref);
}

void ScheduleNode::ReverseComputeInline(const StmtSRef& block_sref) {
  schedule::ReverseComputeInline(this->state, block_sref);
}

/******** Schedule: parallelize / annotate ********/

void ScheduleNode::Vectorize(const StmtSRef& loop_sref) {
  schedule::Vectorize(this->state, loop_sref);
}

void ScheduleNode::Parallel(const StmtSRef& loop_sref) {
  schedule::Parallel(this->state, loop_sref);
}

void ScheduleNode::Unroll(const StmtSRef& loop_sref) { schedule::Unroll(this->state, loop_sref); }

void ScheduleNode::Bind(const StmtSRef& loop_sref, const IterVar& thread) {
  schedule::Bind(this->state, loop_sref, thread);
}

void ScheduleNode::DoubleBuffer(const StmtSRef& block_sref) {
  schedule::DoubleBuffer(this->state, block_sref);
}

void ScheduleNode::Pragma(const StmtSRef& loop_sref, const String& pragma_type,
                          const PrimExpr& pragma_value) {
  schedule::Pragma(this->state, loop_sref, pragma_type, pragma_value);
}

/******** Schedule: cache read/write ********/

StmtSRef ScheduleNode::CacheRead(const StmtSRef& block_sref, int i, const String& storage_scope) {
  return schedule::CacheRead(this->state, block_sref, i, storage_scope);
}

StmtSRef ScheduleNode::CacheWrite(const StmtSRef& block_sref, int i, const String& storage_scope) {
  return schedule::CacheWrite(this->state, block_sref, i, storage_scope);
}

/******** Schedule: reduction ********/

StmtSRef ScheduleNode::RFactor(const StmtSRef& loop_sref, int factor_axis) {
  return schedule::RFactor(this->state, loop_sref, factor_axis);
}

StmtSRef ScheduleNode::DecomposeReduction(const StmtSRef& block_sref,
                                          const Optional<StmtSRef>& loop_sref) {
  return schedule::DecomposeReduction(this->state, block_sref, loop_sref);
}

void ScheduleNode::MergeReduction(const StmtSRef& init_sref, const StmtSRef& update_sref) {
  schedule::MergeReduction(this->state, init_sref, update_sref);
}

/******** Blockize / Tensorize ********/

StmtSRef ScheduleNode::Blockize(const StmtSRef& loop_sref, const String& exec_scope) {
  return schedule::Blockize(this->state, loop_sref, exec_scope);
}

void ScheduleNode::Tensorize(const StmtSRef& loop_sref, const TensorIntrin& intrinsic) {
  schedule::Tensorize(this->state, loop_sref, intrinsic);
}

/**************** FFI ****************/

TVM_REGISTER_NODE_TYPE(BlockRVNode);
TVM_REGISTER_NODE_TYPE(LoopRVNode);
TVM_REGISTER_NODE_TYPE(ScheduleNode);

TVM_REGISTER_GLOBAL("tir.schedule.Schedule").set_body_typed([](PrimFunc func, bool debug_mode) {
  return Schedule(func, debug_mode);
});

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGet")
    .set_body_typed([](Schedule self, ObjectRef obj) -> ObjectRef {
      if (const auto* loop_rv = obj.as<LoopRVNode>()) {
        return self->Get(GetRef<LoopRV>(loop_rv));
      }
      if (const auto* block_rv = obj.as<BlockRVNode>()) {
        return self->Get(GetRef<BlockRV>(block_rv));
      }
      if (const auto* expr_rv = obj.as<PrimExprNode>()) {
        int64_t result = self->Get(GetRef<PrimExpr>(expr_rv));
        return IntImm(DataType::Int(64), result);
      }
      LOG(FATAL) << "TypeError: Cannot evaluate the random variable of type: " << obj->GetTypeKey()
                 << ". Its value is: " << obj;
      throw;
    });

/***** Block/Loop relation *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetBlock").set_body_typed([](Schedule self, String name) {
  return self->GetBlock(name);
});

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetAxes")
    .set_body_typed([](Schedule self, StmtSRef block) { return self->GetAxes(block); });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleGetChildBlocks")
    .set_body_typed([](Schedule self, StmtSRef block_or_loop) {
      return self->GetChildBlocks(block_or_loop);
    });

/***** Schedule: loops *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleFuse")
    .set_body_typed([](Schedule self, StmtSRef outer, StmtSRef inner) {
      return self->Fuse(outer, inner);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleSplit")
    .set_body_typed([](Schedule self, StmtSRef loop_sref, Optional<PrimExpr> nparts,
                       Optional<PrimExpr> factor) {
      const auto* loop = loop_sref->GetStmt<ForNode>();
      if (nparts.defined() && factor.defined()) {
        LOG(FATAL) << "ValueError: `nparts` and `factor` cannot be specified at the same time";
      }
      if (!nparts.defined() && !factor.defined()) {
        LOG(FATAL) << "ValueError: neither `nparts` nor `factor` is specified";
      }
      if (!nparts.defined()) {
        PrimExpr factor_ = factor.value();
        return self->Split(loop_sref, floordiv(loop->extent + factor_ - 1, factor_), factor_);
      } else {
        PrimExpr nparts_ = nparts.value();
        return self->Split(loop_sref, nparts_, floordiv(loop->extent + nparts_ - 1, nparts_));
      }
      return Schedule::Concrete(mod, debug_mode,
                                static_cast<ScheduleErrorRenderLevel>(error_render_level));
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
      if (const auto* block_rv = obj.as<BlockRVNode>()) {
        return self->RemoveRV(GetRef<BlockRV>(block_rv));
      }
      if (const auto* expr_rv = obj.as<ExprRVNode>()) {
        return self->RemoveRV(GetRef<ExprRV>(expr_rv));
      }
      LOG(FATAL) << "TypeError: Invalid type: " << obj->GetTypeKey();
      throw;
    });


TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReorder")
    .set_body_typed([](Schedule self, Array<StmtSRef> order) { self->Reorder(order); });

/***** Schedule: compute location *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleComputeAt")
    .set_body_typed([](Schedule self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
                       bool preserve_unit_loop) {
      self->ComputeAt(block_sref, loop_sref, preserve_unit_loop);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReverseComputeAt")
    .set_body_typed([](Schedule self, StmtSRef block_sref, StmtSRef loop_sref,
                       bool preserve_unit_loop) {
      self->ReverseComputeAt(block_sref, loop_sref, preserve_unit_loop);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleComputeInline")
    .set_body_typed([](Schedule self, StmtSRef block_sref) { self->ComputeInline(block_sref); });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleReverseComputeInline")
    .set_body_typed([](Schedule self, StmtSRef block_sref) {
      self->ReverseComputeInline(block_sref);
    });

/***** Schedule: parallelize / annotate *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleVectorize")
    .set_body_typed([](Schedule self, StmtSRef loop_sref) { self->Vectorize(loop_sref); });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleParallel")
    .set_body_typed([](Schedule self, StmtSRef loop_sref) { self->Parallel(loop_sref); });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleUnroll")
    .set_body_typed([](Schedule self, StmtSRef loop_sref) { self->Unroll(loop_sref); });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleBind")
    .set_body_typed([](Schedule self, StmtSRef loop_sref, IterVar thread) {
      self->Bind(loop_sref, thread);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleDoubleBuffer")
    .set_body_typed([](Schedule self, StmtSRef block_sref) { self->DoubleBuffer(block_sref); });

TVM_REGISTER_GLOBAL("tir.schedule.SchedulePragma")
    .set_body_typed([](Schedule self, StmtSRef loop_sref, String pragma_type,
                       PrimExpr pragma_value) {
      self->Pragma(loop_sref, pragma_type, pragma_value);
    });

/***** Schedule: cache read/write *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCacheRead")
    .set_body_typed([](Schedule self, StmtSRef block_sref, int i, String storage_scope) {
      return self->CacheRead(block_sref, i, storage_scope);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleCacheWrite")
    .set_body_typed([](Schedule self, StmtSRef block_sref, int i, String storage_scope) {
      return self->CacheWrite(block_sref, i, storage_scope);
    });

/***** Schedule: reduction *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleRFactor")
    .set_body_typed([](Schedule self, StmtSRef loop_sref, int factor_axis) {
      return self->RFactor(loop_sref, factor_axis);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleDecomposeReduction")
    .set_body_typed([](Schedule self, StmtSRef block_sref, Optional<StmtSRef> loop_sref) {
      return self->DecomposeReduction(block_sref, loop_sref);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleMergeReduction")
    .set_body_typed([](Schedule self, StmtSRef init_sref, StmtSRef update_sref) {
      return self->MergeReduction(init_sref, update_sref);
    });

/***** Schedule: blockize / tensorize *****/

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleBlockize")
    .set_body_typed([](Schedule self, StmtSRef loop_sref, String exec_scope) {
      return self->Blockize(loop_sref, exec_scope);
    });

TVM_REGISTER_GLOBAL("tir.schedule.ScheduleTensorize")
    .set_body_typed([](Schedule self, StmtSRef loop_sref, TensorIntrin intrinsic) {
      self->Tensorize(loop_sref, intrinsic);
    });
}  // namespace tir
}  // namespace tvm
