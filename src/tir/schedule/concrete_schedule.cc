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
#include "./concrete_schedule.h"

namespace tvm {
namespace tir {

Schedule Schedule::Concrete(IRModule mod, int debug_mode) {
  ObjectPtr<ConcreteScheduleNode> n = make_object<ConcreteScheduleNode>();
  n->state_ = ScheduleState(mod, debug_mode);
  n->symbol_table_ = {};
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  return Schedule(std::move(n));
}

/******** Copy ********/

/*! \brief Helper class to perform a deep copy of the sref tree */
class ScheduleCopier {
  using TSymbolTable = ConcreteScheduleNode::TSymbolTable;
  template <class K, class V>
  using UMap = std::unordered_map<K, V>;
  template <class K, class V>
  using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;

 public:
  static void Copy(const ConcreteScheduleNode* self, ScheduleState* new_state,
                   TSymbolTable* new_symbol_table) {
    const ScheduleState& src_state = self->state_;
    ScheduleCopier copier(src_state);
    ObjectPtr<ScheduleStateNode> n = make_object<ScheduleStateNode>();
    n->mod = src_state->mod;
    n->block_info = copier.Copy(src_state->block_info);
    n->stmt2ref = copier.Copy(src_state->stmt2ref);
    n->debug_mode = src_state->debug_mode;
    *new_state = ScheduleState(std::move(n));
    *new_symbol_table = copier.Copy(self->symbol_table_);
  }

 private:
  /*! \brief Create the copier and properly set up the `old2new_` table */
  explicit ScheduleCopier(const ScheduleState& state) {
    // Create SRef tree without parents
    for (const auto& kv : state->stmt2ref) {
      const StmtSRefNode* sref = kv.second.operator->();
      old2new_.emplace(sref,                          // the old StmtSRef
                       StmtSRef(/*stmt=*/sref->stmt,  // the new StmtSRef
                                /*parent=*/nullptr,   // parent is not set yet
                                /*seq_index=*/sref->seq_index));
    }
    // Fill in the parent field
    // Find out the root along the way
    for (auto& kv : old2new_) {
      const StmtSRefNode* parent = kv.first->parent;
      StmtSRef& sref = kv.second;
      sref->parent = parent ? old2new_.at(parent).get() : nullptr;
    }
  }

  /*! \brief Copy StmtSRef */
  StmtSRef Copy(const StmtSRef& sref) { return old2new_.at(sref.operator->()); }

  /*! \brief Copy StmtSRefNode */
  StmtSRef Copy(const StmtSRefNode* sref) {
    if (old2new_.count(sref)) {
      return old2new_.at(sref);
    }
    // Handle expired sref
    return old2new_[sref] = StmtSRef(nullptr, nullptr, -1);
  }

  /*! \brief Copy Array<StmtSRef> */
  Array<StmtSRef> Copy(const Array<StmtSRef>& list) {
    Array<StmtSRef> result;
    result.reserve(list.size());
    for (const StmtSRef& elem : list) {
      result.push_back(Copy(elem));
    }
    return result;
  }

  /*! \brief Copy Array<Dependency> */
  Array<Dependency> Copy(const Array<Dependency>& list) {
    Array<Dependency> result;
    result.reserve(list.size());
    for (const Dependency& elem : list) {
      result.push_back(Dependency(Copy(elem->src), Copy(elem->dst), elem->kind));
    }
    return result;
  }

  /*! \brief Copy SMap<StmtSRef, Array<Dependency>> */
  SMap<StmtSRef, Array<Dependency>> Copy(const SMap<StmtSRef, Array<Dependency>>& map) {
    SMap<StmtSRef, Array<Dependency>> result;
    result.reserve(map.size());
    for (const auto& kv : map) {
      result[Copy(kv.first)] = Copy(kv.second);
    }
    return result;
  }

  /*! \brief Copy SMap<Buffer, Array<StmtSRef>> */
  SMap<Buffer, Array<StmtSRef>> Copy(const SMap<Buffer, Array<StmtSRef>>& map) {
    SMap<Buffer, Array<StmtSRef>> result;
    result.reserve(map.size());
    for (const auto& kv : map) {
      result[kv.first] = Copy(kv.second);
    }
    return result;
  }

  /*! \brief Copy SMap<StmtSRef, Scope> */
  SMap<StmtSRef, BlockInfo> Copy(const SMap<StmtSRef, BlockInfo>& scopes) {
    SMap<StmtSRef, BlockInfo> result;
    for (const auto& kv : scopes) {
      const StmtSRef& old_sref = kv.first;
      BlockInfo new_info = old_info;
      ObjectPtr<BlockScopeNode> scope = make_object<BlockScopeNode>();
      scope->src2deps = Copy(old_info.scope->src2deps);
      scope->dst2deps = Copy(old_info.scope->dst2deps);
      scope->buffer_writers = Copy(old_info.scope->buffer_writers);
      new_info.scope = BlockScope(std::move(scope));
      result[Copy(old_sref)] = std::move(new_info);
    }
      const StmtNode* stmt = kv.first;
      const StmtSRef& sref = kv.second;
      result.emplace(stmt, Copy(sref));
    }
    return result;
  }

  /*! \brief Copy the symbol table */
  TSymbolTable Copy(const TSymbolTable& tab) {
    TSymbolTable result;
    for (const auto& kv : tab) {
      ObjectRef entry = kv.second;
      if (const auto* sref = entry.as<StmtSRefNode>()) {
        entry = Copy(sref);
      }
      result.Set(kv.first, entry);
    }
    return result;
  }

 private:
  std::unordered_map<const StmtSRefNode*, StmtSRef> old2new_;
};

void ConcreteScheduleNode::Copy(ScheduleState* new_state, TSymbolTable* new_symbol_table) const {
  ScheduleCopier::Copy(this, new_state, new_symbol_table);
}

Schedule ConcreteScheduleNode::Copy() const {
  ObjectPtr<ConcreteScheduleNode> n = make_object<ConcreteScheduleNode>();
  Copy(&n->state_, &n->symbol_table_);
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  return Schedule(std::move(n));
}

/******** Block/Loop relation ********/

BlockRV ConcreteScheduleNode::GetBlock(const String& name, const String& func_name) {
  Array<StmtSRef> blocks = tir::GetBlocks(this->state_, name, func_name);
  CHECK_EQ(blocks.size(), 1) << "ValueError: There are " << blocks.size()
                             << " blocks with the name: " << name;
  return CreateRV<BlockRV>(blocks[0]);
}

Array<LoopRV> ConcreteScheduleNode::GetLoops(const BlockRV& block_rv) {
  return CreateRV<LoopRV>(tir::GetLoops(this->GetSRef(block_rv)));
void ConcreteScheduleNode::Bind(const LoopRV& loop_rv, const IterVar& thread) {
  schedule::Bind(state_, this->GetSRef(loop_rv), thread);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::Bind(const LoopRV& loop_rv, const String& thread) {
  IterVar iter_var(Range(nullptr),  //
                   Var(thread),     //
                   kThreadIndex,    //
                   thread);
  schedule::Bind(state_, this->GetSRef(loop_rv), iter_var);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::DoubleBuffer(const BlockRV& block_rv) {
  schedule::DoubleBuffer(state_, this->GetSRef(block_rv));
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::SetScope(const BlockRV& block_rv, int i, const String& storage_scope) {
  schedule::SetScope(state(), this->GetSRef(block_rv), i, storage_scope);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::Pragma(const LoopRV& loop_rv, const String& pragma_type,
                                  const ExprRV& pragma_value) {
  schedule::Pragma(state_,                  //
                   this->GetSRef(loop_rv),  //
                   pragma_type,             //
                   this->Get(pragma_value));
  this->state_->DebugVerify();
}

/******** Schedule: cache read/write ********/

BlockRV ConcreteScheduleNode::CacheRead(const BlockRV& block_rv, int i,
                                        const String& storage_scope) {
  StmtSRef result = schedule::CacheRead(state_,                   //
                                        this->GetSRef(block_rv),  //
                                        i,                        //
                                        storage_scope);
  this->state_->DebugVerify();
  return SetRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::CacheWrite(const BlockRV& block_rv, int i,
                                         const String& storage_scope) {
  StmtSRef result = schedule::CacheWrite(state_,                   //
                                         this->GetSRef(block_rv),  //
                                         i,                        //
                                         storage_scope);
  this->state_->DebugVerify();
  return SetRV<BlockRV>(result);
}

/******** Schedule: reduction ********/

BlockRV ConcreteScheduleNode::RFactor(const LoopRV& loop_rv, int factor_axis) {
  StmtSRef result = schedule::RFactor(state_, this->GetSRef(loop_rv), factor_axis);
  this->state_->DebugVerify();
  return SetRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::DecomposeReduction(const BlockRV& block_rv,
                                                 const Optional<LoopRV>& opt_loop_rv) {
  Optional<StmtSRef> opt_loop_sref = opt_loop_rv.defined() ?                 //
                                         this->GetSRef(opt_loop_rv.value())  //
                                                           : Optional<StmtSRef>(NullOpt);
  StmtSRef result = schedule::DecomposeReduction(state_,                   //
                                                 this->GetSRef(block_rv),  //
                                                 opt_loop_sref);
  this->state_->DebugVerify();
  return SetRV<BlockRV>(result);
}

void ConcreteScheduleNode::MergeReduction(const BlockRV& init_block_rv,
                                          const BlockRV& update_block_rv) {
  schedule::MergeReduction(state_,                        //
                           this->GetSRef(init_block_rv),  //
                           this->GetSRef(update_block_rv));
  this->state_->DebugVerify();
}

/******** Schedule: blockize / tensorize ********/

BlockRV ConcreteScheduleNode::Blockize(const LoopRV& loop_rv) {
  StmtSRef result = schedule::Blockize(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
  return SetRV<BlockRV>(result);
}

void ConcreteScheduleNode::Tensorize(const LoopRV& loop_rv, const TensorIntrin& intrin) {
  schedule::Tensorize(state_, this->GetSRef(loop_rv), intrin);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::Tensorize(const LoopRV& loop_rv, const String& intrin_name) {
  schedule::Tensorize(state_, this->GetSRef(loop_rv), tir::TensorIntrin::Get(intrin_name));
  this->state_->DebugVerify();
}

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(ConcreteScheduleNode);

}  // namespace tir
}  // namespace tvm
