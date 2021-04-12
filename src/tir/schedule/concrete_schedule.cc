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

/*! \brief Helper class to do StmtSRef translation */
class ScheduleCopier {
  using TSymbolTable = ConcreteScheduleNode::TSymbolTable;
  template <class K, class V>
  using UMap = std::unordered_map<K, V>;
  template <class K, class V>
  using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;

 public:
  static void Copy(const ConcreteScheduleNode* self,
                   ScheduleState* new_state,  //
                   TSymbolTable* new_symbol_table) {
    const ScheduleState& src_state = self->state_;
    ScheduleCopier trans(src_state);
    ObjectPtr<ScheduleStateNode> n = make_object<ScheduleStateNode>();
    n->mod = src_state->mod;
    n->block_info = trans.Trans(src_state->block_info);
    n->stmt2ref = trans.Trans(src_state->stmt2ref);
    n->debug_mode = src_state->debug_mode;
    *new_state = ScheduleState(std::move(n));
    *new_symbol_table = trans.Trans(self->symbol_table_);
  }

 private:
  /*! \brief Create the translator and properly set up the translation table */
  explicit ScheduleCopier(const ScheduleState& state) : trans_() {
    // Create SRef tree without parents
    for (const auto& kv : state->stmt2ref) {
      const StmtSRefNode* sref = kv.second.operator->();
      trans_.emplace(sref,                          // the old StmtSRef
                     StmtSRef(/*stmt=*/sref->stmt,  // the new StmtSRef
                              /*parent=*/nullptr,   // parent is not set yet
                              /*seq_index=*/sref->seq_index));
    }
    // Fill in the parent field
    // Find out the root along the way
    for (auto& kv : trans_) {
      const StmtSRefNode* parent = kv.first->parent;
      StmtSRef& sref = kv.second;
      sref->parent = parent ? trans_.at(parent).get() : nullptr;
    }
  }

  /*! \brief Translate StmtSRef */
  StmtSRef Trans(const StmtSRef& sref) { return trans_.at(sref.operator->()); }

  /*! \brief Translate StmtSRefNode */
  StmtSRef Trans(const StmtSRefNode* sref) {
    if (trans_.count(sref)) {
      return trans_.at(sref);
    }
    // Handle expired sref
    return trans_[sref] = StmtSRef(nullptr, nullptr, -1);
  }

  /*! \brief Translate Array<StmtSRef> */
  Array<StmtSRef> Trans(const Array<StmtSRef>& list) {
    Array<StmtSRef> result;
    result.reserve(list.size());
    for (const StmtSRef& elem : list) {
      result.push_back(Trans(elem));
    }
    return result;
  }

  /*! \brief Translate Array<Dependency> */
  Array<Dependency> Trans(const Array<Dependency>& list) {
    Array<Dependency> result;
    result.reserve(list.size());
    for (const Dependency& elem : list) {
      result.push_back(Dependency(Trans(elem->src), Trans(elem->dst), elem->kind));
    }
    return result;
  }

  /*! \brief Translate SMap<StmtSRef, Array<Dependency>> */
  SMap<StmtSRef, Array<Dependency>> Trans(const SMap<StmtSRef, Array<Dependency>>& map) {
    SMap<StmtSRef, Array<Dependency>> result;
    result.reserve(map.size());
    for (const auto& kv : map) {
      result[Trans(kv.first)] = Trans(kv.second);
    }
    return result;
  }

  /*! \brief Translate SMap<Buffer, Array<StmtSRef>> */
  SMap<Buffer, Array<StmtSRef>> Trans(const SMap<Buffer, Array<StmtSRef>>& map) {
    SMap<Buffer, Array<StmtSRef>> result;
    result.reserve(map.size());
    for (const auto& kv : map) {
      result[kv.first] = Trans(kv.second);
    }
    return result;
  }

  /*! \brief Translate SMap<StmtSRef, Scope> */
  SMap<StmtSRef, BlockInfo> Trans(const SMap<StmtSRef, BlockInfo>& scopes) {
    SMap<StmtSRef, BlockInfo> result;
    for (const auto& kv : scopes) {
      const StmtSRef& old_sref = kv.first;
      const BlockInfo& old_info = kv.second;
      BlockInfo new_info = old_info;
      ObjectPtr<BlockScopeNode> scope = make_object<BlockScopeNode>();
      scope->src2deps = Trans(old_info.scope->src2deps);
      scope->dst2deps = Trans(old_info.scope->dst2deps);
      scope->buffer_writers = Trans(old_info.scope->buffer_writers);
      new_info.scope = BlockScope(std::move(scope));
      result[Trans(old_sref)] = std::move(new_info);
    }
    return result;
  }

  /*! \brief Translate the stmt2ref */
  UMap<const StmtNode*, StmtSRef> Trans(const UMap<const StmtNode*, StmtSRef>& stmt2ref) {
    UMap<const StmtNode*, StmtSRef> result;
    result.reserve(stmt2ref.size());
    for (const auto& kv : stmt2ref) {
      const StmtNode* stmt = kv.first;
      const StmtSRef& sref = kv.second;
      result.emplace(stmt, Trans(sref));
    }
    return result;
  }

  /*! \brief Translate the symbol table */
  TSymbolTable Trans(const TSymbolTable& tab) {
    TSymbolTable result;
    for (const auto& kv : tab) {
      ObjectRef entry = kv.second;
      if (const auto* sref = entry.as<StmtSRefNode>()) {
        entry = Trans(sref);
      }
      result.Set(kv.first, entry);
    }
    return result;
  }

 private:
  std::unordered_map<const StmtSRefNode*, StmtSRef> trans_;
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
  return SetRV<BlockRV>(blocks[0]);
}

Array<LoopRV> ConcreteScheduleNode::GetAxes(const BlockRV& block_rv) {
  return SetRV<LoopRV>(tir::GetAxes(this->GetSRef(block_rv)));
}

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(ConcreteScheduleNode);

}  // namespace tir
}  // namespace tvm
