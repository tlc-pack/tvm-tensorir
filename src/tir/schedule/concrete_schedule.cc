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

#include "./analysis.h"
#include "./primitives/primitives.h"
#include "./utils.h"

namespace tvm {
namespace tir {

Schedule Schedule::Concrete(PrimFunc func, int64_t seed, bool debug_mode) {
  return Schedule::Concrete(IRModule({{GlobalVar("main"), func}}), seed, debug_mode);
}

Schedule Schedule::Concrete(IRModule mod, int64_t seed, bool debug_mode) {
  ObjectPtr<ConcreteScheduleNode> n = make_object<ConcreteScheduleNode>();
  n->state = ScheduleState(mod, debug_mode);
  n->symbol_table = {};
  return Schedule(std::move(n));
}

/******** Copy ********/

/*! \brief Helper class to do StmtSRef translation */
struct SRefTranslator {
  using TSymbolTable = ConcreteScheduleNode::TSymbolTable;
  template <class K, class V>
  using UMap = std::unordered_map<K, V>;
  template <class K, class V>
  using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;

  /*! \brief Create the translator and properly set up the translation table */
  explicit SRefTranslator(const ScheduleState& state) : trans_() {
    // Create SRef tree without parents
    for (const auto& kv : state->stmt2ref) {
      const StmtSRefNode* sref = kv.second.operator->();
      trans_.emplace(sref,                                    // the old StmtSRef
                     StmtSRef(/*stmt=*/sref->stmt,            // the new StmtSRef
                              /*parent=*/nullptr,             // parent is not set yet
                              /*seq_index=*/sref->seq_index,  //
                              /*binding_valid=*/sref->binding_valid));
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
    return trans_[sref] = StmtSRef(nullptr, nullptr, -1, false);
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

  /*! \brief Translate Array<DepEdge> */
  Array<DepEdge> Trans(const Array<DepEdge>& list) {
    Array<DepEdge> result;
    result.reserve(list.size());
    for (const DepEdge& elem : list) {
      result.push_back(DepEdge(Trans(elem->dst), elem->type));
    }
    return result;
  }

  /*! \brief Translate SMap<StmtSRef, Array<DepEdge>> */
  SMap<StmtSRef, Array<DepEdge>> Trans(const SMap<StmtSRef, Array<DepEdge>>& map) {
    SMap<StmtSRef, Array<DepEdge>> result;
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
  SMap<StmtSRef, BlockScope> Trans(const SMap<StmtSRef, BlockScope>& scopes) {
    SMap<StmtSRef, BlockScope> result;
    result.reserve(scopes.size());
    for (const auto& kv : scopes) {
      const StmtSRef& old_sref = kv.first;
      const BlockScope& old_scope = kv.second;
      ObjectPtr<BlockScopeNode> scope = make_object<BlockScopeNode>();
      scope->forward_edges = Trans(old_scope->forward_edges);
      scope->backward_edges = Trans(old_scope->backward_edges);
      scope->buffer_writers = Trans(old_scope->buffer_writers);
      result[Trans(old_sref)] = BlockScope(std::move(scope));
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

Schedule ConcreteScheduleNode::Copy() const {
  const ScheduleState& src_state = this->state;
  SRefTranslator trans(src_state);
  ObjectPtr<ScheduleStateNode> n = make_object<ScheduleStateNode>();
  n->mod = src_state->mod;
  n->scopes = trans.Trans(src_state->scopes);
  n->stmt2ref = trans.Trans(src_state->stmt2ref);
  n->debug_mode = src_state->debug_mode;
  ObjectPtr<ConcreteScheduleNode> p = make_object<ConcreteScheduleNode>();
  p->state = ScheduleState(std::move(n));
  p->symbol_table = trans.Trans(this->symbol_table);
  return Schedule(std::move(p));
}

void ConcreteScheduleNode::Seed(int64_t seed) {
  // do nothing
}

IRModule ConcreteScheduleNode::Module() const { return state->Module(); }

/******** Lookup random variables ********/

Block ConcreteScheduleNode::Get(const BlockRV& block_rv) const {
  StmtSRef sref = this->GetSRef(block_rv);
  const auto* block = TVM_SREF_TO_BLOCK(block, sref);
  return GetRef<Block>(block);
}

For ConcreteScheduleNode::Get(const LoopRV& loop_rv) const {
  StmtSRef sref = this->GetSRef(loop_rv);
  const auto* loop = TVM_SREF_TO_FOR(loop, sref);
  return GetRef<For>(loop);
}

int64_t ConcreteScheduleNode::Get(const Var& var_rv) const {
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

PrimExpr ConcreteScheduleNode::Get(const ExprRV& expr_rv) const {
  // Replace all the Var with their corresponding value in the symbol table
  PrimExpr transformed = Substitute(expr_rv, [this](const Var& var) -> Optional<PrimExpr> {
    int64_t result = this->Get(var);
    return Integer(result);
  });
  return analyzer.Simplify(transformed);
}

StmtSRef ConcreteScheduleNode::GetSRef(const BlockRV& block_rv) const {
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
  if (sref->stmt == nullptr) {
    LOG(FATAL) << "ValueError: The StmtSRef has expired";
  }
  return GetRef<StmtSRef>(sref);
}

StmtSRef ConcreteScheduleNode::GetSRef(const LoopRV& loop_rv) const {
  static StmtSRef inline_mark = StmtSRef::InlineMark();
  static StmtSRef root_mark = StmtSRef::RootMark();
  auto it = this->symbol_table.find(loop_rv);
  if (it == this->symbol_table.end()) {
    LOG(FATAL) << "IndexError: Cannot find corresponding LoopRV: " << loop_rv;
  }
  const ObjectRef& obj = (*it).second;
  if (obj.same_as(inline_mark)) {
    return inline_mark;
  }
  if (obj.same_as(root_mark)) {
    return root_mark;
  }
  const auto* sref = obj.as<StmtSRefNode>();
  if (sref == nullptr) {
    LOG(FATAL) << "ValueError: LoopRV's corresponding type is invalid: "
               << (obj.defined() ? obj->GetTypeKey() : "None");
  }
  if (sref->stmt == nullptr) {
    LOG(FATAL) << "ValueError: The StmtSRef has expired";
  }
  return GetRef<StmtSRef>(sref);
}

StmtSRef ConcreteScheduleNode::GetSRef(const Stmt& stmt) const { return this->GetSRef(stmt.get()); }

StmtSRef ConcreteScheduleNode::GetSRef(const StmtNode* stmt) const {
  auto it = this->state->stmt2ref.find(stmt);
  if (it == this->state->stmt2ref.end()) {
    LOG(FATAL) << "IndexError: The stmt doesn't exist in the IR";
  }
  return it->second;
}

/******** Block/Loop relation ********/

BlockRV ConcreteScheduleNode::GetBlock(const String& name) {
  Array<StmtSRef> blocks = tir::GetBlocks(this->state, name);
  CHECK_EQ(blocks.size(), 1) << "ValueError: There are " << blocks.size()
                             << " blocks with the name: " << name;
  return SetRV<BlockRV>(this, blocks[0]);
}

Array<LoopRV> ConcreteScheduleNode::GetAxes(const BlockRV& block_rv) {
  return SetRV<LoopRV>(this, tir::GetAxes(this->state, this->GetSRef(block_rv)));
}

Array<BlockRV> ConcreteScheduleNode::GetChildBlocks(const BlockRV& block_rv) {
  return SetRV<BlockRV>(this, tir::GetChildBlocks(this->state, this->GetSRef(block_rv), false));
}

Array<BlockRV> ConcreteScheduleNode::GetChildBlocks(const LoopRV& loop_rv) {
  return SetRV<BlockRV>(this, tir::GetChildBlocks(this->state, this->GetSRef(loop_rv), false));
}

Array<BlockRV> ConcreteScheduleNode::GetProducers(const BlockRV& block_rv) {
  return SetRV<BlockRV>(this, tir::GetProducers(this->state, this->GetSRef(block_rv)));
}

Array<BlockRV> ConcreteScheduleNode::GetConsumers(const BlockRV& block_rv) {
  return SetRV<BlockRV>(this, tir::GetConsumers(this->state, this->GetSRef(block_rv)));
}

/******** Schedule: loops ********/

LoopRV ConcreteScheduleNode::Fuse(const Array<LoopRV>& loop_rvs) {
  CHECK(!loop_rvs.empty()) << "ValueError: 'fuse' requires at least 1 loop(s)";
  Array<StmtSRef> loop_srefs = FromRV(this, loop_rvs);
  while (loop_srefs.size() >= 2) {
    StmtSRef inner_sref = loop_srefs.back();
    loop_srefs.pop_back();
    StmtSRef outer_sref = loop_srefs.back();
    loop_srefs.pop_back();
    StmtSRef fused = schedule::Fuse(this->state, outer_sref, inner_sref);
    loop_srefs.push_back(fused);
  }
  return SetRV<LoopRV>(this, loop_srefs[0]);
}

Array<LoopRV> ConcreteScheduleNode::Split(const LoopRV& loop_rv,
                                          const Array<Optional<ExprRV>>& factor_rvs) {
  arith::Analyzer analyzer;
  // Prepare for the splitting
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  const auto* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  PrimExpr len = loop->extent;
  // Find out the None
  int n = factor_rvs.size();
  CHECK_GE(n, 2) << "ValueError: `split` requires at least 2 parts";
  std::vector<PrimExpr> factors;
  factors.reserve(n);
  int p = -1;
  for (int i = 0; i < n; ++i) {
    PrimExpr factor = this->Get(factor_rvs[i].value_or(Integer(-1)));
    if (analyzer.CanProve(factor == -1)) {
      CHECK_EQ(p, -1) << "ValueError: `split` requires at most one `None` factor, but gets: "
                      << factor_rvs;
      p = i;
      factors.emplace_back(Integer(-1));
    } else {
      factors.emplace_back(std::move(factor));
    }
  }
  if (p == -1) {
    PrimExpr prod = factors[0];
    for (int i = 1; i < n; ++i) {
      prod = prod * factors[i];
    }
    if (analyzer.CanProve(prod == len)) {
      p = 0;
      factors[0] = Integer(-1);
    } else {
      LOG(FATAL) << "ValueError: invalid extents for `split`, the loop extent is " << len
                 << ", but extents are: " << Array<PrimExpr>{factors.begin(), factors.end()};
    }
  }
  std::vector<StmtSRef> results(n, StmtSRef{nullptr});
  // Split from right to left
  for (int i = n - 1; i > p; --i) {
    PrimExpr inner_len = factors[i];
    PrimExpr outer_len = floordiv(len + inner_len - 1, inner_len);
    Array<StmtSRef> parts = schedule::Split(this->state,  //
                                            loop_sref,    //
                                            outer_len, inner_len);
    ICHECK_EQ(parts.size(), 2);
    loop_sref = parts[0];
    results[i] = parts[1];
    len = outer_len;
  }
  // Split from left to right
  for (int i = 0; i < p; ++i) {
    PrimExpr outer_len = factors[i];
    PrimExpr inner_len = floordiv(len + outer_len - 1, outer_len);
    Array<StmtSRef> parts = schedule::Split(this->state,  //
                                            loop_sref,    //
                                            outer_len, inner_len);
    ICHECK_EQ(parts.size(), 2);
    results[i] = parts[0];
    loop_sref = parts[1];
    len = inner_len;
  }
  results[p] = loop_sref;
  return SetRV<LoopRV>(this, Array<StmtSRef>{results.begin(), results.end()});
}

void ConcreteScheduleNode::Reorder(const Array<LoopRV>& order) {
  schedule::Reorder(this->state, FromRV(this, order));
}

/******** Schedule: compute location ********/

void ConcreteScheduleNode::ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                     bool preserve_unit_loop) {
  static StmtSRef inline_mark = StmtSRef::InlineMark();
  static StmtSRef root_mark = StmtSRef::RootMark();
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  if (loop_sref.same_as(root_mark)) {
    // do nothing
  } else if (loop_sref.same_as(inline_mark)) {
    schedule::ComputeInline(this->state, this->GetSRef(block_rv));
  } else {
    schedule::ComputeAt(this->state,              //
                        this->GetSRef(block_rv),  //
                        loop_sref,                //
                        preserve_unit_loop);
  }
}

void ConcreteScheduleNode::ReverseComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                            bool preserve_unit_loop) {
  static StmtSRef inline_mark = StmtSRef::InlineMark();
  static StmtSRef root_mark = StmtSRef::RootMark();
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  if (loop_sref.same_as(root_mark)) {
    // do nothing
  } else if (loop_sref.same_as(inline_mark)) {
    schedule::ReverseComputeInline(this->state, this->GetSRef(block_rv));
  } else {
    schedule::ReverseComputeAt(this->state,              //
                               this->GetSRef(block_rv),  //
                               loop_sref,                //
                               preserve_unit_loop);
  }
}

void ConcreteScheduleNode::ComputeInline(const BlockRV& block_rv) {
  schedule::ComputeInline(this->state, this->GetSRef(block_rv));
}

void ConcreteScheduleNode::ReverseComputeInline(const BlockRV& block_rv) {
  schedule::ReverseComputeInline(this->state, this->GetSRef(block_rv));
}

/******** Schedule: parallelize / annotate ********/

void ConcreteScheduleNode::Vectorize(const LoopRV& loop_rv) {
  schedule::Vectorize(this->state, this->GetSRef(loop_rv));
}

void ConcreteScheduleNode::Parallel(const LoopRV& loop_rv) {
  schedule::Parallel(this->state, this->GetSRef(loop_rv));
}

void ConcreteScheduleNode::Unroll(const LoopRV& loop_rv) {
  schedule::Unroll(this->state, this->GetSRef(loop_rv));
}

void ConcreteScheduleNode::Bind(const LoopRV& loop_rv, const IterVar& thread) {
  schedule::Bind(this->state, this->GetSRef(loop_rv), thread);
}

void ConcreteScheduleNode::Bind(const LoopRV& loop_rv, const String& thread) {
  IterVar iter_var(Range(nullptr),  //
                   Var(thread),     //
                   kThreadIndex,    //
                   thread);
  schedule::Bind(this->state, this->GetSRef(loop_rv), iter_var);
}

void ConcreteScheduleNode::DoubleBuffer(const BlockRV& block_rv) {
  schedule::DoubleBuffer(this->state, this->GetSRef(block_rv));
}

void ConcreteScheduleNode::Pragma(const LoopRV& loop_rv, const String& pragma_type,
                                  const ExprRV& pragma_value) {
  schedule::Pragma(this->state,             //
                   this->GetSRef(loop_rv),  //
                   pragma_type,             //
                   this->Get(pragma_value));
}

/******** Schedule: cache read/write ********/

BlockRV ConcreteScheduleNode::CacheRead(const BlockRV& block_rv, int i,
                                        const String& storage_scope) {
  return SetRV<BlockRV>(this, schedule::CacheRead(this->state,              //
                                                  this->GetSRef(block_rv),  //
                                                  i,                        //
                                                  storage_scope));
}

BlockRV ConcreteScheduleNode::CacheWrite(const BlockRV& block_rv, int i,
                                         const String& storage_scope) {
  return SetRV<BlockRV>(this, schedule::CacheWrite(this->state,              //
                                                   this->GetSRef(block_rv),  //
                                                   i,                        //
                                                   storage_scope));
}

/******** Schedule: reduction ********/

BlockRV ConcreteScheduleNode::RFactor(const LoopRV& loop_rv, int factor_axis) {
  return SetRV<BlockRV>(this, schedule::RFactor(this->state, this->GetSRef(loop_rv), factor_axis));
}

BlockRV ConcreteScheduleNode::DecomposeReduction(const BlockRV& block_rv,
                                                 const Optional<LoopRV>& opt_loop_rv) {
  Optional<StmtSRef> opt_loop_sref = opt_loop_rv.defined() ?                 //
                                         this->GetSRef(opt_loop_rv.value())  //
                                                           : Optional<StmtSRef>(NullOpt);
  return SetRV<BlockRV>(this,
                        schedule::DecomposeReduction(this->state,              //
                                                     this->GetSRef(block_rv),  //
                                                     opt_loop_sref));
}

void ConcreteScheduleNode::MergeReduction(const BlockRV& init_block_rv,
                                          const BlockRV& update_block_rv) {
  schedule::MergeReduction(this->state,                   //
                           this->GetSRef(init_block_rv),  //
                           this->GetSRef(update_block_rv));
}

/******** Schedule: blockize / tensorize ********/

BlockRV ConcreteScheduleNode::Blockize(const LoopRV& loop_rv, const String& exec_scope) {
  return SetRV<BlockRV>(this, schedule::Blockize(this->state, this->GetSRef(loop_rv), exec_scope));
}

void ConcreteScheduleNode::Tensorize(const LoopRV& loop_rv, const TensorIntrin& intrin) {
  schedule::Tensorize(this->state, this->GetSRef(loop_rv), intrin);
}

void ConcreteScheduleNode::Tensorize(const LoopRV& loop_rv, const String& intrin_name) {
  schedule::Tensorize(this->state, this->GetSRef(loop_rv), tir::TensorIntrin::Get(intrin_name));
}

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(ConcreteScheduleNode);
TVM_REGISTER_GLOBAL("tir.schedule.Schedule")
    .set_body_typed([](ObjectRef obj, int64_t seed, bool debug_mode) -> Schedule {
      if (const auto* func = obj.as<PrimFuncNode>()) {
        return Schedule::Concrete(GetRef<PrimFunc>(func), seed, debug_mode);
      }
      if (const auto* mod = obj.as<IRModuleNode>()) {
        return Schedule::Concrete(GetRef<IRModule>(mod), seed, debug_mode);
      }
      LOG(FATAL) << "TypeError: Expects `IRModule` or `PrimFunc`, but gets: " << obj->GetTypeKey();
      throw;
    });

}  // namespace tir
}  // namespace tvm
