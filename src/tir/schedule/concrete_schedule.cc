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

#include "./utils.h"

namespace tvm {
namespace tir {

Schedule Schedule::Concrete(PrimFunc func, int64_t seed, int debug_mode) {
  return Schedule::Concrete(IRModule({{GlobalVar("main"), func}}), seed, debug_mode);
}

Schedule Schedule::Concrete(IRModule mod, int64_t seed, int debug_mode) {
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
      const BlockInfo& old_info = kv.second;
      BlockInfo new_info = old_info;
      ObjectPtr<BlockScopeNode> scope = make_object<BlockScopeNode>();
      scope->src2deps = Copy(old_info.scope->src2deps);
      scope->dst2deps = Copy(old_info.scope->dst2deps);
      scope->buffer_writers = Copy(old_info.scope->buffer_writers);
      new_info.scope = BlockScope(std::move(scope));
      result[Copy(old_sref)] = std::move(new_info);
    }
    return result;
  }

  /*! \brief Copy the stmt2ref */
  UMap<const StmtNode*, StmtSRef> Copy(const UMap<const StmtNode*, StmtSRef>& stmt2ref) {
    UMap<const StmtNode*, StmtSRef> result;
    result.reserve(stmt2ref.size());
    for (const auto& kv : stmt2ref) {
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

void ConcreteScheduleNode::MakeCopy(ScheduleState* new_state,
                                    TSymbolTable* new_symbol_table) const {
  ScheduleCopier::Copy(this, new_state, new_symbol_table);
}

Schedule ConcreteScheduleNode::Copy(int64_t new_seed) const {
  ObjectPtr<ConcreteScheduleNode> n = make_object<ConcreteScheduleNode>();
  MakeCopy(&n->state_, &n->symbol_table_);
  n->analyzer_ = std::make_unique<arith::Analyzer>();
  n->sampler_.Seed(new_seed);
  return Schedule(std::move(n));
}

/******** Schedule: Sampling ********/

Array<ExprRV> ConcreteScheduleNode::SamplePerfectTile(const LoopRV& loop_rv, int n,
                                                      int max_innermost_factor,
                                                      Optional<Array<Integer>> decision) {
  return CreateRV(AsArray<int64_t, Integer>(tir::SamplePerfectTile(
      state_, &this->sampler_, this->GetSRef(loop_rv), n, max_innermost_factor, &decision)));
}

ExprRV ConcreteScheduleNode::SampleCategorical(const Array<Integer>& candidates,
                                               const Array<FloatImm>& probs,
                                               Optional<Integer> decision) {
  return CreateRV(tir::SampleCategorical(state_, &this->sampler_, candidates, probs, &decision));
}

LoopRV ConcreteScheduleNode::SampleComputeLocation(const BlockRV& block_rv,
                                                   Optional<Integer> decision) {
  return CreateRV<LoopRV>(
      tir::SampleComputeLocation(state_, &this->sampler_, this->GetSRef(block_rv), &decision));
}

/******** Schedule: Get blocks & loops ********/

BlockRV ConcreteScheduleNode::GetBlock(const String& name, const String& func_name) {
  Array<StmtSRef> blocks = tir::GetBlocks(state_, name);
  CHECK_EQ(blocks.size(), 1) << "ValueError: There are " << blocks.size()
                             << " blocks with the name: " << name;
  return CreateRV<BlockRV>(blocks[0]);
}

Array<LoopRV> ConcreteScheduleNode::GetLoops(const BlockRV& block_rv) {
  return CreateRV<LoopRV>(tir::GetLoops(this->GetSRef(block_rv)));
}

Array<BlockRV> ConcreteScheduleNode::GetChildBlocks(const BlockRV& block_rv) {
  return CreateRV<BlockRV>(tir::GetChildBlocks(state_, this->GetSRef(block_rv), false));
}

Array<BlockRV> ConcreteScheduleNode::GetChildBlocks(const LoopRV& loop_rv) {
  return CreateRV<BlockRV>(tir::GetChildBlocks(state_, this->GetSRef(loop_rv), false));
}

Array<BlockRV> ConcreteScheduleNode::GetProducers(const BlockRV& block_rv) {
  return CreateRV<BlockRV>(tir::GetProducers(state_, this->GetSRef(block_rv)));
}

Array<BlockRV> ConcreteScheduleNode::GetConsumers(const BlockRV& block_rv) {
  return CreateRV<BlockRV>(tir::GetConsumers(state_, this->GetSRef(block_rv)));
}

/******** Schedule: Transform loops ********/

LoopRV ConcreteScheduleNode::Fuse(const Array<LoopRV>& loop_rvs) {
  CHECK(!loop_rvs.empty()) << "ValueError: 'fuse' requires at least 1 loop(s)";
  Array<StmtSRef> loop_srefs = this->GetSRefs(loop_rvs);
  while (loop_srefs.size() >= 2) {
    StmtSRef inner_sref = loop_srefs.back();
    loop_srefs.pop_back();
    StmtSRef outer_sref = loop_srefs.back();
    loop_srefs.pop_back();
    StmtSRef fused = tir::Fuse(state_, outer_sref, inner_sref);
    loop_srefs.push_back(fused);
    this->state_->DebugVerify();
  }
  return CreateRV<LoopRV>(loop_srefs[0]);
}

Array<LoopRV> ConcreteScheduleNode::Split(const LoopRV& loop_rv,
                                          const Array<Optional<ExprRV>>& factor_rvs) {
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
    if (analyzer_->CanProve(factor == -1)) {
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
    if (analyzer_->CanProve(prod == len)) {
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
    Array<StmtSRef> parts = tir::Split(state_,     //
                                       loop_sref,  //
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
    Array<StmtSRef> parts = tir::Split(state_,     //
                                       loop_sref,  //
                                       outer_len, inner_len);
    this->state_->DebugVerify();
    ICHECK_EQ(parts.size(), 2);
    results[i] = parts[0];
    loop_sref = parts[1];
    len = inner_len;
  }
  results[p] = loop_sref;
  return CreateRV<LoopRV>(Array<StmtSRef>{results.begin(), results.end()});
}

void ConcreteScheduleNode::Reorder(const Array<LoopRV>& order) {
  tir::Reorder(state_, this->GetSRefs(order));
}

/******** Schedule: Manipulate ForKind ********/

void ConcreteScheduleNode::Parallel(const LoopRV& loop_rv) {
  tir::Parallel(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::Vectorize(const LoopRV& loop_rv) {
  tir::Vectorize(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::Unroll(const LoopRV& loop_rv) {
  tir::Unroll(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::Bind(const LoopRV& loop_rv, const IterVar& thread) {
  tir::Bind(state_, this->GetSRef(loop_rv), thread);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::Bind(const LoopRV& loop_rv, const String& thread) {
  tir::Bind(state_, this->GetSRef(loop_rv),
            IterVar(/*dom=*/Range(nullptr), /*var=*/Var(thread), /*IterVarType=*/kThreadIndex,
                    /*thread_tag=*/thread));
  this->state_->DebugVerify();
}

/******** Schedule: Insert cache stages ********/

BlockRV ConcreteScheduleNode::CacheRead(const BlockRV& block_rv, int i,
                                        const String& storage_scope) {
  StmtSRef result = tir::CacheRead(state_, this->GetSRef(block_rv), i, storage_scope);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::CacheWrite(const BlockRV& block_rv, int i,
                                         const String& storage_scope) {
  StmtSRef result = tir::CacheWrite(state_, this->GetSRef(block_rv), i, storage_scope);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

/******** Schedule: Compute location ********/

void ConcreteScheduleNode::ComputeAt(const BlockRV& block_rv, const LoopRV& loop_rv,
                                     bool preserve_unit_loop) {
  static StmtSRef inline_mark = StmtSRef::InlineMark();
  static StmtSRef root_mark = StmtSRef::RootMark();
  StmtSRef loop_sref = this->GetSRef(loop_rv);
  if (loop_sref.same_as(root_mark)) {
    // do nothing
  } else if (loop_sref.same_as(inline_mark)) {
    tir::ComputeInline(state_, this->GetSRef(block_rv));
    this->state_->DebugVerify();
  } else {
    tir::ComputeAt(state_, this->GetSRef(block_rv), loop_sref, preserve_unit_loop);
    this->state_->DebugVerify();
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
    tir::ReverseComputeInline(state_, this->GetSRef(block_rv));
    this->state_->DebugVerify();
  } else {
    tir::ReverseComputeAt(state_, this->GetSRef(block_rv), loop_sref, preserve_unit_loop);
    this->state_->DebugVerify();
  }
}

void ConcreteScheduleNode::ComputeInline(const BlockRV& block_rv) {
  tir::ComputeInline(state_, this->GetSRef(block_rv));
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::ReverseComputeInline(const BlockRV& block_rv) {
  tir::ReverseComputeInline(state_, this->GetSRef(block_rv));
  this->state_->DebugVerify();
}

/******** Schedule: Reduction ********/

BlockRV ConcreteScheduleNode::RFactor(const LoopRV& loop_rv, int factor_axis) {
  StmtSRef result = tir::RFactor(state_, this->GetSRef(loop_rv), factor_axis);
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

BlockRV ConcreteScheduleNode::DecomposeReduction(const BlockRV& block_rv,
                                                 const Optional<LoopRV>& opt_loop_rv) {
  StmtSRef result = tir::DecomposeReduction(
      state_, this->GetSRef(block_rv),
      opt_loop_rv.defined() ? this->GetSRef(opt_loop_rv.value()) : Optional<StmtSRef>(NullOpt));
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

void ConcreteScheduleNode::MergeReduction(const BlockRV& init_block_rv,
                                          const BlockRV& update_block_rv) {
  tir::MergeReduction(state_, this->GetSRef(init_block_rv), this->GetSRef(update_block_rv));
  this->state_->DebugVerify();
}

/******** Schedule: Blockize & Tensorize ********/

BlockRV ConcreteScheduleNode::Blockize(const LoopRV& loop_rv) {
  StmtSRef result = tir::Blockize(state_, this->GetSRef(loop_rv));
  this->state_->DebugVerify();
  return CreateRV<BlockRV>(result);
}

void ConcreteScheduleNode::Tensorize(const LoopRV& loop_rv, const TensorIntrin& intrin) {
  tir::Tensorize(state_, this->GetSRef(loop_rv), intrin);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::Tensorize(const LoopRV& loop_rv, const String& intrin_name) {
  tir::Tensorize(state_, this->GetSRef(loop_rv), tir::TensorIntrin::Get(intrin_name));
  this->state_->DebugVerify();
}

/******** Schedule: Annotation ********/

void ConcreteScheduleNode::MarkLoop(const LoopRV& loop_rv, const String& ann_key,
                                    const PrimExpr& ann_val) {
  ICHECK(ann_val->IsInstance<StringImmNode>() || ann_val->IsInstance<IntImmNode>())
      << "TypeError: Only StringImm and IntImm are supported for now, but gets: "
      << ann_val->GetTypeKey();
  tir::MarkLoop(state_, this->GetSRef(loop_rv), ann_key, ann_val);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::MarkBlock(const BlockRV& block_rv, const String& ann_key,
                                     const PrimExpr& ann_val) {
  int64_t value = Downcast<IntImm>(this->Get(ann_val))->value;
  tir::MarkBlock(state_, this->GetSRef(block_rv), ann_key, StringImm(std::to_string(value)));
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::Pragma(const LoopRV& loop_rv, const String& pragma_type,
                                  const ExprRV& pragma_value) {
  tir::Pragma(state_, this->GetSRef(loop_rv), pragma_type, this->Get(pragma_value));
  this->state_->DebugVerify();
}

/******** Schedule: Misc ********/

void ConcreteScheduleNode::DoubleBuffer(const BlockRV& block_rv) {
  tir::DoubleBuffer(state_, this->GetSRef(block_rv));
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::SetScope(const BlockRV& block_rv, int i, const String& storage_scope) {
  tir::SetScope(state(), this->GetSRef(block_rv), i, storage_scope);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::StorageAlign(const BlockRV& block_rv, int buffer_index, int axis,
                                        int factor, int offset) {
  tir::StorageAlign(state_, this->GetSRef(block_rv), buffer_index, axis, factor, offset);
  this->state_->DebugVerify();
}

void ConcreteScheduleNode::InlineArgument(int i, const String& func_name) {
  tir::InlineArgument(state_, i, func_name);
  this->state_->DebugVerify();
}

/******** Instruction traits ********/

struct EnterPostProcTraits : public UnpackedInstTraits<EnterPostProcTraits> {
  static constexpr const char* kName = "EnterPostProc";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 0;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch) { return sch->EnterPostProc(); }

  static String UnpackedAsPython(Array<String> outputs) {
    PythonAPICall py("enter_postproc");
    return py.Str();
  }

  template <typename>
  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND(EnterPostProcTraits);

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(ConcreteScheduleNode);

}  // namespace tir
}  // namespace tvm
