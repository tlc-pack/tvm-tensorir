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
#include "./schedule.h"  // NOLINT(build/include)

#include <tvm/arith/analyzer.h>
#include <tvm/tir/stmt_functor.h>

#include "./sampler.h"

namespace tvm {
namespace meta_schedule {

Schedule::Schedule(tir::PrimFunc orig_func, tir::Schedule sch, Array<Instruction> trace,
                   TSymbolTable sym_tab, Sampler sampler) {
  ObjectPtr<ScheduleNode> n = make_object<ScheduleNode>();
  n->orig_func = std::move(orig_func);
  n->sch = std::move(sch);
  n->trace = std::move(trace);
  n->sym_tab = std::move(sym_tab);
  n->sampler = std::move(sampler);
  data_ = std::move(n);
}

Schedule::Schedule(tir::PrimFunc orig_func)
    : Schedule(orig_func, tir::ScheduleNode::Create(orig_func), {}, {}, Sampler(DeviceRand)) {}

/**************** Utility ****************/

/*! \brief Helper class to do tir::StmtSRef translation */
struct SRefTranslator {
  using TSymbolTable = ScheduleNode::TSymbolTable;
  using StmtSRef = tir::StmtSRef;
  using StmtSRefNode = tir::StmtSRefNode;
  using DepEdge = tir::DepEdge;
  using Buffer = tir::Buffer;
  using Scope = tir::Scope;

  template <class K, class V>
  using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;

  /*! \brief Translate StmtSRef */
  StmtSRef Trans(const StmtSRef& sref) { return trans_.at(sref.operator->()); }

  /*! \brief Translate StmtSRefNode */
  StmtSRef Trans(const StmtSRefNode* sref) { return trans_.at(sref); }

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
    for (const auto& kv : map) {
      result[Trans(kv.first)] = Trans(kv.second);
    }
    return result;
  }

  /*! \brief Translate SMap<Buffer, Array<StmtSRef>> */
  SMap<Buffer, Array<StmtSRef>> Trans(const SMap<Buffer, Array<StmtSRef>>& map) {
    SMap<Buffer, Array<StmtSRef>> result;
    for (const auto& kv : map) {
      result[kv.first] = Trans(kv.second);
    }
    return result;
  }

  /*! \brief Translate SMap<StmtSRef, Scope> */
  SMap<StmtSRef, Scope> Trans(const SMap<StmtSRef, Scope>& scopes) {
    SMap<StmtSRef, Scope> result;
    for (const auto& kv : scopes) {
      Scope& scope = result[Trans(kv.first)] = Scope();
      scope->forward_edges = Trans(kv.second->forward_edges);
      scope->backward_edges = Trans(kv.second->backward_edges);
      scope->buffer_writers = Trans(kv.second->buffer_writers);
    }
    return result;
  }

  /*! \brief Translate the symbol table */
  TSymbolTable Trans(const TSymbolTable& tab) {
    TSymbolTable result = tab;
    for (auto& kv : result) {
      Optional<ObjectRef>& entry = kv.second;
      if (const auto* sref = entry.as<StmtSRefNode>()) {
        entry = Trans(sref);
      }
    }
    return result;
  }

  /*!
   * \brief Translate tir::Schedule
   * \note This method must be called to initialize translation table before other translation
   */
  tir::Schedule Trans(const tir::Schedule& sch) {
    ObjectPtr<tir::ScheduleNode> result = make_object<tir::ScheduleNode>();
    // Create the translation table
    // Fill in result->stmt2ref
    for (const auto& kv : sch->stmt2ref) {
      const StmtSRefNode* sref = kv.second.operator->();
      result->stmt2ref[sref->stmt] = trans_[sref] =
          StmtSRef(/*stmt=*/sref->stmt, /*parent=*/nullptr, /*seq_index=*/sref->seq_index,
                   /*binding_valid=*/sref->binding_valid);
    }
    // Link parents
    // Fill in result->root
    StmtSRef& root = result->root = StmtSRef(nullptr);
    for (auto& kv : trans_) {
      const StmtSRefNode* parent = kv.first->parent;
      StmtSRef& sref = kv.second;
      if (parent == nullptr) {
        sref->parent = nullptr;
        CHECK(!root.defined()) << "InternalError: Two roots are found";
        root = sref;
      } else {
        sref->parent = Trans(parent).operator->();
      }
    }
    CHECK(root.defined()) << "InternalError: No root is found";
    result->func = sch->func;
    result->scopes = Trans(sch->scopes);
    return tir::Schedule(result);
  }

 private:
  std::unordered_map<const StmtSRefNode*, StmtSRef> trans_;
};

Schedule ScheduleNode::copy() const {
  SRefTranslator translator;
  tir::Schedule tir_sch = translator.Trans(this->sch);
  return Schedule(/*orig_func=*/this->orig_func,
                  /*sch=*/tir_sch,
                  /*trace=*/this->trace,
                  /*sym_tab=*/translator.Trans(this->sym_tab),
                  /*sampler=*/this->sampler);
}

/**************** Evaluation ****************/

tir::StmtSRef ScheduleNode::Eval(const BlockRV& block) {
  auto iter = this->sym_tab.find(block);
  CHECK(iter != this->sym_tab.end()) << "IndexError: Cannot find corresponding BlockRV: " << block;
  const Optional<ObjectRef>& obj = iter->second;
  CHECK(obj.defined()) << "ValueError: Corresponding BlockRV's value is not defined: " << block;
  if (const auto* sref = obj.as<tir::StmtSRefNode>()) {
    return GetRef<tir::StmtSRef>(sref);
  }
  LOG(FATAL) << "TypeError: BlockRV's corresponding type is invalid: " << obj->GetTypeKey();
  throw;
}

tir::StmtSRef ScheduleNode::Eval(const LoopRV& loop) {
  auto iter = this->sym_tab.find(loop);
  CHECK(iter != this->sym_tab.end()) << "IndexError: Cannot find corresponding LoopRV: " << loop;
  const Optional<ObjectRef>& obj = iter->second;
  CHECK(obj.defined()) << "ValueError: Corresponding LoopRV's value is not defined: " << loop;
  if (const auto* sref = obj.as<tir::StmtSRefNode>()) {
    return GetRef<tir::StmtSRef>(sref);
  }
  LOG(FATAL) << "TypeError: LoopRV's corresponding type is invalid: " << obj->GetTypeKey();
  throw;
}

int ScheduleNode::Eval(const PrimExpr& expr) {
  arith::Analyzer analyzer;
  // Replace all the tir::Var with their corresponding value in the symbol table
  PrimExpr transformed = tir::Substitute(expr, [this](const tir::Var& var) -> Optional<PrimExpr> {
    auto iter = this->sym_tab.find(var);
    CHECK(iter != this->sym_tab.end()) << "IndexError: Cannot find corresponding ExprRV: " << var;
    const Optional<ObjectRef>& obj = iter->second;
    CHECK(obj.defined()) << "ValueError: Variable \"" << var->name_hint
                         << "\" is not defined in the meta scheduling";
    if (const auto* expr = obj.as<PrimExprNode>()) {
      return GetRef<PrimExpr>(expr);
    }
    LOG(FATAL) << "TypeError: ExprRV's corresponding type is invalid: " << obj->GetTypeKey();
    throw;
  });
  PrimExpr simplified = analyzer.Simplify(transformed);
  const auto* result = simplified.as<IntImmNode>();
  CHECK(result) << "ValueError: Expects Integer, but gets type: " << simplified->GetTypeKey()
                << ", value = " << simplified;
  return result->value;
}

/**************** Sampling ****************/

Array<tir::Var> ScheduleNode::SamplePerfectTile(int n_splits, const LoopRV& loop,
                                                int max_innermost_factor) {
  const auto* tir_loop = Eval(loop)->GetStmt<tir::LoopNode>();
  CHECK(tir_loop);
  int64_t extent;
  {
    const auto* p_extent = tir_loop->extent.as<IntImmNode>();
    CHECK(p_extent);
    extent = p_extent->value;
  }
  // Sample the output
  std::vector<int> samples = sampler.SamplePerfectTile(n_splits, extent, max_innermost_factor);
  // Create the output random variable
  String name_prefix = tir_loop->loop_var->name_hint + ".";
  Array<tir::Var> outputs;
  for (int i = 0; i < n_splits; ++i) {
    tir::Var output(name_prefix + std::to_string(i));
    outputs.push_back(output);
    // Update the symbol table
    this->sym_tab.emplace(output, Integer(samples[i]));
  }
  // Put the instruction in the trace
  this->trace.push_back(
      SamplePerfectTileAttrs::MakeInst(n_splits, loop, max_innermost_factor, outputs));
  return outputs;
}

Array<tir::Var> ScheduleNode::SampleTileFactor(int n_splits, const LoopRV& loop,
                                               const Array<Integer>& where) {
  const auto* tir_loop = Eval(loop)->GetStmt<tir::LoopNode>();
  CHECK(tir_loop);
  int64_t extent;
  std::vector<int> candidates;
  {
    const auto* p_extent = tir_loop->extent.as<IntImmNode>();
    CHECK(p_extent);
    extent = p_extent->value;
    for (const Integer& item : where) {
      candidates.push_back(item);
    }
  }
  // Sample the output
  std::vector<int> samples = sampler.SampleTileFactor(n_splits, extent, candidates);
  // Create the output random variable
  String name_prefix = tir_loop->loop_var->name_hint + ".";
  Array<tir::Var> outputs;
  for (int i = 0; i < n_splits; ++i) {
    tir::Var output(name_prefix + std::to_string(i));
    outputs.push_back(output);
    // Update the symbol table
    this->sym_tab.emplace(output, Integer(samples[i]));
  }
  // Put the instruction in the trace
  this->trace.push_back(SampleTileFactorAttrs::MakeInst(n_splits, loop, where, outputs));
  return outputs;
}

/**************** Block/Loop Relationship ****************/

Optional<BlockRV> ScheduleNode::GetOnlyConsumer(const BlockRV& block) {
  // Find the output from TIR
  tir::StmtSRef block_sref = Eval(block);
  Array<tir::DepEdge> succ_edges = this->sch->GetParentScope(block_sref).GetSuccessors(block_sref);
  std::vector<tir::StmtSRef> result_sref;
  for (const tir::DepEdge edge : succ_edges) {
    if (edge->type == tir::DepType::kRAW || edge->type == tir::DepType::kWAW) {
      result_sref.push_back(edge->dst);
    }
  }
  if (result_sref.size() != 1) {
    return NullOpt;
  }
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.emplace(output, result_sref[0]);
  // Put the instruction in the trace
  this->trace.push_back(GetOnlyConsumerAttrs::MakeInst(block, output));
  return output;
}

BlockRV ScheduleNode::GetBlock(const String& name) {
  // Find the output from TIR
  Array<tir::StmtSRef> tir_result = this->sch->GetBlock(name);
  CHECK(!tir_result.empty()) << "ValueError: Cannot get a block with name: " << name;
  CHECK_EQ(tir_result.size(), 1) << "ValueError: Multiple blocks with the same name: " << name;
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.emplace(output, tir_result[0]);
  // Put the instruction in the trace
  this->trace.push_back(GetBlockAttrs::MakeInst(name, output));
  return output;
}

Array<LoopRV> ScheduleNode::GetAxes(const BlockRV& block) {
  // Find the output from TIR
  Array<tir::StmtSRef> tir_result = this->sch->GetLoopsInScope(Eval(block));
  // Create the output random variable
  Array<LoopRV> outputs;
  for (const tir::StmtSRef& axis : tir_result) {
    LoopRV output;
    outputs.push_back(output);
    // Update the symbol table
    this->sym_tab.emplace(output, axis);
  }
  // Put the instruction in the trace
  this->trace.push_back(GetAxesAttrs::MakeInst(block, outputs));
  return outputs;
}

/**************** Schedule Primitives ****************/

Array<LoopRV> ScheduleNode::Split(const LoopRV& loop, const Array<PrimExpr>& factors) {
  // Find the output from TIR
  std::vector<tir::StmtSRef> tir_result;
  {
    tir::StmtSRef tir_loop = Eval(loop);
    int n_splits = factors.size();
    for (int i = n_splits - 1; i >= 1; --i) {
      int factor = this->Eval(factors[i]);
      const PrimExpr& extent = tir_loop->GetStmt<tir::LoopNode>()->extent;
      PrimExpr nparts = floordiv(extent + factor - 1, factor);
      Array<tir::StmtSRef> split_result = this->sch->split(tir_loop, nparts, factor);
      CHECK_EQ(split_result.size(), 2);
      tir_result.push_back(split_result[1]);
      tir_loop = split_result[0];
    }
    tir_result.push_back(tir_loop);
    std::reverse(tir_result.begin(), tir_result.end());
  }
  // Create the output random variable
  Array<LoopRV> outputs;
  for (const tir::StmtSRef& axis : tir_result) {
    LoopRV output;
    outputs.push_back(output);
    // Update the symbol table
    this->sym_tab.emplace(output, axis);
  }
  // Put the instruction in the trace
  this->trace.push_back(SplitAttrs::MakeInst(loop, factors, outputs));
  return outputs;
}

void ScheduleNode::Reorder(const Array<LoopRV>& after_axes) {
  // Find the inputs to TIR
  std::vector<tir::StmtSRef> tir_inputs;
  for (const LoopRV& loop : after_axes) {
    tir_inputs.push_back(Eval(loop));
  }
  this->sch->reorder(tir_inputs);
  // Put the instruction in the trace
  this->trace.push_back(ReorderAttrs::MakeInst(after_axes));
}

void ScheduleNode::ComputeInline(const BlockRV& block) {
  // Find the inputs to TIR
  tir::StmtSRef block_sref = this->Eval(block);
  this->sch->compute_inline(block_sref);
  // Put the instruction in the trace
  this->trace.push_back(ComputeInlineAttrs::MakeInst(block));
}

BlockRV ScheduleNode::CacheWrite(const BlockRV& block_rv, const String& storage_scope) {
  // Find the output from TIR
  tir::StmtSRef block_sref = this->Eval(block_rv);
  const auto* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block) << "TypeError: Expects block, but gets type: " << block_sref->stmt->GetTypeKey();
  CHECK_EQ(block->writes.size(), 1) << "ValueError: only blocks with a single written is supported";
  tir::StmtSRef tir_result = this->sch->cache_write(block->writes[0]->buffer, storage_scope);
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.emplace(output, tir_result);
  // Put the instruction in the trace
  this->trace.push_back(CacheWriteAttrs::MakeInst(block_rv, storage_scope, output));
  return output;
}

BlockRV ScheduleNode::DecomposeReduction(const BlockRV& block, const LoopRV& loop) {
  // Find the output from TIR
  tir::StmtSRef tir_result = this->sch->decompose_reduction(Eval(block), Eval(loop));
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.emplace(output, tir_result);
  // Put the instruction in the trace
  this->trace.push_back(DecomposeReductionAttrs::MakeInst(block, loop, output));
  return output;
}

/**************** Replay ****************/

void ScheduleNode::ReplayOnce() {
  // Step 1. Create a new schedule to temporarily hold the replay result
  Schedule new_sch(this->orig_func);
  // Maps an old random variable to its corresponding new random variable in the replay
  std::unordered_map<const Object*, const Object*> var_map;
  // Step 2. Replay all the instructions in the trace
  for (const Instruction& prev_inst : this->trace) {
    const Array<ObjectRef>& prev_inputs = prev_inst->inputs;
    const Array<ObjectRef>& prev_outputs = prev_inst->outputs;
    Array<ObjectRef> inputs;
    inputs.reserve(prev_inputs.size());
    for (const ObjectRef& input : prev_inputs) {
      const Object* ptr = input.get();
      CHECK(var_map.count(ptr));
      inputs.push_back(GetRef<ObjectRef>(var_map.at(ptr)));
    }
    Array<ObjectRef> outputs =
        Instruction::ApplyToSchedule(new_sch.operator->(), prev_inst->inst_attrs, inputs);
    CHECK_EQ(prev_outputs.size(), outputs.size()) << "ValueError: Output size mismatch";
    for (int i = 0, n = outputs.size(); i < n; ++i) {
      var_map[prev_outputs[i].get()] = outputs[i].get();
    }
  }
  // Step 3. Re-assign all the variables back according to the symbol table
  this->sch = new_sch->sch;
  for (auto& kv_entry : this->sym_tab) {
    const ObjectRef& old_var = kv_entry.first;
    const ObjectRef& new_var = GetRef<ObjectRef>(var_map.at(old_var.get()));
    kv_entry.second = new_sch->sym_tab.at(new_var);
  }
}

/**************** FFI ****************/

struct Internal {
  /*!
   * \brief FFI function, corresponds to Schedule::Schedule
   * \sa Schedule::Schedule
   */
  static Schedule New(tir::PrimFunc func) { return Schedule(func); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::copy
   * \sa ScheduleNode::Copy
   */
  static Schedule Copy(Schedule sch) { return sch->copy(); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Eval
   * \sa ScheduleNode::Eval
   */
  static ObjectRef Eval(Schedule sch, ObjectRef obj) {
    if (const auto* v = obj.as<BlockRVNode>()) {
      return sch->Eval(GetRef<BlockRV>(v));
    } else if (const auto* v = obj.as<LoopRVNode>()) {
      return sch->Eval(GetRef<LoopRV>(v));
    } else if (const auto* v = obj.as<PrimExprNode>()) {
      return Integer(sch->Eval(GetRef<PrimExpr>(v)));
    }
    LOG(FATAL) << "TypeError: Not a random variable type: " << obj->GetTypeKey();
    throw;
  }
  /**************** Sampling ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::SamplePerfectTile
   * \sa ScheduleNode::SamplePerfectTile
   */
  static Array<tir::Var> SamplePerfectTile(Schedule sch, int n_splits, LoopRV loop,
                                           int max_innermost_factor) {
    return sch->SamplePerfectTile(n_splits, loop, max_innermost_factor);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::SampleTileFactor
   * \sa ScheduleNode::SampleTileFactor
   */
  static Array<tir::Var> SampleTileFactor(Schedule sch, int n_splits, LoopRV loop,
                                          Array<Integer> where) {
    return sch->SampleTileFactor(n_splits, loop, where);
  }
  /**************** Block/Loop Relationship ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::GetOnlyConsumer
   * \sa ScheduleNode::GetOnlyConsumer
   */
  static Optional<BlockRV> GetOnlyConsumer(Schedule sch, BlockRV block) {
    return sch->GetOnlyConsumer(block);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::GetBlock
   * \sa ScheduleNode::GetBlock
   */
  static BlockRV GetBlock(Schedule sch, String name) { return sch->GetBlock(name); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::GetAxes
   * \sa ScheduleNode::GetAxes
   */
  static Array<LoopRV> GetAxes(Schedule sch, BlockRV block) { return sch->GetAxes(block); }
  /**************** Scheduling Primitives ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Split
   * \sa ScheduleNode::Split
   */
  static Array<LoopRV> Split(Schedule sch, LoopRV loop, Array<PrimExpr> factors) {
    return sch->Split(loop, factors);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Reorder
   * \sa ScheduleNode::Reorder
   */
  static void Reorder(Schedule sch, Array<LoopRV> after_axes) { return sch->Reorder(after_axes); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::ComputeInline
   * \sa ScheduleNode::ComputeInline
   */
  static void ComputeInline(Schedule sch, BlockRV block) { sch->ComputeInline(block); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::CacheWrite
   * \sa ScheduleNode::CacheWrite
   */
  static BlockRV CacheWrite(Schedule sch, BlockRV block, String storage_scope) {
    return sch->CacheWrite(block, storage_scope);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::DecomposeReduction
   * \sa ScheduleNode::DecomposeReduction
   */
  static BlockRV DecomposeReduction(Schedule sch, BlockRV block, LoopRV loop) {
    return sch->DecomposeReduction(block, loop);
  }
  /**************** Replay ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::ReplayOnce
   * \sa ScheduleNode::ReplayOnce
   */
  static void ReplayOnce(Schedule sch) { return sch->ReplayOnce(); }
};

TVM_REGISTER_NODE_TYPE(ScheduleNode);
TVM_REGISTER_GLOBAL("meta_schedule.Schedule").set_body_typed(Internal::New);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleCopy").set_body_typed(Internal::Copy);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleEval").set_body_typed(Internal::Eval);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSamplePerfectTile")
    .set_body_typed(Internal::SamplePerfectTile);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSampleTileFactor")
    .set_body_typed(Internal::SampleTileFactor);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetOnlyConsumer")
    .set_body_typed(Internal::GetOnlyConsumer);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetBlock").set_body_typed(Internal::GetBlock);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetAxes").set_body_typed(Internal::GetAxes);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSplit").set_body_typed(Internal::Split);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReorder").set_body_typed(Internal::Reorder);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleComputeInline").set_body_typed(Internal::ComputeInline);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleCacheWrite").set_body_typed(Internal::CacheWrite);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleDecomposeReduction")
    .set_body_typed(Internal::DecomposeReduction);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReplayOnce").set_body_typed(Internal::ReplayOnce);

}  // namespace meta_schedule
}  // namespace tvm
