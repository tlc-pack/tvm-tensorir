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
                   SymbolTable sym_tab, Sampler sampler) {
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

  /*! \brief Translate SymbolTable */
  SymbolTable Trans(const SymbolTable& tab) {
    SymbolTable result = tab;
    for (auto& kv : result) {
      SymbolTableEntry& entry = kv.second;
      if (const auto* sref = entry.value.as<StmtSRefNode>()) {
        entry.value = Trans(sref);
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
  const Optional<ObjectRef> obj = iter->second.value;
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
  const Optional<ObjectRef> obj = iter->second.value;
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
    const Optional<ObjectRef> obj = iter->second.value;
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

Array<tir::Var> ScheduleNode::SampleTileFactor(int n, LoopRV loop, Array<Integer> where) {
  int inst_id = this->trace.size();
  // Sample the output
  std::vector<int> samples;
  {
    const auto* extent = Eval(loop)->GetStmt<tir::LoopNode>()->extent.as<IntImmNode>();
    CHECK(extent);
    std::vector<int> candidates;
    for (const Integer& item : where) {
      candidates.push_back(item);
    }
    samples = sampler.SampleTileFactor(n, extent->value, candidates);
  }
  // Create the output random variable
  String name_prefix = Eval(loop)->GetStmt<tir::LoopNode>()->loop_var->name_hint + ".";
  Array<tir::Var> outputs;
  for (int i = 0; i < n; ++i) {
    tir::Var output(name_prefix + std::to_string(i));
    outputs.push_back(output);
    // Update the symbol table
    Integer value = samples[i];
    this->sym_tab.emplace(output, SymbolTableEntry(inst_id, value));
  }
  // Put the instruction in the trace
  this->trace.push_back(SampleTileFactorInst(loop, where, outputs));
  return outputs;
}

/**************** Block Relationship ****************/

Optional<BlockRV> ScheduleNode::GetOnlyConsumer(const BlockRV& block) {
  int inst_id = this->trace.size();
  // Find the output from TIR
  tir::StmtSRef block_sref = Eval(block);
  Array<tir::DepEdge> succ_edges = this->sch->GetParentScope(block_sref).GetSuccessors(block_sref);
  std::vector<tir::StmtSRef> result_sref;
  for (const tir::DepEdge edge : succ_edges) {
    if (edge->type == tir::DepType::kWAR || edge->type == tir::DepType::kWAW) {
      result_sref.push_back(edge->dst);
    }
  }
  if (result_sref.size() != 1) {
    return NullOpt;
  }
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.emplace(output, SymbolTableEntry(inst_id, result_sref[0]));
  // Put the instruction in the trace
  this->trace.push_back(GetOnlyConsumerInst(block, output));
  return output;
}

/**************** Schedule Primitives ****************/

BlockRV ScheduleNode::GetBlock(const String& name) {
  int inst_id = this->trace.size();
  // Find the output from TIR
  Array<tir::StmtSRef> tir_result = this->sch->GetBlock(name);
  CHECK(!tir_result.empty()) << "ValueError: Cannot get a block with name: " << name;
  CHECK_EQ(tir_result.size(), 1) << "ValueError: Multiple blocks with the same name: " << name;
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.emplace(output, SymbolTableEntry(inst_id, tir_result[0]));
  // Put the instruction in the trace
  this->trace.push_back(GetBlockInst(name, output));
  return output;
}

Array<LoopRV> ScheduleNode::GetAxes(const BlockRV& block) {
  int inst_id = this->trace.size();
  // Find the output from TIR
  Array<tir::StmtSRef> tir_result = this->sch->GetLoopsInScope(Eval(block));
  // Create the output random variable
  Array<LoopRV> outputs;
  for (const tir::StmtSRef& axis : tir_result) {
    LoopRV output;
    outputs.push_back(output);
    // Update the symbol table
    this->sym_tab.emplace(output, SymbolTableEntry(inst_id, axis));
  }
  // Put the instruction in the trace
  this->trace.push_back(GetAxesInst(block, outputs));
  return outputs;
}

Array<LoopRV> ScheduleNode::Split(const LoopRV& loop, const Array<PrimExpr>& factors) {
  int inst_id = this->trace.size();
  // Find the output from TIR
  std::vector<tir::StmtSRef> tir_result;
  {
    tir::StmtSRef tir_loop = Eval(loop);
    int n = factors.size();
    for (int i = n - 1; i >= 1; --i) {
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
    this->sym_tab.emplace(output, SymbolTableEntry(inst_id, axis));
  }
  // Put the instruction in the trace
  this->trace.push_back(SplitInst(loop, factors, outputs));
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
  this->trace.push_back(ReorderInst(after_axes));
}

BlockRV ScheduleNode::DecomposeReduction(const BlockRV& block, const LoopRV& loop) {
  int inst_id = this->trace.size();
  // Find the output from TIR
  tir::StmtSRef tir_result = this->sch->decompose_reduction(Eval(block), Eval(loop));
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.emplace(output, SymbolTableEntry(inst_id, tir_result));
  // Put the instruction in the trace
  this->trace.push_back(DecomposeReductionInst(block, loop, output));
  return output;
}

/**************** Replay ****************/

/*!
 * \brief Store the mapping "old_var => new_var" into var_map
 * \param var_map The old-to-new variable mapping table
 * \param old_var The old variable
 * \param new_var The new variable
 */
void StoreVar(std::unordered_map<const Object*, const Object*>* var_map, const ObjectRef& old_var,
              const ObjectRef& new_var) {
  var_map->emplace(old_var.get(), new_var.get());
}

/*!
 * \brief Store a list of mappings "old_var => new_var" into var_map
 * \tparam TObjectRef Type of the random variable, can be `Block`, `LoopAxis` and `tir::Var`
 * \param var_map The old-to-new variable mapping table
 * \param old_vars The list of old variables
 * \param new_vars The list of new variables
 */
template <class TObjectRef>
void StoreArray(std::unordered_map<const Object*, const Object*>* var_map,
                const Array<TObjectRef>& old_vars, const Array<TObjectRef>& new_vars) {
  CHECK_EQ(old_vars.size(), new_vars.size());
  int n = old_vars.size();
  for (int i = 0; i < n; ++i) {
    StoreVar(var_map, old_vars[i], new_vars[i]);
  }
}

/*!
 * \brief In replay, lookup a random variable in the old-to-new variable mapping table
 * \tparam TObjectRef Type of the random variable, can be `Block`, `LoopAxis` and `tir::Var`
 * \param var_map Maps old variables to new variables
 * \param obj The old variable to be looked up
 * \return The new variable
 */
template <class TObjectRef>
TObjectRef LookupVar(const std::unordered_map<const Object*, const Object*>& var_map,
                     const TObjectRef& obj) {
  using TContainer = typename TObjectRef::ContainerType;
  const Object* ret = var_map.at(obj.get());
  CHECK(ret->IsInstance<TContainer>());
  return GetRef<TObjectRef>(static_cast<const TContainer*>(ret));
}

/*!
 * \brief In replay, lookup a list of random variables in the old-to-new variable mapping table
 * \tparam TObjectRef Type of the random variable, can be `Block`, `LoopAxis` and `tir::Var`
 * \param var_map Maps old variables to new variables
 * \param obj The list of old variables to be looked up
 * \return The list of new variables
 */
template <class TObjectRef>
Array<TObjectRef> LookupArray(const std::unordered_map<const Object*, const Object*>& var_map,
                              const Array<TObjectRef>& objs) {
  Array<TObjectRef> result;
  for (const TObjectRef& obj : objs) {
    result.push_back(LookupVar(var_map, obj));
  }
  return result;
}

void ScheduleNode::ReplayOnce() {
  // Step 1. Create a new schedule to temporarily hold the replay result
  Schedule sch(this->orig_func);
  // Maps an old random variable to its corresponding new random variable in the replay
  std::unordered_map<const Object*, const Object*> var_map;
  // Step 2. Replay all the instructions in the trace
  for (const Instruction& previous_instruction : this->trace) {
    if (const auto* inst = previous_instruction.as<SampleTileFactorInstNode>()) {
      StoreArray(&var_map, inst->outputs,
                 sch->SampleTileFactor(/*n=*/inst->outputs.size(),
                                       /*loop=*/LookupVar(var_map, inst->loop),
                                       /*where=*/inst->where));
    } else if (const auto* inst = previous_instruction.as<GetBlockInstNode>()) {
      StoreVar(&var_map, inst->output, sch->GetBlock(/*name=*/inst->name));
    } else if (const auto* inst = previous_instruction.as<GetAxesInstNode>()) {
      StoreArray(&var_map, inst->outputs, sch->GetAxes(/*block=*/LookupVar(var_map, inst->block)));
    } else if (const auto* inst = previous_instruction.as<SplitInstNode>()) {
      StoreArray(&var_map, inst->outputs,
                 sch->Split(/*loop=*/LookupVar(var_map, inst->loop),
                            /*factors=*/LookupArray(var_map, inst->factors)));
    } else if (const auto* inst = previous_instruction.as<ReorderInstNode>()) {
      sch->Reorder(/*after_axes=*/LookupArray(var_map, inst->after_axes));
    } else if (const auto* inst = previous_instruction.as<DecomposeReductionInstNode>()) {
      StoreVar(&var_map, inst->output,
               sch->DecomposeReduction(/*block=*/LookupVar(var_map, inst->block),
                                       /*loop=*/LookupVar(var_map, inst->loop)));
    } else {
      LOG(FATAL) << "TypeError: Unsupported instruction to be replayed: "
                 << previous_instruction->GetTypeKey();
    }
  }
  // Step 3. Re-assign all the variables back according to the symbol table
  this->sch = sch->sch;
  for (auto& kv_entry : this->sym_tab) {
    const ObjectRef& old_var = kv_entry.first;
    const ObjectRef& new_var = LookupVar(var_map, old_var);
    if (const Optional<ObjectRef>& new_value = sch->sym_tab.at(new_var).value) {
      kv_entry.second.value = new_value.value();
    }
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
   * \brief FFI function, corresponds to Schedule::Eval
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
  /*!
   * \brief FFI function, corresponds to Schedule::SampleTileFactor
   * \sa ScheduleNode::SampleTileFactor
   */
  static Array<tir::Var> SampleTileFactor(Schedule sch, int n, LoopRV loop, Array<Integer> where) {
    return sch->SampleTileFactor(n, loop, where);
  }
  /*!
   * \brief FFI function, corresponds to Schedule::GetBlock
   * \sa ScheduleNode::GetBlock
   */
  static BlockRV GetBlock(Schedule sch, String name) { return sch->GetBlock(name); }
  /*!
   * \brief FFI function, corresponds to Schedule::GetAxes
   * \sa ScheduleNode::GetAxes
   */
  static Array<LoopRV> GetAxes(Schedule sch, BlockRV block) { return sch->GetAxes(block); }
  /*!
   * \brief FFI function, corresponds to Schedule::Split
   * \sa ScheduleNode::Split
   */
  static Array<LoopRV> Split(Schedule sch, LoopRV loop, Array<PrimExpr> factors) {
    return sch->Split(loop, factors);
  }
  /*!
   * \brief FFI function, corresponds to Schedule::Reorder
   * \sa ScheduleNode::Reorder
   */
  static void Reorder(Schedule sch, Array<LoopRV> after_axes) { return sch->Reorder(after_axes); }
  /*!
   * \brief FFI function, corresponds to Schedule::DecomposeReduction
   * \sa ScheduleNode::DecomposeReduction
   */
  static BlockRV DecomposeReduction(Schedule sch, BlockRV block, LoopRV loop) {
    return sch->DecomposeReduction(block, loop);
  }
  /*!
   * \brief FFI function, corresponds to Schedule::ReplayOnce
   * \sa ScheduleNode::ReplayOnce
   */
  static void ReplayOnce(Schedule sch) { return sch->ReplayOnce(); }
};

TVM_REGISTER_NODE_TYPE(ScheduleNode);
TVM_REGISTER_GLOBAL("meta_schedule.Schedule").set_body_typed(Internal::New);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleCopy").set_body_typed(Internal::Copy);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleEval").set_body_typed(Internal::Eval);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSampleTileFactor")
    .set_body_typed(Internal::SampleTileFactor);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetBlock").set_body_typed(Internal::GetBlock);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetAxes").set_body_typed(Internal::GetAxes);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSplit").set_body_typed(Internal::Split);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReorder").set_body_typed(Internal::Reorder);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleDecomposeReduction")
    .set_body_typed(Internal::DecomposeReduction);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReplayOnce").set_body_typed(Internal::ReplayOnce);

}  // namespace meta_schedule
}  // namespace tvm
