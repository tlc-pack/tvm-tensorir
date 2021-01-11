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

#include "./analysis.h"
#include "./sampler.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

Schedule::Schedule(tir::PrimFunc orig_func, tir::Schedule sch, Trace trace, TSymbolTable sym_tab,
                   Optional<Integer> seed) {
  ObjectPtr<ScheduleNode> n = make_object<ScheduleNode>();
  n->orig_func = std::move(orig_func);
  n->sch = std::move(sch);
  n->trace = std::move(trace);
  n->sym_tab = std::move(sym_tab);
  if (seed.defined()) {
    n->sampler.Seed(seed.value()->value);
  }
  data_ = std::move(n);
}

Schedule::Schedule(tir::PrimFunc orig_func, Optional<Integer> seed)
    : Schedule(/*orig_func=*/orig_func, /*sch=*/tir::ScheduleNode::Create(orig_func),
               /*trace=*/Trace(), /*sym_tab=*/{}, /*seed=*/seed) {}

/**************** Utility ****************/

void ScheduleNode::Seed(int seed) { this->sampler.Seed(seed); }

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
  StmtSRef Trans(const StmtSRefNode* sref) {
    if (trans_.count(sref)) {
      return trans_.at(sref);
    }
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
    TSymbolTable result;
    for (const auto& kv : tab) {
      Optional<ObjectRef> entry = kv.second;
      if (const auto* sref = entry.as<StmtSRefNode>()) {
        entry = Trans(sref);
      }
      result.Set(kv.first, entry);
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

Schedule ScheduleNode::Copy(int new_seed) const {
  // TODO(@junrushao1994): translate this->decisions too
  SRefTranslator translator;
  tir::Schedule tir_sch = translator.Trans(this->sch);
  return Schedule(/*orig_func=*/this->orig_func,
                  /*sch=*/tir_sch,
                  /*trace=*/Trace(this->trace->insts, this->trace->decisions),
                  /*sym_tab=*/translator.Trans(this->sym_tab),
                  /*seed=*/Integer(new_seed));
}

/**************** Evaluation of random variables ****************/

tir::StmtSRef ScheduleNode::Eval(const BlockRV& block) {
  auto iter = this->sym_tab.find(block);
  CHECK(iter != this->sym_tab.end()) << "IndexError: Cannot find corresponding BlockRV: " << block;
  const Optional<ObjectRef>& obj = (*iter).second;
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
  const Optional<ObjectRef>& obj = (*iter).second;
  CHECK(obj.defined()) << "ValueError: Corresponding LoopRV's value is not defined: " << loop;
  if (const auto* sref = obj.as<tir::StmtSRefNode>()) {
    return GetRef<tir::StmtSRef>(sref);
  }
  LOG(FATAL) << "TypeError: LoopRV's corresponding type is invalid: " << obj->GetTypeKey();
  throw;
}

ObjectRef ScheduleNode::EvalLoopExtended(const LoopRV& loop) {
  static LoopRV inline_rv = LoopRV::ComputeInlineRV();
  static LoopRV root_rv = LoopRV::ComputeRootRV();
  auto iter = this->sym_tab.find(loop);
  CHECK(iter != this->sym_tab.end()) << "IndexError: Cannot find corresponding LoopRV: " << loop;
  const Optional<ObjectRef>& obj = (*iter).second;
  CHECK(obj.defined()) << "ValueError: Corresponding LoopRV's value is not defined: " << loop;
  if (obj.same_as(inline_rv)) {
    return String(LoopRV::inline_rv);
  }
  if (obj.same_as(root_rv)) {
    return String(LoopRV::root_rv);
  }
  if (const auto* sref = obj.as<tir::StmtSRefNode>()) {
    return GetRef<tir::StmtSRef>(sref);
  }
  LOG(FATAL) << "TypeError: LoopRV's corresponding type is invalid: " << obj->GetTypeKey();
  throw;
}

tir::Buffer ScheduleNode::Eval(const BufferRV& buffer) {
  auto iter = this->sym_tab.find(buffer);
  CHECK(iter != this->sym_tab.end())
      << "IndexError: Cannot find corresponding BufferRV: " << buffer;
  const Optional<ObjectRef>& obj = (*iter).second;
  CHECK(obj.defined()) << "ValueError: Corresponding BufferRV's value is not defined: " << buffer;
  if (const auto* sref = obj.as<tir::BufferNode>()) {
    return GetRef<tir::Buffer>(sref);
  }
  LOG(FATAL) << "TypeError: BufferRV's corresponding type is invalid: " << obj->GetTypeKey();
  throw;
}

int ScheduleNode::Eval(const PrimExpr& expr) {
  arith::Analyzer analyzer;
  // Replace all the tir::Var with their corresponding value in the symbol table
  PrimExpr transformed = tir::Substitute(expr, [this](const tir::Var& var) -> Optional<PrimExpr> {
    auto iter = this->sym_tab.find(var);
    CHECK(iter != this->sym_tab.end())
        << "IndexError: Cannot find corresponding ExprRV: " << var << '@' << var.get();
    const Optional<ObjectRef>& obj = (*iter).second;
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
                                                int max_innermost_factor,
                                                const Optional<Array<ObjectRef>>& decision) {
  const auto* tir_loop = Eval(loop)->GetStmt<tir::LoopNode>();
  CHECK(tir_loop);
  int64_t extent;
  {
    const auto* p_extent = tir_loop->extent.as<IntImmNode>();
    CHECK(p_extent);
    extent = p_extent->value;
  }
  // Sample the output
  std::vector<int> samples =
      decision.defined()                                  //
          ? AsVector<ObjectRef, int>()(decision.value())  //
          : sampler.SamplePerfectTile(n_splits, extent, max_innermost_factor);
  // Create the output random variable
  String name_prefix = tir_loop->loop_var->name_hint + ".";
  Array<tir::Var> outputs;
  for (int i = 0; i < n_splits; ++i) {
    tir::Var output(name_prefix + std::to_string(i));
    outputs.push_back(output);
    // Update the symbol table
    this->sym_tab.Set(output, Integer(samples[i]));
  }
  // Record the instruction
  this->trace->Append(SamplePerfectTileAttrs::Make(n_splits, loop, max_innermost_factor, outputs),
                      AsArray<int, ObjectRef>()(samples));
  return outputs;
}

Array<tir::Var> ScheduleNode::SampleTileFactor(int n_splits, const LoopRV& loop,
                                               const Array<Integer>& where,
                                               const Optional<Array<ObjectRef>>& decision) {
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
  std::vector<int> samples = decision.defined()                                  //
                                 ? AsVector<ObjectRef, int>()(decision.value())  //
                                 : sampler.SampleTileFactor(n_splits, extent, candidates);
  // Create the output random variable
  String name_prefix = tir_loop->loop_var->name_hint + ".";
  Array<tir::Var> outputs;
  for (int i = 0; i < n_splits; ++i) {
    tir::Var output(name_prefix + std::to_string(i));
    outputs.push_back(output);
    // Update the symbol table
    this->sym_tab.Set(output, Integer(samples[i]));
  }
  // Record the instruction
  this->trace->Append(SampleTileFactorAttrs::Make(n_splits, loop, where, outputs),
                      AsArray<int, ObjectRef>()(samples));
  return outputs;
}

tir::Var ScheduleNode::SampleInt(const PrimExpr& min_inclusive, const PrimExpr& max_exclusive,
                                 const Optional<ObjectRef>& decision) {
  int num_min_inclusive = this->Eval(min_inclusive);
  int num_max_exclusive = this->Eval(max_exclusive);
  int sampled = decision.defined()                                //
                    ? Downcast<Integer>(decision.value())->value  //
                    : sampler.SampleInt(num_min_inclusive, num_max_exclusive);
  // Create the output random variable
  tir::Var output("n");
  // Update the symbol table
  this->sym_tab.Set(output, Integer(sampled));
  // Record the instruction
  this->trace->Append(SampleIntAttrs::Make(min_inclusive, max_exclusive, output), Integer(sampled));
  return output;
}

tir::Var ScheduleNode::SampleCategorical(const Array<Integer>& candidates,
                                         const Array<FloatImm>& probs,
                                         const Optional<ObjectRef>& decision) {
  // Sample the output
  CHECK_EQ(candidates.size(), probs.size()) << "ValueError: When sampling ";
  std::vector<double> probs_vec;
  probs_vec.reserve(probs.size());
  for (const FloatImm& prob : probs) {
    probs_vec.push_back(prob->value);
  }
  int sampled = decision.defined()                                //
                    ? Downcast<Integer>(decision.value())->value  //
                    : sampler.MakeMultinomial(probs_vec)();
  int result = candidates[sampled];
  // Create the output random variable
  tir::Var output("n");
  // Update the symbol table
  this->sym_tab.Set(output, Integer(result));
  // Record the instruction
  this->trace->Append(SampleCategoricalAttrs::Make(candidates, probs, output), Integer(sampled));
  return output;
}

LoopRV ScheduleNode::SampleComputeLocation(const BlockRV& block,
                                           const Optional<ObjectRef>& decision) {
  tir::StmtSRef block_sref = Eval(block);
  Array<tir::StmtSRef> loop_srefs = sch->GetLoopsInScope(block_sref);
  int n = loop_srefs.size();
  int i = decision.defined()                                //
              ? Downcast<Integer>(decision.value())->value  //
              : sampler.SampleInt(-2, n);
  // Create the output random variable
  LoopRV output;
  // Update the symbol table
  if (i == -2) {
    this->sym_tab.Set(output, LoopRV::ComputeInlineRV());
  } else if (i == -1) {
    this->sym_tab.Set(output, LoopRV::ComputeRootRV());
  } else {
    this->sym_tab.Set(output, loop_srefs[i]);
  }
  // Record the instruction
  this->trace->Append(SampleComputeLocationAttrs::Make(block, output), Integer(i));
  return output;
}

/**************** Block/Loop Relationship ****************/

Array<BlockRV> ScheduleNode::GetProducers(const BlockRV& block) {
  // Find the output from TIR
  tir::StmtSRef block_sref = Eval(block);
  Array<tir::DepEdge> pred_edges =
      this->sch->GetParentScope(block_sref).GetPredecessors(block_sref);
  // Create the output random variable
  Array<BlockRV> outputs;
  outputs.reserve(pred_edges.size());
  for (const tir::DepEdge edge : pred_edges) {
    if (edge->type == tir::DepType::kRAW || edge->type == tir::DepType::kWAW) {
      // Create the output random variable
      BlockRV output;
      // Update the symbol table
      this->sym_tab.Set(output, edge->dst);
      outputs.push_back(output);
    }
  }
  // Record the instruction
  this->trace->Append(GetProducersAttrs::Make(block, outputs));
  return outputs;
}

Array<BlockRV> ScheduleNode::GetConsumers(const BlockRV& block) {
  // Find the output from TIR
  tir::StmtSRef block_sref = Eval(block);
  Array<tir::DepEdge> succ_edges = this->sch->GetParentScope(block_sref).GetSuccessors(block_sref);
  // Create the output random variable
  Array<BlockRV> outputs;
  outputs.reserve(succ_edges.size());
  for (const tir::DepEdge edge : succ_edges) {
    if (edge->type == tir::DepType::kRAW || edge->type == tir::DepType::kWAW) {
      // Create the output random variable
      BlockRV output;
      // Update the symbol table
      this->sym_tab.Set(output, edge->dst);
      outputs.push_back(output);
    }
  }
  // Record the instruction
  this->trace->Append(GetConsumersAttrs::Make(block, outputs));
  return outputs;
}

BlockRV ScheduleNode::GetBlock(const String& name) {
  // Find the output from TIR
  Array<tir::StmtSRef> tir_result = this->sch->GetBlock(name);
  CHECK(!tir_result.empty()) << "ValueError: Cannot get a block with name: " << name;
  CHECK_EQ(tir_result.size(), 1) << "ValueError: Multiple blocks with the same name: " << name;
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.Set(output, tir_result[0]);
  // Record the instruction
  this->trace->Append(GetBlockAttrs::Make(name, output));
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
    this->sym_tab.Set(output, axis);
  }
  // Record the instruction
  this->trace->Append(GetAxesAttrs::Make(block, outputs));
  return outputs;
}

Array<BufferRV> ScheduleNode::GetReadBuffers(const BlockRV& block_rv) {
  tir::StmtSRef block_sref = Eval(block_rv);
  const tir::BlockNode* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block != nullptr) << "TypeError: Expects `BlockNode`, but gets: "
                          << block_sref->stmt->GetTypeKey();
  // Create the output random variable
  Array<BufferRV> outputs;
  for (const tir::TensorRegion& tensor_region : block->reads) {
    BufferRV output;
    outputs.push_back(output);
    // Update the symbol table
    this->sym_tab.Set(output, tensor_region->buffer);
  }
  // Record the instruction
  this->trace->Append(GetReadBuffersAttrs::Make(block_rv, outputs));
  return outputs;
}

Array<BufferRV> ScheduleNode::GetWriteBuffers(const BlockRV& block_rv) {
  tir::StmtSRef block_sref = Eval(block_rv);
  const tir::BlockNode* block = block_sref->GetStmt<tir::BlockNode>();
  CHECK(block != nullptr) << "TypeError: Expects `BlockNode`, but gets: "
                          << block_sref->stmt->GetTypeKey();
  // Create the output random variable
  Array<BufferRV> outputs;
  for (const tir::TensorRegion& tensor_region : block->writes) {
    BufferRV output;
    outputs.push_back(output);
    // Update the symbol table
    this->sym_tab.Set(output, tensor_region->buffer);
  }
  // Record the instruction
  this->trace->Append(GetWriteBuffersAttrs::Make(block_rv, outputs));
  return outputs;
}

Array<BlockRV> ScheduleNode::GetRootBlocks() {
  Array<tir::StmtSRef> tir_result;
  Array<BlockRV> outputs;
  const auto* root_block = this->sch->root->GetStmt<tir::BlockNode>();
  CHECK(root_block) << "TypeError: Expects Block, but gets: " << root_block;
  tir::PreOrderVisit(root_block->body, [&tir_result, &outputs, this](const ObjectRef& obj) -> bool {
    if (const auto* block = obj.as<tir::BlockNode>()) {
      // Found the output from TIR
      tir::StmtSRef block_sref = this->sch->stmt2ref.at(block);
      tir_result.push_back(block_sref);
      // Create the output random variable
      BlockRV block_rv;
      outputs.push_back(block_rv);
      // Update the symbol table
      this->sym_tab.Set(block_rv, block_sref);
      return false;
    }
    return true;
  });
  // Record the instruction
  this->trace->Append(GetRootBlocksAttrs::Make(outputs));
  return outputs;
}

Array<BlockRV> ScheduleNode::GetLeafBlocks() {
  class BlockVisitor : public tir::StmtVisitor {
   public:
    void VisitStmt_(const tir::BlockNode* block) override {
      if (!stack.empty()) {
        children_counter[stack.back()] += 1;
      }
      children_counter[block] = 0;
      stack.push_back(block);
      tir::StmtVisitor::VisitStmt_(block);
      stack.pop_back();
    }

    std::unordered_map<const tir::BlockNode*, int> children_counter;
    std::vector<const tir::BlockNode*> stack;
  } v;
  const auto* root_block = this->sch->root->GetStmt<tir::BlockNode>();
  CHECK(root_block) << "TypeError: Expects Block, but gets: " << root_block;
  v(GetRef<tir::Block>(root_block));

  Array<tir::StmtSRef> tir_result;
  Array<BlockRV> outputs;
  for (const auto& kv : v.children_counter) {
    if (kv.second != 0) {
      continue;
    }
    // Found the output from TIR
    tir::StmtSRef block_sref = this->sch->stmt2ref.at(kv.first);
    tir_result.push_back(block_sref);
    // Create the output random variable
    BlockRV block_rv;
    outputs.push_back(block_rv);
    // Update the symbol table
    this->sym_tab.Set(block_rv, block_sref);
  }
  // Record the instruction
  this->trace->Append(GetLeafBlocksAttrs::Make(outputs));
  return outputs;
}

/**************** Schedule Primitives ****************/

void ScheduleNode::MarkLoop(const LoopRV& loop, const String& ann_key, const PrimExpr& ann_val) {
  CHECK(ann_val->IsInstance<tir::StringImmNode>() || ann_val->IsInstance<IntImmNode>())
      << "TypeError: Only StringImm and IntImm are supported for now, but gets: "
      << ann_val->GetTypeKey();
  AddAnn(this->sch, this->Eval(loop), ann_key, ann_val);
  // Record the instruction
  this->trace->Append(MarkLoopAttrs::Make(loop, ann_key, ann_val));
}

void ScheduleNode::MarkBlock(const BlockRV& block, const String& ann_key, const PrimExpr& ann_val) {
  int value = this->Eval(ann_val);
  AddAnn(this->sch, this->Eval(block), ann_key, tir::StringImm(std::to_string(value)));
  // Record the instruction
  this->trace->Append(MarkBlockAttrs::Make(block, ann_key, ann_val));
}

LoopRV ScheduleNode::Fuse(const Array<LoopRV>& loops) {
  CHECK(!loops.empty()) << "ValueError: Cannot fuse 0 loops";
  // Output from TIR
  tir::StmtSRef loop_sref = this->Eval(loops[0]);
  for (int i = 1, n = loops.size(); i < n; ++i) {
    loop_sref = this->sch->fuse(loop_sref, this->Eval(loops[i]));
  }
  // Create the output random variable
  LoopRV output;
  // Update the symbol table
  this->sym_tab.Set(output, loop_sref);
  // Record the instruction
  this->trace->Append(FuseAttrs::Make(loops, output));
  return output;
}

Array<LoopRV> ScheduleNode::Split(const LoopRV& loop, const Array<Optional<PrimExpr>>& factors) {
  // Find the output from TIR
  int n_splits = factors.size();
  std::vector<tir::StmtSRef> tir_result;
  int none_idx = -1;
  for (int i = 0, n = factors.size(); i < n; ++i) {
    if (!factors[i].defined()) {
      CHECK_EQ(none_idx, -1) << "ValueError: `split` allows only at most one tile size to be None";
      CHECK(i == 0 || i == n - 1)
          << "ValueError: `split` only allows None to appear at the start or end of the factors";
      none_idx = i;
    }
  }
  if (none_idx == -1 || none_idx == 0) {
    tir::StmtSRef tir_loop = Eval(loop);
    for (int i = n_splits - 1; i >= 1; --i) {
      const PrimExpr& extent = tir_loop->GetStmt<tir::LoopNode>()->extent;
      int factor = this->Eval(factors[i].value());
      PrimExpr nparts = floordiv(extent + factor - 1, factor);
      Array<tir::StmtSRef> split_result = this->sch->split(tir_loop, nparts, factor);
      CHECK_EQ(split_result.size(), 2);
      tir_result.push_back(split_result[1]);
      tir_loop = split_result[0];
    }
    tir_result.push_back(tir_loop);
    std::reverse(tir_result.begin(), tir_result.end());
  } else {
    tir::StmtSRef tir_loop = Eval(loop);
    for (int i = 0; i < n_splits - 1; ++i) {
      const PrimExpr& extent = tir_loop->GetStmt<tir::LoopNode>()->extent;
      int nparts = this->Eval(factors[i].value());
      PrimExpr factor = floordiv(extent + nparts - 1, nparts);
      Array<tir::StmtSRef> split_result = this->sch->split(tir_loop, nparts, factor);
      CHECK_EQ(split_result.size(), 2);
      tir_result.push_back(split_result[0]);
      tir_loop = split_result[1];
    }
    tir_result.push_back(tir_loop);
  }
  // Create the output random variable
  Array<LoopRV> outputs;
  for (const tir::StmtSRef& axis : tir_result) {
    LoopRV output;
    outputs.push_back(output);
    // Update the symbol table
    this->sym_tab.Set(output, axis);
  }
  // Record the instruction
  this->trace->Append(SplitAttrs::Make(loop, factors, outputs));
  return outputs;
}

void ScheduleNode::Reorder(const Array<LoopRV>& after_axes) {
  // Find the inputs to TIR
  std::vector<tir::StmtSRef> tir_inputs;
  for (const LoopRV& loop : after_axes) {
    tir_inputs.push_back(Eval(loop));
  }
  this->sch->reorder(tir_inputs);
  // Record the instruction
  this->trace->Append(ReorderAttrs::Make(after_axes));
}

void ScheduleNode::ComputeAt(const BlockRV& block, const LoopRV& loop) {
  ObjectRef loop_eval = this->EvalLoopExtended(loop);
  if (loop_eval.same_as(LoopRV::ComputeInlineRV())) {
    ComputeInline(block);
    return;
  }
  if (loop_eval.same_as(LoopRV::ComputeRootRV())) {
    return;
  }
  // Find the inputs to TIR
  tir::StmtSRef block_sref = this->Eval(block);
  tir::StmtSRef loop_sref = Downcast<tir::StmtSRef>(loop_eval);
  this->sch->compute_at(block_sref, loop_sref, true);
  // Record the instruction
  this->trace->Append(ComputeAtAttrs::Make(block, loop));
}

void ScheduleNode::ReverseComputeAt(const BlockRV& block, const LoopRV& loop) {
  ObjectRef loop_eval = this->EvalLoopExtended(loop);
  if (loop_eval.same_as(LoopRV::ComputeInlineRV())) {
    ReverseComputeInline(block);
    return;
  }
  if (loop_eval.same_as(LoopRV::ComputeRootRV())) {
    return;
  }
  // Find the inputs to TIR
  tir::StmtSRef block_sref = this->Eval(block);
  tir::StmtSRef loop_sref = Downcast<tir::StmtSRef>(loop_eval);
  this->sch->reverse_compute_at(block_sref, loop_sref, true);
  // Record the instruction
  this->trace->Append(ReverseComputeAtAttrs::Make(block, loop));
}

void ScheduleNode::ComputeInline(const BlockRV& block) {
  // Find the inputs to TIR
  tir::StmtSRef block_sref = this->Eval(block);
  this->sch->compute_inline(block_sref);
  // Record the instruction
  this->trace->Append(ComputeInlineAttrs::Make(block));
}

void ScheduleNode::ReverseComputeInline(const BlockRV& block) {
  // Find the inputs to TIR
  tir::StmtSRef block_sref = this->Eval(block);
  this->sch->reverse_compute_inline(block_sref);
  // Record the instruction
  this->trace->Append(ReverseComputeInlineAttrs::Make(block));
}

BlockRV ScheduleNode::CacheRead(const BlockRV& block, int i, const String& storage_scope) {
  // Find the output from TIR
  tir::StmtSRef tir_result = this->sch->cache_read(Eval(block), i, storage_scope);
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.Set(output, tir_result);
  // Record the instruction
  this->trace->Append(CacheReadAttrs::Make(block, i, storage_scope, output));
  return output;
}

BlockRV ScheduleNode::CacheWrite(const BlockRV& block, int i, const String& storage_scope) {
  // Find the output from TIR
  tir::StmtSRef tir_result = this->sch->cache_write(Eval(block), i, storage_scope);
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.Set(output, tir_result);
  // Record the instruction
  this->trace->Append(CacheWriteAttrs::Make(block, i, storage_scope, output));
  return output;
}

BlockRV ScheduleNode::Blockize(const LoopRV& loop_rv, const String& exec_scope) {
  // Find the output from TIR
  tir::StmtSRef loop_sref = this->Eval(loop_rv);
  tir::StmtSRef tir_result = this->sch->blockize(loop_sref, exec_scope);
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.Set(output, tir_result);
  // Record the instruction
  this->trace->Append(BlockizeAttrs::Make(loop_rv, exec_scope, output));
  return output;
}

BlockRV ScheduleNode::DecomposeReduction(const BlockRV& block, const LoopRV& loop) {
  // Find the output from TIR
  tir::StmtSRef tir_result = this->sch->decompose_reduction(Eval(block), Eval(loop));
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.Set(output, tir_result);
  // Record the instruction
  this->trace->Append(DecomposeReductionAttrs::Make(block, loop, output));
  return output;
}

void ScheduleNode::Parallel(const LoopRV& loop) {
  tir::StmtSRef loop_sref = this->Eval(loop);
  sch->parallel(loop_sref);
  // Record the instruction
  this->trace->Append(ParallelAttrs::Make(loop));
}

void ScheduleNode::Vectorize(const LoopRV& loop) {
  tir::StmtSRef loop_sref = this->Eval(loop);
  sch->vectorize(loop_sref);
  // Record the instruction
  this->trace->Append(VectorizeAttrs::Make(loop));
}

void ScheduleNode::Unroll(const LoopRV& loop) {
  tir::StmtSRef loop_sref = this->Eval(loop);
  sch->unroll(loop_sref);
  // Record the instruction
  this->trace->Append(UnrollAttrs::Make(loop));
}

void ScheduleNode::Bind(const LoopRV& loop, const String& thread_axis) {
  tir::StmtSRef loop_sref = this->Eval(loop);
  tir::IterVar iter_var =
      tir::IterVar(Range(nullptr), tir::Var(thread_axis), tir::kThreadIndex, thread_axis);
  sch->bind(loop_sref, iter_var);
  // Record the instruction
  this->trace->Append(BindAttrs::Make(loop, thread_axis));
}

void ScheduleNode::EnterPostProc() { this->trace->Append(EnterPostProcAttrs::Make()); }

/**************** FFI ****************/

struct Internal {
  /*!
   * \brief FFI function, corresponds to Schedule::Schedule
   * \sa Schedule::Schedule
   */
  static Schedule New(tir::PrimFunc func, Optional<Integer> seed) { return Schedule(func, seed); }
  /**************** Utility ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Seed
   * \sa ScheduleNode::Seed
   */
  static void Seed(Schedule sch, int seed) { return sch->Seed(seed); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::copy
   * \sa ScheduleNode::Copy
   */
  static Schedule Copy(Schedule sch, int new_seed) { return sch->Copy(new_seed); }
  /**************** Evaluation of random variables ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Eval
   * \sa ScheduleNode::Eval
   */
  static ObjectRef Eval(Schedule sch, ObjectRef obj) {
    if (const auto* v = obj.as<BlockRVNode>()) {
      return sch->Eval(GetRef<BlockRV>(v));
    } else if (const auto* v = obj.as<LoopRVNode>()) {
      return sch->EvalLoopExtended(GetRef<LoopRV>(v));
    } else if (const auto* v = obj.as<BufferRVNode>()) {
      return sch->Eval(GetRef<BufferRV>(v));
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
                                           int max_innermost_factor,
                                           Optional<Array<ObjectRef>> decision) {
    return sch->SamplePerfectTile(n_splits, loop, max_innermost_factor, decision);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::SampleTileFactor
   * \sa ScheduleNode::SampleTileFactor
   */
  static Array<tir::Var> SampleTileFactor(Schedule sch, int n_splits, LoopRV loop,
                                          Array<Integer> where,
                                          Optional<Array<ObjectRef>> decision) {
    return sch->SampleTileFactor(n_splits, loop, where, decision);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::SampleInt
   * \sa ScheduleNode::SampleInt
   */
  static tir::Var SampleInt(Schedule sch, PrimExpr min_inclusive, PrimExpr max_exclusive,
                            Optional<ObjectRef> decision) {
    return sch->SampleInt(min_inclusive, max_exclusive, decision);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::SampleCategorical
   * \sa ScheduleNode::SampleCategorical
   */
  static tir::Var SampleCategorical(Schedule sch, Array<Integer> candidates, Array<FloatImm> probs,
                                    Optional<ObjectRef> decision) {
    return sch->SampleCategorical(candidates, probs, decision);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::SampleComputeLocation
   * \sa ScheduleNode::SampleComputeLocation
   */
  static LoopRV SampleComputeLocation(Schedule sch, BlockRV block, Optional<ObjectRef> decision) {
    return sch->SampleComputeLocation(block, decision);
  }
  /**************** Block/Loop Relationship ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::GetProducers
   * \sa ScheduleNode::GetProducers
   */
  static Array<BlockRV> GetProducers(Schedule sch, BlockRV block) {
    return sch->GetProducers(block);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::GetConsumers
   * \sa ScheduleNode::GetConsumers
   */
  static Array<BlockRV> GetConsumers(Schedule sch, BlockRV block) {
    return sch->GetConsumers(block);
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
  /*!
   * \brief FFI function, corresponds to ScheduleNode::GetReadBuffers
   * \sa ScheduleNode::GetReadBuffers
   */
  static Array<BufferRV> GetReadBuffers(Schedule sch, BlockRV block) {
    return sch->GetReadBuffers(block);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::GetWriteBuffers
   * \sa ScheduleNode::GetWriteBuffers
   */
  static Array<BufferRV> GetWriteBuffers(Schedule sch, BlockRV block) {
    return sch->GetWriteBuffers(block);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::GetRootBlocks
   * \sa ScheduleNode::GetRootBlocks
   */
  static Array<BlockRV> GetRootBlocks(Schedule sch) { return sch->GetRootBlocks(); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::GetLeafBlocks
   * \sa ScheduleNode::GetLeafBlocks
   */
  static Array<BlockRV> GetLeafBlocks(Schedule sch) { return sch->GetLeafBlocks(); }
  /**************** Scheduling Primitives ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::MarkLoop
   * \sa ScheduleNode::MarkLoop
   */
  static void MarkLoop(Schedule sch, LoopRV loop, String ann_key, ObjectRef ann_val) {
    if (const auto* str_obj = ann_val.as<StringObj>()) {
      sch->MarkLoop(loop, ann_key, tir::StringImm(GetRef<String>(str_obj)));
    } else {
      sch->MarkLoop(loop, ann_key, Downcast<PrimExpr>(ann_val));
    }
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::MarkBlock
   * \sa ScheduleNode::MarkBlock
   */
  static void MarkBlock(Schedule sch, BlockRV block, String ann_key, PrimExpr ann_val) {
    sch->MarkBlock(block, ann_key, ann_val);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Fuse
   * \sa ScheduleNode::Fuse
   */
  static LoopRV Fuse(Schedule sch, Array<LoopRV> loops) { return sch->Fuse(loops); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Split
   * \sa ScheduleNode::Split
   */
  static Array<LoopRV> Split(Schedule sch, LoopRV loop, Array<Optional<PrimExpr>> factors) {
    return sch->Split(loop, factors);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Reorder
   * \sa ScheduleNode::Reorder
   */
  static void Reorder(Schedule sch, Array<LoopRV> after_axes) { return sch->Reorder(after_axes); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::ComputeAt
   * \sa ScheduleNode::ComputeAt
   */
  static void ComputeAt(Schedule sch, BlockRV block, LoopRV loop) { sch->ComputeAt(block, loop); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::ReverseComputeAt
   * \sa ScheduleNode::ReverseComputeAt
   */
  static void ReverseComputeAt(Schedule sch, BlockRV block, LoopRV loop) {
    sch->ReverseComputeAt(block, loop);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::ComputeInline
   * \sa ScheduleNode::ComputeInline
   */
  static void ComputeInline(Schedule sch, BlockRV block) { sch->ComputeInline(block); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::ReverseComputeInline
   * \sa ScheduleNode::ReverseComputeInline
   */
  static void ReverseComputeInline(Schedule sch, BlockRV block) {
    sch->ReverseComputeInline(block);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::CacheRead
   * \sa ScheduleNode::CacheRead
   */
  static BlockRV CacheRead(Schedule sch, BlockRV block, int i, String storage_scope) {
    return sch->CacheRead(block, i, storage_scope);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::CacheWrite
   * \sa ScheduleNode::CacheWrite
   */
  static BlockRV CacheWrite(Schedule sch, BlockRV block, int i, String storage_scope) {
    return sch->CacheWrite(block, i, storage_scope);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Blockize
   * \sa ScheduleNode::Blockize
   */
  static BlockRV Blockize(Schedule sch, LoopRV loop, String exec_scope) {
    return sch->Blockize(loop, exec_scope);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::DecomposeReduction
   * \sa ScheduleNode::DecomposeReduction
   */
  static BlockRV DecomposeReduction(Schedule sch, BlockRV block, LoopRV loop) {
    return sch->DecomposeReduction(block, loop);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Parallel
   * \sa ScheduleNode::Parallel
   */
  static void Parallel(Schedule sch, LoopRV loop) { sch->Parallel(loop); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Vectorize
   * \sa ScheduleNode::Vectorize
   */
  static void Vectorize(Schedule sch, LoopRV loop) { sch->Vectorize(loop); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Unroll
   * \sa ScheduleNode::Unroll
   */
  static void Unroll(Schedule sch, LoopRV loop) { sch->Unroll(loop); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Bind
   * \sa ScheduleNode::Bind
   */
  static void Bind(Schedule sch, LoopRV loop, String thread_axis) { sch->Bind(loop, thread_axis); }
};

TVM_REGISTER_NODE_TYPE(ScheduleNode);
TVM_REGISTER_GLOBAL("meta_schedule.Schedule").set_body_typed(Internal::New);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSeed").set_body_typed(Internal::Seed);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleCopy").set_body_typed(Internal::Copy);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleEval").set_body_typed(Internal::Eval);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSamplePerfectTile")
    .set_body_typed(Internal::SamplePerfectTile);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSampleTileFactor")
    .set_body_typed(Internal::SampleTileFactor);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSampleInt").set_body_typed(Internal::SampleInt);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSampleCategorical")
    .set_body_typed(Internal::SampleCategorical);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSampleComputeLocation")
    .set_body_typed(Internal::SampleComputeLocation);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetProducers").set_body_typed(Internal::GetProducers);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetConsumers").set_body_typed(Internal::GetConsumers);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetBlock").set_body_typed(Internal::GetBlock);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetAxes").set_body_typed(Internal::GetAxes);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetReadBuffers")
    .set_body_typed(Internal::GetReadBuffers);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetWriteBuffers")
    .set_body_typed(Internal::GetWriteBuffers);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetRootBlocks").set_body_typed(Internal::GetRootBlocks);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetLeafBlocks").set_body_typed(Internal::GetLeafBlocks);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleMarkLoop").set_body_typed(Internal::MarkLoop);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleMarkBlock").set_body_typed(Internal::MarkBlock);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleFuse").set_body_typed(Internal::Fuse);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSplit").set_body_typed(Internal::Split);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReorder").set_body_typed(Internal::Reorder);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleComputeAt").set_body_typed(Internal::ComputeAt);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReverseComputeAt")
    .set_body_typed(Internal::ReverseComputeAt);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleComputeInline").set_body_typed(Internal::ComputeInline);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReverseComputeInline")
    .set_body_typed(Internal::ReverseComputeInline);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleCacheRead").set_body_typed(Internal::CacheRead);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleCacheWrite").set_body_typed(Internal::CacheWrite);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleBlockize").set_body_typed(Internal::Blockize);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleDecomposeReduction")
    .set_body_typed(Internal::DecomposeReduction);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleParallel").set_body_typed(Internal::Parallel);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleVectorize").set_body_typed(Internal::Vectorize);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleUnroll").set_body_typed(Internal::Unroll);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleBind").set_body_typed(Internal::Bind);

}  // namespace meta_schedule
}  // namespace tvm
