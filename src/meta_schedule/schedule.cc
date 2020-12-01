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

Schedule::Schedule(tir::PrimFunc orig_func, tir::Schedule sch, Array<Instruction> trace,
                   Map<Instruction, Array<ObjectRef>> decisions, TSymbolTable sym_tab,
                   Optional<Integer> seed) {
  ObjectPtr<ScheduleNode> n = make_object<ScheduleNode>();
  n->orig_func = std::move(orig_func);
  n->sch = std::move(sch);
  n->trace = std::move(trace);
  n->decisions = std::move(decisions);
  n->sym_tab = std::move(sym_tab);
  if (seed.defined()) {
    n->sampler.Seed(seed.value()->value);
  }
  data_ = std::move(n);
}

Schedule::Schedule(tir::PrimFunc orig_func, Optional<Integer> seed)
    : Schedule(/*orig_func=*/orig_func, /*sch=*/tir::ScheduleNode::Create(orig_func), /*trace=*/{},
               /*decisions=*/{}, /*sym_tab=*/{}, /*seed=*/seed) {}

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
                  /*trace=*/this->trace,
                  /*decisions=*/this->decisions,
                  /*sym_tab=*/translator.Trans(this->sym_tab),
                  /*seed=*/Integer(new_seed));
}

/**************** Serialization ****************/

Schedule ScheduleNode::Import(const Array<ObjectRef>& records, const tir::PrimFunc& orig_func,
                              Optional<Integer> seed) {
  Schedule sch(orig_func, seed);
  Map<String, ObjectRef> named_rvs;
  for (const ObjectRef& record_obj : records) {
    Array<ObjectRef> record = Downcast<Array<ObjectRef>>(record_obj);
    Instruction::ApplyToSchedule(sch.operator->(), record, &named_rvs);
  }
  return sch;
}

Array<ObjectRef> ScheduleNode::Export() const {
  Map<ObjectRef, String> rv_names;
  for (const Instruction& inst : trace) {
    for (const ObjectRef& output : inst->outputs) {
      int i = rv_names.size();
      rv_names.Set(output, "v" + std::to_string(i));
    }
  }
  Array<ObjectRef> records;
  for (const Instruction& inst : trace) {
    if (inst->inst_attrs->IsInstance<EnterPostProcAttrs>()) {
      break;
    }
    Optional<Array<ObjectRef>> decision = decisions.count(inst)
                                              ? Optional<Array<ObjectRef>>(decisions.at(inst))
                                              : Optional<Array<ObjectRef>>(NullOpt);
    records.push_back(inst->Export(rv_names, decision));
  }
  return records;
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
    this->sym_tab.Set(output, Integer(samples[i]));
  }
  // Put the instruction in the trace
  this->trace.push_back(
      SamplePerfectTileAttrs::MakeInst(n_splits, loop, max_innermost_factor, outputs));
  // Put the sampling decision in the decision table
  this->decisions.Set(this->trace.back(), AsArray<int, ObjectRef>()(samples));
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
    this->sym_tab.Set(output, Integer(samples[i]));
  }
  // Put the instruction in the trace
  this->trace.push_back(SampleTileFactorAttrs::MakeInst(n_splits, loop, where, outputs));
  // Put the sampling decision in the decision table
  this->decisions.Set(this->trace.back(), AsArray<int, ObjectRef>()(samples));
  return outputs;
}

tir::Var ScheduleNode::SampleFusibleLoops(const Array<LoopRV>& loops,
                                          const Array<Integer>& loop_types, int max_extent,
                                          bool include_overflow_loop, Order order, Mode mode) {
  CHECK_EQ(loops.size(), loop_types.size())
      << "ValueError: 'loops' and 'loop_types' must have equal number of elements";
  int n_loops = loops.size();
  int i_start, i_end, i_delta;
  if (order == Order::outer_to_inner) {
    // 0 to n_loops - 1, step = 1
    i_start = 0;
    i_end = n_loops;
    i_delta = 1;
  } else if (order == Order::inner_to_order) {
    // n_loops - 1 to 0, step = -1
    i_start = n_loops - 1;
    i_end = -1;
    i_delta = -1;
  } else {
    LOG(FATAL) << "Not reachable";
    throw;
  }
  int n_fusible = 0;
  int64_t prod_extent = 1;
  for (int i = i_start; i != i_end; i += i_delta) {
    // Get the current loop
    const LoopRV& loop_rv = loops[i];
    tir::StmtSRef loop_sref = Eval(loop_rv);
    int loop_type = loop_types[i];
    const auto* loop = loop_sref->GetStmt<tir::LoopNode>();
    CHECK(loop) << "TypeError: Expects Loop, but gets: " << loop_sref->stmt->GetTypeKey();
    // Check if the loop has more than one children
    bool has_multi_children = loop->body->IsInstance<tir::SeqStmtNode>();
    // If scanning from inner to outer, then we cannot fuse a loop who has multiple children
    // But if scanning from outer to inner, we can actually fuse it
    if (has_multi_children && order == Order::inner_to_order) {
      break;
    }
    // Loop cannot have any annotation and must be data parallel
    if (!loop->annotations.empty() || loop_type != tir::IterVarType::kDataPar) {
      break;
    }
    // then this loop can be fused
    const auto* extent = loop->extent.as<IntImmNode>();
    if (prod_extent * extent->value > max_extent) {
      if (include_overflow_loop) {
        prod_extent *= extent->value;
        ++n_fusible;
      }
      break;
    } else {
      prod_extent *= extent->value;
      ++n_fusible;
    }
    // If scanning from outer to inner, then we cannot fuse the next loop if the current loop has
    // multiple children
    if (has_multi_children && order == Order::outer_to_inner) {
      break;
    }
  }
  if (prod_extent == 1) {
    n_fusible = 0;
  }
  if (mode == Mode::rand && n_fusible != 0) {
    n_fusible = sampler.SampleInt(0, n_fusible + 1);
  }
  // Create the output random variable
  tir::Var output("n_fusible");
  // Update the symbol table
  this->sym_tab.Set(output, Integer(n_fusible));
  // Put the instruction in the trace
  this->trace.push_back(
      SampleFusibleLoopsAttrs::MakeInst(loops, loop_types, max_extent, include_overflow_loop,
                                        static_cast<int>(order), static_cast<int>(mode), output));
  // Put the sampling decision in the decision table
  this->decisions.Set(this->trace.back(), {Integer(n_fusible)});
  return output;
}

tir::Var ScheduleNode::SampleCategorical(const Array<Integer>& candidates,
                                         const Array<FloatImm>& probs) {
  // Sample the output
  CHECK_EQ(candidates.size(), probs.size()) << "ValueError: When sampling ";
  std::vector<double> probs_vec;
  probs_vec.reserve(probs.size());
  for (const FloatImm& prob : probs) {
    probs_vec.push_back(prob->value);
  }
  int result = candidates[sampler.MakeMultinomial(probs_vec)()];
  // Create the output random variable
  tir::Var output("n");
  // Update the symbol table
  this->sym_tab.Set(output, Integer(result));
  // Put the instruction in the trace
  this->trace.push_back(SampleCategoricalAttrs::MakeInst(candidates, probs, output));
  // Put the sampling decision in the decision table
  this->decisions.Set(this->trace.back(), {Integer(result)});
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
  // Put the instruction in the trace
  this->trace.push_back(GetProducersAttrs::MakeInst(block, outputs));
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
  // Put the instruction in the trace
  this->trace.push_back(GetConsumersAttrs::MakeInst(block, outputs));
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
    this->sym_tab.Set(output, axis);
  }
  // Put the instruction in the trace
  this->trace.push_back(GetAxesAttrs::MakeInst(block, outputs));
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
  // Put the instruction in the trace
  this->trace.push_back(GetReadBuffersAttrs::MakeInst(block_rv, outputs));
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
  // Put the instruction in the trace
  this->trace.push_back(GetWriteBuffersAttrs::MakeInst(block_rv, outputs));
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
  // Put the instruction in the trace
  this->trace.push_back(GetRootBlocksAttrs::MakeInst(outputs));
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
  // Put the instruction in the trace
  this->trace.push_back(GetLeafBlocksAttrs::MakeInst(outputs));
  return outputs;
}

/**************** Schedule Primitives ****************/

void ScheduleNode::MarkLoopType(const Array<LoopRV>& loops, const String& ann_key,
                                const String& ann_val, const Optional<PrimExpr>& first_n,
                                const Optional<PrimExpr>& last_n) {
  Array<tir::StmtSRef> loop_srefs;
  loop_srefs.reserve(loops.size());
  for (const LoopRV& loop_rv : loops) {
    loop_srefs.push_back(Eval(loop_rv));
  }
  if (first_n.defined()) {
    int n = Eval(first_n.value());
    int st = 0;
    int ed = n;
    AnnotateLoopType(this->sch, {loop_srefs.begin() + st, loop_srefs.begin() + ed},  //
                     ann_key, ann_val);
  }
  if (last_n.defined()) {
    int n = Eval(last_n.value());
    int ed = static_cast<int>(loops.size());
    int st = ed - n;
    AnnotateLoopType(this->sch, {loop_srefs.begin() + st, loop_srefs.begin() + ed},  //
                     ann_key, ann_val);
  }
  // Put the instruction in the trace
  this->trace.push_back(MarkLoopTypeAttrs::MakeInst(
      loops, ann_key, ann_val, first_n.value_or(Integer(0)), last_n.value_or(Integer(0))));
}

void ScheduleNode::MarkBlockType(const BlockRV& block, const String& ann_key,
                                 const String& ann_val) {
  AnnotateBlockType(this->sch, this->Eval(block), ann_key, ann_val);
  // Put the instruction in the trace
  this->trace.push_back(MarkBlockTypeAttrs::MakeInst(block, ann_key, ann_val));
}

LoopRV ScheduleNode::Fuse(const Array<LoopRV>& loops) {
  // Output from TIR
  tir::StmtSRef loop_sref = this->Eval(loops[0]);
  for (int i = 1, n = loops.size(); i < n; ++i) {
    loop_sref = this->sch->fuse(loop_sref, this->Eval(loops[i]));
  }
  // Create the output random variable
  LoopRV output;
  // Update the symbol table
  this->sym_tab.Set(output, loop_sref);
  // Put the instruction in the trace
  this->trace.push_back(FuseAttrs::MakeInst(loops, output));
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

void ScheduleNode::ComputeAt(const BlockRV& block, const LoopRV& loop) {
  // Find the inputs to TIR
  tir::StmtSRef block_sref = this->Eval(block);
  tir::StmtSRef loop_sref = this->Eval(loop);
  this->sch->compute_at(block_sref, loop_sref, true);
  // Put the instruction in the trace
  this->trace.push_back(ComputeAtAttrs::MakeInst(block, loop));
}

void ScheduleNode::ReverseComputeAt(const BlockRV& block, const LoopRV& loop) {
  // Find the inputs to TIR
  tir::StmtSRef block_sref = this->Eval(block);
  tir::StmtSRef loop_sref = this->Eval(loop);
  this->sch->reverse_compute_at(block_sref, loop_sref, true);
  // Put the instruction in the trace
  this->trace.push_back(ReverseComputeAtAttrs::MakeInst(block, loop));
}

void ScheduleNode::ComputeInline(const BlockRV& block) {
  // Find the inputs to TIR
  tir::StmtSRef block_sref = this->Eval(block);
  this->sch->compute_inline(block_sref);
  // Put the instruction in the trace
  this->trace.push_back(ComputeInlineAttrs::MakeInst(block));
}

void ScheduleNode::ReverseComputeInline(const BlockRV& block) {
  // Find the inputs to TIR
  tir::StmtSRef block_sref = this->Eval(block);
  this->sch->reverse_compute_inline(block_sref);
  // Put the instruction in the trace
  this->trace.push_back(ReverseComputeInlineAttrs::MakeInst(block));
}

BlockRV ScheduleNode::CacheRead(const BlockRV& block, int i, const String& storage_scope) {
  // Find the output from TIR
  tir::StmtSRef tir_result = this->sch->cache_read(Eval(block), i, storage_scope);
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.Set(output, tir_result);
  // Put the instruction in the trace
  this->trace.push_back(CacheReadAttrs::MakeInst(block, i, storage_scope, output));
  return output;
}

BlockRV ScheduleNode::CacheWrite(const BlockRV& block, int i, const String& storage_scope) {
  // Find the output from TIR
  tir::StmtSRef tir_result = this->sch->cache_write(Eval(block), i, storage_scope);
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.Set(output, tir_result);
  // Put the instruction in the trace
  this->trace.push_back(CacheWriteAttrs::MakeInst(block, i, storage_scope, output));
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
  // Put the instruction in the trace
  this->trace.push_back(BlockizeAttrs::MakeInst(loop_rv, exec_scope, output));
  return output;
}

BlockRV ScheduleNode::DecomposeReduction(const BlockRV& block, const LoopRV& loop) {
  // Find the output from TIR
  tir::StmtSRef tir_result = this->sch->decompose_reduction(Eval(block), Eval(loop));
  // Create the output random variable
  BlockRV output;
  // Update the symbol table
  this->sym_tab.Set(output, tir_result);
  // Put the instruction in the trace
  this->trace.push_back(DecomposeReductionAttrs::MakeInst(block, loop, output));
  return output;
}

void ScheduleNode::EnterPostProc() { this->trace.push_back(EnterPostProcAttrs::MakeInst()); }

/**************** Trace-related ****************/

void ScheduleNode::MutateDecision(const Instruction& inst,
                                  const Optional<Array<ObjectRef>>& decision) {
  if (decision.defined()) {
    this->decisions.Set(inst, decision.value());
  } else if (this->decisions.count(inst)) {
    this->decisions.erase(inst);
  } else {
    LOG(FATAL) << "ValueError: Cannot find the instruction in decisions";
  }
}

void ScheduleNode::ReSample() { this->Replay(/*follow_decision=*/false); }

void ScheduleNode::ReplayDecision() { this->Replay(/*follow_decision=*/true); }

void ScheduleNode::Replay(bool follow_decision) {
  // Step 1. Create a new schedule to temporarily hold the re-sampling result
  Schedule new_sch(this->orig_func, Integer(this->sampler.ForkSeed()));
  // Maps an old random variable to its corresponding new random variable in the re-sampling
  std::unordered_map<const Object*, const Object*> var_map;
  // Maps an old instruction to its corresponding new instruction
  std::unordered_map<const InstructionNode*, const InstructionNode*> inst_map;

  auto f_var_convert = [&var_map](const tir::Var& var) -> Optional<PrimExpr> {
    const Object* src = var.get();
    if (!var_map.count(src)) {
      return NullOpt;
    }
    const Object* dst = var_map.at(var.get());
    CHECK(dst->IsInstance<tir::VarNode>());
    return GetRef<tir::Var>(static_cast<const tir::VarNode*>(dst));
  };

  auto f_var_map = [&var_map, &f_var_convert](const ObjectRef& obj) -> ObjectRef {
    if (const auto* expr = obj.as<PrimExprNode>()) {
      return tir::Substitute(GetRef<PrimExpr>(expr), f_var_convert);
    } else {
      const Object* src = obj.get();
      CHECK(var_map.count(src));
      const Object* dst = var_map.at(src);
      return GetRef<ObjectRef>(dst);
    }
  };

  // Step 2. Re-do all the instructions in the trace, including sampling instructions
  for (const Instruction& old_inst : this->trace) {
    const Array<ObjectRef>& old_inputs = old_inst->inputs;
    const Array<ObjectRef>& old_outputs = old_inst->outputs;
    // Step 2.1. Construct new inputs
    Array<ObjectRef> new_inputs;
    new_inputs.reserve(old_inputs.size());
    for (const ObjectRef& input : old_inputs) {
      new_inputs.push_back(f_var_map(input));
    }
    // Step 2.2. Construct new outputs
    Array<ObjectRef> new_outputs =
        Instruction::ApplyToSchedule(new_sch.operator->(), old_inst->inst_attrs, new_inputs);
    CHECK_EQ(old_outputs.size(), new_outputs.size()) << "ValueError: Output size mismatch";
    // Step 2.3. Set up correspondence between old and new outputs
    for (int i = 0, n = new_outputs.size(); i < n; ++i) {
      var_map[old_outputs[i].get()] = new_outputs[i].get();
    }
    // Step 2.4. Set up correspondence between old and new instructions
    const Instruction& new_inst = new_sch->trace.back();
    inst_map[old_inst.operator->()] = new_inst.operator->();
    // Step 2.5. Change the decision if we want to follow pre-set decisions
    if (follow_decision && this->decisions.count(old_inst)) {
      Array<ObjectRef> decisions = this->decisions.at(old_inst);
      CHECK_EQ(decisions.size(), new_outputs.size());
      for (int i = 0, n = decisions.size(); i < n; ++i) {
        new_sch->sym_tab.Set(new_outputs[i], decisions[i]);
      }
      new_sch->decisions.Set(new_inst, decisions);
    }
  }
  this->sch = new_sch->sch;
  // Step 3. Re-assign all the variables back according to the symbol table
  {
    TSymbolTable new_sym_tab;
    for (const auto& kv_entry : this->sym_tab) {
      ObjectRef old_var = kv_entry.first;
      ObjectRef new_var = GetRef<ObjectRef>(var_map.at(old_var.get()));
      new_sym_tab.Set(old_var, new_sch->sym_tab.at(new_var));
    }
    this->sym_tab = new_sym_tab;
  }
  // Step 4. Map decisions back
  Map<Instruction, Array<ObjectRef>> decisions;
  for (const Instruction& old_inst : this->trace) {
    Instruction new_inst = GetRef<Instruction>(inst_map.at(old_inst.get()));
    if (new_sch->decisions.count(new_inst)) {
      decisions.Set(old_inst, new_sch->decisions.at(new_inst));
    }
  }
  this->decisions = std::move(decisions);
}

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
  /**************** Serialization ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Import
   * \sa ScheduleNode::Import
   */
  static Schedule Import(Array<ObjectRef> records, tir::PrimFunc orig_func,
                         Optional<Integer> seed) {
    return ScheduleNode::Import(records, orig_func, seed);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Export
   * \sa ScheduleNode::Export
   */
  static Array<ObjectRef> Export(Schedule self) { return self->Export(); }
  /**************** Evaluation of random variables ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::Eval
   * \sa ScheduleNode::Eval
   */
  static ObjectRef Eval(Schedule sch, ObjectRef obj) {
    if (const auto* v = obj.as<BlockRVNode>()) {
      return sch->Eval(GetRef<BlockRV>(v));
    } else if (const auto* v = obj.as<LoopRVNode>()) {
      return sch->Eval(GetRef<LoopRV>(v));
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
  /*!
   * \brief FFI function, corresponds to ScheduleNode::SampleFusibleLoops
   * \sa ScheduleNode::SampleFusibleLoops
   */
  static tir::Var SampleFusibleLoops(Schedule sch, Array<LoopRV> loops, Array<Integer> loop_types,
                                     int max_extent, bool include_overflow_loop, int _order,
                                     int _mode) {
    ScheduleNode::Order order = static_cast<ScheduleNode::Order>(_order);
    ScheduleNode::Mode mode = static_cast<ScheduleNode::Mode>(_mode);
    return sch->SampleFusibleLoops(loops, loop_types, max_extent, include_overflow_loop, order,
                                   mode);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::SampleCategorical
   * \sa ScheduleNode::SampleCategorical
   */
  static tir::Var SampleCategorical(Schedule sch, Array<Integer> candidates,
                                    Array<FloatImm> probs) {
    return sch->SampleCategorical(candidates, probs);
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
   * \brief FFI function, corresponds to ScheduleNode::MarkLoopType
   * \sa ScheduleNode::MarkLoopType
   */
  static void MarkLoopType(Schedule sch, Array<LoopRV> loops, String ann_key, String ann_val,
                           Optional<PrimExpr> first_n, Optional<PrimExpr> last_n) {
    sch->MarkLoopType(loops, ann_key, ann_val, first_n, last_n);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::MarkBlockType
   * \sa ScheduleNode::MarkBlockType
   */
  static void MarkBlockType(Schedule sch, BlockRV block, String ann_key, String ann_val) {
    sch->MarkBlockType(block, ann_key, ann_val);
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
  /**************** Trace-related ****************/
  /*!
   * \brief FFI function, corresponds to ScheduleNode::MutateDecision
   * \sa ScheduleNode::MutateDecision
   */
  static void MutateDecision(Schedule sch, Instruction inst, Optional<Array<ObjectRef>> decision) {
    return sch->MutateDecision(inst, decision);
  }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::ReSample
   * \sa ScheduleNode::ReSample
   */
  static void ReSample(Schedule sch) { sch->ReSample(); }
  /*!
   * \brief FFI function, corresponds to ScheduleNode::ReplayDecision
   * \sa ScheduleNode::ReplayDecision
   */
  static void ReplayDecision(Schedule sch) { sch->ReplayDecision(); }
};

TVM_REGISTER_NODE_TYPE(ScheduleNode);
TVM_REGISTER_GLOBAL("meta_schedule.Schedule").set_body_typed(Internal::New);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSeed").set_body_typed(Internal::Seed);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleCopy").set_body_typed(Internal::Copy);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleImport").set_body_typed(Internal::Import);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleExport").set_body_typed(Internal::Export);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleEval").set_body_typed(Internal::Eval);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSamplePerfectTile")
    .set_body_typed(Internal::SamplePerfectTile);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSampleTileFactor")
    .set_body_typed(Internal::SampleTileFactor);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSampleFusibleLoops")
    .set_body_typed(Internal::SampleFusibleLoops);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSampleCategorical")
    .set_body_typed(Internal::SampleCategorical);
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
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleMarkLoopType").set_body_typed(Internal::MarkLoopType);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleMarkBlockType").set_body_typed(Internal::MarkBlockType);
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
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleMutateDecision")
    .set_body_typed(Internal::MutateDecision);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReSample").set_body_typed(Internal::ReSample);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReplayDecision")
    .set_body_typed(Internal::ReplayDecision);

}  // namespace meta_schedule
}  // namespace tvm
