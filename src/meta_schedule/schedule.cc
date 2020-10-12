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
    : Schedule(orig_func, tir::ScheduleNode::Create(orig_func), {}, {}, {}, seed) {}

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
    CHECK(iter != this->sym_tab.end())
        << "IndexError: Cannot find corresponding ExprRV: " << var << '@' << var.get();
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
    this->sym_tab.emplace(output, Integer(samples[i]));
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
  this->sym_tab.emplace(output, Integer(n_fusible));
  // Put the instruction in the trace
  this->trace.push_back(
      SampleFusibleLoopsAttrs::MakeInst(loops, loop_types, max_extent, include_overflow_loop,
                                        static_cast<int>(order), static_cast<int>(mode), output));
  // Put the sampling decision in the decision table
  this->decisions.Set(this->trace.back(), {Integer(n_fusible)});
  return output;
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
      this->sym_tab.emplace(block_rv, block_sref);
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
    this->sym_tab.emplace(block_rv, block_sref);
  }
  // Put the instruction in the trace
  this->trace.push_back(GetLeafBlocksAttrs::MakeInst(outputs));
  return outputs;
}

/**************** Schedule Primitives ****************/

LoopRV ScheduleNode::Fuse(const Array<LoopRV>& loops, Optional<Range> opt_range) {
  if (!opt_range.defined()) {
    opt_range = Range::FromMinExtent(Integer(0), Integer(loops.size()));
  }
  Range range = opt_range.value();
  int left = this->Eval(range->min);
  int right = this->Eval(range->min + range->extent);
  CHECK(left < right) << "ValueError: Cannot fuse an empty range [" << left << ", " << right << ")";
  // Output from TIR
  tir::StmtSRef loop_sref = this->Eval(loops[left]);
  for (int i = left + 1; i < right; ++i) {
    loop_sref = this->sch->fuse(loop_sref, this->Eval(loops[i]));
  }
  // Create the output random variable
  LoopRV output;
  // Update the symbol table
  this->sym_tab.emplace(output, loop_sref);
  // Put the instruction in the trace
  this->trace.push_back(FuseAttrs::MakeInst(loops, opt_range, output));
  return output;
}

void ScheduleNode::Parallel(const LoopRV& loop) {
  tir::StmtSRef loop_sref = Eval(loop);
  this->sch->parallel(loop_sref);
  // Put the instruction in the trace
  this->trace.push_back(ParallelAttrs::MakeInst(loop));
}

void ScheduleNode::Vectorize(const LoopRV& loop) {
  tir::StmtSRef loop_sref = Eval(loop);
  this->sch->vectorize(loop_sref);
  // Put the instruction in the trace
  this->trace.push_back(VectorizeAttrs::MakeInst(loop));
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

void ScheduleNode::ReverseComputeAt(const BlockRV& block, const LoopRV& loop) {
  // Find the inputs to TIR
  tir::StmtSRef block_sref = this->Eval(block);
  tir::StmtSRef loop_sref = this->Eval(loop);
  this->sch->reverse_compute_at(block_sref, loop_sref);
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

/**************** Trace-related ****************/

void ScheduleNode::MutateDecision(const Instruction& inst,
                                  const Optional<Array<ObjectRef>>& decision) {
  if (decision.defined()) {
    this->decisions.Set(inst, decision.value());
  } else {
    this->decisions.erase(inst);
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
        new_sch->sym_tab[new_outputs[i]] = decisions[i];
      }
      new_sch->decisions.Set(new_inst, decisions);
    }
  }
  // Step 3. Re-assign all the variables back according to the symbol table
  this->sch = new_sch->sch;
  for (auto& kv_entry : this->sym_tab) {
    const ObjectRef& old_var = kv_entry.first;
    const ObjectRef& new_var = GetRef<ObjectRef>(var_map.at(old_var.get()));
    kv_entry.second = new_sch->sym_tab.at(new_var);
  }
  // Step 4. Map decisions back
  Map<Instruction, Array<ObjectRef>> decisions;
  for (auto& kv : this->decisions) {
    const InstructionNode* old_inst = kv.first.operator->();
    const InstructionNode* new_inst = inst_map.at(old_inst);
    const Array<ObjectRef>& decision = new_sch->decisions.at(GetRef<Instruction>(new_inst));
    decisions.Set(GetRef<Instruction>(old_inst), decision);
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
  static tir::Var SampleFusibleLoops(Schedule sch, Array<LoopRV> loops, Array<Integer> loop_types,
                                     int max_extent, bool include_overflow_loop, int _order,
                                     int _mode) {
    ScheduleNode::Order order = static_cast<ScheduleNode::Order>(_order);
    ScheduleNode::Mode mode = static_cast<ScheduleNode::Mode>(_mode);
    return sch->SampleFusibleLoops(loops, loop_types, max_extent, include_overflow_loop, order,
                                   mode);
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
   * \brief FFI function, corresponds to ScheduleNode::Fuse
   * \sa ScheduleNode::Fuse
   */
  static LoopRV Fuse(Schedule sch, Array<LoopRV> loops, Optional<Range> range) {
    return sch->Fuse(loops, range);
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
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleEval").set_body_typed(Internal::Eval);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSamplePerfectTile")
    .set_body_typed(Internal::SamplePerfectTile);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSampleTileFactor")
    .set_body_typed(Internal::SampleTileFactor);
TVM_REGISTER_GLOBAL("meta_schedule.SampleFusibleLoops")
    .set_body_typed(Internal::SampleFusibleLoops);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetOnlyConsumer")
    .set_body_typed(Internal::GetOnlyConsumer);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetBlock").set_body_typed(Internal::GetBlock);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetAxes").set_body_typed(Internal::GetAxes);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetRootBlocks").set_body_typed(Internal::GetRootBlocks);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleGetLeafBlocks").set_body_typed(Internal::GetLeafBlocks);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleFuse").set_body_typed(Internal::Fuse);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleParallel").set_body_typed(Internal::Parallel);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleVectorize").set_body_typed(Internal::Vectorize);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleSplit").set_body_typed(Internal::Split);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReorder").set_body_typed(Internal::Reorder);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReverseComputeAt")
    .set_body_typed(Internal::ReverseComputeAt);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleComputeInline").set_body_typed(Internal::ComputeInline);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReverseComputeInline")
    .set_body_typed(Internal::ReverseComputeInline);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleCacheWrite").set_body_typed(Internal::CacheWrite);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleDecomposeReduction")
    .set_body_typed(Internal::DecomposeReduction);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleMutateDecision")
    .set_body_typed(Internal::MutateDecision);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReSample").set_body_typed(Internal::ReSample);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleReplayDecision")
    .set_body_typed(Internal::ReplayDecision);

}  // namespace meta_schedule
}  // namespace tvm
