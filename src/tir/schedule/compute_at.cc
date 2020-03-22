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

#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/arith/int_set.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/ir_pass.h>
#include "schedule_common.h"

namespace tvm {
namespace tir {

/*! Gather all direct blocks in ast subtree. */
class ChildBlockGatherer : public StmtExprVisitor {
 public:
  ChildBlockGatherer(const Schedule& sch,
                     std::unordered_set<StmtSRef, ObjectHash, ObjectEqual>* child_blocks)
      : sch_(sch), child_blocks_(child_blocks) {}

  void VisitStmt_(const BlockNode* op) final {
    const auto* node = static_cast<const StmtNode*>(op);
    child_blocks_->insert(sch_->stmt2ref.at(node));
  }

 private:
  const Schedule& sch_;
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual>* child_blocks_;
};

/*! To find if there is any dependent block under in the specific subtree */
bool FindAny(const Schedule& sch, const Stmt& stmt, const Array<DepEdge>& edges) {
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> child_blocks;
  ChildBlockGatherer(sch, &child_blocks)(stmt);
  for (const auto& edge : edges) {
    if (child_blocks.count(edge->dst)) return true;
  }
  return false;
}

/*! \note It is incomplete IntSet with limited function.
 *        Perhaps we will public it when it is enhanced
 */
class StrideIntSet {
 public:
  StrideIntSet() = default;
  StrideIntSet(Range iter_range, PrimExpr stride) :
      iter_range_(std::move(iter_range)), stride_(std::move(stride)) {}

  void Union(const StrideIntSet& other) {
    if (stride_.defined()) {
      CHECK(Equal(stride_, other.stride_));
      const Range& rhs_range = other.iter_range_;
      PrimExpr begin = min(iter_range_->min, rhs_range->min);
      PrimExpr extents =
          max(iter_range_->extent + iter_range_->min, rhs_range->extent + rhs_range->min) - begin;
      iter_range_ = Range::make_by_min_extent(begin, extents);
    } else {
      stride_ = other.stride_;
      iter_range_ = other.iter_range_;
    }
  }
  Range iter_range_;
  PrimExpr stride_;
};

/*!
 * \brief Find the minimum region to cover the requirement
 * \param vars The vars whose iter range to be detected
 * \param produces The produce region
 * \param requirements The required region
 * \return The iteration information for each var
 */
std::vector<StrideIntSet> SolveCover(const Array<IterVar>& vars,
                                     const std::vector<Range>& produces,
                                     const std::vector<Range>& requirements) {
  std::vector<StrideIntSet> cover_iters(vars.size());
  std::unordered_map<Var, size_t, ObjectHash, ObjectEqual> var_index;
  arith::Analyzer analyzer;

  for (size_t i = 0; i < vars.size(); ++i) {
    var_index[vars[i]->var] = i;
  }

  // fit requirements one by one
  CHECK_EQ(produces.size(), requirements.size());
  for (size_t i = 0; i < produces.size(); ++i) {
    const auto& produce = produces[i];
    const auto& require = requirements[i];

    CHECK(produce->min.as<VarNode>() != nullptr)
      << "The min of produces range must be a single variable";
    Var var = Downcast<Var>(produce->min);

    CHECK_GT(var_index.count(var), 0) << "Find irrelevant variable in produces";
    size_t id = var_index[var];

    const PrimExpr& base = require->min;
    const PrimExpr& produces_len = produce->extent;
    const PrimExpr& extent = analyzer.Simplify((require->extent + produces_len - 1) / produces_len);
    const PrimExpr& strides = produces_len;

    cover_iters[id].Union(StrideIntSet(Range::make_by_min_extent(base, extent), strides));
  }

  return cover_iters;
}

/*!
 * \brief Regenerate loop and the block realize outside the specific block with iter information
 * \param block_sref The sref of the block
 * \param parent_loop_sref The parent loop where the new loop nesting will be inserted
 * \param iter_domain The iteration information
 * \param insert_pos The insert postion
 * \return The Updated parent loop
 */
Stmt RegenerateLoops(const StmtSRef& block_sref, const StmtSRef& parent_loop_sref,
                     const std::vector<StrideIntSet>& iter_domain, size_t insert_pos) {
  // generate for loops
  std::vector<Var> iter_vars(iter_domain.size());
  const auto* block_realize = GetBlockRealize(block_sref).operator->();
  auto node = make_object<BlockRealizeNode>(*block_realize);
  for (size_t i = iter_domain.size(); i > 0; --i) {
    Var iter_var("ax" + std::to_string(i - 1));
    iter_vars[i - 1] = iter_var;
  }
  for (size_t i = iter_domain.size(); i > 0; --i) {
    const auto& domain = iter_domain[i - 1];
    if (!is_one(domain.iter_range_->extent)) {
      node->binding_values.Set(i - 1, domain.iter_range_->min + iter_vars[i - 1] * domain.stride_);
    } else {
      node->binding_values.Set(i - 1, domain.iter_range_->min);
    }
  }

  Stmt body = Stmt(node);
  for (size_t i = iter_domain.size(); i > 0; --i) {
    const auto& domain = iter_domain[i - 1];
    if (!is_one(domain.iter_range_->extent)) {
      // TODO(Siyuan): support for loop with annotations
      const Var& iter_var = iter_vars[i - 1];
      Loop loop = Loop(iter_var,
                       0,
                       domain.iter_range_->extent,
                       Array<Annotation>(),
                       body);
      body = loop;
    }
  }
  Loop loop = Downcast<Loop>(GetRef<Stmt>(parent_loop_sref->node));
  Array<Stmt> stmts = GetChildren(loop);
  stmts.insert(stmts.begin() + insert_pos, body);

  auto n = make_object<LoopNode>(*loop.operator->());
  n->body = SeqStmt(stmts);
  return Loop(n);
}

/*!
 * \brief Gather the required tensor region for consumer consumer_blocks
 * \param produce_regions The output tensor region of producer consumer_blocks
 * \param lca_loop_sref The lca of producer and consumer
 * \param consumer_blocks The consumer consumer_blocks
 * \return Required with the same order as produce_regions
 */
std::vector<Range> GatherRequirements(const Array<TensorRegion>& produce_regions,
                                      const StmtSRef& lca_loop_sref,
                                      const std::vector<StmtSRef>& consumer_blocks) {
  std::vector<std::vector<arith::IntSet>> require_region(produce_regions.size());
  for (size_t i = 0; i < produce_regions.size(); ++i) {
    const auto& tensor_region = produce_regions[i];
    require_region[i] =
        std::vector<arith::IntSet>(tensor_region->region.size(), arith::IntSet::nothing());
  }

  std::unordered_map<Buffer, size_t, ObjectHash, ObjectEqual> buffer_index;
  for (size_t i = 0; i < produce_regions.size(); ++i) {
    buffer_index[produce_regions[i]->buffer] = i;
  }

  for (const auto& block_sref : consumer_blocks) {
    std::vector<TensorRegion> reads;
    RelaxRegion(block_sref, lca_loop_sref, &reads, nullptr);

    for (const auto& tensor_region : reads) {
      auto it = buffer_index.find(tensor_region->buffer);
      // Only consider the tensor regions which are relative with the block to be `compute_at`
      if (it == buffer_index.end()) continue;
      size_t index = it->second;

      for (size_t i = 0; i < tensor_region->region.size(); ++i) {
        const auto& range = tensor_region->region[i];
        require_region[index][i] =
            arith::Union({require_region[index][i], arith::IntSet::range(range)});
      }
    }
  }

  std::vector<Range> ret;
  for (const auto& region : require_region)
    for (const auto& iset : region) {
      ret.push_back(Range::make_by_min_extent(iset.min(), iset.max() - iset.min() + 1));
    }

  return ret;
}

void Schedule::compute_at(const StmtSRef& block_sref, const StmtSRef& loop_sref) {
  // Equivalence
  // The equivalence is based on three conditions:
  // - Complete block: The only producer for each output tensor and all args are data parallel
  // - Same input: Based on dependency analyse
  // - Output region coverage: Easy to fill the loops because of data-parallel args

  // Check
  const auto* block = DowncastPtr<BlockNode>(block_sref->node);
  const auto* loop = DowncastPtr<LoopNode>(loop_sref->node);
  CHECK(block != nullptr) << block_sref << "is not a block sref";
  CHECK(loop != nullptr) << loop_sref << "is not a loop sref";

  // Check the block and the loop are at the same scope
  CHECK_EQ(GetScope(block_sref), GetScope(loop_sref))
    << "Cannot compute_at between different scope";
  const Scope& scope = operator->()->scopes_.at(GetScope(block_sref));

  // Check complete block
  CHECK(scope.IsComplete(block_sref)) << "Can only compute_at a complete block";

  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> child_blocks;
  ChildBlockGatherer(*this, &child_blocks)(GetRef<Stmt>(loop));
  const auto& predecessors = scope.GetPredecessors(block_sref);
  const auto& successors = scope.GetSuccessors(block_sref);

  // Check the block is not a output block
  std::unordered_set<Buffer, ObjectHash, ObjectEqual> seen_buffer;
  const auto& func = operator->()->func;
  for (const auto& x : block->writes) {
    for (const auto& func_buffer : func->buffer_map)
      CHECK(!x->buffer.same_as(func_buffer.second)) << "Can not compute_at an output block";
  }

  for (const auto& predecessor : predecessors) {
    if (predecessor->type == DepType::kRAW) {
      CHECK(scope.IsComplete(predecessor->dst)) << "The producer must a complete block";
    }
  }

  // Check Region Cover
  CHECK(CheckRegion(block_sref)) << "The producer cannot produce necessary tensor region";

  // Check all successors are in the subtree rooted by loop_sref
  for (const auto& x : successors) {
    if (x->type == DepType::kRAW && !child_blocks.count(x->dst)) {
      LOG(FATAL) << "This block cannot compute at this point because some other " <<
                 "blocks outside the scope of this point are also dependent on this block.";
    }
  }

  // Mutation

  // Find insert position
  // After all predecessors in dependency graph and before all successors in dep graph.
  auto children = GetChildren(GetRef<Stmt>(loop));
  size_t after_pos, before_pos;
  for (after_pos = children.size(); after_pos > 0; --after_pos) {
    if (FindAny(*this, children[after_pos - 1], predecessors)) {
      break;
    }
  }
  for (before_pos = 0; before_pos < children.size(); before_pos++) {
    if (FindAny(*this, children[before_pos], successors)) {
      break;
    }
  }
  if (after_pos > before_pos) {
    LOG(FATAL) << "Cannot satisfy dependency";
  }

  // Gather required region
  std::vector<Range> produces;
  for (const auto& tensor_region : block->writes)
    for (const auto& range : tensor_region->region) {
      produces.push_back(range);
    }
  std::vector<StmtSRef> successor_blocks(successors.size());
  for (size_t i = 0; i < successors.size(); ++i) {
    successor_blocks[i] = successors[i]->dst;
  }
  std::vector<Range> requirements = GatherRequirements(block->writes, loop_sref, successor_blocks);

  // Solve the coverage
  const auto& iter_domain = SolveCover(block->iter_vars, produces, requirements);

  // Regenerate the loop nesting
  Stmt new_stmt = RegenerateLoops(block_sref, loop_sref, iter_domain, after_pos);

  // Mutate the AST with Relace and RemoveLeaf API
  this->Replace(loop_sref, new_stmt);
  this->RemoveLeaf(block_sref);
}

}  // namespace tir
}  // namespace tvm
