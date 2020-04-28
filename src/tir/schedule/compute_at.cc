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
#include <tvm/tir/analysis.h>
#include <tvm/arith/int_set.h>
#include <tvm/arith/analyzer.h>
#include "schedule_common.h"

namespace tvm {
namespace tir {

/*! To find if there is any dependent block under in the specific subtree */
bool FindAny(const ScheduleNode* sch, const Stmt& stmt, const Array<DepEdge>& edges) {
  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> child_blocks;
  ChildBlockGatherer(sch, &child_blocks)(stmt);
  for (const auto& edge : edges) {
    if (child_blocks.count(edge->dst)) return true;
  }
  return false;
}

/*! \note We need the stride factor in order to solve the problem of set patterns in a
 *        tensorized block, but only needed the union function. So this class is kept
 *        private to the file instead of being generic in the IntSet
 */
class StrideIntSet {
 public:
  StrideIntSet() = default;
  StrideIntSet(Range iter_range, PrimExpr stride) :
      iter_range_(std::move(iter_range)), stride_(std::move(stride)) {}

  static StrideIntSet Union(const StrideIntSet& lhs, const StrideIntSet& rhs) {
    StrideIntSet ret;
    if (lhs.stride_.defined()) {
      CHECK(rhs.stride_.defined());
      CHECK(ExprDeepEqual()(lhs.stride_, rhs.stride_));
      const Range& rhs_range = rhs.iter_range_;
      PrimExpr begin = min(lhs.iter_range_->min, rhs_range->min);
      PrimExpr extents =
          max(lhs.iter_range_->extent + lhs.iter_range_->min, rhs_range->extent + rhs_range->min)
              - begin;
      ret.iter_range_ = Range::make_by_min_extent(begin, extents);
      ret.stride_ = lhs.stride_;
    } else {
      ret.stride_ = rhs.stride_;
      ret.iter_range_ = rhs.iter_range_;
    }
    return ret;
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

    cover_iters[id] = StrideIntSet::Union(cover_iters[id],
        StrideIntSet(Range::make_by_min_extent(base, extent), strides));
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

// region cover check
bool ScheduleNode::CheckRegionCover(const StmtSRef& consumer) const {
  if (consumer->parent == nullptr) return true;
  const auto* block = DowncastPtr<BlockNode>(consumer->node);
  StmtSRef scope_sref = GetScope(consumer);
  const Scope& scope = scopes_.at(scope_sref);

  // Gather all the producers
  std::unordered_map<const VarNode*, std::vector<StmtSRef>> producers;
  std::unordered_map<const VarNode*, std::vector<const TensorRegionNode*>> produce_regions;
  const auto& successors = scope.GetSuccessors(consumer);

  for (const auto& edge : successors) {
    if (edge->type == DepType::kRAW) {
      const auto* producer_block = DowncastPtr<BlockNode>(edge->dst->node);
      for (const auto& output_region : producer_block->writes) {
        const auto* bufferVar = output_region->buffer->data.operator->();
        producers[bufferVar].push_back(edge->dst);
        produce_regions[bufferVar].push_back(output_region.operator->());
      }
    }
  }

  for (const auto& input_region : block->reads) {
    const auto* bufferVar = input_region->buffer->data.operator->();
    std::vector<StmtSRef>& nodes = producers[bufferVar];
    if (nodes.empty()) continue;
    std::vector<const TensorRegionNode*> regions = produce_regions[bufferVar];
    // calculate the LCA
    nodes.push_back(consumer);
    const StmtSRef& lca = LowestCommonAncestor(nodes, scope_sref);
    nodes.pop_back();
    // prepare check function
    auto check_cover = [](const TensorRegion& read, const TensorRegion& write) -> bool {
      CHECK_EQ(read->region.size(), write->region.size());
      for (size_t i = 0; i < read->region.size(); ++i) {
        auto read_min = read->region[i]->min;
        auto write_min = write->region[i]->min;
        auto read_max = read_min + read->region[i]->extent;
        auto write_max = write_min + write->region[i]->extent;
        arith::Analyzer analyzer;
        if (!analyzer.CanProve(read_min >= write_min)
            || !analyzer.CanProve(read_max <= write_max)) {
          LOG(WARNING) << "Cannot prove the region cover: producer " << read << " consumer "
                       << write;
          return false;
        }
      }
      return true;
    };
    TensorRegion read = RelaxRegion(consumer, lca, input_region);
    for (size_t i = 0; i < nodes.size(); ++i) {
      TensorRegion write = RelaxRegion(nodes[i], lca, GetRef<TensorRegion>(regions[i]));
      if (!check_cover) return false;
    }
  }

  return true;
}

class StmtReplacer : public StmtMutator {
 public:
  explicit StmtReplacer(const std::unordered_map<Stmt, Stmt, ObjectHash, ObjectEqual>& repalce_map)
      : repalce_map_(repalce_map) {}

  Stmt VisitStmt(const Stmt& stmt) override {
    auto it = repalce_map_.find(stmt);
    if (it == repalce_map_.end()) {
      return StmtMutator::VisitStmt(stmt);
    } else {
      return StmtMutator::VisitStmt(it->second);
    }
  }

 private:
  const std::unordered_map<Stmt, Stmt, ObjectHash, ObjectEqual>& repalce_map_;
};

void ScheduleNode::compute_at(const StmtSRef& block_sref, const StmtSRef& loop_sref) {
  /*!
   * Check:
   *   - check input_block is complete/is a dominant reduction block
   *   - check input_block's RAW predecessors are complete
   *   - check dependency: all input_block's RAW successors are under input_loop
   *   - check one-way fine-grained data flow: all blocks in the same sub tree are complete
   *   - check block is not a output block
   *
   * Mutate:
   *   - generate loops that iterate the whole instance space under
   *     input_loop before all the successors
   *
   * Proof:
   *   - i + ii => input_block only has RAW successors
   *   - i => No other block will write the output of input_block
   *   - ii => No other block will write the input of input_block
   *   - ii + iii + iv + dominance property => input_block will read the same input as before.
   *   - i + iii + iv + v + dominance property => consumers of input_block will
   *     read the same input as before.
   */

  // Check
  const auto* block = DowncastPtr<BlockNode>(block_sref->node);
  const auto* loop = DowncastPtr<LoopNode>(loop_sref->node);
  CHECK(block != nullptr) << block_sref << "is not a block sref";
  CHECK(loop != nullptr) << loop_sref << "is not a loop sref";

  // Check the block and the loop are at the same scope
  CHECK_EQ(GetScope(block_sref), GetScope(loop_sref))
    << "Cannot compute_at between different scope";
  const StmtSRef& scope_sref = GetScope(block_sref);
  const Scope& scope = scopes_.at(scope_sref);
  const auto* scope_block = DowncastPtr<BlockNode>(scope_sref->node);

  // Check complete block
  CHECK(scope.IsComplete(block_sref)) << "Can only compute_at a complete block";

  // Check compact data flow
  StmtSRef sub_tree_root = block_sref;
  while (sub_tree_root.defined()) {
    auto node = GetRef<StmtSRef>(sub_tree_root->parent);
    if (GetRef<Stmt>(node->node).as<BlockNode>()) {
      break;
    } else {
      sub_tree_root = node;
    }
  }
  CHECK(IsCompactDataFlow(sub_tree_root))
    << "Can only compute_at a block from the subtree which is compact data flow";

  std::unordered_set<StmtSRef, ObjectHash, ObjectEqual> child_blocks;
  ChildBlockGatherer(this, &child_blocks)(GetRef<Stmt>(loop));
  const auto& predecessors = scope.GetPredecessors(block_sref);
  const auto& successors = scope.GetSuccessors(block_sref);

  // Check the block is not a output block
  std::unordered_set<Buffer, ObjectHash, ObjectEqual> seen_buffer;
  for (const auto& x : block->writes) {
    for (const auto& output_buffer : scope_block->writes)
      CHECK(!x->buffer.same_as(output_buffer->buffer)) << "Can not compute_at an output block";
  }

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
    if (FindAny(this, children[after_pos - 1], predecessors)) {
      break;
    }
  }
  for (before_pos = 0; before_pos < children.size(); before_pos++) {
    if (FindAny(this, children[before_pos], successors)) {
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
  // Remove leaf
  auto removed = RemoveLeaf(block_sref, root);

  StmtSRef lca = LowestCommonAncestor({block_sref, loop_sref}, root);
  std::unordered_map<Stmt, Stmt, ObjectHash, ObjectEqual> replace_map;
  replace_map[GetRef<Stmt>(loop_sref->node)] = new_stmt;
  replace_map[removed.first] = removed.second;

  // Mutate the AST with Replace
  Stmt replaced_stmt = StmtReplacer(replace_map)(GetRef<Stmt>(lca->node));
  if (replaced_stmt.as<BlockNode>()) {
    Map<Block, Block> block_map;
    auto block_scope = Downcast<Block>(GetRef<Stmt>(scope_sref->node));
    block_map.Set(Downcast<Block>(replaced_stmt), block_scope);
    this->Replace(lca, replaced_stmt, block_map);
  } else {
    this->Replace(lca, replaced_stmt);
  }
}

}  // namespace tir
}  // namespace tvm
