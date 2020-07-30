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

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include "./schedule_common.h"

namespace tvm {
namespace tir {

/*!
 * \brief Helper function to check if there is any edge points to the given set of blocks
 * \param edges A list of edges to be check
 * \param blocks A list of candidate blocks
 * \return True if there is at least one edge that points to a block in the list
 */
bool AnyEdgePointsToABlock(const Array<DepEdge>& edges, const Array<StmtSRef>& blocks) {
  for (const DepEdge& edge : edges) {
    for (const StmtSRef& block : blocks) {
      if (edge->dst.same_as(block)) {
        return true;
      }
    }
  }
  return false;
}

/*!
 * \brief Helper function to check if every edge points to a block in the given set of blocks
 * \param edges A list of edges to be check
 * \param blocks A list of candidate blocks
 * \param raw_edge_only Only consider RAW-dependency edges
 * \return True if all edges that have a corresponding block
 */
bool EachEdgePointsToABlock(const Array<DepEdge>& edges, const Array<StmtSRef>& blocks,
                            bool raw_edge_only) {
  for (const DepEdge& edge : edges) {
    if (raw_edge_only && edge->type != DepType::kRAW) {
      continue;
    }
    bool found = false;
    for (const StmtSRef& block : blocks) {
      if (edge->dst.same_as(block)) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Extract StmtSRef from DepEdgeNode::dst
 * \param edges List of edges to be extracted
 * \return A list of StmtSRef as the result
 */
std::vector<StmtSRef> EdgesToSRefs(const Array<DepEdge>& edges) {
  std::vector<StmtSRef> result;
  result.reserve(edges.size());
  for (const DepEdge& edge : edges) {
    result.push_back(edge->dst);
  }
  return result;
}

/*! \note We need the stride factor in order to solve the problem of set patterns in a
 *        tensorized block, but only needed the union function. So this class is kept
 *        private to the file instead of being generic in the IntSet
 */
class StrideIntSet {
 public:
  StrideIntSet() = default;
  StrideIntSet(PrimExpr min, PrimExpr extent, PrimExpr stride)
      : min_(std::move(min)), extent_(std::move(extent)), stride_(std::move(stride)) {}

  static StrideIntSet Union(const StrideIntSet& lhs, const StrideIntSet& rhs) {
    StrideIntSet ret;
    if (lhs.stride_.defined()) {
      CHECK(rhs.stride_.defined());
      CHECK(ExprDeepEqual()(lhs.stride_, rhs.stride_));
      ret.min_ = min(lhs.min_, rhs.min_);
      ret.extent_ = max(lhs.extent_ + lhs.min_, rhs.extent_ + rhs.min_) - ret.min_;
      ret.stride_ = lhs.stride_;
    } else {
      ret.min_ = rhs.min_;
      ret.extent_ = rhs.extent_;
      ret.stride_ = rhs.stride_;
    }
    return ret;
  }

  PrimExpr min_;
  PrimExpr extent_;
  PrimExpr stride_;
};

/*!
 * \brief Find the minimum region to cover the requirement
 * \param block The producer block to be solved
 * \param consumes The required region
 * \return The iteration information for each var
 */
std::vector<StrideIntSet> SolveCover(const BlockNode* block, const std::vector<Range>& consumes) {
  const Array<IterVar>& iter_vars = block->iter_vars;
  std::vector<StrideIntSet> iter_domain(iter_vars.size());
  std::unordered_map<const VarNode*, int> iter_var_indexer;
  // Maps IterVar::var back to its index in `iter_vars`
  {
    int iter_var_index = 0;
    for (const IterVar& iter_var : iter_vars) {
      iter_var_indexer[iter_var->var.get()] = iter_var_index++;
    }
  }
  // Collect the ranges written in the producer block
  std::vector<Range> produces;
  for (const TensorRegion& write_region : block->writes) {
    for (const Range& range : write_region->region) {
      produces.push_back(range);
    }
  }
  // Fit requirements one by one
  // i.e. range that the producer writes vs. range that the consumer reads
  CHECK_EQ(produces.size(), consumes.size());
  arith::Analyzer analyzer;
  for (int i = 0; i < static_cast<int>(produces.size()); ++i) {
    const Range& produce = produces[i];
    const Range& consume = consumes[i];
    const VarNode* var = produce->min.as<VarNode>();
    CHECK(var != nullptr)
        << "TypeError: The left bound of the range of the producer block must be Var";
    CHECK_GT(iter_var_indexer.count(var), 0) << "Find irrelevant variable in produces";
    StrideIntSet& iset = iter_domain[iter_var_indexer[var]];
    // It changes the consumers range's stride to `produce->extent`
    PrimExpr strides = produce->extent;
    PrimExpr min = consume->min;
    PrimExpr extent = analyzer.Simplify((consume->extent + strides - 1) / strides);
    iset = StrideIntSet::Union(iset, StrideIntSet(min, extent, strides));
  }
  // Rewrite the un-touched iteration domain to the default value
  for (const IterVar& iter_var : iter_vars) {
    StrideIntSet& domain = iter_domain[iter_var_indexer[iter_var->var.get()]];
    if (!domain.min_.defined() || !domain.extent_.defined() || !domain.stride_.defined()) {
      domain.min_ = iter_var->dom->min;
      domain.extent_ = iter_var->dom->extent;
      domain.stride_ = 1;
    }
  }
  return iter_domain;
}

/*!
 * \brief Regenerate loop and the block realize outside the specific block with iter information
 * \param block_sref The sref of the block
 * \param loop_sref The parent loop where the new loop nesting will be inserted
 * \param iter_domain The iteration information
 * \param insert_pos The insert postion
 * \return The Updated parent loop
 */
Loop RegenerateLoops(const StmtSRef& block_sref, const StmtSRef& loop_sref, int insert_pos,
                     const std::vector<StrideIntSet>& iter_domain) {
  const LoopNode* loop = loop_sref->GetStmt<LoopNode>();
  int n_iter_domain = iter_domain.size();
  // Step 1. Construct loop variables
  std::vector<Var> loop_vars;
  for (int i = 0; i < n_iter_domain; ++i) {
    loop_vars.emplace_back("ax" + std::to_string(i));
  }
  // Step 2. Create a new BlockRealizeNode
  ObjectPtr<BlockRealizeNode> realize =
      make_object<BlockRealizeNode>(*GetBlockRealize(block_sref).get());
  for (int i = 0; i < n_iter_domain; ++i) {
    const StrideIntSet& domain = iter_domain[i];
    // Add binding for each block var
    if (!is_one(domain.extent_)) {
      realize->binding_values.Set(i, domain.min_ + loop_vars[i] * domain.stride_);
    } else {
      realize->binding_values.Set(i, domain.min_);
    }
  }
  // Step 3. Create loops above the BlockRealizeNode
  Stmt body = Stmt(realize);
  for (int i = iter_domain.size(); i > 0; --i) {
    const StrideIntSet& domain = iter_domain[i - 1];
    if (!is_one(domain.extent_)) {
      // TODO(Siyuan): support for loop with annotations
      body = Loop(loop_vars[i - 1], 0, domain.extent_, {}, body);
    }
  }
  // Step 3. Insert the new statement into the children of the loop
  Array<Stmt> stmts = GetChildren(GetRef<Stmt>(loop));
  stmts.insert(stmts.begin() + insert_pos, body);
  // Step 4. Create a new loop with those statements as new children to substitute loop_sref->stmt
  ObjectPtr<LoopNode> n = make_object<LoopNode>(*loop);
  n->body = SeqStmt(stmts);
  return Loop(n);
}

/*!
 * \brief For each buffer written by the producer block, accumulate the rnages on it that are read
 * by the consumer block
 * \param produced_regions The output tensor region of producer consumer_blocks
 * \param lca_loop_sref The lca of producer and consumer
 * \param consumer_blocks The consumer consumer_blocks
 * \return Required with the same order as produce_regions TODO
 */
std::vector<Range> GatherRequirements(const Array<TensorRegion>& produced_regions,
                                      const StmtSRef& lca_loop_sref,
                                      const std::vector<StmtSRef>& consumer_blocks) {
  // For write domain in produce_regions, initiate an empty IntSet for it
  std::vector<std::vector<arith::IntSet>> produced_region_reads;
  for (const TensorRegion& region : produced_regions) {
    produced_region_reads.emplace_back(region->region.size(), arith::IntSet::Nothing());
  }
  // Maps a tensor region's buffer into its index
  std::unordered_map<const BufferNode*, int> buffer_indexer;
  {
    int buffer_index = 0;
    for (const TensorRegion& region : produced_regions) {
      buffer_indexer[region->buffer.get()] = buffer_index++;
    }
  }
  // For each consumer's reading region
  for (const StmtSRef& block_sref : consumer_blocks) {
    std::vector<TensorRegion> reads;
    RelaxRegion(block_sref, lca_loop_sref, &reads, nullptr);
    for (const TensorRegion& region : reads) {
      const BufferNode* buffer = region->buffer.get();
      if (!buffer_indexer.count(buffer)) {
        continue;
      }
      // Find the corresponding buffer
      int buffer_index = buffer_indexer[buffer];
      int range_index = 0;
      // Accumuate the read range into its corresponding buffer
      for (const Range& range : region->region) {
        arith::IntSet& iset = produced_region_reads[buffer_index][range_index];
        iset = arith::Union({iset, arith::IntSet::FromRange(range)});
        ++range_index;
      }
    }
  }
  // Flatten the regions into a list and return
  arith::Analyzer analyzer;
  std::vector<Range> ret;
  for (const std::vector<arith::IntSet>& region_reads : produced_region_reads) {
    for (const arith::IntSet& iset : region_reads) {
      PrimExpr min = iset.min();
      PrimExpr extent = analyzer.Simplify(iset.max() - iset.min() + 1);
      ret.push_back(Range::FromMinExtent(min, extent));
    }
  }
  return ret;
}

class StmtReplacer : public StmtMutator {
 public:
  explicit StmtReplacer(const std::unordered_map<const StmtNode*, const StmtNode*>& repalce_map)
      : replace_map(repalce_map) {}

  Stmt VisitStmt(const Stmt& stmt) override {
    auto it = replace_map.find(stmt.get());
    if (it == replace_map.end()) {
      return StmtMutator::VisitStmt(stmt);
    } else {
      return StmtMutator::VisitStmt(GetRef<Stmt>(it->second));
    }
  }

  const std::unordered_map<const StmtNode*, const StmtNode*>& replace_map;
};

/*!
 * \brief Get the subtree the node is in in its parent's scope
 * \param node The node to query
 * \return StmtSRef indicating the subtree the node is in in its parent's scope
 */
StmtSRef GetSubTreeOfParent(const StmtSRef& node) {
  const StmtSRefNode* child = node.get();
  const StmtSRefNode* parent;
  while (!(parent = child->parent)->stmt->IsInstance<BlockNode>()) {
    child = parent;
  }
  return GetRef<StmtSRef>(child);
}

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
  const auto* block = block_sref->GetStmt<BlockNode>();
  const auto* loop = loop_sref->GetStmt<LoopNode>();
  CHECK(block != nullptr) << "TypeError: 'compute_at' expects 'block' to be a block, but get type: "
                          << block_sref->stmt->GetTypeKey();
  CHECK(loop != nullptr) << "TypeError: 'compute_at' expects 'loop' to be a loop, but get type: "
                         << loop_sref->stmt->GetTypeKey();
  const StmtSRef& parent_block_sref = GetParentBlockSRef(block_sref);
  const BlockNode* parent_block = parent_block_sref->GetStmt<BlockNode>();
  const Scope& scope = scopes.at(parent_block_sref);
  Array<DepEdge> edges_to_pred = scope.GetPredecessors(block_sref);
  Array<DepEdge> edges_to_succ = scope.GetSuccessors(block_sref);
  // Cond 0. `block` and `loop` are in the same scope
  CHECK_EQ(parent_block_sref.get(), GetParentBlockSRef(loop_sref).get())
      << "ValueError: 'compute_at' expects 'block' and 'loop' be in the same block";
  // Cond 1. 'block' is complete/reduction block
  CHECK(scope.IsComplete(block_sref) || scope.IsReduction(block_sref))
      << "ValueError: 'compute_at' expects 'block' to be a complete or reduction block";
  // Cond 3. Check all RAW successors are in the subtree rooted by loop_sref
  CHECK(EachEdgePointsToABlock(edges_to_succ, GetChildBlocks(loop_sref), /*raw_edge_only=*/true))
      << "ValueError: 'compute_at' does not apply to a block that some other "
      << "blocks outside the scope depends on";
  // Cond 4. The subtree has compact data flow
  CHECK(scope.IsCompactDataFlow(GetSubTreeOfParent(block_sref), this))
      << "ValueError: 'compute_at' expects the subtree of 'block' to have compact dataflow";
  // Cond 5. Check the block is not a output block
  for (const TensorRegion& parent_write : parent_block->writes) {
    for (const TensorRegion& write : block->writes) {
      CHECK_NE(write->buffer.get(), parent_write->buffer.get())
          << "ValueError: 'compute_at' does not work on an output block";
    }
  }
  // Mutation
  // Step 1. Find insertion position
  int insert_pos;
  {
    // After all predecessors in dependency graph
    Array<Stmt> loop_body = GetChildren(GetRef<Stmt>(loop));
    int n_stmts = loop_body.size();
    for (insert_pos = n_stmts; insert_pos > 0; --insert_pos) {
      const StmtNode* stmt = loop_body[insert_pos - 1].get();
      if (AnyEdgePointsToABlock(edges_to_pred, GetChildBlocks(stmt2ref.at(stmt)))) {
        break;
      }
    }
    // Before all successors in dep graph.
    int before_pos;
    for (before_pos = 0; before_pos < n_stmts; before_pos++) {
      const StmtNode* stmt = loop_body[before_pos].get();
      if (AnyEdgePointsToABlock(edges_to_pred, GetChildBlocks(stmt2ref.at(stmt)))) {
        break;
      }
    }
    CHECK(insert_pos <= before_pos)
        << "ValueError: 'compute_at' cannot find an insertion point that satisfies dependency";
  }
  // Generate new LoopNode to substitte loop_sref->stmt
  Loop new_loop = RegenerateLoops(
      block_sref, loop_sref, insert_pos,
      SolveCover(block, GatherRequirements(/*produced_regions=*/block->writes,
                                           /*lca_loop_sref=*/loop_sref,
                                           /*consumer_blocks=*/EdgesToSRefs(edges_to_succ))));
  // Remove leaf
  std::pair<Stmt, Stmt> removed = RemoveLeaf(block_sref, this->root);
  std::unordered_map<const StmtNode*, const StmtNode*> replace_map = {
      {removed.first.get(), removed.second.get()},
      {loop_sref->stmt, new_loop.get()},
  };
  // Mutate the AST with Replace
  StmtSRef lca = LowestCommonAncestor({block_sref, loop_sref}, this->root);
  Stmt replaced = StmtReplacer(replace_map)(GetRef<Stmt>(lca->stmt));
  if (const auto* replaced_block = replaced.as<BlockNode>()) {
    this->Replace(lca, replaced, {{GetRef<Block>(replaced_block), GetRef<Block>(parent_block)}});
  } else {
    this->Replace(lca, replaced);
  }
}

}  // namespace tir
}  // namespace tvm
