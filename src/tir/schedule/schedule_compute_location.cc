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
#include "../../arith/pattern_match.h"
#include "./schedule_common.h"

namespace tvm {
namespace tir {

using BufferRegionMap =
    std::unordered_map<Buffer, std::vector<Range>, ObjectPtrHash, ObjectPtrEqual>;

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

/*!
 * \brief Find the minimum region to cover the requirement
 * \param block The producer block to be solved
 * \param consumes The required region
 * \param gather_write Use write region to solve
 * \param buf The buffer in interest
 * \return The iteration information for each var
 */
std::vector<arith::IntSet> SolveCover(const BlockNode* block, const BufferRegionMap& consumes,
                                      bool gather_write) {
  const Array<IterVar>& iter_vars = block->iter_vars;
  // initialize the iter domain
  std::vector<arith::IntSet> iter_domain(iter_vars.size());
  for (size_t i = 0; i < iter_vars.size(); ++i) iter_domain[i] = arith::IntSet::Nothing();
  // Maps IterVar::var back to its index in `iter_vars`
  std::unordered_map<const VarNode*, int> iter_var_indexer;
  int iter_var_index = 0;
  for (const IterVar& iter_var : iter_vars) {
    iter_var_indexer[iter_var->var.get()] = iter_var_index++;
  }
  // Collect the ranges written in the producer block
  BufferRegionMap produces;
  const auto& tensor_regions = gather_write ? block->writes : block->reads;
  for (const TensorRegion& tensor_region : tensor_regions) {
    std::vector<Range> region;
    for (const Range& range : tensor_region->region) {
      region.push_back(range);
    }
    produces[tensor_region->buffer] = std::move(region);
  }
  // Fit requirements one by one
  // i.e. range that the producer writes vs. range that the consumer reads
  arith::Analyzer analyzer;
  for (auto it : produces) {
    auto itt = consumes.find(it.first);
    if (itt != consumes.end()) {
      CHECK_EQ(it.second.size(), itt->second.size());
      for (size_t i = 0; i < it.second.size(); ++i) {
        const Range& produce = it.second[i];
        const Range& consume = itt->second[i];
        PrimExpr min, extent;
        arith::PVar<Var> v;
        arith::PVar<PrimExpr> c;
        if (c.Match(produce->extent) &&
            ((v * c).Match(produce->min) || (c * v).Match(produce->min))) {
          min = div(consume->min, c.Eval());
          extent = analyzer.Simplify((consume->extent + c.Eval() - 1) / c.Eval());
        } else if (is_one(produce->extent) && v.Match(produce->min)) {
          min = consume->min;
          extent = consume->extent;
        } else {
          LOG(FATAL) << "ValueError: TensorRegion pattern match failed";
        }
        const auto* var = v.Eval().get();
        if (iter_var_indexer.count(var)) {
          arith::IntSet& iset = iter_domain[iter_var_indexer[var]];
          iset = arith::Union({iset, arith::IntSet::FromRange(Range::FromMinExtent(min, extent))});
        } else {
          CHECK(analyzer.CanProve(produce->min - consume->min == 0));
          CHECK(analyzer.CanProve(produce->extent - produce->extent == 0));
        }
      }
    }
  }
  // Rewrite the un-touched iteration domain to the default value
  for (const IterVar& iter_var : iter_vars) {
    arith::IntSet& domain = iter_domain[iter_var_indexer[iter_var->var.get()]];
    if (domain.IsNothing()) {
      domain =
          arith::IntSet::FromRange(Range::FromMinExtent(iter_var->dom->min, iter_var->dom->extent));
    }
  }
  return iter_domain;
}

/*!
 * \brief Regenerate loop and the block realize outside the specific block with iter information
 * \param block_sref The sref of the block
 * \param loop_sref The parent loop where the new loop nesting will be inserted
 * \param insert_pos The insert postion
 * \param iter_domain The iteration information
 * \param preserve_trivial_loop Keep the trivial loops whose extent is 1
 * \return The Updated parent loop
 */
Loop RegenerateLoops(const StmtSRef& block_sref, const StmtSRef& loop_sref, int insert_pos,
                     const std::vector<arith::IntSet>& iter_domain, bool preserve_trivial_loop) {
  const auto* loop = loop_sref->GetStmt<LoopNode>();
  int n_iter_domain = iter_domain.size();
  // Step 1. Construct loop variables
  std::vector<Var> loop_vars;
  loop_vars.reserve(n_iter_domain);
  for (int i = 0; i < n_iter_domain; ++i) {
    loop_vars.emplace_back("ax" + std::to_string(i));
  }
  // Step 2. Create a new BlockRealizeNode
  arith::Analyzer analyzer;
  ObjectPtr<BlockRealizeNode> realize =
      make_object<BlockRealizeNode>(*GetBlockRealize(block_sref).get());
  for (int i = 0; i < n_iter_domain; ++i) {
    const arith::IntSet& domain = iter_domain[i];
    // Add binding for each block var
    PrimExpr extent = analyzer.Simplify(domain.max() - domain.min() + 1);
    if (!is_one(extent)) {
      realize->binding_values.Set(i, domain.min() + loop_vars[i]);
    } else {
      realize->binding_values.Set(i, domain.min());
    }
  }
  // Step 3. Create loops above the BlockRealizeNode
  Stmt body = Stmt(realize);
  for (int i = iter_domain.size(); i > 0; --i) {
    const arith::IntSet& domain = iter_domain[i - 1];
    PrimExpr extent = analyzer.Simplify(domain.max() - domain.min() + 1);
    if (preserve_trivial_loop || !is_one(extent)) {
      // TODO(Siyuan): support for loop with annotations
      body = Loop(loop_vars[i - 1], 0, extent, {}, body);
    }
  }
  // Step 3. Insert the new statement into the children of the loop
  Array<Stmt> stmts = GetChildren(GetRef<Stmt>(loop), true);
  stmts.insert(stmts.begin() + insert_pos, body);
  // Step 4. Create a new loop with those statements as new children to substitute loop_sref->stmt
  ObjectPtr<LoopNode> n = make_object<LoopNode>(*loop);
  n->body = SeqStmt(stmts);
  return Loop(n);
}

/*!
 * \brief For each buffer written by the producer block, accumulate the ranges on it that are read
 * by the consumer block
 * \param produced_regions The output tensor region of producer consumer_blocks
 * \param lca_loop_sref The lca of producer and consumer
 * \param consumer_blocks The consumer consumer_blocks
 * \param relax_vars The additional vars should be relaxed according to execution scope
 * \param gather_read If true(false), gather the read(write) region of consumer_blocks
 * \param buf The buffer in interest
 * \return Required with the same order as produce_regions
 */
BufferRegionMap GatherRequirements(const Array<TensorRegion>& produced_regions,
                                   const StmtSRef& lca_loop_sref,
                                   const std::vector<StmtSRef>& consumer_blocks,
                                   const std::unordered_map<const VarNode*, Range>& relax_vars,
                                   bool gather_read) {
  // For write domain in produce_regions, initiate an empty IntSet for it
  std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>
      produced_region_reads;
  for (const TensorRegion& region : produced_regions) {
    std::vector<arith::IntSet> produced_region_read(region->region.size(),
                                                    arith::IntSet::Nothing());
    produced_region_reads[region->buffer] = std::move(produced_region_read);
  }
  // For each consumer's reading region
  for (const StmtSRef& block_sref : consumer_blocks) {
    std::vector<TensorRegion> relaxed;
    if (gather_read) {
      RelaxRegion(block_sref, lca_loop_sref, &relaxed, nullptr, relax_vars);
    } else {
      RelaxRegion(block_sref, lca_loop_sref, nullptr, &relaxed, relax_vars);
    }
    for (const TensorRegion& region : relaxed) {
      if (produced_region_reads.count(region->buffer)) {
        // Accumulate the read range into its corresponding buffer
        for (size_t i = 0; i < region->region.size(); ++i) {
          arith::IntSet& iset = produced_region_reads[region->buffer][i];
          iset = arith::Union({iset, arith::IntSet::FromRange(region->region[i])});
        }
      }
    }
  }
  // Flatten the regions into a list and return
  arith::Analyzer analyzer;
  BufferRegionMap ret;
  for (const auto& it : produced_region_reads) {
    std::vector<Range> ret_buf;
    for (const arith::IntSet& iset : it.second)
      if (!iset.IsNothing()) {
        PrimExpr min = iset.min();
        PrimExpr extent = analyzer.Simplify(iset.max() - iset.min() + 1);
        ret_buf.push_back(Range::FromMinExtent(min, extent));
      }
    if (!ret_buf.empty()) {
      ret[it.first] = std::move(ret_buf);
    }
  }
  return ret;
}

/*!
 * \brief Get the subtree the node is in in its parent's scope
 * \param node The node to query
 * \return StmtSRef indicating the subtree the node is in in its parent's scope
 */
StmtSRef GetSubTreeOfParent(const StmtSRef& node) {
  const StmtSRefNode* child = node.operator->();
  const StmtSRefNode* parent;
  while (!(parent = child->parent)->stmt->IsInstance<BlockNode>()) {
    child = parent;
  }
  return GetRef<StmtSRef>(child);
}

std::unordered_map<const VarNode*, Range> RelaxForExecScope(const StmtSRef& loop_sref,
                                                            const StmtSRef& block_sref) {
  std::unordered_map<const VarNode*, Range> relax_var;
  const auto* block = block_sref->GetStmt<BlockNode>();
  const BlockRealize& realize = GetBlockRealize(block_sref);
  const String& exe_scope = realize->exec_scope;
  StmtSRef sref = loop_sref;

  auto update_for_gpu = [&block, &exe_scope](const LoopNode* loop) -> bool {
    CHECK_EQ(block->writes.size(), 1)
        << "InternalError: Only block with one write buffer can be compute_at";
    std::string write_scope = block->writes[0]->buffer->scope;

    std::string thread_tag;
    for (const auto& annotation : loop->annotations)
      if (annotation->attr_key == attr::loop_type) {
        thread_tag = Downcast<StringImm>(annotation->value)->value;
      }
    if (exe_scope == "gpu_thread" || exe_scope.empty()) {
      if (write_scope == "shared" &&
          (thread_tag.substr(0, 9) == "threadIdx" || thread_tag == "vthread")) {
        return true;
      } else if ((write_scope == "global" || write_scope.empty()) &&
                 (thread_tag.substr(0, 9) == "threadIdx" ||
                  thread_tag.substr(0, 9) == "blockIdx")) {
        return true;
      }
    }
    return false;
  };

  while (sref.defined()) {
    if (const auto* loop = sref->GetStmt<LoopNode>()) {
      if (update_for_gpu(loop)) {
        relax_var[loop->loop_var.get()] = Range::FromMinExtent(loop->min, loop->extent);
      }
    }
    sref = GetRef<StmtSRef>(sref->parent);
  }

  return relax_var;
}

void ScheduleNode::compute_at(const StmtSRef& block_sref, const StmtSRef& loop_sref,
                              bool preserve_trivial_loop) {
  /*!
   * Check:
   *   - check input_block is complete/is a dominant reduction block
   *   - check dependency: all input_block's RAW successors are under input_loop
   *   - check all blocks in the same sub tree are complete
   *   - check block is not an output block
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
  const auto* parent_block = parent_block_sref->GetStmt<BlockNode>();
  const Scope& scope = scopes.at(parent_block_sref);
  Array<DepEdge> edges_to_pred = scope->GetPredecessors(block_sref);
  Array<DepEdge> edges_to_succ = scope->GetSuccessors(block_sref);
  // Cond 0. `block` and `loop` are in the same scope
  CHECK_EQ(parent_block_sref.get(), GetParentBlockSRef(loop_sref).get())
      << "ValueError: 'compute_at' expects 'block' and 'loop' be in the same block";
  // Cond 1. 'block' is complete/reduction block
  CHECK(scope->IsComplete(block_sref) || scope->IsReduction(block_sref))
      << "ValueError: 'compute_at' expects 'block' to be a complete or reduction block";
  // Cond 2. Check all RAW successors are in the subtree rooted by loop_sref
  CHECK(EachEdgePointsToABlock(edges_to_succ, GetChildBlocks(loop_sref), /*raw_edge_only=*/true))
      << "ValueError: 'compute_at' does not apply to a block that some other "
      << "blocks outside the scope depends on";
  // Cond 3. The subtree has compact data flow
  CHECK(scope->IsCompactDataFlow(GetSubTreeOfParent(block_sref), this))
      << "ValueError: 'compute_at' expects the subtree of 'block' to have compact dataflow";
  // Cond 4. Check the block is not a output block
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
      if (AnyEdgePointsToABlock(edges_to_succ, GetChildBlocks(stmt2ref.at(stmt)))) {
        break;
      }
    }
    CHECK(insert_pos <= before_pos)
        << "ValueError: 'compute_at' cannot find an insertion point that satisfies dependency";
  }
  // Generate new LoopNode to substitute loop_sref->stmt
  Loop new_loop = RegenerateLoops(
      block_sref, loop_sref, insert_pos,
      SolveCover(block,
                 GatherRequirements(/*produced_regions=*/block->writes,
                                    /*lca_loop_sref=*/loop_sref,
                                    /*consumer_blocks=*/EdgesToSRefs(edges_to_succ),
                                    /*relax_vars=*/RelaxForExecScope(loop_sref, block_sref),
                                    /*gather_read=*/true),
                 true),
      preserve_trivial_loop);
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

void ScheduleNode::reverse_compute_at(const StmtSRef& block_sref, const StmtSRef& loop_sref,
                                      bool preserve_trivial_loop) {
  /*!
   * Check:
   *   - check input_block is complete/is a dominant reduction block
   *   - check all input_block's RAW predecessors are under input_loop
   *   - check all blocks in the same sub tree are complete
   *   - check all input_block's RAW predecessors are complete/dominant reduction block
   *
   * Mutate:
   *   - generate loops that iterate the whole instance space under
   *     input_loop after all the predecessors
   */
  const auto* block = block_sref->GetStmt<BlockNode>();
  const auto* loop = loop_sref->GetStmt<LoopNode>();
  CHECK(block != nullptr)
      << "TypeError: 'reverse_compute_at' expects 'block' to be a block, but get type: "
      << block_sref->stmt->GetTypeKey();
  CHECK(loop != nullptr)
      << "TypeError: 'reverse_compute_at' expects 'loop' to be a loop, but get type: "
      << loop_sref->stmt->GetTypeKey();
  const StmtSRef& parent_block_sref = GetParentBlockSRef(block_sref);
  const auto* parent_block = parent_block_sref->GetStmt<BlockNode>();
  const Scope& scope = scopes.at(parent_block_sref);
  Array<DepEdge> edges_to_pred = scope->GetPredecessors(block_sref);
  Array<DepEdge> edges_to_succ = scope->GetSuccessors(block_sref);
  // Cond 0. `block` and `loop` are in the same scope
  CHECK_EQ(parent_block_sref.get(), GetParentBlockSRef(loop_sref).get())
      << "ValueError: 'reverse_compute_at' expects 'block' and 'loop' be in the same block";
  // Cond 1. 'block' is complete/reduction block
  CHECK(scope->IsComplete(block_sref) || scope->IsReduction(block_sref))
      << "ValueError: 'reverse_compute_at' expects 'block' to be a complete or reduction block";
  // Cond 2. Check all RAW predecessors are in the subtree rooted by loop_sref
  CHECK(EachEdgePointsToABlock(edges_to_pred, GetChildBlocks(loop_sref), /*raw_edge_only=*/true))
      << "ValueError: 'reverse_compute_at' does not apply to a block that some other "
      << "blocks outside the scope depends on";
  // Cond 3. The subtree has compact data flow
  CHECK(scope->IsCompactDataFlow(GetSubTreeOfParent(block_sref), this))
      << "ValueError: 'reverse_compute_at' expects the subtree of 'block' to have compact dataflow";
  // Cond 4. Check there is only one RAW predecessor
  CHECK_EQ(edges_to_pred.size(), 1)
      << "ValueError: 'reverse_compute_at' expects only one producer of current block";
  // Cond 5. Check the RAW predecessor is complete/reduction block
  CHECK(scope->IsComplete(edges_to_pred[0]->dst) || scope->IsReduction(edges_to_pred[0]->dst))
      << "ValueError: 'reverse_compute_at' expects producers of 'block' to be a complete or "
         "reduction block";
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
      if (AnyEdgePointsToABlock(edges_to_succ, GetChildBlocks(stmt2ref.at(stmt)))) {
        break;
      }
    }
    CHECK(insert_pos <= before_pos) << "ValueError: 'reverse_compute_at' cannot find an insertion "
                                       "point that satisfies dependency";
  }
  // Generate new LoopNode to substitute loop_sref->stmt
  Loop new_loop =
      RegenerateLoops(block_sref, loop_sref, insert_pos,
                      SolveCover(block,
                                 GatherRequirements(/*produced_regions=*/block->reads,
                                                    /*lca_loop_sref=*/loop_sref,
                                                    /*consumer_blocks=*/EdgesToSRefs(edges_to_pred),
                                                    /*relax_vars=*/{},
                                                    /*gather_read=*/false),
                                 false),
                      preserve_trivial_loop);
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
