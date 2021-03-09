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
#include "../../../arith/pattern_match.h"
#include "../analysis.h"
#include "../utils.h"
#include "./primitives.h"

namespace tvm {
namespace tir {
namespace schedule {

using BufferRegionMap =
    std::unordered_map<Buffer, std::vector<Range>, ObjectPtrHash, ObjectPtrEqual>;

Array<Var> GatherVars(const ObjectRef& stmt_or_expr) {
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> result;
  PreOrderVisit(stmt_or_expr, [&result](const ObjectRef& node) -> bool {
    if (const auto* var = node.as<VarNode>()) {
      result.insert(GetRef<Var>(var));
    }
    return true;
  });
  return std::vector<Var>(result.begin(), result.end());
}

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
    if (raw_edge_only && edge->type != DepKind::kRAW) {
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
  const auto& buffer_regions = gather_write ? block->writes : block->reads;
  for (const BufferRegion& buffer_region : buffer_regions) {
    std::vector<Range> region;
    for (const Range& range : buffer_region->region) {
      region.push_back(range);
    }
    produces[buffer_region->buffer] = std::move(region);
  }
  // Fit requirements one by one
  // i.e. range that the producer writes vs. range that the consumer reads
  arith::Analyzer analyzer;
  for (auto it : produces) {
    auto itt = consumes.find(it.first);
    if (itt != consumes.end()) {
      ICHECK_EQ(it.second.size(), itt->second.size());
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
          LOG(FATAL) << "ValueError: BufferRegion pattern match failed";
        }
        const auto* var = v.Eval().get();
        if (iter_var_indexer.count(var)) {
          arith::IntSet& iset = iter_domain[iter_var_indexer[var]];
          iset = arith::Union({iset, arith::IntSet::FromRange(Range::FromMinExtent(min, extent))});
        } else {
          ICHECK(analyzer.CanProve(produce->min - consume->min == 0));
          ICHECK(analyzer.CanProve(produce->extent - produce->extent == 0));
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
For RegenerateLoops(const StmtSRef& block_sref, const StmtSRef& loop_sref, int insert_pos,
                    const std::vector<arith::IntSet>& iter_domain, bool preserve_trivial_loop) {
  const auto* loop = loop_sref->GetStmt<ForNode>();
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
      body = For(loop_vars[i - 1], 0, extent, {}, body);
    }
  }
  // Step 3. Insert the new statement into the children of the loop
  Array<Stmt> stmts = GetChildren(GetRef<Stmt>(loop), true);
  stmts.insert(stmts.begin() + insert_pos, body);
  // Step 4. Create a new loop with those statements as new children to substitute loop_sref->stmt
  ObjectPtr<ForNode> n = make_object<ForNode>(*loop);
  n->body = SeqStmt(stmts);
  return For(n);
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
BufferRegionMap GatherRequirements(const Array<BufferRegion>& produced_regions,
                                   const StmtSRef& lca_loop_sref,
                                   const std::vector<StmtSRef>& consumer_blocks,
                                   const std::unordered_map<const VarNode*, Range>& relax_vars,
                                   bool gather_read) {
  // For write domain in produce_regions, initiate an empty IntSet for it
  std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>
      produced_region_reads;
  for (const BufferRegion& region : produced_regions) {
    std::vector<arith::IntSet> produced_region_read(region->region.size(),
                                                    arith::IntSet::Nothing());
    produced_region_reads[region->buffer] = std::move(produced_region_read);
  }
  // For each consumer's reading region
  for (const StmtSRef& block_sref : consumer_blocks) {
    std::vector<BufferRegion> relaxed;
    if (gather_read) {
      RelaxRegion(block_sref, lca_loop_sref, &relaxed, nullptr, relax_vars);
    } else {
      RelaxRegion(block_sref, lca_loop_sref, nullptr, &relaxed, relax_vars);
    }
    for (const BufferRegion& region : relaxed) {
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
  const String& exec_scope = block->exec_scope;
  StmtSRef sref = loop_sref;

  auto update_for_gpu = [&block, &exec_scope](const ForNode* loop) -> bool {
    CHECK_EQ(block->writes.size(), 1)
        << "ValueError: Only block with one write buffer can be compute_at";
    std::string write_scope = block->writes[0]->buffer->scope;

    std::string thread_tag =
        loop->thread_binding.defined() ? loop->thread_binding.value()->thread_tag : "";
    if (exec_scope == "gpu_thread" || exec_scope.empty()) {
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
    if (const auto* loop = sref->GetStmt<ForNode>()) {
      if (update_for_gpu(loop)) {
        relax_var[loop->loop_var.get()] = Range::FromMinExtent(loop->min, loop->extent);
      }
    }
    sref = GetRef<StmtSRef>(sref->parent);
  }

  return relax_var;
}

bool CheckCompactDataFlow(const ScheduleState& self, const BlockScope& scope,
                          const StmtSRef& block_sref) {
  StmtSRef subtree_sref = GetSubTreeOfParent(block_sref);
  Array<StmtSRef> children = GetChildBlocks(self, subtree_sref, true);
  return scope->IsCompactDataFlow(subtree_sref, children);
}

class StatementInliner : public StmtExprMutator {
 public:
  explicit StatementInliner(const BlockNode* block, Map<Block, Block>* block_sref_map,
                            const std::unordered_map<const StmtNode*, const StmtNode*>& replace_map)
      : block_(block), block_sref_map_(block_sref_map), replace_map_(replace_map) {
    const auto store = block_->body.as<BufferStoreNode>();
    value_ = store->value;
    ICHECK_EQ(block_->writes.size(), 1);
    for (const auto& index : store->indices) {
      const auto* variable = index.as<VarNode>();
      CHECK(variable) << "Only support inline direct access block";
      Var var = GetRef<Var>(variable);
      vars_.push_back(var);
    }
    Array<Var> value_vars = GatherVars(value_);
    for (const auto& x : value_vars) {
      ICHECK(std::find_if(vars_.begin(), vars_.end(),
                          [=](const Var& var) -> bool { return var.same_as(x); }) != vars_.end())
          << "Not All variable in value can be replaced by index vars";
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    // Find the original block before leaf removing
    bool is_scope_block = is_scope_block_;
    is_scope_block_ = false;
    const StmtNode* node = op;
    for (const auto& pair : replace_map_) {
      if (pair.second == op) {
        node = pair.first;
        break;
      }
    }
    Block origin_block = Downcast<Block>(GetRef<Stmt>(node));
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);

    // Update Allocation
    const Buffer& buffer = block_->writes[0]->buffer;
    Array<Buffer> alloc_buffers;
    for (const auto alloc_buf : op->alloc_buffers) {
      if (alloc_buf != buffer) alloc_buffers.push_back(alloc_buf);
    }

    Array<BufferRegion> reads(nullptr);
    if (is_scope_block) {
      reads = op->reads;
    } else {
      // Update read region only for none-scope block
      BlockReadWriteCollector block_read_write_collector(alloc_buffers);
      block_read_write_collector(op->body);
      reads = block_read_write_collector.reads();
    }

    auto block_node = make_object<BlockNode>(*op);
    block_node->reads = reads;
    block_node->alloc_buffers = alloc_buffers;
    Block block(block_node);

    block_sref_map_->Set(origin_block, block);
    return std::move(block);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    const Buffer& buffer = block_->writes[0]->buffer;
    if (buffer.same_as(op->buffer)) {
      std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> v_map;
      for (size_t i = 0; i < op->indices.size(); ++i) {
        v_map[vars_[i]] = op->indices[i];
      }
      return Substitute(value_, v_map);
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  /*! The block realize of the block to be inlined*/
  const BlockNode* block_;
  /*! The block vars of the block to be inlined*/
  Array<Var> vars_;
  /*! The buffer store value*/
  PrimExpr value_;
  /*! The block sref map using in Replace */
  Map<Block, Block>* block_sref_map_;
  const std::unordered_map<const StmtNode*, const StmtNode*>& replace_map_;
  /*! Whether this block is the scope block (first visited block)*/
  bool is_scope_block_ = true;
};

class ReverseStatementInliner : public StmtExprMutator {
 public:
  explicit ReverseStatementInliner(const BlockNode* block, const BlockNode* producer,
                                   Map<Block, Block>* block_sref_map)
      : block_(block), producer_(producer), block_sref_map_(block_sref_map) {
    // Check BufferStore of producer is like Buffer[v0, v1, ...]
    const auto* store = producer_->body.as<BufferStoreNode>();
    value_ = store->value;
    ICHECK_EQ(producer_->writes.size(), 1);
    for (const auto& index : store->indices) {
      const auto* variable = index.as<VarNode>();
      CHECK(variable)
          << "ValueError: 'reverse_compute_inline' only supports inline direct access block";
      Var var = GetRef<Var>(variable);
      new_vars_.push_back(var);
      old_vars_.push_back(NullValue<Var>());
    }
    Array<Var> value_vars = GatherVars(store->value);
    for (const auto& x : value_vars) {
      ICHECK(std::find_if(new_vars_.begin(), new_vars_.end(),
                          [=](const Var& var) -> bool { return var.same_as(x); }) !=
             new_vars_.end())
          << "ValueError: Not all variable in value can be replaced by index vars";
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    bool is_producer = op == producer_;
    Block origin_producer = Downcast<Block>(GetRef<Stmt>(producer_));
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockNode>();
    ICHECK(op != nullptr);
    // update allocation
    const Buffer& buffer = producer_->writes[0]->buffer;
    Array<Buffer> alloc_buffers;
    for (const auto alloc_buf : op->alloc_buffers) {
      if (alloc_buf != buffer) alloc_buffers.push_back(alloc_buf);
    }
    // update read/write region
    BlockReadWriteCollector block_read_write_collector(alloc_buffers);
    block_read_write_collector(op->body);

    auto block_node = make_object<BlockNode>(*op);
    block_node->reads = block_read_write_collector.reads();
    block_node->writes = block_read_write_collector.writes();
    block_node->alloc_buffers = alloc_buffers;
    Block block(block_node);

    if (is_producer) block_sref_map_->Set(origin_producer, block);
    return std::move(Block(block));
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    const Buffer& buffer = producer_->writes[0]->buffer;
    if (buffer.same_as(op->buffer)) {
      // find the BufferStore of producer, now check the BufferLoad inside the old store
      const auto* old_store = block_->body.as<BufferStoreNode>();
      PrimExpr value = VisitExpr(old_store->value);
      // check BufferStore of block is substitutable
      auto v_map = [&](const Var& var) -> Optional<PrimExpr> {
        for (size_t i = 0; i < old_vars_.size(); ++i) {
          if (old_vars_[i].same_as(var) || new_vars_[i].same_as(var)) return new_vars_[i];
        }
        LOG(FATAL) << "ValueError: indices not match";
        return NullOpt;
      };
      std::vector<PrimExpr> new_indices;
      for (const auto& index : old_store->indices) new_indices.push_back(Substitute(index, v_map));
      return BufferStore(old_store->buffer, Substitute(value, v_map), new_indices);
    }
    return GetRef<Stmt>(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    const Buffer& buffer = producer_->writes[0]->buffer;
    if (buffer.same_as(op->buffer)) {
      for (size_t i = 0; i < op->indices.size(); ++i) {
        const auto* var = op->indices[i].as<VarNode>();
        ICHECK(var) << "ValueError: indices not match";
        if (!old_vars_[i].defined()) {
          old_vars_[i] = GetRef<Var>(var);
        } else {
          ICHECK(old_vars_[i].same_as(GetRef<Var>(var))) << "ValueError: indices not match";
        }
      }
      return value_;
    }
    return GetRef<PrimExpr>(op);
  }

 private:
  /*! The block to be reverse inlined*/
  const BlockNode* block_;
  /*! The producer of the block to be reverse inlined*/
  const BlockNode* producer_;
  /*! The block vars in block_*/
  std::vector<Var> old_vars_;
  /*! The block vars in producer_*/
  std::vector<Var> new_vars_;
  /*! The buffer store value*/
  PrimExpr value_;
  /*! The block sref map using in Replace */
  Map<Block, Block>* block_sref_map_;
};

void ComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
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
  const auto* loop = loop_sref->GetStmt<ForNode>();
  CHECK(block != nullptr) << "TypeError: 'compute_at' expects 'block' to be a block, but get type: "
                          << block_sref->stmt->GetTypeKey();
  CHECK(loop != nullptr) << "TypeError: 'compute_at' expects 'loop' to be a loop, but get type: "
                         << loop_sref->stmt->GetTypeKey();
  const StmtSRef& parent_block_sref = GetScopeRoot(block_sref);
  const auto* parent_block = parent_block_sref->GetStmt<BlockNode>();
  const BlockScope& scope = self->scopes.at(parent_block_sref);
  Array<DepEdge> edges_to_pred = scope->GetPredecessors(block_sref);
  Array<DepEdge> edges_to_succ = scope->GetSuccessors(block_sref);
  // Cond 0. `block` and `loop` are in the same scope
  CHECK_EQ(parent_block_sref.get(), GetScopeRoot(loop_sref).get())
      << "ValueError: 'compute_at' expects 'block' and 'loop' be in the same block";
  // Cond 1. 'block' is complete/reduction block
  CHECK(scope->IsComplete(block_sref) || scope->IsReduction(block_sref))
      << "ValueError: 'compute_at' expects 'block' to be a complete or reduction block";
  // Cond 2. Check all RAW successors are in the subtree rooted by loop_sref
  CHECK(EachEdgePointsToABlock(edges_to_succ, GetChildBlocks(self, loop_sref, true),
                               /*raw_edge_only=*/true))
      << "ValueError: 'compute_at' does not apply to a block that some other "
      << "blocks outside the scope depends on";
  // Cond 3. The subtree has compact data flow
  CHECK(CheckCompactDataFlow(self, scope, block_sref))
      << "ValueError: 'compute_at' expects the subtree of 'block' to have compact dataflow";
  // Cond 4. Check the block is not a output block
  for (const BufferRegion& parent_write : parent_block->writes) {
    for (const BufferRegion& write : block->writes) {
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
      if (AnyEdgePointsToABlock(edges_to_pred,
                                GetChildBlocks(self, self->stmt2ref.at(stmt), true))) {
        break;
      }
    }
    // Before all successors in dep graph.
    int before_pos;
    for (before_pos = 0; before_pos < n_stmts; before_pos++) {
      const StmtNode* stmt = loop_body[before_pos].get();
      if (AnyEdgePointsToABlock(edges_to_succ,
                                GetChildBlocks(self, self->stmt2ref.at(stmt), true))) {
        break;
      }
    }
    CHECK(insert_pos <= before_pos)
        << "ValueError: 'compute_at' cannot find an insertion point that satisfies dependency";
  }
  // Generate new LoopNode to substitute loop_sref->stmt
  For new_loop = RegenerateLoops(
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
  StmtSRef root = GetSRefTreeRoot(block_sref);
  std::pair<Stmt, Stmt> removed = RemoveLeaf(block_sref, root);
  std::unordered_map<const StmtNode*, const StmtNode*> replace_map = {
      {removed.first.get(), removed.second.get()},
      {loop_sref->stmt, new_loop.get()},
  };
  // Mutate the AST with Replace
  StmtSRef lca = LowestCommonAncestor({block_sref, loop_sref}, root);
  Stmt replaced = StmtReplacer(replace_map)(GetRef<Stmt>(lca->stmt));
  if (const auto* replaced_block = replaced.as<BlockNode>()) {
    self->Replace(lca, replaced, {{GetRef<Block>(parent_block), GetRef<Block>(replaced_block)}});
  } else {
    self->Replace(lca, replaced, {});
  }
}

void ReverseComputeAt(ScheduleState self, const StmtSRef& block_sref, const StmtSRef& loop_sref,
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
  const auto* loop = loop_sref->GetStmt<ForNode>();
  CHECK(block != nullptr)
      << "TypeError: 'reverse_compute_at' expects 'block' to be a block, but get type: "
      << block_sref->stmt->GetTypeKey();
  CHECK(loop != nullptr)
      << "TypeError: 'reverse_compute_at' expects 'loop' to be a loop, but get type: "
      << loop_sref->stmt->GetTypeKey();
  const StmtSRef& parent_block_sref = GetScopeRoot(block_sref);
  const auto* parent_block = parent_block_sref->GetStmt<BlockNode>();
  const BlockScope& scope = self->scopes.at(parent_block_sref);
  Array<DepEdge> edges_to_pred = scope->GetPredecessors(block_sref);
  Array<DepEdge> edges_to_succ = scope->GetSuccessors(block_sref);
  // Cond 0. `block` and `loop` are in the same scope
  CHECK_EQ(parent_block_sref.get(), GetScopeRoot(loop_sref).get())
      << "ValueError: 'reverse_compute_at' expects 'block' and 'loop' be in the same block";
  // Cond 1. 'block' is complete/reduction block
  CHECK(scope->IsComplete(block_sref) || scope->IsReduction(block_sref))
      << "ValueError: 'reverse_compute_at' expects 'block' to be a complete or reduction block";
  // Cond 2. Check all RAW predecessors are in the subtree rooted by loop_sref
  CHECK(EachEdgePointsToABlock(edges_to_pred, GetChildBlocks(self, loop_sref, true),
                               /*raw_edge_only=*/true))
      << "ValueError: 'reverse_compute_at' does not apply to a block that some other "
      << "blocks outside the scope depends on";
  // Cond 3. The subtree has compact data flow
  CHECK(CheckCompactDataFlow(self, scope, block_sref))
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
      if (AnyEdgePointsToABlock(edges_to_pred,
                                GetChildBlocks(self, self->stmt2ref.at(stmt), true))) {
        break;
      }
    }
    // Before all successors in dep graph.
    int before_pos;
    for (before_pos = 0; before_pos < n_stmts; before_pos++) {
      const StmtNode* stmt = loop_body[before_pos].get();
      if (AnyEdgePointsToABlock(edges_to_succ,
                                GetChildBlocks(self, self->stmt2ref.at(stmt), true))) {
        break;
      }
    }
    CHECK(insert_pos <= before_pos) << "ValueError: 'reverse_compute_at' cannot find an insertion "
                                       "point that satisfies dependency";
  }
  // Generate new LoopNode to substitute loop_sref->stmt
  For new_loop =
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
  StmtSRef root = GetSRefTreeRoot(block_sref);
  std::pair<Stmt, Stmt> removed = RemoveLeaf(block_sref, root);
  std::unordered_map<const StmtNode*, const StmtNode*> replace_map = {
      {removed.first.get(), removed.second.get()},
      {loop_sref->stmt, new_loop.get()},
  };
  // Mutate the AST with Replace
  StmtSRef lca = LowestCommonAncestor({block_sref, loop_sref}, root);
  Stmt replaced = StmtReplacer(replace_map)(GetRef<Stmt>(lca->stmt));
  if (const auto* replaced_block = replaced.as<BlockNode>()) {
    self->Replace(lca, replaced, {{GetRef<Block>(parent_block), GetRef<Block>(replaced_block)}});
  } else {
    self->Replace(lca, replaced, {});
  }
}

void ComputeInline(ScheduleState self, const StmtSRef& block_sref) {
  /*!
   * Check:
   *    1. The inner stmt of block_sref if a BufferStore
   *    2. block_sref if a complete Block
   */
  const auto* block = block_sref->GetStmt<BlockNode>();
  const StmtSRef& scope_block_sref = GetScopeRoot(block_sref);
  const auto* scope_block = scope_block_sref->GetStmt<BlockNode>();
  const BlockScope& scope = self->scopes.at(scope_block_sref);
  CHECK(block->body.as<BufferStoreNode>())
      << "ValueError: 'compute_inline' can only inline single assignment statement";
  CHECK_EQ(block->writes.size(), 1)
      << "ValueError: 'compute_inline' can only inline statement with one output";
  CHECK(scope->IsComplete(block_sref))
      << "ValueError: 'compute_inline' can only inline a complete block";
  // Remove leaf
  std::pair<Stmt, Stmt> removed = RemoveLeaf(block_sref, scope_block_sref);
  std::unordered_map<const StmtNode*, const StmtNode*> replace_map = {
      {removed.first.get(), removed.second.get()}};
  Stmt replaced = StmtReplacer(replace_map)(GetRef<Stmt>(scope_block));
  // Inline
  Map<Block, Block> block_sref_map;
  StatementInliner inliner(block, &block_sref_map, replace_map);
  Stmt inlined_stmt = inliner(replaced);
  self->Replace(scope_block_sref, inlined_stmt, block_sref_map);
}

void ReverseComputeInline(ScheduleState self, const StmtSRef& block_sref) {
  /*!
   * Check:
   *    1. block_sref is complete
   *    2. The inner stmt of block_sref is a BufferStore
   *    3. block_sref has only one producer
   *    4. The producer is complete
   *    5. The inner stmt of producer is a BufferStore
   *    6. The producer has only one consumer(which is block_sref)
   */
  const auto* block = block_sref->GetStmt<BlockNode>();
  CHECK(block != nullptr)
      << "TypeError: 'reverse_compute_at' expects 'block' to be a block, but get type: "
      << block_sref->stmt->GetTypeKey();
  const StmtSRef& scope_block_sref = GetScopeRoot(block_sref);
  const auto* scope_block = scope_block_sref->GetStmt<BlockNode>();
  const BlockScope& scope = self->scopes.at(scope_block_sref);
  // Cond 1. Check block_sref is complete
  CHECK(scope->IsComplete(block_sref))
      << "ValueError: 'reverse_compute_inline' expects the 'block' to be a complete block";
  // Cond 2. The inner stmt of block_sref if a BufferStore
  CHECK(block->body.as<BufferStoreNode>())
      << "ValueError: 'reverse_compute_inline' expects the 'block' contains a single BufferStore";
  // Cond 3. block_sref has only one RAW producer
  const auto& producers = scope->GetPredecessors(block_sref);
  CHECK_EQ(producers.size(), 1)
      << "ValueError: 'reverse_compute_inline' expects the 'block' has only one producer";
  CHECK(producers[0]->type == DepKind::kRAW)
      << "ValueError: 'reverse_compute_inline' expects the 'block' has only one producer";
  const StmtSRef& producer_sref = producers[0]->dst;
  // Cond 4. The producer is complete
  CHECK(scope->IsComplete(producer_sref))
      << "ValueError: 'reverse_compute_inline' expects the producer of 'block' to be complete";
  // Cond 5. The inner stmt of producer is a BufferStore
  const auto* producer = producer_sref->GetStmt<BlockNode>();
  CHECK(producer->body.as<BufferStoreNode>())
      << "ValueError: 'reverse_compute_inline' expects the producer of 'block' to contain a single "
         "BufferStore";
  // Cond 6. The producer has only one consumer(which is block_sref)
  const auto& consumers = scope->GetSuccessors(producer_sref);
  CHECK_EQ(consumers.size(), 1) << "ValueError: 'reverse_compute_inline' expects 'block' is the "
                                   "only consumer of its producer";
  CHECK_EQ(consumers[0]->dst, block_sref) << "ValueError: 'reverse_compute_inline' expects 'block' "
                                             "is the only consumer of its producer";

  // Remove leaf
  std::pair<Stmt, Stmt> removed = RemoveLeaf(block_sref, scope_block_sref);
  std::unordered_map<const StmtNode*, const StmtNode*> replace_map = {
      {removed.first.get(), removed.second.get()}};
  Stmt replaced = StmtReplacer(replace_map)(GetRef<Stmt>(scope_block));
  // Inline
  Map<Block, Block> block_sref_map;
  ReverseStatementInliner inliner(block, producer, &block_sref_map);
  Stmt inlined_stmt = inliner(replaced);
  self->Replace(scope_block_sref, inlined_stmt, block_sref_map);
}

}  // namespace schedule
}  // namespace tir
}  // namespace tvm
