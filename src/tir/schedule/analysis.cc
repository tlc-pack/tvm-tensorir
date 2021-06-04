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
#include "./utils.h"

namespace tvm {
namespace tir {

/******** ContainsVar ********/

bool ContainsVar(const ObjectRef& stmt_or_expr, const Array<Var>& vars) {
  std::unordered_set<const VarNode*> vars_set;
  vars_set.reserve(vars.size());
  for (const Var& var : vars) {
    vars_set.insert(var.get());
  }
  return ContainsVar(stmt_or_expr, vars_set);
}

bool ContainsVar(const ObjectRef& stmt_or_expr, const Var& var) {
  return ContainsVar(stmt_or_expr, {var.get()});
}

bool ContainsVar(const ObjectRef& stmt_or_expr, const std::unordered_set<const VarNode*>& vars) {
  bool found = false;
  auto f_find = [&found, &vars](const ObjectRef& obj) -> bool {
    if (found) {
      return false;
    }
    if (const VarNode* var = obj.as<VarNode>()) {
      if (vars.count(var)) {
        found = true;
        return false;
      }
    }
    return true;
  };
  PreOrderVisit(stmt_or_expr, f_find);
  return found;
}

std::unordered_set<const VarNode*> Vars(const ObjectRef& stmt_or_expr) {
  std::unordered_set<const VarNode*> result;
  auto f_visit = [&result](const ObjectRef& obj) -> void {
    if (const auto* var = obj.as<VarNode>()) {
      result.insert(var);
    }
  };
  PostOrderVisit(stmt_or_expr, f_visit);
  return result;
}

bool ValidateBlockBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges) {
  if (loop_var_ranges.empty()) {
    return true;
  }
  arith::Analyzer analyzer;
  Array<arith::IterSumExpr> results = arith::DetectIterMap(
      /*leaf_iters=*/realize->block->iter_vars,
      /*bindings=*/realize->iter_values,
      /*root_iters=*/loop_var_ranges,
      /*input_pred=*/realize->predicate, /*analyzer=*/&analyzer);
  if (results.empty()) {
    return false;
  }
  for (const arith::IterSumExpr& sum_expr : results) {
    const Array<arith::IterSplitExpr>& args = sum_expr->args;
    if (args.empty()) {
      continue;
    }
    if (!is_one(args[0]->scale)) {
      return false;
    }
  }
  return true;
}

bool RegionCoveredConsumer(const ScheduleState& self, const StmtSRef& consumer_block_sref,
                           const StmtSRef& scope_root) {
  if (consumer_block_sref->parent == nullptr) {
    return true;
  }
  const auto* consumer_block = TVM_SREF_TO_BLOCK(consumer_block, consumer_block_sref);
  BlockScope scope = self->GetBlockScope(scope_root);
  // Step 1. Gather all the producers
  struct Producer {
    /*! \brief The block that writes the buffer */
    StmtSRef block_sref;
    /*! \brief The region the buffer is written */
    BufferRegion region;
    /*! \brief Constructor */
    explicit Producer(StmtSRef block_sref, BufferRegion region)
        : block_sref(std::move(block_sref)), region(std::move(region)) {}
  };
  // Maps a buffer var to its producers
  std::unordered_map<const VarNode*, std::vector<Producer>> buffer_producers;
  // Collect all producers to a buffer by enumerating all RAW predecessors of the consumer
  for (const Dependency& edge : scope->GetDepsByDst(consumer_block_sref)) {
    // i.e. the RAW predecessor is producer
    if (edge->kind == DepKind::kRAW) {
      const StmtSRef& producer_block_sref = edge->src;
      const auto* producer_block = TVM_SREF_TO_BLOCK(producer_block, producer_block_sref);
      for (const BufferRegion& output_region : producer_block->writes) {
        const VarNode* buffer_var = output_region->buffer->data.get();
        buffer_producers[buffer_var].emplace_back(producer_block_sref, output_region);
      }
    }
  }
  // Step 2. For each buffer that the consumer reads, check the region cover property
  arith::Analyzer analyzer;
  for (const BufferRegion& consumer_region : consumer_block->reads) {
    // Step 2.1. Find the producers of the buffer
    const VarNode* buffer_var = consumer_region->buffer->data.get();
    auto it = buffer_producers.find(buffer_var);
    if (it == buffer_producers.end()) {
      continue;
    }
    const std::vector<Producer>& producers = it->second;
    // Step 2.2. Figure out LCA of consumer and all producers
    StmtSRef lca{nullptr};
    {
      std::vector<StmtSRef> inputs;
      inputs.reserve(producers.size() + 1);
      inputs.emplace_back(consumer_block_sref);
      for (const Producer& producer : producers) {
        inputs.emplace_back(producer.block_sref);
      }
      lca = LowestCommonAncestor(inputs, scope_root);
    }
    // Step 2.3. Relax the read region with the loops under LCA
    BufferRegion read = RelaxRegion(consumer_block_sref, lca, consumer_region);
    int ndim = read->region.size();
    for (const Producer& producer : producers) {
      // Relax the write region with the loops under LCA
      BufferRegion write = RelaxRegion(producer.block_sref, lca, producer.region);
      ICHECK_EQ(read->region.size(), write->region.size())
          << "ValueError: Inconsistent rank of the same buffer between reads and writes";
      // Check if the write domain covers the read domain
      for (int i = 0; i < ndim; ++i) {
        PrimExpr read_min = read->region[i]->min;
        PrimExpr read_max = read_min + read->region[i]->extent;
        PrimExpr write_min = write->region[i]->min;
        PrimExpr write_max = write_min + write->region[i]->extent;
        PrimExpr cond = (write_min <= read_min) && (read_max <= write_max);
        if (!analyzer.CanProve(cond)) {
          return false;
        }
      }
    }
  }
  return true;
}

class SRefTreeVerifier : public StmtVisitor {
 public:
  static void Verify(const ScheduleStateNode* self) { SRefTreeVerifier(self).Verify(); }

 private:
  /*! \brief Constructor */
  explicit SRefTreeVerifier(const ScheduleStateNode* self) : self_(self) {}

  void Verify() {
    VisitPrimFuncs(self_->mod, [this](const PrimFuncNode* func) { this->VisitStmt(func->body); });
    ICHECK_EQ(n_sref_visited_, static_cast<int>(self_->stmt2ref.size()));
    for (const auto& kv : self_->block_info) {
      const StmtSRef& sref = kv.first;
      ICHECK(sref->stmt != nullptr)
          << "InternalError: An expired sref is found in the block_scope mapping";
      ICHECK(self_->stmt2ref.count(sref->stmt))
          << "InternalError: The sref points to a statement that does not exist in stmt2ref";
      const StmtSRef& sref2 = self_->stmt2ref.at(sref->stmt);
      ICHECK(sref.same_as(sref2))
          << "InternalError: The sref points to a statement whose corresponding sref in stmt2ref "
             "is not the same object as itself";
    }
    ICHECK_EQ(n_block_sref_visited_, static_cast<int>(self_->block_info.size()));
  }

  void VisitStmt_(const BlockNode* block) override {
    if (init_block_depth_) {
      ICHECK(!self_->stmt2ref.count(block)) << "InternalError: A block inside init block has its "
                                               "corresponding sref, which is not allowed";
      StmtVisitor::VisitStmt_(block);
      return;
    }
    ICHECK(self_->stmt2ref.count(block))
        << "InternalError: A BlockNode should appear in sref map, but it didn't\n"
        << GetRef<Stmt>(block);
    ++n_sref_visited_;
    ++n_block_sref_visited_;
    const StmtSRef& sref = self_->stmt2ref.at(block);
    ICHECK(self_->block_info.count(sref))
        << "InternalError: Cannot find scope information of the BlockNode:\n"
        << GetRef<Stmt>(block);
    ICHECK(sref->parent == ancestors_.back())
        << "InternalError: Parent information mismatch for BlockNode:\n"
        << GetRef<Stmt>(block) << "\nIts parent is supposed to be:\n"
        << GetRef<Stmt>(ancestors_.back()->stmt) << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? Optional<Stmt>(GetRef<Stmt>(sref->parent->stmt))
                         : Optional<Stmt>(NullOpt));
    ancestors_.push_back(sref.operator->());
    if (block->init.defined()) {
      ++init_block_depth_;
      VisitStmt(block->init.value());
      --init_block_depth_;
    }
    VisitStmt(block->body);
    ancestors_.pop_back();
  }

  void VisitStmt_(const ForNode* loop) override {
    if (init_block_depth_) {
      ICHECK(!self_->stmt2ref.count(loop)) << "InternalError: A loop inside init block has its "
                                              "corresponding sref, which is not allowed";
      StmtVisitor::VisitStmt_(loop);
      return;
    }
    ICHECK(self_->stmt2ref.count(loop))
        << "InternalError: A ForNode should appear in sref map, but it didn't\n"
        << GetRef<Stmt>(loop);
    ++n_sref_visited_;
    const StmtSRef& sref = self_->stmt2ref.at(loop);
    Optional<Stmt> stmt = NullOpt;
    ICHECK(sref->parent == ancestors_.back())
        << "InternalError: Parent information mismatch for ForNode:\n"
        << GetRef<Stmt>(loop) << "\nIts parent is supposed to be:\n"
        << GetRef<Stmt>(ancestors_.back()->stmt) << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? Optional<Stmt>(GetRef<Stmt>(sref->parent->stmt))
                         : Optional<Stmt>(NullOpt));
    ancestors_.push_back(sref.operator->());
    StmtVisitor::VisitStmt_(loop);
    ancestors_.pop_back();
  }

  void VisitStmt_(const SeqStmtNode* seq_stmt) override {
    // Verify seq_index
    if (init_block_depth_) {
      StmtVisitor::VisitStmt_(seq_stmt);
      return;
    }
    int n = static_cast<int>(seq_stmt->seq.size());
    for (int i = 0; i < n; ++i) {
      const Stmt& child = seq_stmt->seq[i];
      StmtSRef sref{nullptr};
      if (const auto* realize = child.as<BlockRealizeNode>()) {
        const auto* block = realize->block.get();
        ICHECK(self_->stmt2ref.count(block));
        sref = self_->stmt2ref.at(block);
      } else if (child->IsInstance<ForNode>()) {
        ICHECK(self_->stmt2ref.count(child.get()));
        sref = self_->stmt2ref.at(child.get());
      } else {
        continue;
      }
      ICHECK_EQ(sref->seq_index, i) << "InternalError: A StmtSRef has incorrect seq_index";
    }
    StmtVisitor::VisitStmt_(seq_stmt);
  }

  /*! \brief The schedule it belongs to */
  const ScheduleStateNode* self_;
  /*! \brief Parent information during the visit */
  std::vector<const StmtSRefNode*> ancestors_ = {nullptr};
  /*! \brief If the visitor is currently in the init block */
  int init_block_depth_ = 0;
  /*! \brief Number of srefs that are visited */
  int n_sref_visited_ = 0;
  /*! \brief Number of block srefs that are visited */
  int n_block_sref_visited_ = 0;
};

void VerifySRefTree(const ScheduleState& self) { SRefTreeVerifier::Verify(self.get()); }

class BlockInfoVerifier : public StmtVisitor {
 public:
  static void Verify(const ScheduleStateNode* self) { BlockInfoVerifier(self).Verify(); }

 private:
  /*! \brief Constructor */
  explicit BlockInfoVerifier(const ScheduleStateNode* self) : self_(self) {}

  void Verify() {
    VisitPrimFuncs(self_->mod, [this](const PrimFuncNode* func) { this->VisitStmt(func->body); });
  }

  void CheckAffineBinding(const BlockRealizeNode* realize) const {
    const auto* block = realize->block.get();
    StmtSRef block_sref = self_->stmt2ref.at(block);
    Map<Var, Range> loop_var_ranges;
    for (StmtSRefNode* loop_sref = block_sref->parent; loop_sref != nullptr;
         loop_sref = loop_sref->parent) {
      if (const auto* loop = loop_sref->StmtAs<ForNode>()) {
        loop_var_ranges.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
      } else {
        break;
      }
    }
    bool affine_binding = ValidateBlockBinding(GetRef<BlockRealize>(realize), loop_var_ranges);
    ICHECK_EQ(affine_binding, self_->IsAffineBlockBinding(block_sref))
        << "Block: " << realize->block->name_hint << "\n"
        << Repr(self_->mod) << "\n"
        << loop_var_ranges;
  }

  void VisitStmt_(const BlockRealizeNode* realize) override {
    this->VisitStmt(realize->block->body);
    CheckAffineBinding(realize);
  }
  /*! \brief The schedule it belongs to */
  const ScheduleStateNode* self_;
};

void VerifyBlockInfo(const ScheduleState& self) { BlockInfoVerifier::Verify(self.get()); }

StmtSRef GetScopeRoot(const StmtSRef& sref) {
  for (const StmtSRefNode* p = sref->parent; p != nullptr; p = p->parent) {
    if (p->stmt->IsInstance<BlockNode>()) {
      return GetRef<StmtSRef>(p);
    }
  }
  ICHECK(false) << "Cannot get a scope block of a root block";
  throw;
}

Array<StmtSRef> GetBlocks(const ScheduleState& self, const String& name) {
  Array<StmtSRef> result;
  for (const auto& kv : self->block_info) {
    const StmtSRef& block_sref = kv.first;
    const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
    if (block->name_hint == name) {
      result.push_back(block_sref);
    }
  }
  return result;
}

bool IsCompactDataFlow(const ScheduleState& self, const StmtSRef& scope_root,
                       const Array<StmtSRef>& child_blocks) {
  for (const StmtSRef& block : child_blocks) {
    if (!CompleteBlock(self, block, scope_root) && !ReductionBlock(self, block, scope_root)) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Check if each reduction instance is valid. Particularly, check:
 * 1) Each iteration variable is either data parallel or reduction
 * 2) Indices used to access the output buffer are not related to or affected by reduction iteration
 * variables.
 * \param iter_vars Iteration variables of the reduction
 * \param output_buffer_indices Indices used to access the output buffer
 * \return A boolean indicating if the reduction instance is valid
 */
bool CheckReductionInstance(const Array<IterVar>& iter_vars,
                            const Array<PrimExpr>& output_buffer_indices) {
  std::unordered_set<const VarNode*> reduction_block_vars;
  reduction_block_vars.reserve(iter_vars.size());
  // Check 1. Each iter_var can only be data parallel or reduction
  for (const IterVar& iter_var : iter_vars) {
    IterVarType kind = iter_var->iter_type;
    if (kind != kDataPar && kind != kCommReduce) {
      return false;
    }
    if (kind == kCommReduce) {
      reduction_block_vars.insert(iter_var->var.get());
    }
  }
  // Check 2. Each reduction iter_var should not be used to index output buffer
  for (const PrimExpr& idx : output_buffer_indices) {
    if (ContainsVar(idx, reduction_block_vars)) {
      return false;
    }
  }
  return true;
}

bool IsDominant(const BlockScope& self, const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // Cond 1. Block is the only writer to its outputs
  const auto& buffer_writers = self->buffer_writers;
  for (const BufferRegion& write_region : block->writes) {
    ICHECK(buffer_writers.count(write_region->buffer))
        << "InternalError: buffer \"" << write_region->buffer->name
        << "\" does not exist in the current scope, when querying block:\n"
        << GetRef<Block>(block);
    // Check if the buffer is only written once (by the given block)
    if (buffer_writers.at(write_region->buffer).size() != 1) {
      return false;
    }
  }
  return true;
}

bool CompleteBlock(const ScheduleState& self, const StmtSRef& block_sref,
                   const StmtSRef& scope_root) {
  BlockScope scope = self->GetBlockScope(scope_root);
  // Cond 2. Check if all the block vars are data parallel
  const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return false;
    }
  }
  // Cond 1. A complete block must be dominate
  if (!IsDominant(scope, block_sref)) {
    return false;
  }
  // Cond 3. Check if there is no overlap between buffers read and buffers written
  for (const BufferRegion& write : block->writes) {
    const Buffer& buffer = write->buffer;
    for (const BufferRegion& read : block->reads) {
      if (buffer.same_as(read->buffer)) {
        return false;
      }
    }
  }
  return true;
}

bool ReductionBlock(const ScheduleState& self, const StmtSRef& block_sref,
                    const StmtSRef& scope_root) {
  BlockScope scope = self->GetBlockScope(scope_root);
  // Cond 3. Block binding is valid iter affine map
  // TODO
  // if (!this->IsAffineBlockBinding(block_sref)) {
  //   return false;
  // }
  // Cond 4. Check whether the block body has the init statement.
  const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
  if (!block->init.defined()) {
    return false;
  }
  // Cond 2. All block vars are either data parallel or reduction
  const Array<IterVar>& iter_vars = block->iter_vars;
  for (const IterVar& iter_var : iter_vars) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
  }
  // Cond 1. Dominate block
  if (!IsDominant(scope, block_sref)) {
    return false;
  }
  // Cond 5. All reduction vars should not affect indexing the output buffer
  std::unordered_set<const BufferNode*> buffer_written;
  buffer_written.reserve(block->writes.size());
  for (const BufferRegion& write_region : block->writes) {
    buffer_written.insert(write_region->buffer.get());
  }
  bool not_affected = true;
  PreOrderVisit(block->body, [&not_affected, &iter_vars, &buffer_written](const ObjectRef& obj) {
    if (!not_affected) {
      return false;
    }
    if (const auto* store = obj.as<BufferStoreNode>()) {
      // Only consider buffers written by the block
      if (buffer_written.count(store->buffer.get())) {
        if (!CheckReductionInstance(iter_vars, store->indices)) {
          not_affected = false;
        }
      } else {
        LOG(FATAL) << "InternalError: A write buffer is not in the block signature: "
                   << store->buffer;
      }
      return false;
    }
    return true;
  });
  return not_affected;
}

bool CanMergeReduction(const ScheduleState& self, const StmtSRef& init_block_sref,
                       const StmtSRef& update_block_sref, const StmtSRef& scope_root) {
  BlockScope scope = self->GetBlockScope(scope_root);
  const auto* init = TVM_SREF_TO_BLOCK(init, init_block_sref);
  const auto* update = TVM_SREF_TO_BLOCK(update, update_block_sref);
  // Cond 1. Check the binding of update block is valid
  if (!self->IsAffineBlockBinding(update_block_sref)) {
    return false;
  }
  // Cond 2. Check init_block and update_block are the only two producers for their output buffer
  for (const BufferRegion& write_region : update->writes) {
    const Array<StmtSRef>& writers = scope->buffer_writers.at(write_region->buffer);
    if (writers.size() != 2) {
      return false;
    }
    if (!writers[0].same_as(init_block_sref) && !writers[0].same_as(update_block_sref)) {
      return false;
    }
    if (!writers[1].same_as(init_block_sref) && !writers[1].same_as(update_block_sref)) {
      return false;
    }
  }
  // Cond 3. init and update share the same buffer
  const auto* init_body = TVM_TYPE_AS(init_body, init->body, BufferStoreNode);
  const auto* update_body = TVM_TYPE_AS(update_body, update->body, BufferStoreNode);
  if (!init_body->buffer.same_as(update_body->buffer)) {
    return false;
  }
  // Access must be the same dimensional
  ICHECK_EQ(init_body->indices.size(), update_body->indices.size())
      << "InternalError: indexing to the same buffer with different dimensions";
  // Cond 4. All block vars of update_block are either data parallel or reduction,
  // and reduction vars of update_block should not affect indexing the output buffer
  return CheckReductionInstance(update->iter_vars, update_body->indices);
}

Array<StmtSRef> GetAxes(const ScheduleState& self, const StmtSRef& block_sref) {
  std::vector<StmtSRef> result;
  for (StmtSRefNode* parent = block_sref->parent; parent && parent->stmt->IsInstance<ForNode>();
       parent = parent->parent) {
    result.push_back(GetRef<StmtSRef>(parent));
  }
  return {result.rbegin(), result.rend()};
}

Array<StmtSRef> GetChildBlocks(const ScheduleState& self, const StmtSRef& parent_sref,
                               bool inclusive) {
  struct Collector : public StmtVisitor {
   private:
    void VisitStmt_(const BlockNode* block) final { result.push_back(self->stmt2ref.at(block)); }

   public:
    explicit Collector(const ScheduleState& self) : self(self) {}

    const ScheduleState& self;
    Array<StmtSRef> result;
  };
  Collector collector(self);
  if (inclusive) {
    collector(GetRef<Stmt>(parent_sref->stmt));
  } else if (parent_sref->stmt->IsInstance<ForNode>()) {
    const auto* loop = static_cast<const ForNode*>(parent_sref->stmt);
    collector(loop->body);
  } else if (parent_sref->stmt->IsInstance<BlockNode>()) {
    const auto* block = static_cast<const BlockNode*>(parent_sref->stmt);
    collector(block->body);
  }
  return std::move(collector.result);
}

Array<StmtSRef> GetProducers(const ScheduleState& self, const StmtSRef& block_sref) {
  Array<Dependency> pred_edges = self->GetBlockScope(GetScopeRoot(block_sref))  //
                                     ->GetDepsByDst(block_sref);
  Array<StmtSRef> results;
  results.reserve(pred_edges.size());
  for (const Dependency& edge : pred_edges) {
    if (edge->kind == DepKind::kRAW || edge->kind == DepKind::kWAW) {
      results.push_back(edge->src);
    }
  }
  return results;
}

Array<StmtSRef> GetConsumers(const ScheduleState& self, const StmtSRef& block_sref) {
  Array<Dependency> succ_edges = self->GetBlockScope(GetScopeRoot(block_sref))  //
                                     ->GetDepsBySrc(block_sref);
  Array<StmtSRef> results;
  results.reserve(succ_edges.size());
  for (const Dependency& edge : succ_edges) {
    if (edge->kind == DepKind::kRAW || edge->kind == DepKind::kWAW) {
      results.push_back(edge->dst);
    }
  }
  return results;
}

bool HasSingleChild(const StmtSRef& loop_or_block_sref) {
  const StmtNode* body = nullptr;
  if (const auto* loop = loop_or_block_sref->StmtAs<ForNode>()) {
    body = loop->body.get();
  } else if (const auto* block = loop_or_block_sref->StmtAs<BlockNode>()) {
    body = block->body.get();
  } else {
    LOG(FATAL) << "TypeError: Unable to recognize the type of `loop_or_block_sref`: "
               << loop_or_block_sref->stmt->GetTypeKey();
  }
  if (body->IsInstance<SeqStmtNode>()) {
    const auto* seq_stmt = static_cast<const SeqStmtNode*>(body);
    return seq_stmt->seq.size() == 1;
  }
  return true;
}

IterVarType GetLoopIterType(const ScheduleState& self, const StmtSRef& loop_sref) {
  int n_spatial = 0;
  int n_reduce = 0;
  int n_other = 0;
  const auto* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  const Var& loop_var = loop->loop_var;
  auto f_visit = [&loop_var, &n_spatial, &n_reduce, &n_other](const ObjectRef& obj) -> bool {
    if (const auto* realize = obj.as<BlockRealizeNode>()) {
      const BlockNode* block = realize->block.get();
      // Number of block vars and their bindings
      ICHECK_EQ(realize->iter_values.size(), block->iter_vars.size());
      int n = realize->iter_values.size();
      for (int i = 0; i < n; ++i) {
        const IterVar& iter_var = block->iter_vars[i];
        const PrimExpr& binding = realize->iter_values[i];
        // Categorize the current block var
        int* ref = nullptr;
        if (iter_var->iter_type == IterVarType::kDataPar) {
          ref = &n_spatial;
        } else if (iter_var->iter_type == IterVarType::kCommReduce) {
          ref = &n_reduce;
        } else {
          ref = &n_other;
        }
        // Visit the binding to see if `loop_var` appears
        PostOrderVisit(binding, [&ref, &loop_var](const ObjectRef& obj) -> void {
          if (obj.same_as(loop_var)) {
            (*ref) += 1;
          }
        });
      }
      return false;
    }
    return true;
  };
  PreOrderVisit(loop->body, f_visit);
  if (n_other) {
    return IterVarType::kOpaque;
  } else if (n_spatial && n_reduce) {
    return IterVarType::kOpaque;
  } else if (n_reduce) {
    return IterVarType::kCommReduce;
  } else if (loop->kind == ForKind::kUnrolled) {
    return IterVarType::kUnrolled;
  } else if (loop->kind == ForKind::kVectorized) {
    return IterVarType::kVectorized;
  } else if (loop->kind == ForKind::kParallel) {
    return IterVarType::kParallelized;
  }
  return IterVarType::kDataPar;
}

Array<StmtSRef> CollectComputeLocation(const ScheduleState& self, const StmtSRef& block_sref) {
  Array<StmtSRef> loop_srefs = GetAxes(self, block_sref);
  Array<StmtSRef> result;
  result.reserve(loop_srefs.size());
  bool visited_reduce = false;
  for (const StmtSRef& loop_sref : loop_srefs) {
    const auto* loop = TVM_SREF_TO_FOR(loop, loop_sref);
    IterVarType iter_type = GetLoopIterType(self, loop_sref);
    if (iter_type == IterVarType::kDataPar) {
      if (visited_reduce) {
        break;
      }
    } else {
      visited_reduce = true;
    }
    result.push_back(loop_sref);
  }
  return result;
}

StmtSRef GetSRefTreeRoot(const StmtSRef& sref) {
  const StmtSRefNode* p = sref.get();
  for (; p->parent != nullptr; p = p->parent) {
  }
  return GetRef<StmtSRef>(p);
}

const PrimFuncNode* GetRootPrimFunc(const ScheduleState& self, const StmtSRef& sref) {
  const StmtSRefNode* p = sref.get();
  for (; p->parent != nullptr; p = p->parent) {
  }
  for (const auto& kv : self->mod->functions) {
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<PrimFuncNode>()) {
      if (const auto* realize = func->body.as<BlockRealizeNode>()) {
        if (realize->block.get() == p->stmt) {
          return func;
        }
      }
    }
  }
  LOG(FATAL) << "IndexError: Could not get the correpsonding function in the schedule state of the "
                "statement:\n"
             << GetRef<Stmt>(sref->stmt);
  throw;
}

}  // namespace tir
}  // namespace tvm
