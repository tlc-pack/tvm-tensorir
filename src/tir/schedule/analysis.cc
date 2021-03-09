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

bool ValidateBlockBinding(const BlockRealize& realize, const Map<Var, Range>& loop_var_ranges) {
  arith::Analyzer analyzer;
  Array<arith::IterSumExpr> results = arith::DetectIterMap(
      /*leaf_iters=*/realize->block->iter_vars,
      /*bindings=*/realize->binding_values,
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

void VerifyRegionCover(const ScheduleState& self, const StmtSRef& consumer_block_sref) {
  if (consumer_block_sref->parent == nullptr) {
    return;
  }
  const auto* consumer_block = consumer_block_sref->GetStmt<BlockNode>();
  const StmtSRef& parent_block_sref = GetScopeRoot(consumer_block_sref);
  // Gather all the producers
  struct Producer {
    /*! \brief The block that writes the buffer */
    StmtSRef block_sref;
    /*! \brief The region the buffer is written */
    BufferRegion region;
    /*! \brief Constructor */
    Producer(const StmtSRef& block_sref, const BufferRegion& region)
        : block_sref(block_sref), region(region) {}
  };
  // Maps a buffer var to its producers
  std::unordered_map<const VarNode*, std::vector<Producer>> buffer_producers;
  // Collect all producers to a buffer by enumerating all RAW predecessors of the consumer
  for (const Dependency& edge :
       self->scopes.at(parent_block_sref)->GetPredecessors(consumer_block_sref)) {
    if (edge->kind != DepKind::kRAW) {
      continue;
    }
    // i.e. the RAW predecessor is producer
    const StmtSRef& producer_block_sref = edge->dst;
    for (const BufferRegion& output_region : producer_block_sref->GetStmt<BlockNode>()->writes) {
      const VarNode* buffer_var = output_region->buffer->data.get();
      buffer_producers[buffer_var].emplace_back(producer_block_sref, output_region);
    }
  }
  // Check the region cover property for each buffer that the consumer reads
  for (const BufferRegion& consumer_region : consumer_block->reads) {
    const VarNode* buffer_var = consumer_region->buffer->data.get();
    if (!buffer_producers.count(buffer_var)) {
      continue;
    }
    // Producers of the current buffer
    const std::vector<Producer>& producers = buffer_producers.at(buffer_var);
    // Figure out LCA of consumer and all producers
    StmtSRef lca = [&producers, &consumer_block_sref, &parent_block_sref]() {
      // inputs include consumer and all producers
      std::vector<StmtSRef> inputs = {consumer_block_sref};
      for (const Producer& producer : producers) {
        inputs.push_back(producer.block_sref);
      }
      return LowestCommonAncestor(inputs, parent_block_sref);
    }();
    arith::Analyzer analyzer;
    // Relax the read region with the loops under LCA
    BufferRegion read = RelaxRegion(consumer_block_sref, lca, consumer_region);
    int ndim = read->region.size();
    for (const Producer& producer : producers) {
      // Relax the write region with the loops under LCA
      BufferRegion write = RelaxRegion(producer.block_sref, lca, producer.region);
      CHECK_EQ(read->region.size(), write->region.size())
          << "ValueError: Inconsistent rank of the same buffer between reads and writes";
      // Check if the write domain covers the read domain
      for (int i = 0; i < ndim; ++i) {
        PrimExpr read_min = read->region[i]->min;
        PrimExpr read_max = read_min + read->region[i]->extent;
        PrimExpr write_min = write->region[i]->min;
        PrimExpr write_max = write_min + write->region[i]->extent;
        if (!analyzer.CanProve(write_min <= read_min) ||
            !analyzer.CanProve(read_max <= write_max)) {
          LOG(FATAL) << "ValueError: Cannot prove the region cover property on dimension " << i
                     << "\nThe producer is:\n  " << write << ", write range: [" << write_min << ", "
                     << write_max << ")"
                     << "\nThe consumer is:\n  " << read << ", read range: [" << read_min << ","
                     << read_max << ")";
        }
      }
    }
  }
}

void VerifySRefTree(const ScheduleState& self) {
  /*!
   * \brief A helper class to validate correctness of StmtSRef
   * TODO(@junrushao1994): refactor this
   */
  class SRefTreeVerifier : public StmtVisitor {
   public:
    static void Verify(const ScheduleStateNode* self) { SRefTreeVerifier(self).Verify(); }

   private:
    /*! \brief Constructor */
    explicit SRefTreeVerifier(const ScheduleStateNode* self)
        : self_(self),
          ancestors_{nullptr},
          is_in_init_block_(0),
          n_sref_visited_(0),
          n_block_sref_visited_(0) {}

    void Verify() {
      for (const auto& kv : self_->mod->functions) {
        const BaseFunc& base_func = kv.second;
        if (const auto* func = base_func.as<PrimFuncNode>()) {
          VisitStmt(func->body);
        }
      }
      ICHECK_EQ(n_sref_visited_, static_cast<int>(self_->stmt2ref.size()));
      for (const auto& kv : self_->scopes) {
        const StmtSRef& sref = kv.first;
        ICHECK(sref->stmt != nullptr);
        ICHECK(self_->stmt2ref.count(sref->stmt));
        const StmtSRef& sref2 = self_->stmt2ref.at(sref->stmt);
        ICHECK(sref.same_as(sref2));
      }
      ICHECK_EQ(n_block_sref_visited_, static_cast<int>(self_->scopes.size()));
    }

    // Valida each block
    void VisitStmt_(const BlockNode* block) override {
      if (is_in_init_block_) {
        ICHECK(!self_->stmt2ref.count(block));
        StmtVisitor::VisitStmt_(block);
        return;
      }
      ICHECK(self_->stmt2ref.count(block))
          << "InternalError: A BlockNode should appear in sref map, but it didn't\n"
          << GetRef<Stmt>(block);
      ++n_sref_visited_;
      ++n_block_sref_visited_;
      const StmtSRef& sref = self_->stmt2ref.at(block);
      ICHECK(self_->scopes.count(sref))
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
        ++is_in_init_block_;
        VisitStmt(block->init.value());
        --is_in_init_block_;
      }
      VisitStmt(block->body);
      ancestors_.pop_back();
    }

    // Validate each loop
    void VisitStmt_(const ForNode* loop) override {
      if (is_in_init_block_) {
        ICHECK(!self_->stmt2ref.count(loop));
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

    // Validate seq_index
    void VisitStmt_(const SeqStmtNode* seq_stmt) override {
      if (is_in_init_block_) {
        StmtVisitor::VisitStmt_(seq_stmt);
        return;
      }
      int n = seq_stmt->seq.size();
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
    std::vector<const StmtSRefNode*> ancestors_;
    /*! \brief If the visitor is currently in the init block */
    int is_in_init_block_;
    /*! \brief Number of srefs that are visited */
    int n_sref_visited_;
    /*! \brief Number of block srefs that are visited */
    int n_block_sref_visited_;
  };
  SRefTreeVerifier::Verify(self.get());
}

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
  for (const auto& kv : self->scopes) {
    const StmtSRef& block_sref = kv.first;
    const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
    if (block->name_hint == name) {
      result.push_back(block_sref);
    }
  }
  return result;
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
  Array<Dependency> pred_edges = self->scopes
                                     .at(GetScopeRoot(block_sref))  //
                                     ->GetPredecessors(block_sref);
  Array<StmtSRef> results;
  results.reserve(pred_edges.size());
  for (const Dependency& edge : pred_edges) {
    if (edge->kind == DepKind::kRAW || edge->kind == DepKind::kWAW) {
      results.push_back(edge->dst);
    }
  }
  return results;
}

Array<StmtSRef> GetConsumers(const ScheduleState& self, const StmtSRef& block_sref) {
  Array<Dependency> succ_edges = self->scopes
                                     .at(GetScopeRoot(block_sref))  //
                                     ->GetSuccessors(block_sref);
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
  if (const auto* loop = loop_or_block_sref->GetStmt<ForNode>()) {
    body = loop->body.get();
  } else if (const auto* block = loop_or_block_sref->GetStmt<BlockNode>()) {
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
      ICHECK_EQ(realize->binding_values.size(), block->iter_vars.size());
      int n = realize->binding_values.size();
      for (int i = 0; i < n; ++i) {
        const IterVar& iter_var = block->iter_vars[i];
        const PrimExpr& binding = realize->binding_values[i];
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
    // If the loop has multiple children, then do not go into it anymore
    if (!HasSingleChild(loop_sref)) {
      break;
    }
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
