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
#include "./analysis.h"

namespace tvm {
namespace tir {

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

StmtSRef GetScopeSRef(const StmtSRef& sref) {
  for (const StmtSRefNode* p = sref->parent; p != nullptr; p = p->parent) {
    if (p->stmt->IsInstance<BlockNode>()) {
      return GetRef<StmtSRef>(p);
    }
  }
  ICHECK(false) << "Cannot get a scope block of a root block";
  throw;
}

void VerifyRegionCover(const ScheduleState& self, const StmtSRef& consumer_block_sref) {
  if (consumer_block_sref->parent == nullptr) {
    return;
  }
  const auto* consumer_block = consumer_block_sref->GetStmt<BlockNode>();
  const StmtSRef& parent_block_sref = GetScopeSRef(consumer_block_sref);
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
  for (const DepEdge& edge :
       self->scopes.at(parent_block_sref)->GetPredecessors(consumer_block_sref)) {
    if (edge->type != DepType::kRAW) {
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
      VisitStmt(self_->func->body);
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

}  // namespace tir
}  // namespace tvm
