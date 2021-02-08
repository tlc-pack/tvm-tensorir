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

#include <tvm/arith/iter_affine_map.h>

#include "../../arith/pattern_match.h"
#include "./schedule_common.h"

namespace tvm {
namespace tir {

/*! \brief A helper class to validate loops and store them into StmtSRefNode::binding_valid */
class LoopValidator : public StmtVisitor {
 public:
  /*! \brief Constructor */
  explicit LoopValidator(std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref,
                         const ScheduleNode* schedule_node)
      : stmt2ref_(stmt2ref), schedule_node_(schedule_node) {}

  // Validate loop binding for each block
  void VisitStmt_(const BlockRealizeNode* realize) final {
    // Check StmtSRef's binding validity on all blocks
    stmt2ref_->at(realize->block.get())->binding_valid = ValidateBlockBinding(realize);
    StmtVisitor::VisitStmt_(realize);
  }

  /*! \brief Validate the binding of a given block */
  bool ValidateBlockBinding(const BlockRealizeNode* realize) {
    // validate the bindings to loop variables
    auto loops = schedule_node_->GetLoopsInScope(stmt2ref_->at(realize->block.get()));
    std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> loop_vars;
    for (const auto& loop_sref : loops) {
      const auto* loop = loop_sref->GetStmt<LoopNode>();
      loop_vars.emplace(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
    std::vector<PrimExpr> bindings{realize->binding_values.begin(), realize->binding_values.end()};
    arith::Analyzer analyzer;
    auto results = arith::DetectIterMap(realize->block->iter_vars, bindings, loop_vars,
                                        realize->predicate, &analyzer);
    if (!results.empty()) {
      for (const auto& sum_expr : results) {
        if (!sum_expr->args.empty() && !is_one(sum_expr->args[0]->scale)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

 private:
  /*! \brief ScheduleNode::stmt2ref whose StmtSRef::binding_valid needs updating */
  std::unordered_map<const StmtNode*, StmtSRef>* stmt2ref_;
  /*! \brief Pointer to ScheduleNode */
  const ScheduleNode* schedule_node_;
  /*! \brief The block vars in the ancestor blocks */
  std::unordered_set<const VarNode*> ancestor_block_vars_;
  /*! \brief An analyzer used to simplify expressions */
  arith::Analyzer analyzer_;
};

void ScheduleNode::ValidateLoops() {
  LoopValidator validator(&stmt2ref, this);
  validator(func->body);
}

bool ScheduleNode::ValidateRegionCover(const StmtSRef& consumer_block_sref) const {
  if (consumer_block_sref->parent == nullptr) {
    return true;
  }
  const auto* consumer_block = consumer_block_sref->GetStmt<BlockNode>();
  const StmtSRef& parent_block_sref = GetParentBlockSRef(consumer_block_sref);
  // Gather all the producers
  struct Producer {
    /*! \brief The block that writes the buffer */
    StmtSRef block_sref;
    /*! \brief The region the buffer is written */
    TensorRegion region;
    /*! \brief Constructor */
    Producer(const StmtSRef& block_sref, const TensorRegion& region)
        : block_sref(block_sref), region(region) {}
  };
  // Maps a buffer var to its producers
  std::unordered_map<const VarNode*, std::vector<Producer>> buffer_producers;
  // Collect all producers to a buffer by enumerating all RAW predecessors of the consumer
  for (const DepEdge& edge : scopes.at(parent_block_sref).GetPredecessors(consumer_block_sref)) {
    if (edge->type != DepType::kRAW) {
      continue;
    }
    // i.e. the RAW predecessor is producer
    const StmtSRef& producer_block_sref = edge->dst;
    for (const TensorRegion& output_region : producer_block_sref->GetStmt<BlockNode>()->writes) {
      const VarNode* buffer_var = output_region->buffer->data.get();
      buffer_producers[buffer_var].emplace_back(producer_block_sref, output_region);
    }
  }
  // Check the region cover property for each buffer that the consumer reads
  for (const TensorRegion& consumer_region : consumer_block->reads) {
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
    TensorRegion read = RelaxRegion(consumer_block_sref, lca, consumer_region);
    int ndim = read->region.size();
    for (const Producer& producer : producers) {
      // Relax the write region with the loops under LCA
      TensorRegion write = RelaxRegion(producer.block_sref, lca, producer.region);
      CHECK_EQ(read->region.size(), write->region.size())
          << "InternalError: Inconsistent rank of the same buffer between reads and writes";
      // Check if the write domain covers the read domain
      for (int i = 0; i < ndim; ++i) {
        PrimExpr read_min = read->region[i]->min;
        PrimExpr read_max = read_min + read->region[i]->extent;
        PrimExpr write_min = write->region[i]->min;
        PrimExpr write_max = write_min + write->region[i]->extent;
        if (!analyzer.CanProve(write_min <= read_min) ||
            !analyzer.CanProve(read_max <= write_max)) {
          LOG(WARNING) << "InternalError: Cannot prove the region cover property on dimension " << i
                       << "\nThe producer is:\n  " << write << ", write range: [" << write_min
                       << ", " << write_max << ")"
                       << "\nThe consumer is:\n  " << read << ", read range: [" << read_min << ", "
                       << read_max << ")";
          return false;
        }
      }
    }
  }
  return true;
}

class GPUValidator : public StmtVisitor {
 public:
  void VisitStmt_(const LoopNode* loop) final {
    std::string thread_tag;
    for (const auto& annotation : loop->annotations) {
      if (annotation->attr_key == attr::loop_type) {
        thread_tag = Downcast<StringImm>(annotation->value)->value;
      }
    }

    bool new_kernel = false;
    if ((IsBlockIdx(thread_tag) || IsThreadIdx(thread_tag)) && thread_tag != "vthread") {
      // Check thread binding extents are same in one single kernel

      // If there is no binding, we can regard it as a new kernel
      new_kernel = thread_extents_.empty();

      auto it = thread_extents_.find(thread_tag);
      if (it != thread_extents_.end()) {
        CHECK(ExprDeepEqual()(loop->extent, it->second))
            << "All loops with the same thread binding must have the same extent, but get "
            << loop->extent << " vs " << it->second;
      } else {
        thread_extents_[thread_tag] = loop->extent;
      }
    }

    // Check execution scope and
    if ((current_scope_ == "gpu_thread") && (IsBlockIdx(thread_tag) || IsThreadIdx(thread_tag))) {
      // If the current scope is gpu_thread, any inside threadIdx or blockIdx is illegal.
      LOG(FATAL) << "threadIdx or blockIdx can not be binded under the exec_scope gpu_thread";
    } else if (current_scope_ == "gpu_warp" &&
               ((IsBlockIdx(thread_tag) || IsThreadIdx(thread_tag)) &&
                (thread_tag != "threadIdx.x" || !ExprDeepEqual()(loop->extent, 32)))) {
      LOG(FATAL) << "threadIdx or blockIdx can not be binded under the exec_scope "
                    "gpu_thread except threadIdx.x with extents 32";
    } else if (current_scope_ == "gpu_block" && IsBlockIdx(thread_tag)) {
      // If the current scope is gpu_block, any inside blockIdx is illegal.
      LOG(FATAL) << "blockIdx can not be binded under the exec_scope gpu_block";
    }

    bool contain_thread_x = contain_thread_x_ || thread_tag == "threadIdx.x";
    std::swap(contain_thread_x, contain_thread_x_);
    StmtVisitor::VisitStmt_(loop);
    std::swap(contain_thread_x, contain_thread_x_);

    if (new_kernel) {
      if (check_thread_x_) {
        auto it = thread_extents_.find("threadIdx.x");
        CHECK(it != thread_extents_.end())
            << "can not find threadIdx.x but find warp level execution scope";
        CHECK(ExprDeepEqual()(it->second, 32))
            << "threadIdx.x extent is expected to be 32 with warp level block but get "
            << it->second;
      }
      check_thread_x_ = false;
      thread_extents_.clear();
    }
  }

  void VisitStmt_(const BlockRealizeNode* realize) final {
    std::string exec_scope = realize->exec_scope;
    std::string current_scope;
    std::swap(current_scope, current_scope_);

    if (!exec_scope.empty() && !current_scope.empty()) {
      if (exec_scope == "gpu_block") {
        CHECK(current_scope == "gpu_block" || current_scope == "gpu_global");
      } else if (exec_scope == "gpu_warp") {
        CHECK(current_scope == "gpu_warp" || current_scope == "gpu_block" ||
              current_scope == "gpu_global");
      } else if (exec_scope == "gpu_warp") {
        CHECK(exec_scope == "gpu_thread" || current_scope == "gpu_warp" ||
              current_scope == "gpu_block" || current_scope == "gpu_global");
      }
    }
    if (exec_scope == "gpu_warp") {
      check_thread_x_ = true;
      CHECK(!contain_thread_x_);
    }
    current_scope_ = exec_scope;
    StmtVisitor::VisitStmt_(realize);
    std::swap(current_scope, current_scope_);
  }

  /*! \brief The final result */

 private:
  static inline bool IsThreadIdx(const std::string& thread_tag) {
    return thread_tag.substr(0, 9) == "threadIdx" || thread_tag.substr(0, 7) == "vthread";
  }

  static inline bool IsBlockIdx(const std::string& thread_tag) {
    return thread_tag.substr(0, 9) == "BlockIdx";
  }

  /*! \brief The current execution scope (gpu_global, gpu_block, gpu_warp or gpu_thread) */
  std::string current_scope_ = "gpu_global";
  /*! \brief The extents of each threadIdx or blockIdx */
  std::unordered_map<std::string, PrimExpr> thread_extents_;
  /*! \brief Whether need to check threadIdx.x extents = 32 */
  bool check_thread_x_ = false;
  /*! \brief The loop stack from current node up to root contain thread_x */
  bool contain_thread_x_ = false;
};

void ScheduleNode::ValidateHierarchy(const PrimFunc& f) {
  GPUValidator gpu_validator;
  gpu_validator(f->body);
}

/*! \brief A helper class to validate correctness of StmtSRef */
class SRefValidator : public StmtVisitor {
 public:
  /*! \brief Constructor */
  explicit SRefValidator(const ScheduleNode* sch) : sch(sch), ancestors({nullptr}) {}
  // Valida each block
  void VisitStmt_(const BlockNode* block) override {
    CHECK(sch->stmt2ref.count(block))
        << "InternalError: A BlockNode should appear in sref map, but it didn't\n"
        << GetRef<Stmt>(block);
    const StmtSRef& sref = sch->stmt2ref.at(block);
    CHECK(sch->scopes.count(sref))
        << "InternalError: Cannot find scope information of the BlockNode:\n"
        << GetRef<Stmt>(block);
    CHECK(sref->parent == ancestors.back())
        << "InternalError: Parent information mismatch for BlockNode:\n"
        << GetRef<Stmt>(block) << "\nIts parent is supposed to be:\n"
        << GetRef<Stmt>(ancestors.back()->stmt) << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? Optional<Stmt>(GetRef<Stmt>(sref->parent->stmt))
                         : Optional<Stmt>(NullOpt));
    ancestors.push_back(sref.operator->());
    StmtVisitor::VisitStmt_(block);
    ancestors.pop_back();
  }
  // Validate each loop
  void VisitStmt_(const LoopNode* loop) override {
    CHECK(sch->stmt2ref.count(loop))
        << "InternalError: A LoopNode should appear in sref map, but it didn't\n"
        << GetRef<Stmt>(loop);
    const StmtSRef& sref = sch->stmt2ref.at(loop);
    Optional<Stmt> stmt = NullOpt;
    CHECK(sref->parent == ancestors.back())
        << "InternalError: Parent information mismatch for LoopNode:\n"
        << GetRef<Stmt>(loop) << "\nIts parent is supposed to be:\n"
        << GetRef<Stmt>(ancestors.back()->stmt) << "\nHowever, its parent is incorrect and is:\n"
        << (sref->parent ? Optional<Stmt>(GetRef<Stmt>(sref->parent->stmt))
                         : Optional<Stmt>(NullOpt));
    ancestors.push_back(sref.operator->());
    StmtVisitor::VisitStmt_(loop);
    ancestors.pop_back();
  }
  /*! \brief The schedule it belongs to */
  const ScheduleNode* sch;
  /*! \brief Parent information during the visit */
  std::vector<const StmtSRefNode*> ancestors;
};

bool ScheduleNode::ValidateSRef() const {
  SRefValidator(this)(func->body);
  return true;
}

TVM_REGISTER_GLOBAL("tir.schedule.ValidateSRef")
    .set_body_typed<bool(Schedule)>([](Schedule schedule) { return schedule->ValidateSRef(); });

TVM_REGISTER_GLOBAL("tir.schedule.ValidateHierarchy")
    .set_body_typed<void(PrimFunc)>([](PrimFunc f) { return ScheduleNode::ValidateHierarchy(f); });

}  // namespace tir
}  // namespace tvm
