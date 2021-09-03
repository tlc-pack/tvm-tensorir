/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

/*!
 * \file allreduce_transform.cc
 */

#include <tvm/arith/int_set.h>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../schedule/analysis.h"

namespace tvm {
namespace tir {

/*!
 * \brief Given a reduction identity and a reduction combiner, detect the corresponding commutative
 * reducer, and extract the combiner lhs and combiner rhs
 * \param identity The reduction identity to be analyzed
 * \param combiner The reduction combiner to be analyzed
 * \return The corresponding CommReducer, the combiner lhs and the combiner rhs
 * \throw ScheduleError If no corresponding commutative reducer can be matched
 */
std::tuple<CommReducer, PrimExpr, PrimExpr> GetReducerAndCombinerLhsRhs(
    const PrimExpr& identity, const BufferStore& combiner) {
  CommReducer reducer{nullptr};
  PrimExpr combiner_lhs{nullptr}, combiner_rhs{nullptr};
  bool matched = FromIdentityCombiner(identity, combiner, &reducer, &combiner_lhs, &combiner_rhs);
  if (!matched) {
    LOG(FATAL) << "No matched reducer for identity " << identity << " and combiner " << combiner
               << "In this case rfactor cannot be applied. You can check tvm::tir::ReducerRegistry "
                  "for default reducers or registering new reducers.";
  }
  return std::make_tuple(std::move(reducer), std::move(combiner_lhs), std::move(combiner_rhs));
}

/*!
 * \brief Detect allreduce and then transform.
 */
class AllReduceTransformer : public StmtExprMutator {
 public:
  AllReduceTransformer() = default;

 public:
#define TVM_ALLREDUCE_VISIT_SIMPLE_BODY(Type)                               \
  Stmt VisitStmt_(const Type* op) override {                                \
    if (status == kDetecting) {                                             \
      stmt_stack_.push_back(GetRef<Stmt>(op));                              \
      Stmt res_stmt = StmtMutator::VisitStmt_(op);                          \
      const auto* res = res_stmt.as<Type>();                                \
      ICHECK(res != nullptr);                                               \
      ICHECK(!stmt_stack_.empty())                                          \
          << "Size of stmt_stack_ is expected to be positive, but it is 0"; \
      stmt_stack_.pop_back();                                               \
                                                                            \
      ObjectPtr<Type> n = CopyOnWrite(res);                                 \
      AddStatements(GetRef<Stmt>(op), op->body, res->body, n->body);        \
      return Stmt(n);                                                       \
    } else {                                                                \
      return StmtMutator::VisitStmt_(op);                                   \
    }                                                                       \
  }

  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(AttrStmtNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(LetStmtNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(ForNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(AllocateNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(BufferRealizeNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(AssertStmtNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(ProducerRealizeNode);

#undef TVM_ALLREDUCE_VISIT_SIMPLE_BODY

  Stmt VisitStmt_(const IfThenElseNode* op) override {
    if (status == kDetecting) {
      stmt_stack_.push_back(GetRef<Stmt>(op));
      Stmt res_stmt = StmtMutator::VisitStmt_(op);
      const auto* res = res_stmt.as<IfThenElseNode>();
      ICHECK(res != nullptr);
      ICHECK(!stmt_stack_.empty()) << "Size of stmt_stack_ is expected to be positive, but it is 0";
      stmt_stack_.pop_back();

      ObjectPtr<IfThenElseNode> n = CopyOnWrite(res);
      AddStatements(GetRef<Stmt>(op), op->then_case, res->then_case, n->then_case);
      AddStatements(GetRef<Stmt>(op), op->else_case, res->else_case, n->else_case);
      return Stmt(n);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const SeqStmtNode* op) override {
    if (status == kDetecting) {
      Stmt res_stmt = StmtMutator::VisitStmt_(op);
      const auto* res = res_stmt.as<SeqStmtNode>();
      ICHECK(res != nullptr);
      ICHECK(!stmt_stack_.empty()) << "Size of stmt_stack_ is expected to be positive, but it is 0";

      std::vector<Stmt> seq;
      for (const Stmt& stmt : res->seq) {
        seq.emplace_back(stmt);
      }

      ObjectPtr<SeqStmtNode> n = CopyOnWrite(res);
      for (size_t i = 0; i < seq.size(); ++i) {
        AddStatements(GetRef<Stmt>(op), op->seq[i], res->seq[i], seq[i]);
      }
      n->seq = seq;
      return Stmt(n);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) override {
    if (status == kDetecting) {
      stmt_stack_.push_back(GetRef<Stmt>(op));
      Stmt res = StmtMutator::VisitStmt_(op);
      ICHECK(!stmt_stack_.empty()) << "Size of stmt_stack_ is expected to be positive, but it is 0";
      stmt_stack_.pop_back();
      return res;
    } else if (status == kMutatingBlock_nor_red) {
      // Mutate buffer, indices and value
      ICHECK(op->buffer.same_as(write_buffer));
      ICHECK(normal_reduce.defined());

      ObjectPtr<BufferStoreNode> n = CopyOnWrite(op);
      PrimExpr value = this->VisitExpr(op->value);

      n->buffer = normal_reduce.value();
      n->indices = {0};
      n->value = value;
      return Stmt(n);
    } else if (status == kMutatingBlock_red_tmp) {
      return StmtMutator::VisitStmt_(op);
    } else {
      LOG(FATAL);
      throw;
    }
  }

  Stmt VisitStmt_(const BlockNode* op) override {
    if (status == kDetecting) {
      std::deque<Stmt> tmp_stmt_stack_;
      std::swap(stmt_stack_, tmp_stmt_stack_);
      stmt_stack_.push_back(GetRef<Stmt>(op));
      Stmt res_stmt = StmtMutator::VisitStmt_(op);
      const auto* res = res_stmt.as<BlockNode>();
      ICHECK(res != nullptr);
      ICHECK_EQ(stmt_stack_.size(), 1);
      std::swap(stmt_stack_, tmp_stmt_stack_);

      ObjectPtr<BlockNode> n = CopyOnWrite(res);
      std::vector<Buffer> allocations;
      for (const Buffer& allocation : res->alloc_buffers) {
        allocations.emplace_back(allocation);
      }
      const std::vector<Buffer>& new_allos = new_allocations_[GetRef<Block>(op)];
      for (const Buffer& allocation : new_allos) {
        allocations.emplace_back(allocation);
      }
      n->alloc_buffers = allocations;

      return Stmt(n);
    } else if (status == kMutatingBlock_nor_red) {
      // Mutate body, init, reads and writes.
      ObjectPtr<BlockNode> n = CopyOnWrite(op);
      // 1. Mutate body.
      ICHECK(op->body.as<BufferStoreNode>() != nullptr);
      Stmt body = this->VisitStmt(op->body);

      // 2. Mutate init.
      ICHECK(op->init.defined());
      ICHECK(op->init.value().as<BufferStoreNode>() != nullptr);
      Stmt init = this->VisitStmt(op->init.value());

      // 3. Mutate reads.
      ICHECK(normal_reduce.defined());
      std::vector<BufferRegion> reads;
      for (const BufferRegion& region : op->reads) {
        if (region->buffer.same_as(write_buffer)) {
          reads.emplace_back(BufferRegion(normal_reduce.value(), {Range(0, 1)}));
        } else {
          reads.emplace_back(region);
        }
      }

      // 4. Mutate writes.
      ICHECK_EQ(op->writes.size(), 1);
      ICHECK(op->writes[0]->buffer.same_as(write_buffer));

      n->body = body;
      n->init = NullOpt;
      n->reads = reads;
      n->writes = {BufferRegion(normal_reduce.value(), {Range(0, 1)})};
      n->name_hint = nor_red_name;
      return SeqStmt({init, Stmt(n)});
    } else if (status == kMutatingBlock_red_tmp) {
      ObjectPtr<BlockNode> n = CopyOnWrite(op);

      // 1. reads: remove the original write buffer
      std::vector<BufferRegion> reads;
      for (const BufferRegion& region : op->reads) {
        if (!region->buffer.same_as(write_buffer)) {
          reads.emplace_back(region);
        }
      }
      n->reads = reads;

      // 2. writes: reduce_temp[0]
      ICHECK(reduce_temp.defined());
      n->writes = {BufferRegion(reduce_temp.value(), {Range(0, 1)})};

      // 3. body: red_tmp_block_body
      // 4. init: NullOpt
      // 5. tag: red_tmp_name
      n->body = red_tmp_block_body.value();
      n->init = NullOpt;
      n->name_hint = red_tmp_name;
      return Stmt(n);
    } else {
      LOG(FATAL);
      throw;
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) override {
    ICHECK_EQ(status, kDetecting);
    const auto* block_op = op->block.as<BlockNode>();

    // Step 1. Check whether it is a reduction block.
    if (!block_op->init.defined()) {
      return StmtMutator::VisitStmt_(op);
    }

    // Step 2. Mark the binding values of reduction IterVars "reduction relative".
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reduction_relative_;
    for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
      const IterVar& block_var = block_op->iter_vars[i];
      const PrimExpr& binding_value = op->iter_values[i];
      ICHECK(block_var.as<IterVarNode>() != nullptr);
      ICHECK(binding_value.as<PrimExprNode>() != nullptr);

      if (block_var->iter_type == kCommReduce) {
        PreOrderVisit(binding_value, [&reduction_relative_](const ObjectRef& node) {
          if (const auto* var = node.as<VarNode>()) {
            reduction_relative_.insert(GetRef<Var>(var));
            return false;
          }
          return true;
        });
      }
    }

    // Step 3. Check whether any of the reduction relative loops is bound to threadIdx.
    //         If so, allreduce is needed.
    bool appear = false;  // Whether a reduction relative loop is met.
    bool deep_normal_loop = false;
    size_t num_bound_rela = 0;
    size_t num_tot_rela = 0;
    for (const Stmt& stmt : stmt_stack_) {
      const auto* loop = stmt.as<ForNode>();
      if (loop == nullptr) {
        continue;
      }
      if (reduction_relative_.find(loop->loop_var) == reduction_relative_.end()) {
        // This loop var is not reduction relative.
        if (appear) {
          deep_normal_loop = true;
        }
        continue;
      }

      appear = true;
      num_tot_rela++;
      std::string thread_tag =
          loop->thread_binding.defined() ? loop->thread_binding.value()->thread_tag : "";
      if (thread_tag.substr(0, 9) == "threadIdx") {
        ICHECK(thread_tag == "threadIdx.x" || thread_tag == "threadIdx.y" ||
               thread_tag == "threadIdx.z");
        num_bound_rela++;
      }
    }
    CHECK_LE(num_bound_rela, num_tot_rela);
    if (num_bound_rela == 0) {
      // None of the reduction relative loops is bound to threadIdx.
      return StmtMutator::VisitStmt_(op);
    }

    // Step 4. Check whether there is a not-reduction-relative loop is deeper than some reduction
    //         relative loops. In this case, allreduce cannot be supported.
    ICHECK(!deep_normal_loop)
        << "Normal loops should not be deeper than a reduction loop bound to threadIdx.";

    // Step 5. If the block has multiple write region, allreduce cannot be supported.
    ICHECK_EQ(block_op->writes.size(), 1)
        << "The block should not have multiple write region when allreduce is needed";

    // Step 6. If one of block_op->init or block_op->body is not BufferStore, allreduce cannot
    //         be supported.
    const auto* init_body = block_op->init.value().as<BufferStoreNode>();
    const auto* update_body = block_op->body.as<BufferStoreNode>();
    ICHECK(init_body && update_body)
        << R"(The "init" and "body" should be BufferStore when allreduce is needed.)";
    ICHECK(init_body->buffer.same_as(update_body->buffer))
        << R"(The write buffer of "init" and "body" should be the same.)";

    // Step 7. If the reduction can not be represented by a CommReducer, allreduce cannot
    //         be supported.
    CommReducer reducer;
    PrimExpr combiner_lhs, combiner_rhs;
    std::tie(reducer, combiner_lhs, combiner_rhs) =
        GetReducerAndCombinerLhsRhs(init_body->value, GetRef<BufferStore>(update_body));
    PrimExpr update_value = combiner_rhs;

    const bool need_normal_reduce = num_bound_rela < num_tot_rela;  // In this case, buffer
    // normal_reduce is needed.

    // Step 8. Take the stmt above the first reduction relative loop.
    size_t par_idx;
    CHECK_GE(stmt_stack_.size(), 1);
    for (par_idx = 0; par_idx + 1 < stmt_stack_.size(); ++par_idx) {
      const auto* loop = stmt_stack_[par_idx + 1].as<ForNode>();
      if (loop == nullptr) {
        continue;
      }
      if (reduction_relative_.find(loop->loop_var) != reduction_relative_.end()) {
        break;
      }
    }
    CHECK_LT(par_idx + 1, stmt_stack_.size());
    const Stmt& par_stmt = stmt_stack_[par_idx];
    auto red_loop = Downcast<For>(stmt_stack_[par_idx + 1]);
    const auto* top_block = stmt_stack_[0].as<BlockNode>();
    ICHECK(top_block != nullptr);

    // Step 9. Create buffers of normal_reduce and reduce_temp.
    std::vector<Buffer>& allos = bufs_to_allo_[par_stmt][red_loop];
    std::vector<Buffer>& allocations_ = new_allocations_[GetRef<Block>(top_block)];
    write_buffer = block_op->writes[0]->buffer;

    DataType dtype = write_buffer->dtype;
    nor_red_name = "normal_reduce_temp" + std::to_string(allreduce_id);
    red_tmp_name = "reduce_temp" + std::to_string(allreduce_id);
    if (need_normal_reduce) {
      normal_reduce = AddBufferAllocation(nor_red_name, allos, allocations_, dtype);
    }
    reduce_temp = AddBufferAllocation(red_tmp_name, allos, allocations_, dtype);
    allreduce_id++;

    std::string block_name = block_op->name_hint;
    if (need_normal_reduce) {
      // Step a. Mutate the original block if normal_reduce is needed.
      status = kMutatingBlock_nor_red;
      Stmt mutate_res = this->VisitStmt(op->block);
      ICHECK(mutate_res.as<SeqStmtNode>() != nullptr);
      SeqStmt mutate_stmts = Downcast<SeqStmt>(mutate_res);
      ICHECK(mutate_stmts->seq.size() == 2);
      ICHECK(mutate_stmts->seq[1].as<BlockNode>() != nullptr);

      Block reduction_block = Downcast<Block>(mutate_stmts->seq[1]);
      ObjectPtr<BlockRealizeNode> n = CopyOnWrite(op);
      n->block = reduction_block;
      status = kDetecting;

      ICHECK(mutate_stmts->seq[0].as<BufferStoreNode>() != nullptr);
      BufferStore reduction_init = Downcast<BufferStore>(mutate_stmts->seq[0]);
      std::vector<BufferStore>& inits = inits_to_add_[par_stmt][red_loop];
      inits.emplace_back(reduction_init);

      // Step b. Create a block/blockRealize: normal_reduce -> reduce_temp.
      ICHECK(normal_reduce.defined());
      Array<BufferRegion> reads = {BufferRegion(normal_reduce.value(), {Range(0, 1)})};

      ICHECK(reduce_temp.defined());
      Array<BufferRegion> writes = {BufferRegion(reduce_temp.value(), {Range(0, 1)})};

      std::vector<For>& loops = loops_to_bind_[par_stmt][red_loop];
      std::vector<PrimExpr> reduce_args;
      reduce_args.emplace_back(make_const(DataType::UInt(32), static_cast<uint32_t>(1)));
      reduce_args.emplace_back(BufferLoad(normal_reduce.value(), {0}));
      reduce_args.emplace_back(const_true());
      reduce_args.emplace_back(reduce_temp.value()->data);
      for (size_t i = par_idx + 1; i < stmt_stack_.size(); ++i) {
        const auto* loop = stmt_stack_[i].as<ForNode>();
        ICHECK(loop != nullptr);
        std::string thread_tag =
            loop->thread_binding.defined() ? loop->thread_binding.value()->thread_tag : "";
        if (thread_tag.substr(0, 9) == "threadIdx") {
          reduce_args.emplace_back(loop->loop_var);
          loops.emplace_back(GetRef<For>(loop));
          already_bound_loop_vars_.insert(loop->loop_var);
        }
      }
      PrimExpr call = Call(DataType::Handle(), tir::builtin::tvm_thread_allreduce(), reduce_args);
      Stmt body0 = Evaluate(call);
      body0 = AttrStmt(reducer, tir::attr::reduce_scope, make_zero(DataType::Handle()), body0);
      body0 = Block({}, reads, writes, red_tmp_name, body0);
      body0 = BlockRealize({}, const_true(), GetRef<Block>(body0.as<BlockNode>()));

      // Step c. Create block/blockRealize: reduce_temp -> the original write buffer.
      std::vector<IterVar> iter_vars;
      std::vector<PrimExpr> iter_values;
      ICHECK_EQ(block_op->iter_vars.size(), op->iter_values.size());
      for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
        const auto* iter_var = block_op->iter_vars[i].as<IterVarNode>();
        const auto* value = op->iter_values[i].as<PrimExprNode>();
        ICHECK(iter_var != nullptr);
        ICHECK(value != nullptr);
        if (iter_var->iter_type == kCommReduce) {
          continue;
        }
        auto new_iter_var = make_object<IterVarNode>(*iter_var);
        new_iter_var->var = Var(make_object<VarNode>(*iter_var->var.get()));
        iter_vars.emplace_back(IterVar(new_iter_var));
        iter_values.emplace_back(GetRef<PrimExpr>(value));
      }

      reads = {BufferRegion(reduce_temp.value(), {Range(0, 1)})};

      writes.clear();
      for (const BufferRegion& write : block_op->writes) {
        auto new_write = make_object<BufferRegionNode>(*write.as<BufferRegionNode>());
        writes.push_back(GetRef<BufferRegion>(new_write.get()));
      }

      // Add store predicate.
      PrimExpr predicate = const_true();
      for (size_t i = par_idx + 1; i < stmt_stack_.size(); ++i) {
        const auto* loop = stmt_stack_[i].as<ForNode>();
        ICHECK(loop != nullptr);
        std::string thread_tag =
            loop->thread_binding.defined() ? loop->thread_binding.value()->thread_tag : "";
        if (thread_tag.substr(0, 9) == "threadIdx") {
          predicate = And(predicate, EQ(loop->loop_var, loop->min));
        }
      }

      Stmt body1 =
          BufferStore(write_buffer, BufferLoad(reduce_temp.value(), {0}), update_body->indices);
      body1 = Block(iter_vars, reads, writes, block_name, body1);
      body1 = BlockRealize(iter_values, predicate, GetRef<Block>(body1.as<BlockNode>()));

      // Step d. Append the stmts above to the list.
      std::vector<Stmt>& new_stmts_ = stmts_to_append_[par_stmt][red_loop];
      new_stmts_.emplace_back(body0);
      new_stmts_.emplace_back(body1);

      return Stmt(n);
    } else {
      ICHECK(reduce_temp.defined());
      // Step a. Mutate op and block_op to become the original read buffer -> reduce_temp.
      std::vector<For>& loops = loops_to_bind_[par_stmt][red_loop];
      std::vector<PrimExpr> reduce_args;
      std::unordered_map<const VarNode*, PrimExpr> loop_var_map_;
      reduce_args.emplace_back(make_const(DataType::UInt(32), static_cast<uint32_t>(1)));
      reduce_args.emplace_back(update_value);
      reduce_args.emplace_back(const_true());
      reduce_args.emplace_back(reduce_temp.value()->data);
      for (size_t i = par_idx + 1; i < stmt_stack_.size(); ++i) {
        const auto* loop = stmt_stack_[i].as<ForNode>();
        ICHECK(loop != nullptr);
        reduce_args.emplace_back(loop->loop_var);
        loops.emplace_back(GetRef<For>(loop));
        already_bound_loop_vars_.insert(loop->loop_var);
      }
      PrimExpr call = Call(DataType::Handle(), tir::builtin::tvm_thread_allreduce(), reduce_args);
      ICHECK(!red_tmp_block_body.defined());
      red_tmp_block_body = Evaluate(call);
      red_tmp_block_body = AttrStmt(reducer, tir::attr::reduce_scope,
                                    make_zero(DataType::Handle()), red_tmp_block_body.value());

      status = kMutatingBlock_red_tmp;
      Block reduction_block = Downcast<Block>(this->VisitStmt(op->block));
      ObjectPtr<BlockRealizeNode> n = CopyOnWrite(op);
      n->block = reduction_block;
      n->predicate = const_true();
      red_tmp_block_body = NullOpt;
      status = kDetecting;

      // Step b. Create block/blockRealize: reduce_temp -> the original write buffer.
      std::vector<IterVar> iter_vars;
      std::vector<PrimExpr> iter_values;
      ICHECK_EQ(block_op->iter_vars.size(), op->iter_values.size());
      for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
        const auto* iter_var = block_op->iter_vars[i].as<IterVarNode>();
        const auto* value = op->iter_values[i].as<PrimExprNode>();
        ICHECK(iter_var != nullptr);
        ICHECK(value != nullptr);
        if (iter_var->iter_type == kCommReduce) {
          continue;
        }
        auto new_iter_var = make_object<IterVarNode>(*iter_var);
        new_iter_var->var = Var(make_object<VarNode>(*iter_var->var.get()));
        iter_vars.emplace_back(IterVar(new_iter_var));
        iter_values.emplace_back(GetRef<PrimExpr>(value));
      }

      Array<BufferRegion> reads = {BufferRegion(reduce_temp.value(), {Range(0, 1)})};

      std::vector<BufferRegion> writes;
      for (const BufferRegion& write : block_op->writes) {
        auto new_write = make_object<BufferRegionNode>(*write.as<BufferRegionNode>());
        writes.push_back(GetRef<BufferRegion>(new_write.get()));
      }

      // Add store predicate.
      PrimExpr predicate = const_true();
      for (size_t i = par_idx + 1; i < stmt_stack_.size(); ++i) {
        const auto* loop = stmt_stack_[i].as<ForNode>();
        ICHECK(loop != nullptr);
        predicate = And(predicate, EQ(loop->loop_var, loop->min));
      }

      Stmt body =
          BufferStore(write_buffer, BufferLoad(reduce_temp.value(), {0}), update_body->indices);
      body = Block(iter_vars, reads, writes, block_name, body);
      body = BlockRealize(iter_values, predicate, GetRef<Block>(body.as<BlockNode>()));

      // Step c. Append the stmt above to the list.
      std::vector<Stmt>& new_stmts_ = stmts_to_append_[par_stmt][red_loop];
      new_stmts_.emplace_back(body);

      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    if (status == kMutatingBlock_nor_red) {
      // Mutate buffer and indices.
      ObjectPtr<BufferLoadNode> n = make_object<BufferLoadNode>(*op);

      if (op->buffer.same_as(write_buffer)) {
        n->buffer = normal_reduce.value();
        n->indices = {0};
      }
      return PrimExpr(n);
    } else {
      return ExprMutator::VisitExpr_(op);
    }
  }

  /*! Save the bound loops whose corresponding AttrStmt are already added into the ir. */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> already_bound_loop_vars_;

 private:
  /*! \brief Three states indicating the status of the mutating process. */
  enum MutatorStatus { kDetecting, kMutatingBlock_nor_red, kMutatingBlock_red_tmp };
  MutatorStatus status = kDetecting;

  /*! \brief A stack recording the statements above the current statement. */
  std::deque<Stmt> stmt_stack_;

  /*! \brief Total number of allreduce. Name of normal_reduce_temp and reduce_temp. */
  size_t allreduce_id = 0;
  std::string nor_red_name, red_tmp_name;

  /*! \brief Buffers of normal_reduce_temp and reduce_temp. */
  Optional<Buffer> normal_reduce = NullOpt;
  Optional<Buffer> reduce_temp = NullOpt;

  /*! \brief The write buffer of the original block. */
  Buffer write_buffer;

  /*! \brief The block of reduce_temp when normal_reduce_temp is not needed. */
  Optional<Stmt> red_tmp_block_body = NullOpt;

  /*! \brief The map/set to save the statements to be inserted. */
  std::unordered_map<Block, std::vector<Buffer>, ObjectPtrHash, ObjectPtrEqual> new_allocations_;
  std::unordered_map<
      Stmt, std::unordered_map<For, std::vector<BufferStore>, ObjectPtrHash, ObjectPtrEqual>,
      ObjectPtrHash, ObjectPtrEqual>
      inits_to_add_;
  std::unordered_map<Stmt,
                     std::unordered_map<For, std::vector<Stmt>, ObjectPtrHash, ObjectPtrEqual>,
                     ObjectPtrHash, ObjectPtrEqual>
      stmts_to_append_;
  std::unordered_map<Stmt, std::unordered_map<For, std::vector<For>, ObjectPtrHash, ObjectPtrEqual>,
                     ObjectPtrHash, ObjectPtrEqual>
      loops_to_bind_;
  std::unordered_map<Stmt,
                     std::unordered_map<For, std::vector<Buffer>, ObjectPtrHash, ObjectPtrEqual>,
                     ObjectPtrHash, ObjectPtrEqual>
      bufs_to_allo_;

  static Buffer AddBufferAllocation(const std::string& name, std::vector<Buffer>& allos,
                                    std::vector<Buffer>& allocations_, const DataType& dtype) {
    Var var(name, PointerType(PrimType(dtype),"local"));
    Buffer buf(var, dtype, {1}, {1}, PrimExpr(), name, 0, 0, kDefault);

    allos.emplace_back(buf);
    allocations_.emplace_back(buf);
    return buf;
  }

  void AddStatements(const Stmt& op_stmt, const Stmt& loop_stmt, const Stmt& stmt_ori, Stmt& stmt) {
    struct LaunchedThreadRemover : public StmtMutator {
      Stmt VisitStmt_(const ForNode* op) final {
        Stmt stmt = StmtMutator::VisitStmt_(op);
        For loop = Downcast<For>(stmt);
        return loop->kind == ForKind::kThreadBinding ? loop->body : loop;
      }

      Stmt VisitStmt_(const BlockRealizeNode* block_realize) final {
        return GetRef<Stmt>(block_realize);
      }
    };

    const auto* loop = loop_stmt.as<ForNode>();
    if (loop != nullptr) {
      For loop_stmt_ = Downcast<For>(loop_stmt);
      const std::vector<Stmt>& new_stmts_ = stmts_to_append_[op_stmt][loop_stmt_];
      const std::vector<For>& loops = loops_to_bind_[op_stmt][loop_stmt_];
      const std::vector<BufferStore>& inits = inits_to_add_[op_stmt][loop_stmt_];
      const std::vector<Buffer>& allos = bufs_to_allo_[op_stmt][loop_stmt_];
      if (!new_stmts_.empty()) {
        std::vector<Stmt> stmts;
        // Add init to the very beginning.
        if (!inits.empty()) {
          ICHECK_EQ(inits.size(), 1);
          stmts.emplace_back(inits[0]);
        }
        // Append the original statement and the new statements.
        stmts.emplace_back(LaunchedThreadRemover()(stmt_ori));
        for (const Stmt& stmt_ : new_stmts_) {
          stmts.emplace_back(stmt_);
        }
        stmt = SeqStmt(stmts);
        // Wrap the result with allocation statements.
        ICHECK(!allos.empty());
        for (auto it = allos.rbegin(); it != allos.rend(); it++) {
          Buffer allo = *it;
          stmt = Allocate(allo->data, allo->dtype, {1}, const_true(), stmt);
          std::string scope = allo.scope();
          stmt = AttrStmt(allo->data, attr::storage_scope, StringImm(scope), stmt);
        }
        // Wrap the result with loop binding attributes.
        ICHECK(!loops.empty());
        for (auto it = loops.rbegin(); it != loops.rend(); it++) {
          For loop_ = *it;
          loop_.CopyOnWrite()->body = stmt;
          stmt = loop_;
        }
      }
    }
  }
};

PrimFunc AllreduceTransform(PrimFunc f) {
  if (!f->body->IsInstance<BlockRealizeNode>()) {
    return f;
  }
  PrimFuncNode* fptr = f.CopyOnWrite();

  // Resolve allreduce.
  AllReduceTransformer allreduce_transformer;
  fptr->body = allreduce_transformer(fptr->body);

  return f;
}

namespace transform {

Pass AllreduceTransform() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return AllreduceTransform(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.AllreduceTransform", {});
}

TVM_REGISTER_GLOBAL("tir.transform.AllreduceTransform").set_body_typed(AllreduceTransform);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
