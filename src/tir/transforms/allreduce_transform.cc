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
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

/*!
 * \brief Detect allreduce and then transform.
 */
class AllReduceTransformer : public StmtExprMutator {
 public:
  AllReduceTransformer() = default;

 public:
#define TVM_ALLREDUCE_VISIT_SIMPLE_BODY(Type)                                                      \
  Stmt VisitStmt_(const Type* op) override {                                                       \
    if (status == kDetecting) {                                                                    \
      stmt_stack_.push_back(GetRef<Stmt>(op));                                                     \
      Stmt res_stmt = StmtMutator::VisitStmt_(op);                                                 \
      const auto* res = res_stmt.as<Type>();                                                       \
      CHECK(res != nullptr);                                                                       \
      CHECK(!stmt_stack_.empty()) << "Size of stmt_stack_ is expected to be positive, but it is 0";\
      stmt_stack_.pop_back();                                                                      \
                                                                                                   \
      ObjectPtr<Type> n = CopyOnWrite(res);                                                        \
      AddStatements(GetRef<Stmt>(op), op->body, res->body, n->body);                               \
      return Stmt(n);                                                                              \
    } else {                                                                                       \
      return StmtMutator::VisitStmt_(op);                                                          \
    }                                                                                              \
  }

  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(AttrStmtNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(LetStmtNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(ForNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(AllocateNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(BufferRealizeNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(AssertStmtNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(ProducerRealizeNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(LoopNode);

#undef TVM_ALLREDUCE_VISIT_SIMPLE_BODY

  Stmt VisitStmt_(const IfThenElseNode* op) override {
    if (status == kDetecting) {
      stmt_stack_.push_back(GetRef<Stmt>(op));
      Stmt res_stmt = StmtMutator::VisitStmt_(op);
      const auto* res = res_stmt.as<IfThenElseNode>();
      CHECK(res != nullptr);
      CHECK(!stmt_stack_.empty()) << "Size of stmt_stack_ is expected to be positive, but it is 0";
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
      stmt_stack_.push_back(GetRef<Stmt>(op));
      Stmt res_stmt = StmtMutator::VisitStmt_(op);
      const auto* res = res_stmt.as<SeqStmtNode>();
      CHECK(res != nullptr);
      CHECK(!stmt_stack_.empty()) << "Size of stmt_stack_ is expected to be positive, but it is 0";
      stmt_stack_.pop_back();

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
      CHECK(!stmt_stack_.empty()) << "Size of stmt_stack_ is expected to be positive, but it is 0";
      stmt_stack_.pop_back();
      return res;
    } else if (status == kMutatingBlock_nor_red) {
      // Mutate buffer, indices and value
      CHECK(op->buffer.same_as(write_buffer));
      CHECK(normal_reduce.defined());

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
      CHECK(res != nullptr);
      CHECK_EQ(stmt_stack_.size(), 1);
      std::swap(stmt_stack_, tmp_stmt_stack_);

      ObjectPtr<BlockNode> n = CopyOnWrite(res);
      std::vector<BufferAllocate> allocations;
      for (const BufferAllocate& allocation : res->allocations) {
        allocations.emplace_back(allocation);
      }
      const std::vector<BufferAllocate>& new_allos = new_allocations_[GetRef<Block>(op)];
      for (const BufferAllocate& allocation : new_allos) {
        allocations.emplace_back(allocation);
      }
      n->allocations = allocations;

      return Stmt(n);
    } else if (status == kMutatingBlock_nor_red) {
      // Mutate body, init, reads and writes.
      ObjectPtr<BlockNode> n = CopyOnWrite(op);
      // 1. Mutate body.
      CHECK(op->body.as<BufferStoreNode>() != nullptr);
      Stmt body = this->VisitStmt(op->body);

      // 2. Mutate init.
      CHECK(op->init.defined());
      CHECK(op->init.value().as<BufferStoreNode>() != nullptr);
      Stmt init = this->VisitStmt(op->init.value());

      // 3. Mutate reads.
      CHECK(normal_reduce.defined());
      std::vector<TensorRegion> reads;
      for (const TensorRegion& region : op->reads) {
        if (region->buffer.same_as(write_buffer)) {
          reads.emplace_back(TensorRegion(normal_reduce.value(), {Range(0, 1)}));
        } else {
          reads.emplace_back(region);
        }
      }

      // 4. Mutate writes.
      CHECK_EQ(op->writes.size(), 1);
      CHECK(op->writes[0]->buffer.same_as(write_buffer));

      n->body = body;
      n->init = NullOpt;
      n->reads = reads;
      n->writes = {TensorRegion(normal_reduce.value(), {Range(0, 1)})};
      n->tag = nor_red_name;
      return SeqStmt({init, Stmt(n)});
    } else if (status == kMutatingBlock_red_tmp) {
      ObjectPtr<BlockNode> n = CopyOnWrite(op);

      // 1. reads: remove the original write buffer
      std::vector<TensorRegion> reads;
      for (const TensorRegion& region : op->reads) {
        if (!region->buffer.same_as(write_buffer)) {
          reads.emplace_back(region);
        }
      }
      n->reads = reads;

      // 2. writes: reduce_temp[0]
      CHECK(reduce_temp.defined());
      n->writes = {TensorRegion(reduce_temp.value(), {Range(0, 1)})};

      // 3. body: red_tmp_block_body
      // 4. init: NullOpt
      // 5. tag: red_tmp_name
      n->body = red_tmp_block_body.value();
      n->init = NullOpt;
      n->tag = red_tmp_name;
      return Stmt(n);
    } else {
      LOG(FATAL);
      throw;
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) override {
    CHECK_EQ(status, kDetecting);
    const auto* block_op = op->block.as<BlockNode>();

    // Step 1. Check whether it is a reduction block.
    if (!block_op->init.defined()) {
      return StmtMutator::VisitStmt_(op);
    }

    // Step 2. Mark the binding values of reduction IterVars "reduction relative".
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reduction_relative_;
    for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
      const IterVar& block_var = block_op->iter_vars[i];
      const PrimExpr& binding_value = op->binding_values[i];
      CHECK(block_var.as<IterVarNode>() != nullptr);
      CHECK(binding_value.as<PrimExprNode>() != nullptr);

      if (block_var->iter_type == kCommReduce) {
        PreOrderVisit(binding_value, [&reduction_relative_] (const ObjectRef& node) {
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
    bool appear = false; // Whether a reduction relative loop is met.
    bool deep_normal_loop = false;
    size_t num_bound_rela = 0;
    size_t num_tot_rela = 0;
    for (const Stmt& stmt : stmt_stack_) {
      const auto* loop = stmt.as<LoopNode>();
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
      for (const Annotation& annotation : loop->annotations) {
        if (annotation->attr_key == attr::loop_type) {
          std::string thread_tag = Downcast<StringImm>(annotation->value)->value;
          if (thread_tag.substr(0, 9) == "threadIdx") {
            CHECK(thread_tag == "threadIdx.x" || thread_tag == "threadIdx.y"
                  || thread_tag == "threadIdx.z");
            num_bound_rela++;
          }
        }
      }
    }
    CHECK_LE(num_bound_rela, num_tot_rela);
    if (num_bound_rela == 0) {
      // None of the reduction relative loops is bound to threadIdx.
      return StmtMutator::VisitStmt_(op);
    }

    // Step 4. Check whether there is a not-reduction-relative loop is deeper than some reduction
    //         relative loops. In this case, allreduce cannot be supported.
    CHECK(!deep_normal_loop)
        << "Normal loops should not be deeper than a reduction loop bound to threadIdx.";

    // Step 5. If the block has multiple write region, allreduce cannot be supported.
    CHECK_EQ(block_op->writes.size(), 1)
        << "The block should not have multiple write region when allreduce is needed";

    // Step 6. If one of block_op->init or block_op->body is not BufferStore, allreduce cannot
    //         be supported.
    const auto* init_body = block_op->init.value().as<BufferStoreNode>();
    const auto* update_body = block_op->body.as<BufferStoreNode>();
    CHECK(init_body && update_body)
        << R"(The "init" and "body" should be BufferStore when allreduce is needed.)";
    CHECK(init_body->buffer.same_as(update_body->buffer))
        << R"(The write buffer of "init" and "body" should be the same.)";

    // Step 7. If the reduction can not be represented by a CommReducer, allreduce cannot
    //         be supported.
    Optional<CommReducer> optional_reducer;
    Optional<PrimExpr> reducer_lhs, reducer_rhs;
    CommReducer::FromInitUpdate(init_body->value, GetRef<BufferStore>(update_body),
                                optional_reducer, reducer_lhs, reducer_rhs, Span());
    CHECK(optional_reducer.defined())
        << "Cannot find a commutative reducer when allreduce is needed.";
    const auto* reducer = optional_reducer.value().as<CommReducerNode>();
    CHECK(reducer_lhs.defined() && reducer_rhs.defined());
    PrimExpr update_value = reducer_rhs.value();


    const bool need_normal_reduce = num_bound_rela < num_tot_rela; // In this case, buffer
    // normal_reduce is needed.

    // Step 8. Take the stmt above the first reduction relative loop.
    size_t par_idx;
    CHECK_GE(stmt_stack_.size(), 1);
    for (par_idx = 0; par_idx + 1 < stmt_stack_.size(); ++par_idx) {
      const auto* loop = stmt_stack_[par_idx + 1].as<LoopNode>();
      if (loop == nullptr) {
        continue;
      }
      if (reduction_relative_.find(loop->loop_var) != reduction_relative_.end()) {
        break;
      }
    }
    CHECK_LT(par_idx + 1, stmt_stack_.size());
    const Stmt& par_stmt = stmt_stack_[par_idx];
    Loop red_loop = GetRef<Loop>(stmt_stack_[par_idx + 1].as<LoopNode>());
    const auto* top_block = stmt_stack_[0].as<BlockNode>();
    CHECK(top_block != nullptr);

    // Step 9. Create buffers of normal_reduce and reduce_temp.
    std::vector<BufferAllocate>& allos = bufs_to_allo_[par_stmt][red_loop];
    std::vector<BufferAllocate>& allocations_ = new_allocations_[GetRef<Block>(top_block)];
    write_buffer = block_op->writes[0]->buffer;

    DataType dtype = write_buffer->dtype;
    nor_red_name = "normal_reduce_temp" + std::to_string(allreduce_id);
    red_tmp_name = "reduce_temp" + std::to_string(allreduce_id);
    if (need_normal_reduce) {
      normal_reduce = AddBufferAllocation(nor_red_name, allos, allocations_, dtype);
    }
    reduce_temp = AddBufferAllocation(red_tmp_name, allos, allocations_, dtype);
    allreduce_id++;

    std::string block_name = block_op->tag;
    std::string exec_scope = op->exec_scope;
    if (need_normal_reduce) {
      // Step a. Mutate the original block if normal_reduce is needed.
      status = kMutatingBlock_nor_red;
      Stmt mutate_res = this->VisitStmt(op->block);
      CHECK(mutate_res.as<SeqStmtNode>() != nullptr);
      SeqStmt mutate_stmts = Downcast<SeqStmt>(mutate_res);
      CHECK(mutate_stmts->seq.size() == 2);
      CHECK(mutate_stmts->seq[1].as<BlockNode>() != nullptr);

      Block reduction_block = Downcast<Block>(mutate_stmts->seq[1]);
      ObjectPtr<BlockRealizeNode> n = CopyOnWrite(op);
      n->block = reduction_block;
      status = kDetecting;

      CHECK(mutate_stmts->seq[0].as<BufferStoreNode>() != nullptr);
      BufferStore reduction_init = Downcast<BufferStore>(mutate_stmts->seq[0]);
      std::vector<BufferStore>& inits = inits_to_add_[par_stmt][red_loop];
      inits.emplace_back(reduction_init);

      // Step b. Create a block/blockRealize: normal_reduce -> reduce_temp.
      CHECK(normal_reduce.defined());
      Array<TensorRegion> reads = {TensorRegion(normal_reduce.value(), {Range(0, 1)})};

      CHECK(reduce_temp.defined());
      Array<TensorRegion> writes = {TensorRegion(reduce_temp.value(), {Range(0, 1)})};

      std::vector<Loop>& loops = loops_to_bind_[par_stmt][red_loop];
      std::vector<PrimExpr> reduce_args;
      reduce_args.emplace_back(make_const(DataType::UInt(32), static_cast<uint32_t>(1)));
      reduce_args.emplace_back(BufferLoad(normal_reduce.value(), {0}));
      reduce_args.emplace_back(const_true());
      reduce_args.emplace_back(reduce_temp.value()->data);
      for (size_t i = par_idx + 1; i < stmt_stack_.size(); ++i) {
        const auto* loop = stmt_stack_[i].as<LoopNode>();
        CHECK(loop != nullptr);
        for (const Annotation& annotation : loop->annotations) {
          if (annotation->attr_key == attr::loop_type) {
            std::string thread_tag = Downcast<StringImm>(annotation->value)->value;
            if (thread_tag.substr(0, 9) == "threadIdx") {
              reduce_args.emplace_back(loop->loop_var);
              loops.emplace_back(GetRef<Loop>(loop));
              already_bound_loop_vars_.insert(loop->loop_var);
            }
          }
        }
      }
      PrimExpr call = Call(DataType::Handle(), tir::builtin::tvm_thread_allreduce(), reduce_args);
      Stmt body0 = Evaluate(call);
      body0 = AttrStmt(GetRef<CommReducer>(reducer), tir::attr::reduce_scope,
                       make_zero(DataType::Handle()), body0);
      body0 = Block({}, reads, writes, body0, {}, {}, red_tmp_name, NullOpt);
      body0 = BlockRealize({}, const_true(), GetRef<Block>(body0.as<BlockNode>()), "");

      // Step c. Create block/blockRealize: reduce_temp -> the original write buffer.
      std::vector<IterVar> iter_vars;
      std::vector<PrimExpr> binding_values;
      CHECK_EQ(block_op->iter_vars.size(), op->binding_values.size());
      for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
        const auto* iter_var = block_op->iter_vars[i].as<IterVarNode>();
        const auto* value = op->binding_values[i].as<PrimExprNode>();
        CHECK(iter_var != nullptr);
        CHECK(value != nullptr);
        if (iter_var->iter_type == kCommReduce) {
          continue;
        }
        ObjectPtr<IterVarNode> new_iter_var = CopyOnWrite(iter_var);
        new_iter_var->var = Var(CopyOnWrite(iter_var->var.as<VarNode>()));
        iter_vars.emplace_back(IterVar(new_iter_var));
        binding_values.emplace_back(GetRef<PrimExpr>(value));
      }

      reads = {TensorRegion(reduce_temp.value(), {Range(0, 1)})};

      writes.clear();
      for (const TensorRegion& write : block_op->writes) {
        auto new_write = make_object<TensorRegionNode>(*write.as<TensorRegionNode>());
        writes.push_back(GetRef<TensorRegion>(new_write.get()));
      }

      // Add store predicate.
      PrimExpr predicate = op->predicate;
      for (size_t i = par_idx + 1; i < stmt_stack_.size(); ++i) {
        const auto* loop = stmt_stack_[i].as<LoopNode>();
        CHECK(loop != nullptr);
        for (const Annotation& annotation : loop->annotations) {
          if (annotation->attr_key == attr::loop_type) {
            std::string thread_tag = Downcast<StringImm>(annotation->value)->value;
            if (thread_tag.substr(0, 9) == "threadIdx") {
              predicate = And(predicate, EQ(loop->loop_var, loop->min));
            }
          }
        }
      }

      Stmt body1 = BufferStore(write_buffer, BufferLoad(reduce_temp.value(), {0}),
                               update_body->indices);
      body1 = Block(iter_vars, reads, writes, body1, {}, {}, block_name, NullOpt);
      body1 = BlockRealize(binding_values, predicate,
                           GetRef<Block>(body1.as<BlockNode>()), exec_scope);

      // Step d. Append the stmts above to the list.
      std::vector<Stmt>& new_stmts_ = stmts_to_append_[par_stmt][red_loop];
      new_stmts_.emplace_back(body0);
      new_stmts_.emplace_back(body1);

      return Stmt(n);
    } else {
      CHECK(reduce_temp.defined());
      // Step a. Mutate op and block_op to become the original read buffer -> reduce_temp.
      std::vector<Loop>& loops = loops_to_bind_[par_stmt][red_loop];
      std::vector<PrimExpr> reduce_args;
      std::unordered_map<const VarNode*, PrimExpr> loop_var_map_;
      reduce_args.emplace_back(make_const(DataType::UInt(32), static_cast<uint32_t>(1)));
      reduce_args.emplace_back(update_value);
      reduce_args.emplace_back(const_true());
      reduce_args.emplace_back(reduce_temp.value()->data);
      for (size_t i = par_idx + 1; i < stmt_stack_.size(); ++i) {
        const auto* loop = stmt_stack_[i].as<LoopNode>();
        CHECK(loop != nullptr);
        reduce_args.emplace_back(loop->loop_var);
        loops.emplace_back(GetRef<Loop>(loop));
        already_bound_loop_vars_.insert(loop->loop_var);
      }
      PrimExpr call = Call(DataType::Handle(), tir::builtin::tvm_thread_allreduce(), reduce_args);
      CHECK(!red_tmp_block_body.defined());
      red_tmp_block_body = Evaluate(call);
      red_tmp_block_body = AttrStmt(GetRef<CommReducer>(reducer), tir::attr::reduce_scope,
                                    make_zero(DataType::Handle()), red_tmp_block_body.value());

      status = kMutatingBlock_red_tmp;
      Block reduction_block = Downcast<Block>(this->VisitStmt(op->block));
      ObjectPtr<BlockRealizeNode> n = CopyOnWrite(op);
      n->block = reduction_block;
      n->predicate = const_true();
      n->exec_scope = "";
      red_tmp_block_body = NullOpt;
      status = kDetecting;

      // Step b. Create block/blockRealize: reduce_temp -> the original write buffer.
      std::vector<IterVar> iter_vars;
      std::vector<PrimExpr> binding_values;
      CHECK_EQ(block_op->iter_vars.size(), op->binding_values.size());
      for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
        const auto* iter_var = block_op->iter_vars[i].as<IterVarNode>();
        const auto* value = op->binding_values[i].as<PrimExprNode>();
        CHECK(iter_var != nullptr);
        CHECK(value != nullptr);
        if (iter_var->iter_type == kCommReduce) {
          continue;
        }
        ObjectPtr<IterVarNode> new_iter_var = CopyOnWrite(iter_var);
        new_iter_var->var = Var(CopyOnWrite(iter_var->var.as<VarNode>()));
        iter_vars.emplace_back(IterVar(new_iter_var));
        binding_values.emplace_back(GetRef<PrimExpr>(value));
      }

      Array<TensorRegion> reads = {TensorRegion(reduce_temp.value(), {Range(0, 1)})};

      std::vector<TensorRegion> writes;
      for (const TensorRegion& write : block_op->writes) {
        auto new_write = make_object<TensorRegionNode>(*write.as<TensorRegionNode>());
        writes.push_back(GetRef<TensorRegion>(new_write.get()));
      }

      // Add store predicate.
      PrimExpr predicate = op->predicate;
      for (size_t i = par_idx + 1; i < stmt_stack_.size(); ++i) {
        const auto* loop = stmt_stack_[i].as<LoopNode>();
        CHECK(loop != nullptr);
        predicate = And(predicate, EQ(loop->loop_var, loop->min));
      }

      Stmt body = BufferStore(write_buffer, BufferLoad(reduce_temp.value(), {0}),
                              update_body->indices);
      body = Block(iter_vars, reads, writes, body, {}, {}, block_name, NullOpt);
      body = BlockRealize(binding_values, predicate,
                          GetRef<Block>(body.as<BlockNode>()), exec_scope);

      // Step c. Append the stmt above to the list.
      std::vector<Stmt>& new_stmts_ = stmts_to_append_[par_stmt][red_loop];
      new_stmts_.emplace_back(body);

      return Stmt(n);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) override {
    if (status == kMutatingBlock_nor_red) {
      // Mutate buffer and indices.
      ObjectPtr<BufferLoadNode> n = CopyOnWrite(op);

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
  enum MutatorStatus {
    kDetecting,
    kMutatingBlock_nor_red,
    kMutatingBlock_red_tmp
  };
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
  std::unordered_map<Block,
  std::vector<BufferAllocate>, ObjectPtrHash, ObjectPtrEqual> new_allocations_;
  std::unordered_map<Stmt, std::unordered_map<Loop, std::vector<BufferStore>,
  ObjectPtrHash, ObjectPtrEqual>, ObjectPtrHash, ObjectPtrEqual> inits_to_add_;
  std::unordered_map<Stmt, std::unordered_map<Loop, std::vector<Stmt>,
  ObjectPtrHash, ObjectPtrEqual>, ObjectPtrHash, ObjectPtrEqual> stmts_to_append_;
  std::unordered_map<Stmt, std::unordered_map<Loop, std::vector<Loop>,
  ObjectPtrHash, ObjectPtrEqual>, ObjectPtrHash, ObjectPtrEqual> loops_to_bind_;
  std::unordered_map<Stmt, std::unordered_map<Loop, std::vector<BufferAllocate>,
  ObjectPtrHash, ObjectPtrEqual>, ObjectPtrHash, ObjectPtrEqual> bufs_to_allo_;

  static Buffer AddBufferAllocation(const std::string& name,
                                    std::vector<BufferAllocate>& allos,
                                    std::vector<BufferAllocate>& allocations_,
                                    const DataType& dtype) {
    Var var(name, PointerType(PrimType(dtype)));
    Buffer buf(var, dtype, {1}, {1}, PrimExpr(), name, "local", 0, 0, kDefault);
    BufferAllocate allo(buf, "local");

    allos.emplace_back(allo);
    allocations_.emplace_back(allo);
    return buf;
  }

  void AddStatements(const Stmt& op_stmt, const Stmt& loop_stmt,
                     const Stmt& stmt_ori, Stmt& stmt) {
    const auto* loop = loop_stmt.as<LoopNode>();
    if (loop != nullptr) {
      Loop loop_stmt_ = Downcast<Loop>(loop_stmt);
      const std::vector<Stmt>& new_stmts_ = stmts_to_append_[op_stmt][loop_stmt_];
      const std::vector<Loop>& loops = loops_to_bind_[op_stmt][loop_stmt_];
      const std::vector<BufferStore>& inits = inits_to_add_[op_stmt][loop_stmt_];
      const std::vector<BufferAllocate>& allos = bufs_to_allo_[op_stmt][loop_stmt_];
      if (!new_stmts_.empty()) {
        std::vector<Stmt> stmts;
        // Add init to the very beginning.
        if (!inits.empty()) {
          CHECK_EQ(inits.size(), 1);
          stmts.emplace_back(inits[0]);
        }
        // Append the original statement and the new statements.
        stmts.emplace_back(stmt_ori);
        for (const Stmt& stmt_ : new_stmts_) {
          stmts.emplace_back(stmt_);
        }
        stmt = SeqStmt(stmts);
        // Wrap the result with allocation statements.
        CHECK(!allos.empty());
        for (auto it = allos.rbegin(); it != allos.rend(); it++) {
          BufferAllocate allo = *it;
          stmt = Allocate(allo->buffer->data, allo->buffer->dtype, {1}, const_true(), stmt);
          std::string scope = allo->scope;
          stmt = AttrStmt(allo->buffer->data, attr::storage_scope, StringImm(scope), stmt);
        }
        // Wrap the result with loop binding attributes.
        CHECK(!loops.empty());
        for (auto it = loops.rbegin(); it != loops.rend(); it++) {
          Loop loop_ = *it;
          for (const Annotation& annotation : loop_->annotations) {
            if (annotation->attr_key == attr::loop_type) {
              std::string thread_tag = Downcast<StringImm>(annotation->value)->value;
              if (thread_tag.substr(0, 9) == "threadIdx") {
                stmt = AttrStmt(IterVar(Range(loop_->min, loop_->extent), loop_->loop_var,
                                        IterVarType::kThreadIndex, thread_tag),
                                attr::thread_extent, loop_->extent, stmt);
              }
            }
          }
        }
      }
    }
  }
};

PrimFunc AllreduceTransform(PrimFunc f) {
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
