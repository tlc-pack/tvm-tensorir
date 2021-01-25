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
 * \file buffer_flatten.cc
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
#define TVM_ALLREDUCE_VISIT_NO_BODY(Type)                                                          \
  Stmt VisitStmt_(const Type* op) override {                                                       \
    if (status == kDetecting) {                                                                    \
      stmt_stack_.push_back(GetRef<Stmt>(op));                                                     \
      Stmt res = StmtMutator::VisitStmt_(op);                                                      \
      CHECK(!stmt_stack_.empty()) << "Size of stmt_stack_ is expected to be positive, but it is 0";\
      stmt_stack_.pop_back();                                                                      \
      return res;                                                                                  \
    } else {                                                                                       \
      return StmtMutator::VisitStmt_(op);                                                          \
    }                                                                                              \
  }

#define TVM_ALLREDUCE_VISIT_SIMPLE_BODY(Type)                                                      \
  Stmt VisitStmt_(const Type* op) override {                                                       \
    if (status == kDetecting) {                                                                    \
      stmt_stack_.push_back(GetRef<Stmt>(op));                                                     \
      Stmt res_stmt = StmtMutator::VisitStmt_(op);                                                 \
      const auto* res = res_stmt.as<Type>();                                                       \
      CHECK(res);                                                                                  \
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

  TVM_ALLREDUCE_VISIT_NO_BODY(StoreNode);
  TVM_ALLREDUCE_VISIT_NO_BODY(ProducerStoreNode);
  TVM_ALLREDUCE_VISIT_NO_BODY(PrefetchNode);
  TVM_ALLREDUCE_VISIT_NO_BODY(EvaluateNode);
  TVM_ALLREDUCE_VISIT_NO_BODY(BufferAllocateNode);

  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(AttrStmtNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(LetStmtNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(ForNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(AllocateNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(BufferRealizeNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(AssertStmtNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(ProducerRealizeNode);
  TVM_ALLREDUCE_VISIT_SIMPLE_BODY(LoopNode);

#undef TVM_ALLREDUCE_VISIT_SIMPLE_BODY
#undef TVM_ALLREDUCE_VISIT_NO_BODY

  Stmt VisitStmt_(const IfThenElseNode* op) override {
    if (status == kDetecting) {
      stmt_stack_.push_back(GetRef<Stmt>(op));
      Stmt res_stmt = StmtMutator::VisitStmt_(op);
      const auto* res = res_stmt.as<IfThenElseNode>();
      CHECK(res);
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
      CHECK(res);
      CHECK(!stmt_stack_.empty()) << "Size of stmt_stack_ is expected to be positive, but it is 0";
      stmt_stack_.pop_back();

      std::vector<Stmt> seq;
      for (const auto& stmt : res->seq) {
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
      CHECK(res);
      CHECK_EQ(stmt_stack_.size(), 1);
      std::swap(stmt_stack_, tmp_stmt_stack_);

      ObjectPtr<BlockNode> n = CopyOnWrite(res);
      std::vector<BufferAllocate> allocations;
      for (const auto& allocation : res->allocations) {
        allocations.emplace_back(allocation);
      }
      const std::vector<BufferAllocate>& new_allos = new_allocations_[GetRef<Block>(op)];
      for (const auto& allocation : new_allos) {
        allocations.emplace_back(allocation);
      }
      n->allocations = allocations;

      return Stmt(n);
    } else if (status == kMutatingBlock_nor_red) {
      // Mutate body, init, reads and writes.
      ObjectPtr<BlockNode> n = CopyOnWrite(op);
      // 1. Mutate body.
      CHECK(op->body.as<BufferStoreNode>());
      Stmt body = this->VisitStmt(op->body);

      // 2. Mutate init.
      CHECK(op->init);
      CHECK(op->init.value().as<BufferStoreNode>());
      Stmt init = this->VisitStmt(op->init.value());

      // 3. Mutate reads.
      CHECK(normal_reduce.defined());
      std::vector<TensorRegion> reads;
      for (const auto& region : op->reads) {
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
      for (const auto& region : op->reads) {
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
    if (block_op->init == nullptr) {
      return StmtMutator::VisitStmt_(op);
    }

    // Step 2. Mark the binding values of reduction IterVars "reduction relative".
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> reduction_relative_;
    for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
      const IterVar& block_var = block_op->iter_vars[i];
      const PrimExpr& binding_value = op->binding_values[i];
      CHECK(block_var.as<IterVarNode>());
      CHECK(binding_value.as<PrimExprNode>());

      if (block_var->iter_type != kCommReduce) {
        continue;
      }
      PreOrderVisit(binding_value, [&reduction_relative_] (const ObjectRef& node) {
        if (const auto* var = node.as<VarNode>()) {
          reduction_relative_.insert(GetRef<Var>(var));
          return false;
        }
        return true;
      });
    }

    // Step 3. Check whether any of the reduction relative loops is bound to threadIdx.
    //         If so, allreduce is needed.
    bool appear = false; // Whether a reduction relative loop is met.
    bool deep_normal_loop = false;
    size_t num_bound_rela = 0;
    size_t num_tot_rela = 0;
    for (const auto& stmt : stmt_stack_) {
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
      for (const auto& annotation : loop->annotations) {
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
    std::tuple<Optional<CommReducer>, PrimExpr, PrimExpr> reducer_res =
        FromInitUpdate(init_body->value, GetRef<BufferStore>(update_body));
    Optional<CommReducer> optional_reducer = std::get<0>(reducer_res);
    CHECK(optional_reducer.defined())
        << "Cannot find a commutative reducer when allreduce is needed.";
    const auto* reducer = optional_reducer.value().as<CommReducerNode>();
    PrimExpr update_value = std::get<2>(reducer_res);
    CHECK(update_value.as<PrimExprNode>());


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
    CHECK(top_block);

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
      CHECK(mutate_res.as<SeqStmtNode>());
      SeqStmt mutate_stmts = Downcast<SeqStmt>(mutate_res);
      CHECK(mutate_stmts->seq.size() == 2);
      CHECK(mutate_stmts->seq[1].as<BlockNode>());

      Block reduction_block = Downcast<Block>(mutate_stmts->seq[1]);
      ObjectPtr<BlockRealizeNode> n = CopyOnWrite(op);
      n->block = reduction_block;
      status = kDetecting;

      CHECK(mutate_stmts->seq[0].as<BufferStoreNode>());
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
        CHECK(loop);
        for (const auto& annotation : loop->annotations) {
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
      body0 = Block({}, reads, writes, body0, {}, {}, red_tmp_name);
      body0 = BlockRealize({}, const_true(), GetRef<Block>(body0.as<BlockNode>()), "");

      // Step c. Create block/blockRealize: reduce_temp -> the original write buffer.
      std::vector<IterVar> iter_vars;
      std::vector<PrimExpr> binding_values;
      CHECK_EQ(block_op->iter_vars.size(), op->binding_values.size());
      for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
        const auto* iter_var = block_op->iter_vars[i].as<IterVarNode>();
        const auto* value = op->binding_values[i].as<PrimExprNode>();
        CHECK(iter_var);
        CHECK(value);
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
      for (const auto& write : block_op->writes) {
        auto new_write = make_object<TensorRegionNode>(*write.as<TensorRegionNode>());
        writes.push_back(GetRef<TensorRegion>(new_write.get()));
      }

      Stmt body1 = BufferStore(write_buffer, BufferLoad(reduce_temp.value(), {0}),
                         update_body->indices);
      body1 = Block(iter_vars, reads, writes, body1, {}, {}, block_name);
      body1 = BlockRealize(binding_values, op->predicate,
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
        CHECK(loop);
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
        CHECK(iter_var);
        CHECK(value);
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
      for (const auto& write : block_op->writes) {
        auto new_write = make_object<TensorRegionNode>(*write.as<TensorRegionNode>());
        writes.push_back(GetRef<TensorRegion>(new_write.get()));
      }

      Stmt body = BufferStore(write_buffer, BufferLoad(reduce_temp.value(), {0}),
                               update_body->indices);
      body = Block(iter_vars, reads, writes, body, {}, {}, block_name);
      body = BlockRealize(binding_values, op->predicate,
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
    if (loop) {
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
        for (const auto& stmt_ : new_stmts_) {
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
          for (const auto& annotation : loop_->annotations) {
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

/*!
 * \brief Transform block with init into actual computation
 */
class ReductionTransformer : public StmtExprMutator {
 public:
  ReductionTransformer() = default;

  Stmt VisitStmt_(const BlockNode* op) override {
    Block res = Downcast<Block>(StmtMutator::VisitStmt_(op));
    if (op->init) {
      PrimExpr condition = Bool(true);
      for (const auto& var : res->iter_vars) {
        if (var->iter_type == IterVarType::kCommReduce) {
          condition = And(condition, EQ(var, var->dom->min));
        }
      }
      Stmt init = op->init.value();
      if (!is_one(condition)) init = IfThenElse(condition, init);
      res.CopyOnWrite()->body = SeqStmt::Flatten(init, op->body);
      res.CopyOnWrite()->init = NullOpt;
    }
    return std::move(res);
  }
};

/*!
 * \brief Detecting the LCA of buffer access points of
 *        buffers for calculating the realize region
 */
class LCADetector : public StmtExprVisitor {
 public:
  explicit LCADetector(const Map<Var, Buffer>& func_args) {
    for (const auto& x : func_args) {
      arg_buffers_.insert(x.second);
      buffers_lca_[x.second] = NullValue<ObjectRef>();
    }
  }

  // Update parent and depth information for each AST node

  void VisitStmt_(const LoopNode* op) final {
    Stmt n = GetRef<Stmt>(op);
    ast_scopes_info_[n] = ScopeInfo{scope_, depth_};
    ++depth_;
    std::swap(scope_, n);
    StmtExprVisitor::VisitStmt_(op);
    std::swap(scope_, n);
    --depth_;
  }

  // Update LCA when visiting BufferLoad and BufferStore
  template <typename T>
  void VisitBuffer(T op) {
    Buffer buffer = op->buffer;
    ObjectRef n = GetRef<ObjectRef>(op);
    ast_scopes_info_[n] = ScopeInfo{scope_, depth_};
    // No need to update LCA if the buffer is in the func args (function input/output buffer)
    if (arg_buffers_.count(buffer)) return;
    if (buffers_lca_.count(buffer)) {
      buffers_lca_[buffer] = LowestCommonAncestor(GetRef<ObjectRef>(op), buffers_lca_[buffer]);
    } else {
      buffers_lca_[buffer] = GetRef<ObjectRef>(op);
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    VisitBuffer(op);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode* op) final {
    VisitBuffer(op);
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief The map from Buffer to its LCA Stmt/Expr */
  std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual> buffers_lca_;
  /*! \brief The Buffer in function args */
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> arg_buffers_;

 private:
  /*! \brief The AST node information for querying LCA */
  struct ScopeInfo {
    // The parent loop node
    Stmt parent_scope;
    // The scope depth in the AST
    size_t depth;
  };

  /*! \brief The current scope initializing with Null */
  Stmt scope_{NullValue<Stmt>()};
  /*! \brief The current DFS depth */
  size_t depth_{0};
  /*! \brief The parent and depth info of each Loop/BufferLoad/BufferStore Node */
  std::unordered_map<ObjectRef, ScopeInfo, ObjectPtrHash, ObjectPtrEqual> ast_scopes_info_;

  ObjectRef LowestCommonAncestor(ObjectRef lhs, ObjectRef rhs) {
    if (!lhs.defined() || !rhs.defined()) return NullValue<ObjectRef>();
    CHECK(ast_scopes_info_.count(lhs));
    CHECK(ast_scopes_info_.count(rhs));
    while (ast_scopes_info_[lhs].depth > ast_scopes_info_[rhs].depth) {
      lhs = ast_scopes_info_[lhs].parent_scope;
    }
    while (ast_scopes_info_[lhs].depth < ast_scopes_info_[rhs].depth) {
      rhs = ast_scopes_info_[rhs].parent_scope;
    }
    while (!lhs.same_as(rhs)) {
      lhs = ast_scopes_info_[lhs].parent_scope;
      rhs = ast_scopes_info_[rhs].parent_scope;
    }
    return lhs;
  }
};

/*!
 * \brief Gather the used region of each buffers.
 */
class RegionGatherer : public StmtExprVisitor {
 public:
  RegionGatherer(
      const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca,
      const Map<Var, Buffer>& func_args)
      : buffers_lca_(buffers_lca) {
    for (const auto& arg : func_args) {
      std::vector<arith::IntSet> region;
      for (const auto& size : arg.second->shape) {
        region.push_back(arith::IntSet::FromRange(Range::FromMinExtent(0, size)));
      }
      buffers_region_[arg.second] = region;
    }
  }

  void VisitStmt_(const LoopNode* op) final {
    Loop loop = GetRef<Loop>(op);
    loop_stack_.push_back(loop);
    if (op->annotations.empty() && is_one(op->extent)) {
      unit_loops_[op->loop_var.get()] = op->min;
    }
    StmtExprVisitor::VisitStmt_(op);
    loop_stack_.pop_back();
  }

  void VisitStmt_(const BlockRealizeNode* op) final {
    const auto* block_op = op->block.as<BlockNode>();
    for (size_t i = 0; i < block_op->iter_vars.size(); ++i) {
      const auto& iter = block_op->iter_vars[i];
      const auto& v = op->binding_values[i];
      block_var_[iter->var.get()] = Substitute(Substitute(v, block_var_), unit_loops_);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BlockNode* op) final {
    for (const auto& tensor_region : op->reads) {
      VisitBufferRegion(tensor_region);
    }
    for (const auto& tensor_region : op->writes) {
      VisitBufferRegion(tensor_region);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferAllocateNode* op) final {
    std::vector<arith::IntSet> empty_region(op->buffer->shape.size(), arith::IntSet::Nothing());
    // Initialize the buffer region with empty region.
    buffers_region_[op->buffer] = empty_region;
    StmtExprVisitor::VisitStmt_(op);
  }

  /*! \brief The used region of each Buffer */
  std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>
      buffers_region_;
  /*! \brief The map from block vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> block_var_;
  /*! \brief The map from unit lopo vars to the expr value */
  std::unordered_map<const VarNode*, PrimExpr> unit_loops_;

 private:
  const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca_;

  /*! \brief The loops from the current node up to the root */
  std::vector<Loop> loop_stack_;

  void VisitBufferRegion(const TensorRegion& tensor_region) {
    auto it = buffers_region_.find(tensor_region->buffer);
    CHECK(it != buffers_region_.end());
    const auto& region = GatherRegion(tensor_region);
    auto& buffer_region = it->second;
    CHECK_EQ(buffer_region.size(), region.size());
    for (size_t i = 0; i < region.size(); ++i) {
      buffer_region[i] = arith::Union({buffer_region[i], region[i]});
    }
  }

  /*!
   * \brief Gather used buffer region
   */
  std::vector<arith::IntSet> GatherRegion(const TensorRegion& tensor_region) {
    std::unordered_map<const VarNode*, arith::IntSet> dom_map;
    auto it = buffers_lca_.find(tensor_region->buffer);
    CHECK(it != buffers_lca_.end());
    const auto& lca = it->second;
    // Every loop will be relaxed if the lca is the root
    bool need_relax = !lca.defined();
    for (size_t i = 0; i < loop_stack_.size(); ++i) {
      const Loop& loop = loop_stack_[i];
      const VarNode* var = loop->loop_var.get();
      if (need_relax || (tensor_region->buffer->scope == "shared" && IsThreadBinded(loop))) {
        dom_map[var] = arith::IntSet::FromRange(Range::FromMinExtent(loop->min, loop->extent));
      }
      if (loop.same_as(lca)) need_relax = true;
    }
    std::vector<arith::IntSet> region;
    for (const auto& range : tensor_region->region) {
      Range r = Range::FromMinExtent(Substitute(Substitute(range->min, block_var_), unit_loops_),
                                     Substitute(Substitute(range->extent, block_var_), unit_loops_));
      region.push_back(arith::EvalSet(r, dom_map));
    }
    return region;
  }

  static bool IsThreadBinded(const Loop& loop) {
    for (const auto& annotation : loop->annotations)
      if (annotation->attr_key == attr::loop_type) {
        std::string thread_tag = Downcast<StringImm>(annotation->value)->value;
        if (thread_tag.substr(0, 9) == "threadIdx" || thread_tag.substr(0, 7) == "vthread")
          return true;
      }
    return false;
  }
};

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into one-dimension Load/Store
 */
class BufferFlattener : public StmtExprMutator {
 public:
  BufferFlattener(
      const std::unordered_map<const VarNode*, PrimExpr>& block_var,
      const std::unordered_map<const VarNode*, PrimExpr>& unit_loops,
      const std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>&
      buffers_region,
      const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca,
      const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& arg_buffers,
      const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& already_bound_loop_vars_)
      : buffers_region_(buffers_region),
        block_var_(block_var),
        unit_loops_(unit_loops),
        buffers_lca_(buffers_lca),
        arg_buffers_(arg_buffers),
        already_bound_loop_vars_(already_bound_loop_vars_) {}

  Stmt VisitStmt(const Stmt& stmt) override {
    Stmt body = StmtMutator::VisitStmt(stmt);
    return body;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // Handle allocations
    const auto* block_op = op->block.as<BlockNode>();
    Stmt old_stmt = GetRef<Stmt>(block_op);
    CHECK(block_op != nullptr);
    for (size_t i = block_op->allocations.size(); i > 0; --i) {
      const auto& buffer = block_op->allocations[i - 1]->buffer;
      const std::string name = std::string(buffer->name);
      if (name.substr(0, 18) == "normal_reduce_temp" || name.substr(0, 11) == "reduce_temp") {
        continue;
      }
      if (buffers_lca_.at(buffer).defined()) {
        pending_allocate_[buffer] = block_op->allocations[i - 1];
      }
    }
    // visit body
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BlockRealizeNode>();
    CHECK(op != nullptr);
    block_op = op->block.as<BlockNode>();
    CHECK(block_op != nullptr);
    Stmt body = block_op->body;
    // Handle block predicate
    if (!is_one(op->predicate)) {
      body = IfThenElse(op->predicate, body);
    }

    for (size_t i = block_op->allocations.size(); i > 0; --i) {
      const auto& n = block_op->allocations[i - 1];
      const std::string name = std::string(n->buffer->name);
      if (name.substr(0, 18) == "normal_reduce_temp" || name.substr(0, 11) == "reduce_temp") {
        continue;
      }
      if (!buffers_lca_.at(n->buffer).defined() || buffers_lca_.at(n->buffer).same_as(old_stmt)) {
        PrimExpr extents = 1;
        for (const auto& extent : buffers_region_.at(n->buffer)) {
          extents *= extent.max() - extent.min() + 1;
        }
        body = Allocate(n->buffer->data, n->buffer->dtype, {extents}, const_true(), body);

        // Change empty scope into global
        std::string scope = n->scope.empty() ? "global" : n->scope;
        body = AttrStmt(n->buffer->data, attr::storage_scope, StringImm(scope), body);
      }
    }

    return body;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    // Replace the block var with its value
    auto it = block_var_.find(op);
    if (it != block_var_.end()) {
      return Substitute(it->second, unit_loops_);
    } else {
      return Substitute(GetRef<PrimExpr>(op), unit_loops_);
    }
  }

  Stmt VisitStmt_(const LoopNode* op) final {
    Stmt old_stmt = GetRef<Stmt>(op);

    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<LoopNode>();
    CHECK(op != nullptr);

    std::string thread_tag;
    bool thread_binded = false;

    ForType for_type = ForType::Serial;
    for (const auto& annotation : op->annotations) {
      if (annotation->attr_key == tir::attr::loop_type) {
        std::string type = Downcast<StringImm>(annotation->value)->value;
        if (type == "unroll") {
          for_type = ForType::Unrolled;
        } else if (type == "vectorize") {
          for_type = ForType::Vectorized;
        } else if (type == "parallel") {
          for_type = ForType::Parallel;
        } else {
          thread_binded = true;
          thread_tag = Downcast<StringImm>(annotation->value)->value;
        }
      }
    }

    Stmt body = op->body;
    for (auto it = pending_allocate_.begin(); it != pending_allocate_.end();) {
      if (old_stmt.same_as(buffers_lca_.at(it->first))) {
        PrimExpr extents = 1;
        const auto& n = it->second;
        for (const auto& extent : buffers_region_.at(n->buffer)) {
          extents *= extent.max() - extent.min() + 1;
        }
        body = Allocate(n->buffer->data, n->buffer->dtype, {extents}, const_true(), body);
        // Change empty scope into global
        std::string scope = n->scope.empty() ? "global" : n->scope;
        body = AttrStmt(n->buffer->data, attr::storage_scope, StringImm(scope), body);
        pending_allocate_.erase(it++);
      } else {
        it++;
      }
    }

    Stmt for_stmt;
    if (thread_binded) {
      if (!already_bound_loop_vars_.count(op->loop_var)) {
        for_stmt = AttrStmt(IterVar(Range(op->min, op->extent), op->loop_var,
                                    IterVarType::kThreadIndex, thread_tag),
                            thread_tag == "vthread" ? attr::virtual_thread : attr::thread_extent,
                            op->extent, body);
      } else {
        for_stmt = body;
      }
    } else if (is_one(op->extent) && op->annotations.empty()) {
      return body;
    } else {
      for_stmt = For(op->loop_var, op->min, op->extent, for_type, DeviceAPI::None, body);
    }

    for (const auto& annotation : op->annotations) {
      if (attr::IsPragmaKey(annotation->attr_key)) {
        for_stmt = AttrStmt(op->loop_var, annotation->attr_key, annotation->value, for_stmt);
      }
    }

    return for_stmt;
  }

  // TODO(Siyuan): add support for For and AttrStmt
  Stmt VisitStmt_(const ForNode* op) final {
    LOG(FATAL) << "For is not allowed in TIR schedule for now.";
    return Stmt();
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<BufferStoreNode>();
    CHECK(op != nullptr);
    auto begins = ComputeRelativeIndices(op->buffer, op->indices);
    Buffer new_buffer = ReshapeBuffer(op->buffer, this->buffers_region_.at(op->buffer));
    return new_buffer.vstore(begins, op->value);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    op = expr.as<BufferLoadNode>();
    auto begins = ComputeRelativeIndices(op->buffer, op->indices);
    Buffer new_buffer = ReshapeBuffer(op->buffer, this->buffers_region_.at(op->buffer));
    return new_buffer.vload(begins, op->dtype);
  }

  PrimExpr VisitExpr_(const CallNode* op) final {
    if (op->op.same_as(builtin::get_elem_offset())) {
      CHECK_EQ(op->args.size(), 1);
      const auto* buffer_load = op->args[0].as<BufferLoadNode>();
      CHECK(buffer_load != nullptr);
      Load load = Downcast<Load>(VisitExpr(op->args[0]));
      return load->index;
    } else {
      return StmtExprMutator::VisitExpr_(op);
    }
  }

 private:
  const std::unordered_map<Buffer, std::vector<arith::IntSet>, ObjectPtrHash, ObjectPtrEqual>&
      buffers_region_;
  const std::unordered_map<const VarNode*, PrimExpr>& block_var_;
  const std::unordered_map<const VarNode*, PrimExpr>& unit_loops_;
  const std::unordered_map<Buffer, ObjectRef, ObjectPtrHash, ObjectPtrEqual>& buffers_lca_;
  const std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>& arg_buffers_;
  const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& already_bound_loop_vars_;

  std::unordered_map<Buffer, BufferAllocate, ObjectPtrHash, ObjectPtrEqual> pending_allocate_;

  /*!
   * \brief Create a buffer with alternative shape
   */
  Buffer ReshapeBuffer(const Buffer& buffer, const std::vector<arith::IntSet>& region) {
    if (arg_buffers_.count(buffer)) return buffer;
    auto n = runtime::make_object<BufferNode>(*(buffer.operator->()));
    Array<PrimExpr> shape;
    for (const auto& i : region) {
      shape.push_back(i.max() - i.min() + 1);
    }
    n->shape = std::move(shape);
    return Buffer(n);
  }

  /*!
   * \brief Transform indices from the absolute indices to relative indices
   * \note T can be BufferLoad or BufferStore
   */
  std::vector<PrimExpr> ComputeRelativeIndices(const Buffer& buffer,
                                               const Array<PrimExpr>& indices) {
    auto it = buffers_region_.find(buffer);
    CHECK(it != buffers_region_.end());
    const auto& region = it->second;
    std::vector<PrimExpr> new_indices;
    for (size_t i = 0; i < region.size(); ++i) {
      if (arg_buffers_.count(buffer)) {
        new_indices.push_back(indices[i]);
      } else {
        new_indices.push_back(indices[i] - region[i].min());
      }
    }
    return new_indices;
  }
};

PrimFunc BufferFlatten(PrimFunc f) {
  auto fptr = f.CopyOnWrite();

  AllReduceTransformer allreduce_transformer;
  fptr->body = allreduce_transformer(fptr->body);

  // Check memory and execution hierarchy
  ScheduleNode::ValidateHierarchy(f);

  // Transform the reduction calls to BufferStore
  ReductionTransformer reduction_transformer;
  fptr->body = reduction_transformer(fptr->body);

  // Find the LCA of each Buffer access
  LCADetector lca_detector(fptr->buffer_map);
  lca_detector(fptr->body);

  // Recalculate the buffer region
  RegionGatherer region_gatherer(lca_detector.buffers_lca_, fptr->buffer_map);
  region_gatherer(fptr->body);

  // Transform BufferLoad/BufferStore into Load/Store
  BufferFlattener flattener(region_gatherer.block_var_, region_gatherer.unit_loops_,
                            region_gatherer.buffers_region_, lca_detector.buffers_lca_,
                            lca_detector.arg_buffers_,
                            allreduce_transformer.already_bound_loop_vars_);
  fptr->body = flattener(fptr->body);

  return f;
}

namespace transform {

Pass BufferFlatten() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return BufferFlatten(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.BufferFlatten", {});
}

TVM_REGISTER_GLOBAL("tir.transform.BufferFlatten").set_body_typed(BufferFlatten);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
