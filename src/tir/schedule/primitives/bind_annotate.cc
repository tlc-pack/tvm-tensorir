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
#include "../analysis.h"
#include "../utils.h"
#include "./primitives.h"

namespace tvm {
namespace tir {
namespace schedule {

/*!
 * \brief Checks if a loop variable is parallelizable.
 * \param loop_var The loop variable
 * \param block_realize The block realize node under the loop. It is possible that there are
 * multiple blocks, and in this case, we should invoke this function multiple times.
 * \param schedule The schedule object
 * \param anno_value The annotation anno_value
 * \return A boolean indicating if the loop var is parallelizable
 */
bool IsLoopVarParallelizable(const ScheduleState self, const Var& loop_var,
                             const Stmt& block_realize, const Optional<IterVar>& thread_binding) {
  const BlockRealizeNode* realize = block_realize.as<BlockRealizeNode>();
  ICHECK(realize != nullptr)
      << "InternalError: in IsLoopVarParallelizable, expect BlockRealize, but get type: "
      << block_realize->GetTypeKey();
  const BlockNode* block = realize->block.get();
  // Cond 1. Binding is validated
  if (!self->stmt2ref.at(block)->binding_valid) {
    return false;
  }
  CHECK_EQ(realize->binding_values.size(), block->iter_vars.size())
      << "InternalError: BlockRealize is inconsistent with its Block";
  int n = realize->binding_values.size();
  // Cond 2. For each iter var that is not data parallel, the binding does not involve loop_var
  std::string thread_tag = thread_binding.defined() ? thread_binding.value()->thread_tag : "";
  for (int i = 0; i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = realize->binding_values[i];
    bool contains = StmtExprContainsVar(binding, loop_var);
    if (contains && iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
    if (contains && iter_var->iter_type == kCommReduce && thread_tag.substr(0, 9) != "threadIdx") {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Create a new loop with the given annotation added
 * \param loop The loop with original annotation
 * \param attr_key The annotation key to be added
 * \param attr_value The annotation value to be added
 * \return A new loop with the given annotation as its last annotation
 */
For WithAnnotation(const ForNode* loop, const String& attr_key, const ObjectRef& attr_value) {
  Map<String, ObjectRef> annotations = loop->annotations;
  annotations.Set(attr_key, attr_value);
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->annotations = std::move(annotations);
  return For(new_loop);
}

/*!
 * \brief Create a new block with the given annotation added
 * \param block The block with original annotation
 * \param attr_key The annotation key to be added
 * \param attr_value The annotation value to be added
 * \return A new block with the given annotation as its last annotation
 */
Block WithAnnotation(const BlockNode* block, const String& attr_key, const ObjectRef& attr_value) {
  Map<String, ObjectRef> annotations = block->annotations;
  annotations.Set(attr_key, attr_value);
  ObjectPtr<BlockNode> new_block = make_object<BlockNode>(*block);
  new_block->annotations = std::move(annotations);
  return Block(new_block);
}

void ParallelCompute(ScheduleState self, const StmtSRef& loop_sref, const ForKind& for_kind,
                     const Optional<IterVar>& thread_binding) {
  /*!
   * Check:
   * - 1. check the block under is complete block or reduction block
   * - 2. check `input_loop` is bound and only bound to `data_par` block_vars
   * - 3. check the loops of reduction blocks are validatable
   * Mutate:
   * - 4. set Annotation on the loop
   * Proof:
   * We prove by showing that there are no data flows between `input_loop=i` and`input_loop=j`,
   * and we show this by induction on the number of blocks.
   *
   * If there is only one block below
   * - The block is complete. All the instances are independent of each other.
   * - The block is reduction. `input_loop` bound and only bound to `data_par` blocks + loops of
   * reduction blocks are validatable => instances of `input_loop=i` will write different positions
   * with instances of `input_loop=j`, hence they are independent.
   *
   * If there's a new block coming in. Consider its instances under `input_loop=i`.
   * - If the producer is complete. Producer instances under `input_loop=j` may write the positions
   * that new instances under `input_loop=i`  may read, but it will read the same value produced by
   * the producer under `input_loop=i` since it's complete.
   * - If the producer is reduction. Producer instances under `input_loop=j` will never write the
   * positions that new instances under `input_loop=j` may read. Hence no data flow.
   */
  const auto* loop = loop_sref->GetStmt<ForNode>();
  CHECK(loop != nullptr) << "TypeError: Parallel compute applies only to a loop, but get: "
                         << loop_sref->stmt->GetTypeKey();
  // Now only support:
  //   1. All the blocks are complete below
  //   2. A single block below the loop
  const BlockScope& scope = self->scopes.at(GetScopeRoot(loop_sref));
  bool is_compact_dataflow =
      scope->IsCompactDataFlow(loop_sref, GetChildBlocks(self, loop_sref, false));
  if (!is_compact_dataflow) {
    Array<Stmt> single_child = GetChildren(GetRef<Stmt>(loop), true);
    // TODO(@junrushao1994): I am not super convinced by the checks here, revisit later
    CHECK(single_child.size() == 1)
        << "ValueError: loop with variable \"" << loop->loop_var << "\" cannot be parallelized, "
        << "because it does not satisfy one-way fine-grained dataflow "
           "condition, and has more than 1 child block";
    const auto* realize = single_child[0].as<BlockRealizeNode>();
    CHECK(realize != nullptr) << "TypeError: Expects 'BlockRealizeNode', but gets: "
                              << single_child[0]->GetTypeKey();
    CHECK(IsLoopVarParallelizable(self, loop->loop_var, GetRef<Stmt>(realize), thread_binding))
        << "ValueError: loop with variable \"" << loop->loop_var
        << "\" cannot be parallelized because of block:\n"
        << GetRef<Stmt>(realize);
  } else {
    PreOrderVisit(GetRef<Stmt>(loop), [self, &loop, thread_binding](const ObjectRef& node) {
      if (const auto* realize = node.as<BlockRealizeNode>()) {
        CHECK(IsLoopVarParallelizable(self, loop->loop_var, GetRef<Stmt>(realize), thread_binding))
            << "ValueError: loop with variable \"" << loop->loop_var
            << "\" cannot be parallelized because of block:\n"
            << GetRef<Stmt>(realize);
        return false;
      }
      return true;
    });
  }
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->kind = for_kind;
  if (thread_binding.defined()) {
    new_loop->thread_binding = thread_binding;
  }
  self->Replace(loop_sref, For(new_loop), {});
}

void Vectorize(ScheduleState self, const StmtSRef& loop_sref) {
  if (is_one(loop_sref->GetStmt<ForNode>()->extent)) {
    return;
  }
  ParallelCompute(self, loop_sref, ForKind::kVectorized, NullOpt);
}

void Parallel(ScheduleState self, const StmtSRef& loop_sref) {
  ParallelCompute(self, loop_sref, ForKind::kParallel, NullOpt);
}

void Unroll(ScheduleState self, const StmtSRef& loop_sref) {
  const auto* loop = loop_sref->GetStmt<ForNode>();
  CHECK(loop != nullptr) << "TypeError: Unroll expects a loop, but get type: "
                         << loop_sref->stmt->GetTypeKey();
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->kind = ForKind::kUnrolled;
  self->Replace(loop_sref, For(new_loop), {});
}

void Bind(ScheduleState self, const StmtSRef& loop_sref, const IterVar& thread) {
  const auto* loop = loop_sref->GetStmt<ForNode>();
  CHECK(loop != nullptr) << "Parallel-like compute expect a loop";
  if (thread->dom.defined()) {
    CHECK(ExprDeepEqual()(loop->extent, thread->dom->extent))
        << "Thread axis extent and loop extent mismatch";
  }
  ParallelCompute(self, loop_sref, ForKind::kThreadBinding, thread);
}

void Pragma(ScheduleState self, const StmtSRef& loop_sref, const String& pragma_type,
            const PrimExpr& pragma_value) {
  const auto* loop_ptr = loop_sref->GetStmt<ForNode>();
  CHECK(loop_ptr) << "TypeError: pragma expects a Loop as its first argument";
  self->Replace(loop_sref, WithAnnotation(loop_ptr, "pragma_" + pragma_type, pragma_value), {});
}

void DoubleBuffer(ScheduleState self, const StmtSRef& block_sref) {
  const auto* block_ptr = block_sref->GetStmt<BlockNode>();
  CHECK(block_ptr) << "TypeError: double_buffer expects 'block' as its argument";
  const StmtSRef& parent_block_sref = GetScopeRoot(block_sref);
  const auto* parent_block = parent_block_sref->GetStmt<BlockNode>();
  const BlockScope& scope = self->scopes.at(parent_block_sref);
  CHECK(scope->IsComplete(block_sref))
      << "ValueError: 'double_buffer' expects 'block' to be a complete block";
  for (const BufferRegion& parent_write : parent_block->writes) {
    for (const BufferRegion& write : block_ptr->writes) {
      CHECK_NE(write->buffer.get(), parent_write->buffer.get())
          << "ValueError: 'double_buffer' does not work on an output block";
    }
  }
  CHECK_EQ(block_ptr->writes.size(), 1)
      << "ValueError: 'double_buffer' expects 'block' with only one write buffer";
  Block new_block =
      WithAnnotation(block_ptr, tir::attr::double_buffer_scope, IntImm(DataType::Int(32), 1));
  self->Replace(block_sref, new_block, {{new_block, GetRef<Block>(block_ptr)}});
}

}  // namespace schedule
}  // namespace tir
}  // namespace tvm
