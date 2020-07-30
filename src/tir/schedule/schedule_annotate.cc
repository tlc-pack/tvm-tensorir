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
#include "./schedule_common.h"

namespace tvm {
namespace tir {

/*!
 * \brief Checks if a loop variable is parallelizable.
 * If fvisit returns false, it stops visit the children on that node.
 * \param loop_var The loop variable
 * \param block_realize The block realize node under the loop. It is possible that there are
 * multiple blocks, and in this case, we should invoke this function multiple times.
 * \param schedule The schedule object
 * \return A boolean indicating if the loop var is parallelizable
 */
bool IsLoopVarParallelizable(const Var& loop_var, const Stmt& block_realize,
                             const ScheduleNode* schedule) {
  const BlockRealizeNode* realize = block_realize.as<BlockRealizeNode>();
  CHECK(realize != nullptr)
      << "InternalError: in IsLoopVarParallelizable, expect BlockRealize, but get type: "
      << block_realize->GetTypeKey();
  const BlockNode* block = realize->block.get();
  // Cond 1. Binding is validated
  if (!schedule->stmt2ref.at(block)->binding_valid) {
    return false;
  }
  CHECK_EQ(realize->binding_values.size(), block->iter_vars.size())
      << "InternalError: BlockRealize is inconsistent with its Block";
  int n = realize->binding_values.size();
  // Cond 2. For each iter var that is not data parallel, the binding does not involve loop_var
  for (int i = 0; i < n; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = realize->binding_values[i];
    if (iter_var->iter_type != kDataPar && ExprContainsVar(binding, loop_var)) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Create a new loop with the given annotation added
 * \param loop The loop with original annotation
 * \param annotation The annotation to be added
 * \return A new loop with the given annotation as its last annotation
 */
Loop WithAnnotation(const LoopNode* loop, const Annotation& annotation) {
  ObjectPtr<LoopNode> new_loop = make_object<LoopNode>(*loop);
  new_loop->annotations.push_back(annotation);
  return Loop(new_loop);
}

void ScheduleNode::ParallelCompute(const StmtSRef& loop_sref, const Annotation& annotation) {
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
  const auto* loop = loop_sref->GetStmt<LoopNode>();
  CHECK(loop != nullptr) << "TypeError: Parallel compute applies only to a loop, but get: "
                         << loop_sref->stmt->GetTypeKey();
  CHECK(loop->annotations.empty())
      << "ValueError: Cannot apply parallelization to a loop that already has annotations: "
      << loop->annotations;
  // Now only support:
  //   1. All the blocks are complete below
  //   2. A single block below the loop
  // TODO(bohan): support reduction later
  bool is_compact_dataflow = GetParentScope(loop_sref).IsCompactDataFlow(loop_sref, this);
  if (!is_compact_dataflow) {
    Array<Stmt> single_child = GetChildren(GetRef<Stmt>(loop), true);
    // TODO(@junrushao1994): I am not super convinced by the checks here, revisit later
    CHECK(single_child.size() == 1)
        << "ValueError: loop with variable \"" << loop->loop_var << "\" cannot be parallelized, "
        << "because it does not satisfy one-way fine-grained dataflow "
           "condition, and has more than 1 child block";
    const auto* realize = single_child.as<BlockRealizeNode>();
    CHECK(IsLoopVarParallelizable(loop->loop_var, GetRef<Stmt>(realize), this))
        << "ValueError: loop with variable \"" << loop->loop_var
        << "\" cannot be parallelized because of block:\n"
        << GetRef<Stmt>(realize);
  } else {
    PreOrderVisit(GetRef<Stmt>(loop), [&loop, this](const ObjectRef& node) {
      if (const auto* realize = node.as<BlockRealizeNode>()) {
        CHECK(IsLoopVarParallelizable(loop->loop_var, GetRef<Stmt>(realize), this))
            << "ValueError: loop with variable \"" << loop->loop_var
            << "\" cannot be parallelized because of block:\n"
            << GetRef<Stmt>(realize);
        return false;
      }
      return true;
    });
  }
  this->Replace(loop_sref, WithAnnotation(loop, annotation));
}

void ScheduleNode::vectorize(const StmtSRef& loop_sref) {
  ParallelCompute(loop_sref, Annotation(attr::loop_type, StringImm("vectorize")));
}

void ScheduleNode::parallel(const StmtSRef& loop_sref) {
  ParallelCompute(loop_sref, Annotation(attr::loop_type, StringImm("parallel")));
}

void ScheduleNode::unroll(const StmtSRef& loop_sref) {
  const auto* loop = loop_sref->GetStmt<LoopNode>();
  CHECK(loop != nullptr) << "TypeError: Unroll expects a loop, but get type: "
                         << loop_sref->stmt->GetTypeKey();
  CHECK(loop->annotations.empty())
      << "ValueError: Cannot apply unrolling to a loop that already has annotations: "
      << loop->annotations;
  this->Replace(loop_sref, WithAnnotation(loop, Annotation(attr::loop_type, StringImm("unroll"))));
}

}  // namespace tir
}  // namespace tvm
