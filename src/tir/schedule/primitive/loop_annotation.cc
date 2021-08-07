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
#include "../utils.h"

namespace tvm {
namespace tir {

class NotAffineBindingError : public ScheduleError {
 public:
  explicit NotAffineBindingError(IRModule mod, Block block)
      : mod_(std::move(mod)), block_(std::move(block)) {}
  String FastErrorString() const override {
    return "ScheduleError: The blocks underlying the loop to be parallelized are required to have "
           "affine bindings, but some such block does not have";
  }
  String DetailRenderTemplate() const override {
    return "The blocks underlying the loop to be parallelized are required to have affine "
           "bindings, however block {0} does not have affine bindings";
  }
  IRModule mod() const override { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const override { return {block_}; }
  IRModule mod_;
  Block block_;
};

class CannotParallelizeError : public ScheduleError {
 public:
  explicit CannotParallelizeError(IRModule mod, Var loop_var, Block block)
      : mod_(std::move(mod)), loop_var_(std::move(loop_var)), block_(std::move(block)) {}
  String FastErrorString() const override {
    return "ScheduleError: The parallelization cannot be fulfilled because the loop cannot be "
           "parallelized with regard to some of its underlying block";
  }
  String DetailRenderTemplate() const override {
    std::ostringstream os;
    os << "The loop cannot be parallelized with regard to block {0}, because one of the two "
          "following reasons:\n1) some block iter whose binding contains the loop var "
       << loop_var_ << " is neither data-parallel nor reduction block iter\n2) the loop var "
       << loop_var_
       << " is contained in some reduction block iter's binding, while the context thread is not "
          "`threadIdx`";
    return os.str();
  }
  IRModule mod() const override { return mod_; }
  Array<ObjectRef> LocationsOfInterest() const override { return {block_}; }
  IRModule mod_;
  Var loop_var_;
  Block block_;
};

/*!
 * \brief Check if a loop is parallelizable with regard to a specific block
 * \details There are two conditions:
 * 1) The block is required to have affine bindings, and
 * 2) For each block iter whose binding contains the input loop variable, either
 *   - the block iter is data parallel, or
 *   - the block iter is a reduction block iter, and the input `thread_tag` starts with "threadIdx"
 *   in case of cross-thread reduction.
 * \param self The schedule state
 * \param loop_var The loop variable of the loop to be checked
 * \param block_realize The block-realize of the block to be checked
 * \param thread_tag The tag of the thread in GPU to be bound, which is an empty string if the
 * parallelization is not for GPU
 * \throws ScheduleError If the input loop is not parallelizable with regard to the input block
 */
void CheckLoopParallelizableInBlock(const ScheduleState& self, const Var& loop_var,
                                    const BlockRealize& block_realize,
                                    const std::string& thread_tag) {
  const Block& block = block_realize->block;

  // Cond 1. The block is required to have affine bindings.
  if (!self->IsAffineBlockBinding(self->stmt2ref.at(block_realize->block.get()))) {
    throw NotAffineBindingError(self->mod, block);
  }

  // Cond 2. For each block iter whose binding contains `loop_var`, only two cases are allowed.
  ICHECK_EQ(block->iter_vars.size(), block_realize->iter_values.size());
  int n_iters = static_cast<int>(block->iter_vars.size());
  for (int i = 0; i < n_iters; ++i) {
    const IterVar& iter_var = block->iter_vars[i];
    const PrimExpr& binding = block_realize->iter_values[i];

    if (!UsesVar(binding, [v = loop_var.get()](const VarNode* var) { return var == v; })) {
      continue;
    }
    // Only two cases are allowed:
    // - The block iter is data parallel, or
    // - The block iter is a reduction block iter, and the `thread_tag` starts with "threadIdx"
    // in case of cross-thread reduction.
    IterVarType iter_type = iter_var->iter_type;
    if (!(iter_type == kDataPar ||
          (iter_type == kCommReduce && thread_tag.substr(0, 9) == "threadIdx"))) {
      throw CannotParallelizeError(self->mod, loop_var, block);
    }
  }
}

/*!
 * \brief For each block (recursive) under the given loop, check whether the input loop is
 * parallelizable with regard to the block
 * \param self The schedule state
 * \param loop The loop to be checked
 * \param thread_tag The tag of the thread in GPU to be bound, which is an empty string if the
 * parallelization is not for GPU
 */
void CheckParallelizability(const ScheduleState& self, const For& loop,
                            const std::string& thread_tag) {
  PreOrderVisit(loop, [self, loop_var = loop->loop_var, thread_tag](const ObjectRef& node) {
    if (const auto* realize = node.as<BlockRealizeNode>()) {
      CheckLoopParallelizableInBlock(self, loop_var, GetRef<BlockRealize>(realize), thread_tag);
      return false;
    }
    return true;
  });
}

/*!
 * \brief Parallelize a given loop using the given kind of parallelization
 * \param self The schedule state
 * \param loop_sref The sref of the loop to be parallelized
 * \param for_kind The type of the parallelization (only `kParallel`, `kVectorized` and
 * `kThreadBinding` are allowed)
 * \param thread The context thread that the input loop is bound to, which is not `NullOpt`
 * only when `for_kind` is `kThreadBinding`
 */
void ParallelizeComputation(const ScheduleState& self, const StmtSRef& loop_sref,
                            const ForKind& for_kind, const Optional<IterVar>& thread) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  // If `loop` has extent 1, and `for_kind` is `kParallel` or `kVectorized`, just return.
  if (is_one(loop->extent) &&
      (for_kind == ForKind::kParallel || for_kind == ForKind::kVectorized)) {
    return;
  }

  /*
   * Check:
   * - 1. the subtree rooted from the input loop in sref tree has compact data flow
   * - 2. all the blocks under the given loop have affine block bindings
   * - 3. the input loop can be only bound to data-parallel block iters, or the loop can be bound to
   * reduction block iter if `thread` is `threadIdx.x/y/z` in case of cross-thread reduction
   * When the above conditions are all satisfied, this input loop can be parallelized.
   */
  // Step 1. Check whether the subtree rooted from the `loop` in sref tree has compact data flow.
  CheckSRefSubtreeCompactDataFlow(self, loop_sref);

  // Step 2. Check whether the loop can be parallelized with regard to each underlying block.
  CheckParallelizability(self, GetRef<For>(loop),
                         thread.defined() ? thread.value()->thread_tag : "");

  // Step 3. Loop update and IR replacement
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->kind = for_kind;
  new_loop->thread_binding = thread;
  self->Replace(loop_sref, For(new_loop), {});
}

void Parallel(ScheduleState self, const StmtSRef& loop_sref) {
  ParallelizeComputation(self, loop_sref, ForKind::kParallel, NullOpt);
}

void Vectorize(ScheduleState self, const StmtSRef& loop_sref) {
  ParallelizeComputation(self, loop_sref, ForKind::kVectorized, NullOpt);
}

void Bind(ScheduleState self, const StmtSRef& loop_sref, const IterVar& thread) {
  ParallelizeComputation(self, loop_sref, ForKind::kThreadBinding, thread);
}

void Unroll(ScheduleState self, const StmtSRef& loop_sref) {
  const ForNode* loop = TVM_SREF_TO_FOR(loop, loop_sref);
  if (is_one(loop->extent)) {
    return;
  }
  ObjectPtr<ForNode> new_loop = make_object<ForNode>(*loop);
  new_loop->kind = ForKind::kUnrolled;
  self->Replace(loop_sref, For(new_loop), {});
}

/******** Instruction Registration ********/

struct ParallelTraits : public UnpackedInstTraits<ParallelTraits> {
  static constexpr const char* kName = "Parallel";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) {
    return sch->Parallel(loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("parallel");
    py.Input("loop", loop_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct VectorizeTraits : public UnpackedInstTraits<VectorizeTraits> {
  static constexpr const char* kName = "Vectorize";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) {
    return sch->Vectorize(loop_rv);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("vectorize");
    py.Input("loop", loop_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct BindTraits : public UnpackedInstTraits<BindTraits> {
  static constexpr const char* kName = "Bind";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, String thread) {
    return sch->Bind(loop_rv, thread);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, String thread) {
    PythonAPICall py("bind");
    py.Input("loop", loop_rv);
    py.Input("thread", thread);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

struct UnrollTraits : public UnpackedInstTraits<UnrollTraits> {
  static constexpr const char* kName = "Unroll";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 0;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv) { return sch->Unroll(loop_rv); }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv) {
    PythonAPICall py("unroll");
    py.Input("loop", loop_rv);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(ParallelTraits);
TVM_REGISTER_INST_KIND_TRAITS(VectorizeTraits);
TVM_REGISTER_INST_KIND_TRAITS(BindTraits);
TVM_REGISTER_INST_KIND_TRAITS(UnrollTraits);

}  // namespace tir
}  // namespace tvm