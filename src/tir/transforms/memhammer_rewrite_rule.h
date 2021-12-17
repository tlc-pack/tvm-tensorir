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
#include "../../../include/tvm/arith/iter_affine_map.h"
#include "../../../include/tvm/runtime/registry.h"
#include "../../../include/tvm/target/target.h"
#include "../../../include/tvm/tir/expr.h"
#include "../../../include/tvm/tir/op.h"
#include "../../../include/tvm/tir/stmt_functor.h"
#include "../../../include/tvm/tir/transform.h"

#include "../schedule/utils.h"

namespace tvm {
namespace tir {

struct ConstraintSet{

};

class RewriteRule {
 private:
  /*!
   * \brief Rewrite the stmt under certain constraints
   * \param stmt The stmt
   * \param constraints The constraints of the rewrite
   * \param output Some additional information that the rewrite rule produces. (including the new
   *               buffer to be allocated, etc.)
   * \return the stmt after rewrite
   */
  virtual Stmt Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
                       Map<String, ObjectRef>* output) const = 0;
  /*!
   * \brief Whether the rewrite rule can be applied to the stmt under certain constraints
   * \param stmt The stmt
   * \param constraints The constraints of the rewrite
   * \return A boolean flag indicating whether the rule can be applied
   */
  virtual bool CanApply(const Stmt& stmt, const Map<String, ObjectRef>& constraints) const {
    return true;
  }

 public:
  inline Stmt Apply(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
                    Map<String, ObjectRef>* output) const {
    if (CanApply(stmt, constraints)) {
      return Rewrite(stmt, constraints, output);
    } else {
      return stmt;
    }
  }
};

/*!
 * \brief Coalesce and vectorize memory access.
 */
class CoalescedAccess : public RewriteRule {
 public:
  Stmt Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
               Map<String, ObjectRef>* output) const final;
  bool CanApply(const Stmt& stmt, const Map<String, ObjectRef>& constraints) const final {
    String src = Downcast<String>(constraints["src_scope"]);
    String tgt = Downcast<String>(constraints["tgt_scope"]);
    runtime::StorageScope src_scope = runtime::StorageScope::Create(src);
    runtime::StorageScope tgt_scope = runtime::StorageScope::Create(tgt);
    return (src_scope.rank == runtime::StorageRank::kGlobal &&
            tgt_scope.rank == runtime::StorageRank::kShared) ||
           (src_scope.rank == runtime::StorageRank::kShared &&
            tgt_scope.rank == runtime::StorageRank::kGlobal);
  }
};

/*!
 * \brief Transform from A[f(i,j)] = B[i,j] to A[i,j] = B[f^{-1}(i,j)]
 */
class InverseMapping : public RewriteRule {
 public:
  Stmt Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
               Map<String, ObjectRef>* output) const final;
  bool CanApply(const Stmt& stmt, const Map<String, ObjectRef>& constraints) const final {
    String src = Downcast<String>(constraints["src_scope"]);
    String tgt = Downcast<String>(constraints["tgt_scope"]);
    runtime::StorageScope src_scope = runtime::StorageScope::Create(src);
    runtime::StorageScope tgt_scope = runtime::StorageScope::Create(tgt);
    return src_scope.rank == runtime::StorageRank::kShared &&
           tgt_scope.rank == runtime::StorageRank::kGlobal;
  }
};

/*!
 * \brief Create a local stage when loading from global memory to shared memory.
 */
class CreateLocalStage : public RewriteRule {
 public:
  Stmt Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
               Map<String, ObjectRef>* output) const final;
  bool CanApply(const Stmt& stmt, const Map<String, ObjectRef>& constraints) const final {
    String src = Downcast<String>(constraints["src_scope"]);
    String tgt = Downcast<String>(constraints["tgt_scope"]);
    PrimExpr has_local_stage =
        Downcast<PrimExpr>(constraints.Get("local_stage").value_or(Integer(0)));
    runtime::StorageScope src_scope = runtime::StorageScope::Create(src);
    runtime::StorageScope tgt_scope = runtime::StorageScope::Create(tgt);
    return src_scope.rank == runtime::StorageRank::kGlobal &&
           tgt_scope.rank == runtime::StorageRank::kShared && is_one(has_local_stage);
  }
};

/*!
 * \brief Add a cache stage in shared memory. Perform tensor core rewrite for wmma->shared, and
 *  perform coalescing and vectorizing for shared->global.
 */
class WmmaToGlobal : public RewriteRule {
 public:
  Stmt Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
               Map<String, ObjectRef>* output) const final;
  bool CanApply(const Stmt& stmt, const Map<String, ObjectRef>& constraints) const final {
    String src = Downcast<String>(constraints["src_scope"]);
    String tgt = Downcast<String>(constraints["tgt_scope"]);
    runtime::StorageScope src_scope = runtime::StorageScope::Create(src);
    runtime::StorageScope tgt_scope = runtime::StorageScope::Create(tgt);
    return src_scope.rank == runtime::StorageRank::kWMMAAccumulator &&
           tgt_scope.rank == runtime::StorageRank::kGlobal;
  }
};

/*!
 * \brief Rewrite shared->wmma data copy with load_matrix_sync
 */
class SharedToWmma : public RewriteRule {
 public:
  Stmt Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
               Map<String, ObjectRef>* output) const final;
  bool CanApply(const Stmt& stmt, const Map<String, ObjectRef>& constraints) const final {
    String src = Downcast<String>(constraints["src_scope"]);
    String tgt = Downcast<String>(constraints["tgt_scope"]);
    runtime::StorageScope src_scope = runtime::StorageScope::Create(src);
    runtime::StorageScope tgt_scope = runtime::StorageScope::Create(tgt);
    return src_scope.rank == runtime::StorageRank::kShared &&
           (tgt_scope.rank == runtime::StorageRank::kWMMAMatrixA ||
            tgt_scope.rank == runtime::StorageRank::kWMMAMatrixB);
  }
};

/*!
 * \brief Rewrite wmma->shared data copy with store_matrix_sync
 */
class WmmaToShared : public RewriteRule {
 public:
  Stmt Rewrite(const Stmt& stmt, const Map<String, ObjectRef>& constraints,
               Map<String, ObjectRef>* output) const final;
  bool CanApply(const Stmt& stmt, const Map<String, ObjectRef>& constraints) const final {
    String src = Downcast<String>(constraints["src_scope"]);
    String tgt = Downcast<String>(constraints["tgt_scope"]);
    runtime::StorageScope src_scope = runtime::StorageScope::Create(src);
    runtime::StorageScope tgt_scope = runtime::StorageScope::Create(tgt);
    return src_scope.rank == runtime::StorageRank::kWMMAAccumulator &&
           tgt_scope.rank == runtime::StorageRank::kShared;
  }
};

}  // namespace tir
}  // namespace tvm
