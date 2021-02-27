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
#ifndef TVM_TIR_SCHEDULE_STATE_H_
#define TVM_TIR_SCHEDULE_STATE_H_

#include <tvm/tir/function.h>
#include <tvm/tir/schedule/block_scope.h>

#include <unordered_map>

namespace tvm {
namespace tir {

// TODO(@junrushao1994): change `std::unordered_map` to `Map`?

class ScheduleStateNode : public runtime::Object {
 public:
  /*! \brief The function to be scheduled */
  PrimFunc func;
  /*! \brief The root of schedulable reference tree */
  StmtSRef root;
  /*! \brief The block scopes of each block */
  std::unordered_map<StmtSRef, BlockScope, ObjectPtrHash, ObjectPtrEqual> scopes;
  /*! \brief The mapping from stmt to its schedulable reference node */
  std::unordered_map<const StmtNode*, StmtSRef> stmt2ref;
  /*! \brief In debug mode, we will check correctness after each replacement */
  bool debug_mode;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("func", &func);
    v->Visit("root", &root);
    // `scopes` is not visited
    // `stmt2ref` is not visited
    v->Visit("debug_mode", &debug_mode);
  }

  /*!
   * \brief Replace part of pointed by `src_sref` AST with a new statement `tgt_stmt`.
   * Only 3 replacement types are allowed from `src_sref->stmt` to `tgt_stmt`.
   * 1) Block -> Block
   * 2) Loop -> Loop
   * 3) Loop -> BlockRealize
   * \param src_sref The sref of the statement to be replaced
   * \param tgt_stmt The statement to be replaced to
   * \param block_sref_reuse Maps an new block (replaced to) back to an old block (to be replaced),
   * and enforces reuse of srefs between them (rather than create new srefs)
   * i.e. after being replaced, the sref that points to the old block will point to the new one
   * \note `loop_sref_reuse` will be automatically detected via loop vars
   *
   * TODO(@junrushao1994): change the semantic of `block_sref_reuse`
   * from "new -> old" to "old -> new"
   */
  TVM_DLL void Replace(const tir::StmtSRef& src_sref, const Stmt& tgt_stmt,
                       const Map<Block, Block>& block_sref_reuse);

  static constexpr const char* _type_key = "tir.ScheduleState";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleStateNode, runtime::Object);
};

class ScheduleState : public runtime::ObjectRef {
 public:
  /*!
   * \brief Construct a schedule from a PrimFunc
   * \param func The PrimFunc to be created
   */
  TVM_DLL explicit ScheduleState(PrimFunc func, bool debug_mode);
  /*!
   * \brief Constructor
   * \param func The function to be scheduled
   * \param root brief The root of schedulable reference tree
   * \param scopes The mapping from stmt to its schedulable reference node
   * \param stmt2ref The block scopes of each block
   */
  TVM_DLL explicit ScheduleState(
      PrimFunc func, StmtSRef root,
      std::unordered_map<StmtSRef, BlockScope, ObjectPtrHash, ObjectPtrEqual> scopes,
      std::unordered_map<const StmtNode*, StmtSRef> stmt2ref, bool debug_mode);

  ScheduleStateNode* get() { return static_cast<ScheduleStateNode*>(data_.get()); }
  const ScheduleStateNode* get() const {
    return static_cast<const ScheduleStateNode*>(data_.get());
  }
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ScheduleState, runtime::ObjectRef, ScheduleStateNode);
};

}  // namespace tir
}  // namespace tvm

#endif  // TVM_TIR_SCHEDULE_STATE_H_
