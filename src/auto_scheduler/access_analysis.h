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
#ifndef SRC_AUTO_SCHEDULER_ACCESS_ANALYSIS_H_ /* TODO(@junrushao1994): guard name convention */
#define SRC_AUTO_SCHEDULER_ACCESS_ANALYSIS_H_

#include <tvm/ir/expr.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace auto_scheduler {

class LoopTreeNode;
class LoopTree;

class BaseAccessPatternNode : public Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "auto_scheduler.BaseAccessPattern";
  TVM_DECLARE_BASE_OBJECT_INFO(BaseAccessPatternNode, Object);
};

class BaseAccessPattern : public ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(BaseAccessPattern, ObjectRef, BaseAccessPatternNode);

 protected:
  BaseAccessPattern() = default;
};

class DummyAccessPatternNode : public BaseAccessPatternNode {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "auto_scheduler.DummyAccessPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(DummyAccessPatternNode, BaseAccessPatternNode);
};

class DummyAccessPattern : public BaseAccessPattern {
 public:
  DummyAccessPattern() = delete;
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(DummyAccessPattern, BaseAccessPattern,
                                            DummyAccessPatternNode);
};

class LeafAccessPatternNode : public BaseAccessPatternNode {
 public:
  /*! \brief Number of statements in the block */
  int num_stmts;
  /*! \brief If there is branching statement/intrinsic */
  bool has_branch;
  /*!
   * \brief If there is an expensive operator, for now, it covers:
   * 1) exp
   */
  bool has_expensive_op;
  /*!
   * \brief Indicating all the axes in the output buffer are one of those
   * 1) constants;
   * 2) block vars.
   */
  bool all_trivial_store;
  /*! \brief Block vars used in trivial store */
  Array<tir::Var> block_vars_in_trivial_store;
  /*! \brief If every non-constant load axis can be mapped to a store axis */
  bool lsmap_exists;
  /*!
   * \brief True if every store axis has >= 1 corresponding load axes;
   * False if an axis has 0 corresponding load axes.
   */
  bool lsmap_surjective;
  /*!
   * \brief True if every store axis has <= 1 corresponding load axes;
   * False if an axis has >= 2 corresponding load axes
   */
  bool lsmap_injective;
  /*! \brief All store axes are mapped in the same order of load axes */
  bool lsmap_ordered;
  /*! \brief Accumulate the number of store axes that don't appear in buffer load */
  int num_axes_reuse;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("num_stmts", &num_stmts);
    v->Visit("has_branch", &has_branch);
    v->Visit("has_expensive_op", &has_expensive_op);
    v->Visit("all_trivial_store", &all_trivial_store);
    v->Visit("block_vars_in_trivial_store", &block_vars_in_trivial_store);
    v->Visit("lsmap_exists", &lsmap_exists);
    v->Visit("lsmap_surjective", &lsmap_surjective);
    v->Visit("lsmap_injective", &lsmap_injective);
    v->Visit("lsmap_ordered", &lsmap_ordered);
    v->Visit("num_axes_reuse", &num_axes_reuse);
  }

  static constexpr const char* _type_key = "auto_scheduler.LeafAccessPattern";
  TVM_DECLARE_FINAL_OBJECT_INFO(LeafAccessPatternNode, BaseAccessPatternNode);
};

class LeafAccessPattern : public BaseAccessPattern {
 public:
  explicit LeafAccessPattern(const LoopTreeNode* node);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(LeafAccessPattern, BaseAccessPattern,
                                            LeafAccessPatternNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // SRC_AUTO_SCHEDULER_ACCESS_ANALYSIS_H_
