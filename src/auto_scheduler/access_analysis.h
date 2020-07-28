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
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace auto_scheduler {

// Forward declaration
class LoopTreeNode;
class LoopTree;

/*!
 * \brief Data structure representing an interval [min, min + extent).
 * Different from tir::Range, we allow extent to be NullOpt, and in this case,
 * it means a single point.
 */
class RangeNode : public Object {
 public:
  /*! \brief Beginning of the node */
  PrimExpr min;
  /*! \brief The extend of range, NullOpt if it is a point */
  Optional<PrimExpr> extent;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("min", &min);
    v->Visit("extent", &extent);
  }

  static constexpr const char* _type_key = "auto_scheduler.RangeNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(RangeNode, Object);
};

/*! \brief Managed reference to RangeNode. */
class Range : public ObjectRef {
 public:
  explicit Range(PrimExpr min, Optional<PrimExpr> extent);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(Range, ObjectRef, RangeNode);
};

/*! \brief Cartesian product of ranges that represents a high-dimensional region of a buffer */
using Domain = Array<Range>;

/*! \brief Records buffer accesses in a certain loop scope */
class ScopeAccessNode : public Object {
 public:
  Map<tir::Var, Array<Domain>> read_doms;
  Map<tir::Var, Array<Domain>> write_doms;
  std::unordered_map<const tir::VarNode*, std::vector<const LoopTreeNode*>> producer_siblings;
  std::unordered_map<const tir::VarNode*, std::vector<const LoopTreeNode*>> consumer_siblings;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("read_doms", &read_doms);
    v->Visit("write_doms", &write_doms);
    // `producer_siblings` is not visited
    // `consumer_siblings` is not visited
  }

  bool IsInjective(bool* axis_missing, bool* axis_duplicated, bool* axes_in_order) const;

  static constexpr const char* _type_key = "auto_scheduler.ScopeAccess";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScopeAccessNode, Object);
};

/*! \brief Managed reference to ScopeAccessNode */
class ScopeAccess : public ObjectRef {
 public:
  ScopeAccess(
      Map<tir::Var, Array<Domain>> read_doms, Map<tir::Var, Array<Domain>> write_doms,
      std::unordered_map<const tir::VarNode*, std::vector<const LoopTreeNode*>> producer_siblings,
      std::unordered_map<const tir::VarNode*, std::vector<const LoopTreeNode*>> consumer_siblings);

  static ScopeAccess FromLoopTreeLeaf(const LoopTree& leaf);
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(ScopeAccess, ObjectRef, ScopeAccessNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // SRC_AUTO_SCHEDULER_ACCESS_ANALYSIS_H_
