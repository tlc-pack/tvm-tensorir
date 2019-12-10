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

/*!
 *  Copyright (c) by Contributors
 * \file schedule_tree.h
 */

#ifndef TVM_TE_SCHEDULE_TREE_H_
#define TVM_TE_SCHEDULE_TREE_H_

#include <tvm/te/ir.h>
#include <vector>

namespace tvm {
namespace te {

class ScheduleTreeNode;

class ScheduleTreeNodeRef : public NodeRef {
 public:
  ScheduleTreeNodeRef() {}
  explicit ScheduleTreeNodeRef(NodePtr<Node> n): NodeRef(n) {}

  const ScheduleTreeNode* operator->() const;
  ScheduleTreeNode* operator->();
  operator bool() const { return this->defined(); }
  using ContainerType = ScheduleTreeNode;

};
class ScheduleTreeNode : public Node {
 public:
  ScheduleTreeNodeRef father;
  virtual void VisitAttrs(AttrVisitor* v) {}
  static constexpr const char* _type_key = "te.ScheduleTreeNode";
  TVM_DECLARE_NODE_TYPE_INFO(ScheduleTreeNode, Node);
  virtual const StmtNode* stmt() const { return nullptr; }
};



class BlockTreeNodeRef;
class BlockTreeNode : public ScheduleTreeNode {
 public:
  /*! \brief The corresponding block in AST.
   * We use weak reference here to reduce copy construct in AST
   */
  const BlockNode* block;
  void VisitAttrs(AttrVisitor* v) {}
  static constexpr const char* _type_key = "te.BlockTreeNode";
  TVM_DECLARE_NODE_TYPE_INFO(BlockTreeNode, Node);
  const StmtNode* stmt() const final {
    return static_cast<const StmtNode*>(block);
  }
};

class BlockTreeNodeRef : public ScheduleTreeNodeRef {
 public:
  explicit BlockTreeNodeRef(const BlockNode* block,
                           ScheduleTreeNodeRef father);
  TVM_DEFINE_NODE_REF_METHODS(BlockTreeNodeRef, ScheduleTreeNodeRef, BlockTreeNode);
  inline BlockTreeNode* operator->() {
    return static_cast<BlockTreeNode*>(data_.get());
  }
};

class AxisTreeNodeRef;
class AxisTreeNode : public ScheduleTreeNode {
 public:
  /*! \brief The corresponding loop in AST.
   * We use weak reference here to reduce copy construct in AST
   */
  const LoopNode* loop;
  void VisitAttrs(AttrVisitor* v) {}
  static constexpr const char* _type_key = "te.AxisTreeNode";
  TVM_DECLARE_NODE_TYPE_INFO(AxisTreeNode, Node);

  const StmtNode* stmt() const final {
    return static_cast<const StmtNode*>(loop);
  }
};

class AxisTreeNodeRef : public ScheduleTreeNodeRef {
 public:
  explicit AxisTreeNodeRef(const LoopNode* loop,
                           ScheduleTreeNodeRef father);
  TVM_DEFINE_NODE_REF_METHODS(AxisTreeNodeRef, ScheduleTreeNodeRef, AxisTreeNode);
  inline AxisTreeNode* operator->() {
    return static_cast<AxisTreeNode*>(data_.get());
  }
};

}
}

#endif  //TVM_TE_SCHEDULE_TREE_H_
