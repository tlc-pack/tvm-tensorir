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

#include <tvm/te/schedule_tree.h>

namespace tvm {
namespace te {

const ScheduleTreeNode* ScheduleTreeNodeRef::operator->() const {
  return static_cast<const ScheduleTreeNode*>(data_.get());
}

ScheduleTreeNode* ScheduleTreeNodeRef::operator->() {
  return static_cast<ScheduleTreeNode*>(data_.get());
}

AxisTreeNodeRef::AxisTreeNodeRef(const LoopNode* loop,
                                 ScheduleTreeNodeRef father) {
  NodePtr<AxisTreeNode> node = make_node<AxisTreeNode>();
  node->loop = loop;
  node->father = father;
  data_ = std::move(node);
}

BlockTreeNodeRef::BlockTreeNodeRef(const BlockNode* block,
                                   ScheduleTreeNodeRef father) {
  NodePtr<BlockTreeNode> node = make_node<BlockTreeNode>();
  node->block = block;
  node->father = father;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(ScheduleTreeNode);
TVM_REGISTER_NODE_TYPE(BlockTreeNode);
TVM_REGISTER_NODE_TYPE(AxisTreeNode);
}  // namespace te
}  // namespace tvm
