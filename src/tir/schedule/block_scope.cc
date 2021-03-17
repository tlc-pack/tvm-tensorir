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
#include "./utils.h"

namespace tvm {
namespace tir {

/******** Utility functions ********/

template <class K, class V>
using SMap = std::unordered_map<K, V, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief Add a dependency edge.
 * \param src The source of the dependency
 * \param dst The destination of the dependecy
 * \param kind Type of the dependency
 */
void AddEdge(BlockScopeNode* self, const StmtSRef& src, const StmtSRef& dst, DepKind kind) {
  if (!src.same_as(dst)) {
    Dependency dep(src, dst, kind);
    self->src2deps[src].push_back(dep);
    self->dst2deps[dst].push_back(dep);
  }
}

/*!
 * \brief Add a new child block, which is assumed to be the last one if visiting subtrees
 * from left to right, then update the `buffer_writers`, `buffer_readers` and the dependency graph
 * \param self The block scope to be updated
 * \param child_block_sref The child block to be added
 * \param _buffer_readers An auxiliary data structure that maps existing buffers to a list of blocks
 * that reads it
 */
void AddLastChildBlock(BlockScopeNode* self, const StmtSRef& child_block_sref,
                       SMap<Buffer, Array<StmtSRef>>* _buffer_readers) {
  const BlockNode* child_block = TVM_SREF_TO_BLOCK(child_block, child_block_sref);
  SMap<Buffer, Array<StmtSRef>>& buffer_readers = *_buffer_readers;
  SMap<Buffer, Array<StmtSRef>>& buffer_writers = self->buffer_writers;
  // Step 1. Update `buffer_readers` and `buffer_writer` for each buffer
  for (const BufferRegion& region : child_block->writes) {
    buffer_writers[region->buffer].push_back(child_block_sref);
  }
  for (const BufferRegion& region : child_block->reads) {
    buffer_readers[region->buffer].push_back(child_block_sref);
  }
  // Check and update block dependencies: RAW, WAW, WAR.
  // Note: AddEdge is effectively NOP on self-loops
  // Step 2. Update RAW dependency
  for (const BufferRegion& region : child_block->reads) {
    if (buffer_writers.count(region->buffer)) {
      for (const StmtSRef& from : buffer_writers[region->buffer]) {
        AddEdge(self, from, child_block_sref, DepKind::kRAW);
      }
    }
  }
  // Step 3. Update WAW dependency
  for (const BufferRegion& region : child_block->writes) {
    if (buffer_writers.count(region->buffer)) {
      for (const StmtSRef& from : buffer_writers[region->buffer]) {
        AddEdge(self, from, child_block_sref, DepKind::kWAW);
      }
    }
  }
  // Step 4. Check WAR dependency: not allowed in the IR
  for (const BufferRegion& region : child_block->writes) {
    if (buffer_readers.count(region->buffer)) {
      for (const StmtSRef& from : buffer_readers[region->buffer]) {
        CHECK(from.same_as(child_block_sref)) << "ValueError: WAR dependency is not allowed";
      }
    }
  }
}

/******** Constructors ********/

StmtSRef::StmtSRef(const StmtNode* stmt, StmtSRefNode* parent, int64_t seq_index) {
  ObjectPtr<StmtSRefNode> n = make_object<StmtSRefNode>();
  n->stmt = stmt;
  n->parent = parent;
  n->seq_index = seq_index;
  data_ = std::move(n);
}

StmtSRef StmtSRef::InlineMark() {
  static StmtSRef result(nullptr, nullptr, -1);
  return result;
}

StmtSRef StmtSRef::RootMark() {
  static StmtSRef result(nullptr, nullptr, -1);
  return result;
}

Dependency::Dependency(StmtSRef src, StmtSRef dst, DepKind kind) {
  ObjectPtr<DependencyNode> node = make_object<DependencyNode>();
  node->src = std::move(src);
  node->dst = std::move(dst);
  node->kind = kind;
  data_ = std::move(node);
}

BlockScope::BlockScope() { data_ = make_object<BlockScopeNode>(); }

BlockScope::BlockScope(const Array<StmtSRef>& child_block_srefs) {
  ObjectPtr<BlockScopeNode> n = make_object<BlockScopeNode>();
  SMap<Buffer, Array<StmtSRef>> buffer_readers;
  for (const StmtSRef& block_sref : child_block_srefs) {
    AddLastChildBlock(n.get(), block_sref, &buffer_readers);
  }
  data_ = std::move(n);
}

/******** Dependency ********/

Array<Dependency> BlockScopeNode::GetDepsBySrc(const StmtSRef& block_sref) const {
  const std::unordered_map<StmtSRef, Array<Dependency>, ObjectPtrHash, ObjectPtrEqual>& edges =
      this->src2deps;
  auto iter = edges.find(block_sref);
  if (iter != edges.end()) {
    return iter->second;
  } else {
    return {};
  }
}

Array<Dependency> BlockScopeNode::GetDepsByDst(const StmtSRef& block_sref) const {
  const std::unordered_map<StmtSRef, Array<Dependency>, ObjectPtrHash, ObjectPtrEqual>& edges =
      this->dst2deps;
  auto iter = edges.find(block_sref);
  if (iter != edges.end()) {
    return iter->second;
  } else {
    return {};
  }
}

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(StmtSRefNode);
TVM_REGISTER_NODE_TYPE(DependencyNode);
TVM_REGISTER_NODE_TYPE(BlockScopeNode);

TVM_REGISTER_GLOBAL("tir.schedule.StmtSRefStmt")
    .set_body_typed([](StmtSRef sref) -> Optional<Stmt> {
      return GetRef<Optional<Stmt>>(sref->stmt);
    });
TVM_REGISTER_GLOBAL("tir.schedule.StmtSRefParent")
    .set_body_typed([](StmtSRef sref) -> Optional<StmtSRef> {
      return GetRef<Optional<StmtSRef>>(sref->parent);
    });
TVM_REGISTER_GLOBAL("tir.schedule.StmtSRefRootMark")  //
    .set_body_typed(StmtSRef::RootMark);
TVM_REGISTER_GLOBAL("tir.schedule.StmtSRefInlineMark")  //
    .set_body_typed(StmtSRef::InlineMark);
TVM_REGISTER_GLOBAL("tir.schedule.BlockScopeGetDepsBySrc")
    .set_body_method<BlockScope>(&BlockScopeNode::GetDepsBySrc);
TVM_REGISTER_GLOBAL("tir.schedule.BlockScopeGetDepsByDst")
    .set_body_method<BlockScope>(&BlockScopeNode::GetDepsByDst);

}  // namespace tir
}  // namespace tvm
