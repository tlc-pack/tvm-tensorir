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

using TBufferReaderWriter =
    std::unordered_map<Buffer, Array<StmtSRef>, ObjectPtrHash, ObjectPtrEqual>;

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
                       TBufferReaderWriter* _buffer_readers) {
  const BlockNode* child_block = TVM_SREF_TO_BLOCK(child_block, child_block_sref);
  TBufferReaderWriter& buffer_readers = *_buffer_readers;
  TBufferReaderWriter& buffer_writers = self->buffer_writers;
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

/*!
 * \brief Check if each reduction instance is valid. Particularly, check:
 * 1) Each iteration variable is either data parallel or reduction
 * 2) Indices used to access the output buffer are not related to or affected by reduction iteration
 * variables.
 * \param iter_vars Iteration variables of the reduction
 * \param output_buffer_indices Indices used to access the output buffer
 * \return A boolean indicating if the reduction instance is valid
 */
bool CheckReductionInstance(const Array<IterVar>& iter_vars,
                            const Array<PrimExpr>& output_buffer_indices) {
  std::unordered_set<const VarNode*> reduction_block_vars;
  reduction_block_vars.reserve(iter_vars.size());
  // Check 1. Each iter_var can only be data parallel or reduction
  for (const IterVar& iter_var : iter_vars) {
    IterVarType kind = iter_var->iter_type;
    if (kind != kDataPar && kind != kCommReduce) {
      return false;
    }
    if (kind == kCommReduce) {
      reduction_block_vars.insert(iter_var->var.get());
    }
  }
  // Check 2. Each reduction iter_var should not be used to index output buffer
  for (const PrimExpr& idx : output_buffer_indices) {
    if (ContainsVar(idx, reduction_block_vars)) {
      return false;
    }
  }
  return true;
}

/******** Constructors ********/

StmtSRef::StmtSRef(const StmtNode* stmt, StmtSRefNode* parent, int64_t seq_index,
                   bool affine_block_binding) {
  ObjectPtr<StmtSRefNode> n = make_object<StmtSRefNode>();
  n->stmt = stmt;
  n->parent = parent;
  n->seq_index = seq_index;
  n->affine_block_binding = affine_block_binding;
  data_ = std::move(n);
}

StmtSRef StmtSRef::InlineMark() {
  static StmtSRef result(nullptr, nullptr, -1, false);
  return result;
}

StmtSRef StmtSRef::RootMark() {
  static StmtSRef result(nullptr, nullptr, -1, false);
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
  TBufferReaderWriter buffer_readers;
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

/******** Property of a block ********/

bool IsDominant(const BlockScopeNode* self, const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // Cond 1. Block is the only writer to its outputs
  const TBufferReaderWriter& buffer_writers = self->buffer_writers;
  for (const BufferRegion& write_region : block->writes) {
    ICHECK(buffer_writers.count(write_region->buffer))
        << "InternalError: buffer \"" << write_region->buffer->name
        << "\" does not exist in the current scope, when querying block:\n"
        << GetRef<Block>(block);
    // Check if the buffer is only written once (by the given block)
    if (buffer_writers.at(write_region->buffer).size() != 1) {
      return false;
    }
  }
  return true;
}

bool BlockScopeNode::IsComplete(const StmtSRef& block_sref) const {
  // Cond 2. Check if all the block vars are data parallel
  const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
  for (const IterVar& iter_var : block->iter_vars) {
    if (iter_var->iter_type != kDataPar) {
      return false;
    }
  }
  // Cond 1. A complete block must be dominate
  if (!IsDominant(this, block_sref)) {
    return false;
  }
  // Cond 3. Check if there is no overlap between buffers read and buffers written
  for (const BufferRegion& write : block->writes) {
    const Buffer& buffer = write->buffer;
    for (const BufferRegion& read : block->reads) {
      if (buffer.same_as(read->buffer)) {
        return false;
      }
    }
  }
  return true;
}

bool BlockScopeNode::IsReduction(const StmtSRef& block_sref) const {
  // Cond 3. Block binding is valid iter affine map
  if (!block_sref->affine_block_binding) {
    return false;
  }
  // Cond 4. Check whether the block body has the init statement.
  const auto* block = TVM_SREF_TO_BLOCK(block, block_sref);
  if (!block->init.defined()) {
    return false;
  }
  // Cond 2. All block vars are either data parallel or reduction
  const Array<IterVar>& iter_vars = block->iter_vars;
  for (const IterVar& iter_var : iter_vars) {
    if (iter_var->iter_type != kDataPar && iter_var->iter_type != kCommReduce) {
      return false;
    }
  }
  // Cond 1. Dominate block
  if (!IsDominant(this, block_sref)) {
    return false;
  }
  // Cond 5. All reduction vars should not affect indexing the output buffer
  std::unordered_set<const BufferNode*> buffer_written;
  buffer_written.reserve(block->writes.size());
  for (const BufferRegion& write_region : block->writes) {
    buffer_written.insert(write_region->buffer.get());
  }
  bool not_affected = true;
  PreOrderVisit(block->body, [&not_affected, &iter_vars, &buffer_written](const ObjectRef& obj) {
    if (!not_affected) {
      return false;
    }
    if (const auto* store = obj.as<BufferStoreNode>()) {
      // Only consider buffers written by the block
      if (buffer_written.count(store->buffer.get())) {
        if (!CheckReductionInstance(iter_vars, store->indices)) {
          not_affected = false;
        }
      } else {
        LOG(FATAL) << "InternalError: A write buffer is not in the block signature: "
                   << store->buffer;
      }
      return false;
    }
    return true;
  });
  return not_affected;
}

/******** Inter-block properties ********/

bool BlockScopeNode::CanMergeReduction(const StmtSRef& init_sref,
                                       const StmtSRef& update_sref) const {
  const auto* init = init_sref->GetStmt<BlockNode>();
  const auto* update = update_sref->GetStmt<BlockNode>();
  ICHECK(init != nullptr) << "InternalError: Scope::CanMergeReduction only accepts tir::Block as "
                             "init_block, but get type:"
                          << init_sref->stmt->GetTypeKey();
  ICHECK(update != nullptr) << "InternalError: Scope::CanMergeReduction only accepts tir::Block as "
                               "update_block, but get type:"
                            << update_sref->stmt->GetTypeKey();
  // Cond 1. Check the binding of update block is valid
  if (!update_sref->affine_block_binding) {
    return false;
  }
  // Cond 2. Check init_block and update_block are the only two producers for their output buffer
  for (const BufferRegion& write_region : update->writes) {
    const Array<StmtSRef>& writers = this->buffer_writers.at(write_region->buffer);
    if (writers.size() != 2) {
      return false;
    }
    if (!writers[0].same_as(init_sref) && !writers[0].same_as(update_sref)) {
      return false;
    }
    if (!writers[1].same_as(init_sref) && !writers[1].same_as(update_sref)) {
      return false;
    }
  }
  // Cond 3. init and update share the same buffer
  const auto* init_body = init->body.as<BufferStoreNode>();
  const auto* update_body = update->body.as<BufferStoreNode>();
  ICHECK(init_body != nullptr)
      << "InternalError: init_block should contain only a BufferStore as its lhs, but get type: "
      << init->body->GetTypeKey();
  ICHECK(update_body != nullptr)
      << "InternalError: update_block should contain only a BufferStore as its lhs, but get type:"
      << update->body->GetTypeKey();
  if (!init_body->buffer.same_as(update_body->buffer)) {
    return false;
  }
  // Access must be the same dimensional
  ICHECK_EQ(init_body->indices.size(), update_body->indices.size())
      << "InternalError: indexing to the same buffer with different dimensions";
  // Cond 4. All block vars of update_block are either data parallel or reduction,
  // and reduction vars of update_block should not affect indexing the output buffer
  return CheckReductionInstance(update->iter_vars, update_body->indices);
}

/******** FFI ********/

TVM_REGISTER_NODE_TYPE(StmtSRefNode);
TVM_REGISTER_NODE_TYPE(DependencyNode);
TVM_REGISTER_NODE_TYPE(BlockScopeNode);

TVM_REGISTER_GLOBAL("tir.schedule.StmtSRefStmt")
    .set_body_typed([](StmtSRef sref) -> Optional<Stmt> {
      return sref->stmt != nullptr ? GetRef<Stmt>(sref->stmt) : Optional<Stmt>(NullOpt);
    });
TVM_REGISTER_GLOBAL("tir.schedule.StmtSRefParent")
    .set_body_typed([](StmtSRef sref) -> Optional<StmtSRef> {
      return sref->parent != nullptr ? GetRef<StmtSRef>(sref->parent) : Optional<StmtSRef>(NullOpt);
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
