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

#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include "../ir/functor_common.h"
#include "./schedule_common.h"

namespace tvm {
namespace tir {

using ExprComparator = ExprFunctor<bool(const PrimExpr& n, const PrimExpr& other)>;
using StmtComparator = StmtFunctor<bool(const Stmt& n, const Stmt& other)>;

// Deep comparison to check if two IR graph are equivalent
class TensorizeComparator : public ExprComparator, public StmtComparator {
 public:
  explicit TensorizeComparator(bool assert_mode = true) : assert_mode_(assert_mode) {}

  bool VisitExpr(const PrimExpr& n, const PrimExpr& other) override {
    bool equal = StructuralEqual().EqualWithMap(n, other, equal_map_);
    if (!equal && assert_mode_)
      LOG(FATAL) << "Exprs are not matching between:" << n << " and " << other;
    return equal;
  }

  bool VisitStmt(const Stmt& n, const Stmt& other) override {
    if (n.same_as(other)) return true;
    if (n->type_index() != other->type_index()) return false;
    bool equal = StmtComparator::VisitStmt(n, other);
    if (!equal && assert_mode_)
      LOG(FATAL) << "Stmts are not matching between:\n" << n << "\nand\n" << other;
    return equal;
  }

  bool VisitStmt_(const LoopNode* op, const Stmt& other) final {
    const auto* rhs = other.as<LoopNode>();
    if (!DefEqual(op->loop_var, rhs->loop_var)) return false;
    if (!VisitExpr(op->min, rhs->min)) return false;
    if (!VisitExpr(op->extent, rhs->extent)) return false;
    if (!VisitStmt(op->body, rhs->body)) return false;
    return CompareArray(op->annotations, rhs->annotations, &TensorizeComparator::CompareAnnotation);
  }

  bool VisitStmt_(const SeqStmtNode* op, const Stmt& other) final {
    const auto* rhs = other.as<SeqStmtNode>();
    return CompareArray(op->seq, rhs->seq, &TensorizeComparator::VisitStmt);
  }

  bool VisitStmt_(const BufferAllocateNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BufferAllocateNode>();
    return CompareBuffer(op->buffer, rhs->buffer) && op->scope == rhs->scope;
  }

  bool VisitStmt_(const BufferStoreNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BufferStoreNode>();
    return CompareBuffer(op->buffer, rhs->buffer) &&
           CompareArray(op->indices, rhs->indices, &TensorizeComparator::VisitExpr) &&
           VisitExpr(op->value, rhs->value);
  }

  bool VisitStmt_(const BlockRealizeNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BlockRealizeNode>();
    // Skip Compare binding values.
    return VisitExpr(op->predicate, rhs->predicate) && VisitStmt(op->block, rhs->block);
  }

  bool VisitStmt_(const BlockNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BlockNode>();
    // Check block equal
    // All iter var and buffer region should matches including the order

    // Check iterVar
    // need to use DefEqual to remap vars
    if (op->iter_vars.size() != rhs->iter_vars.size()) return false;
    for (size_t i = 0; i < op->iter_vars.size(); ++i) {
      auto lhs_var = op->iter_vars[i], rhs_var = rhs->iter_vars[i];
      // Skip iter dom
      if (!DefEqual(lhs_var->var, rhs_var->var)) return false;
      if (lhs_var->iter_type != rhs_var->iter_type) return false;
    }
    if (!CompareArray(op->writes, rhs->writes, &TensorizeComparator::CompareTensorRegion))
      return false;
    if (!CompareArray(op->reads, rhs->reads, &TensorizeComparator::CompareTensorRegion))
      return false;
    if (!CompareArray(op->annotations, rhs->annotations, &TensorizeComparator::CompareAnnotation))
      return false;
    return VisitStmt(op->body, rhs->body);
  }

  // Map from rhs buffer to lhs buffer
  std::unordered_map<Buffer, Buffer, ObjectHash, ObjectEqual> rhs_buffer_map_;

 private:
  bool DefEqual(const ObjectRef& lhs, const ObjectRef& rhs) {
    if (lhs.same_as(rhs)) return true;
    if (lhs->type_index() != rhs->type_index()) return false;
    auto it = equal_map_.find(lhs);
    // If there is already a mapping
    if (it != equal_map_.end()) return it->second.same_as(rhs);
    equal_map_[lhs] = rhs;
    return true;
  }

  bool CompareAnnotation(const Annotation& lhs, const Annotation& rhs) {
    if (lhs.same_as(rhs)) return true;
    return VisitExpr(lhs->value, rhs->value) && lhs->attr_key == rhs->attr_key;
  }

  bool CompareBuffer(const Buffer& lhs, const Buffer& rhs) {
    if (lhs.same_as(rhs)) return true;
    // Remap both buffer itself and buffer data
    // Skip buffer shape
    bool equal = DefEqual(lhs, rhs) && DefEqual(lhs->data, rhs->data) &&
                 lhs->buffer_type == rhs->buffer_type && CompareType(lhs->dtype, rhs->dtype) &&
                 lhs->scope == rhs->scope;
    if (equal) rhs_buffer_map_[rhs] = lhs;
    return equal;
  }

  bool CompareTensorRegion(const TensorRegion& lhs, const TensorRegion& rhs) {
    return CompareBuffer(lhs->buffer, rhs->buffer) && CompareRegion(lhs->region, rhs->region);
  }

  template <typename T, typename F>
  bool CompareArray(const Array<T>& lhs, const Array<T>& rhs, F cmp) {
    if (lhs.same_as(rhs)) return true;
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!(this->*cmp)(lhs[i], rhs[i])) return false;
    }
    return true;
  }

  bool CompareRegion(const Region& lhs, const Region& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
      if (!VisitExpr(lhs[i]->min, rhs[i]->min)) return false;
      if (!VisitExpr(lhs[i]->extent, rhs[i]->extent)) return false;
    }
    return true;
  }

  bool CompareType(const DataType& lhs, const DataType& rhs) {
    if (lhs == rhs) return true;
    return lhs.code() == rhs.code() && lhs.bits() == rhs.bits() && lhs.lanes() == rhs.lanes();
  }

  // variable remap if any
  std::unordered_map<ObjectRef, ObjectRef, ObjectPtrHash, ObjectPtrEqual> equal_map_;
  bool assert_mode_;
};

void BufferRemap(const TensorIntrin& intrinsic,
                 std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* buffer_map) {
  CHECK_EQ(intrinsic->description->params.size(), intrinsic->implementation->params.size());
  for (size_t i = 0; i < intrinsic->description->params.size(); ++i) {
    const auto& lhs_var = intrinsic->description->params[i];
    const auto& lhs_buffer = intrinsic->description->buffer_map[lhs_var];
    const auto& rhs_var = intrinsic->implementation->params[i];
    const auto& rhs_buffer = intrinsic->implementation->buffer_map[rhs_var];
    (*buffer_map)[rhs_buffer] = lhs_buffer;
  }
}

// Replace buffer with its data, element_offset
class BufferReplacer : public StmtExprMutator {
 public:
  explicit BufferReplacer(
      const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>& buffer_map,
      const std::unordered_map<const VarNode*, const PrimExprNode*>& var_map)
      : buffer_map_(buffer_map), var_map_(var_map) {}

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto s = StmtExprMutator::VisitStmt_(op);
    op = s.as<BufferStoreNode>();
    CHECK(op);
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      return Stmt(n);
    } else {
      return GetRef<Stmt>(op);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto s = StmtExprMutator::VisitExpr_(op);
    op = s.as<BufferLoadNode>();
    CHECK(op);
    auto it = buffer_map_.find(op->buffer);
    if (it != buffer_map_.end()) {
      auto n = CopyOnWrite(op);
      n->buffer = it->second;
      return PrimExpr(n);
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = var_map_.find(op);
    if (it != var_map_.end()) {
      return GetRef<PrimExpr>(it->second);
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    auto s = StmtExprMutator::VisitStmt_(op);
    op = s.as<BlockNode>();
    CHECK(op);
    auto reads = UpdateBufferViaMap(op->reads);
    auto writes = UpdateBufferViaMap(op->writes);
    if (reads.same_as(op->reads) && writes.same_as(op->writes)) {
      return GetRef<Block>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      return Block(n);
    }
  }

 private:
  const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>& buffer_map_;
  const std::unordered_map<const VarNode*, const PrimExprNode*>& var_map_;

  Array<TensorRegion> UpdateBufferViaMap(const Array<TensorRegion>& tensor_regions) {
    auto fmutate = [this](const TensorRegion& tensor_region) {
      auto it = buffer_map_.find(tensor_region->buffer);
      if (it != buffer_map_.end()) {
        auto n = CopyOnWrite(tensor_region.operator->());
        n->buffer = it->second;
        return TensorRegion(n);
      } else {
        return tensor_region;
      }
    };
    return MutateArray(tensor_regions, fmutate, allow_copy_on_write_);
  }
};

void ScheduleNode::tensorize(const StmtSRef& sref, const TensorIntrin& intrinsic) {
  /*!
   * Check:
   *   - Check buffer binding, including type, alignment, shape and etc.
   *   - Check the sub AST is equal to the description function.
   *
   * Mutate:
   *   - Blockize the sub AST (please refer blockize for details)
   *   - Bind buffers
   *   - Mutate implement function with buffer binding
   *   - Replace the sub tree with the mutated function.
   */
  // TODO(Siyuan): fix range
  const auto* loop = sref->GetStmt<LoopNode>();
  CHECK(loop) << "Only support tensorize a loop for now";

  const StmtSRef& block_sref = blockize(sref, "");
  const BlockRealize& block_realize = GetBlockRealize(block_sref);
  const Block& block = block_realize->block;
  const auto* intrin_block_realize = intrinsic->implementation->body.as<BlockRealizeNode>();
  const Block& intrin_block = intrin_block_realize->block;
  TensorizeComparator comparator;

  bool equal = comparator.VisitStmt(block_realize, intrinsic->description->body);
  CHECK(equal) << "The AST subtree does not match intrinsic description";

  // Map from intrinsic func buffer to description func buffer
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> intrin_buffer_map;
  BufferRemap(intrinsic, &intrin_buffer_map);

  // Map form intrinsic func buffer to current AST buffer
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map;
  for (const auto& pair : intrin_buffer_map) {
    auto it = comparator.rhs_buffer_map_.find(pair.second);
    CHECK(it != comparator.rhs_buffer_map_.end());
    buffer_map[pair.first] = it->second;
  }

  std::unordered_map<Buffer, PrimExpr, ObjectHash, ObjectEqual> element_offset;
  auto get_element_offset = [&element_offset](const Array<TensorRegion>& old_regions,
                                              const Array<TensorRegion>& new_regions) {
    CHECK_EQ(old_regions.size(), new_regions.size());
    for (size_t i = 0; i < old_regions.size(); ++i) {
      const auto& old_region = old_regions[i];
      const auto& new_region = new_regions[i];
      PrimExpr offset = 0, stride = 1;
      const auto& buffer = old_region->buffer;
      const auto& region = new_region->region;
      for (size_t i = region.size(); i > 0; --i) {
        offset = region[i - 1]->min * stride + offset;
        stride *= buffer->shape[i - 1];
      }
      auto it = element_offset.find(buffer);
      if (it != element_offset.end()) {
        CHECK(ExprDeepEqual()(it->second, offset));
      } else {
        element_offset[buffer] = offset;
      }
    }
  };
  get_element_offset(block->reads, intrin_block->reads);
  get_element_offset(block->writes, intrin_block->writes);

  std::unordered_map<const VarNode*, const PrimExprNode*> var_map;
  auto update_var_map = [&var_map](const PrimExpr& lhs, const PrimExpr& rhs) {
    if (const auto* var = lhs.as<VarNode>()) {
      var_map[var] = rhs.get();
    }
  };

  for (const auto& pair : buffer_map) {
    update_var_map(pair.first->data, pair.second->data);

    auto it = element_offset.find(pair.second);
    CHECK(it != element_offset.end());
    auto offset = it->second;
    // TODO(Siyuan): add data alignment assert

    // Update elem_offset
    const auto& lhs_offset = pair.first->elem_offset;
    if (const auto* var = lhs_offset.as<VarNode>()) {
      var_map[var] = offset.get();
    }
  }

  // Update block var remapping
  for (size_t i = 0; i < block->iter_vars.size(); ++i) {
    var_map[block->iter_vars[i]->var.get()] = intrin_block->iter_vars[i]->var.get();
  }

  CHECK(intrin_block_realize);
  // Mutate description function
  Block new_block =
      Downcast<Block>(BufferReplacer(buffer_map, var_map)(intrin_block_realize->block));

  // Replace
  Map<Block, Block> block_map;
  block_map.Set(new_block, block_realize->block);
  this->Replace(block_sref, new_block, block_map);
}

}  // namespace tir
}  // namespace tvm
