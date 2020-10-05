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
#include <tvm/tir/builtin.h>
#include <tvm/tir/schedule.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>

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
    bool equal = (n->type_index() == other->type_index()) && ExprComparator::VisitExpr(n, other);
    if (!equal && assert_mode_)
      LOG(INFO) << "Exprs are not matching between:" << n << " and " << other;
    return equal;
  }

  // Stmts
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
    return CompareBufferAccess(op, rhs) && VisitExpr(op->value, rhs->value);
  }

  bool VisitStmt_(const BlockRealizeNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BlockRealizeNode>();
    // Skip Compare binding values if the block is scope block (the outermost one).
    if (!is_scope_block) {
      size_t offset = op->binding_values.size() - rhs->binding_values.size();
      if (rhs->binding_values.size() > op->binding_values.size()) return false;
      for (size_t i = 0; i < rhs->binding_values.size(); i++) {
        if (!VisitExpr(op->binding_values[i + offset], rhs->binding_values[i])) return false;
      }
      const Block& block = op->block;
      for (size_t i = 0; i < offset; i++) {
        Var block_var = Downcast<Var>(op->binding_values[i]);
        auto it = equal_map_.find(block_var);
        equal_map_[block->iter_vars[i]->var] = (it == equal_map_.end() ? block_var : it->second);
      }
    }

    return VisitExpr(op->predicate, rhs->predicate) && op->exec_scope == rhs->exec_scope &&
           VisitStmt(op->block, rhs->block);
  }

  bool VisitStmt_(const BlockNode* op, const Stmt& other) final {
    const auto* rhs = other.as<BlockNode>();
    // Check block equal
    // All iter var and buffer region should matches including the order

    // Check iterVar
    // need to use DefEqual to remap vars
    // Note:
    //    We only compare the inner most several axis
    if (op->iter_vars.size() < rhs->iter_vars.size()) return false;

    size_t offset = op->iter_vars.size() - rhs->iter_vars.size();
    for (size_t i = 0; i < rhs->iter_vars.size(); ++i) {
      auto lhs_var = op->iter_vars[i + offset], rhs_var = rhs->iter_vars[i];
      // Skip iter dom
      if (!DefEqual(lhs_var->var, rhs_var->var)) return false;
      if (lhs_var->iter_type != rhs_var->iter_type) return false;
    }

    for (size_t i = 0; i < offset; i++) {
      if (is_scope_block) {
        extra_block_vars_.push_back(op->iter_vars[i]);
      }
    }
    is_scope_block = false;
    if (!CompareArray(op->writes, rhs->writes, &TensorizeComparator::CompareTensorRegion))
      return false;
    if (!CompareArray(op->reads, rhs->reads, &TensorizeComparator::CompareTensorRegion))
      return false;
    if (!CompareArray(op->annotations, rhs->annotations, &TensorizeComparator::CompareAnnotation))
      return false;
    return VisitStmt(op->body, rhs->body);
  }

  // Map from rhs buffer to lhs buffer
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> rhs_buffer_map_;
  // Buffer indices mapping
  std::unordered_map<Buffer, std::vector<PrimExpr>, ObjectPtrHash, ObjectPtrEqual> buffer_indices_;
  std::vector<IterVar> extra_block_vars_;

// Exprs
#define TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(OpName)             \
  bool VisitExpr_(const OpName* op, const PrimExpr& other) final { \
    const auto* rhs = other.as<OpName>();                          \
    return VisitExpr(op->a, rhs->a) && VisitExpr(op->b, rhs->b);   \
  }

  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(AddNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(SubNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MulNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(DivNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(ModNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(EQNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(NENode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(LTNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(LENode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(GTNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(GENode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(AndNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(OrNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MinNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(MaxNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(FloorDivNode);
  TVM_DECLARE_TENSORIZE_COMPARATOR_BINOP(FloorModNode);

  bool VisitExpr_(const IntImmNode* op, const PrimExpr& other) final {
    const auto* rhs = other.as<IntImmNode>();
    return CompareType(op->dtype, rhs->dtype) && op->value == rhs->value;
  }

  bool VisitExpr_(const FloatImmNode* op, const PrimExpr& other) final {
    const auto* rhs = other.as<FloatImmNode>();
    return CompareType(op->dtype, rhs->dtype) && op->value == rhs->value;
  }

  bool VisitExpr_(const CastNode* op, const PrimExpr& other) final {
    const auto* rhs = other.as<CastNode>();
    return CompareType(op->dtype, rhs->dtype) && VisitExpr(op->value, rhs->value);
  }

  bool VisitExpr_(const VarNode* op, const PrimExpr& other) final {
    const auto* rhs = other.as<VarNode>();
    auto lhs = GetRef<Var>(op);
    if (lhs.same_as(other)) return true;
    if (!CompareType(op->dtype, rhs->dtype)) return false;
    auto it = equal_map_.find(lhs);
    return it != equal_map_.end() && it->second.same_as(other);
  }

  bool VisitExpr_(const BufferLoadNode* op, const PrimExpr& other) final {
    const auto* rhs = other.as<BufferLoadNode>();
    return CompareBufferAccess(op, rhs);
  }

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
                 CompareType(lhs->dtype, rhs->dtype) && lhs->scope == rhs->scope;
    if (equal)
      rhs_buffer_map_[rhs] = lhs;
    else if (assert_mode_) {
      LOG(FATAL) << "Buffers are not matching between:" << lhs << " and " << rhs;
    }
    return equal;
  }

  bool CompareTensorRegion(const TensorRegion& lhs, const TensorRegion& rhs) {
    // Only for block region declaration
    if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;
    // Number of indices in desc_block must be smaller than it in AST
    if (rhs->region.size() > lhs->region.size()) return false;
    size_t offset = lhs->region.size() - rhs->region.size();
    bool need_update = false;
    if (auto it = buffer_indices_.find(lhs->buffer) == buffer_indices_.end()) {
      need_update = true;
      buffer_indices_[lhs->buffer] = std::vector<PrimExpr>();
    } else {
      if (offset != buffer_indices_[lhs->buffer].size()) return false;
    }
    std::vector<PrimExpr>& indices = buffer_indices_[lhs->buffer];
    for (size_t i = 0; i < offset; ++i) {
      const Range& range = lhs->region[i];
      // High-dim region must be element-wise
      if (!is_one(range->extent)) return false;
      if (need_update) {
        indices.push_back(range->min);
      } else {
        // The order matters since we only map inner block_var to outside block_var
        if (!VisitExpr(range->min, indices[i])) return false;
      }
    }

    for (size_t i = 0; i < rhs->region.size(); ++i) {
      if (!CompareRange(lhs->region[i + offset], rhs->region[i])) return false;
    }

    return true;
  }

  // Only for BufferStoreNode and BufferLoadNode
  template <typename T>
  bool CompareBufferAccess(const T* lhs, const T* rhs) {
    if (!CompareBuffer(lhs->buffer, rhs->buffer)) return false;

    if (rhs->indices.size() > lhs->indices.size()) return false;
    size_t offset = lhs->indices.size() - rhs->indices.size();

    for (size_t i = 0; i < rhs->indices.size(); ++i) {
      if (!VisitExpr(lhs->indices[i + offset], rhs->indices[i])) return false;
    }
    return true;
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

  bool CompareRange(const Range& lhs, const Range& rhs) {
    return VisitExpr(lhs->min, rhs->min) && VisitExpr(lhs->extent, rhs->extent);
  }

  bool CompareType(const DataType& lhs, const DataType& rhs) {
    if (lhs == rhs) return true;
    return lhs.code() == rhs.code() && lhs.bits() == rhs.bits() && lhs.lanes() == rhs.lanes();
  }

  // variable remap if any
  std::unordered_map<ObjectRef, ObjectRef, ObjectPtrHash, ObjectPtrEqual> equal_map_;
  bool assert_mode_;
  bool is_scope_block = true;
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
      const std::unordered_map<const VarNode*, const PrimExprNode*>& var_map,
      std::vector<IterVar>&& extra_block_vars,
      const std::unordered_map<Buffer, std::vector<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>&
          buffer_indices)
      : buffer_map_(buffer_map),
        var_map_(var_map),
        extra_block_vars_(std::move(extra_block_vars)),
        buffer_indices_(buffer_indices) {}

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
      auto it2 = block_var_map_.find(op);
      if (it2 != block_var_map_.find(op)) {
        return GetRef<PrimExpr>(it2->second);
      } else {
        return GetRef<PrimExpr>(op);
      }
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    std::vector<IterVar> extra_block_var;
    std::unordered_map<const VarNode*, const PrimExprNode*> block_var_map;
    for (const auto& iter_var : extra_block_vars_) {
      auto n = runtime::make_object<IterVarNode>(*(iter_var.get()));
      IterVar block_var(n);
      extra_block_var.push_back(block_var);
      block_var_map[iter_var->var.get()] = block_var->var.get();
    }
    std::swap(block_var_map, block_var_map_);
    auto s = StmtExprMutator::VisitStmt_(op);
    op = s.as<BlockNode>();
    CHECK(op);

    auto iter_vars = op->iter_vars;
    iter_vars.insert(iter_vars.begin(), extra_block_var.begin(), extra_block_var.end());
    auto reads = UpdateBufferViaMap(op->reads);
    auto writes = UpdateBufferViaMap(op->writes);

    std::swap(block_var_map, block_var_map_);

    if (reads.same_as(op->reads) && writes.same_as(op->writes) &&
        iter_vars.same_as(op->iter_vars)) {
      return GetRef<Block>(op);
    } else {
      auto n = CopyOnWrite(op);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->iter_vars = std::move(iter_vars);
      return Block(n);
    }
  }

 private:
  const std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>& buffer_map_;
  const std::unordered_map<const VarNode*, const PrimExprNode*>& var_map_;
  std::unordered_map<const VarNode*, const PrimExprNode*> block_var_map_;
  const std::vector<IterVar>& extra_block_vars_;
  const std::unordered_map<Buffer, std::vector<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>&
      buffer_indices_;

  Array<TensorRegion> UpdateBufferViaMap(const Array<TensorRegion>& tensor_regions) {
    auto fmutate = [this](const TensorRegion& tensor_region) {
      auto it = buffer_map_.find(tensor_region->buffer);
      if (it != buffer_map_.end()) {
        auto n = CopyOnWrite(tensor_region.operator->());
        n->buffer = it->second;
        auto it2 = buffer_indices_.find(n->buffer);
        if (it2 != buffer_indices_.end()) {
          Region region;
          for (const auto& min : it2->second) {
            region.push_back(Range::FromMinExtent(VisitExpr(min), 1));
          }
          n->region.insert(n->region.begin(), region.begin(), region.end());
        }
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

  const auto* intrin_block_realize = intrinsic->implementation->body.as<BlockRealizeNode>();
  const Block& intrin_block = intrin_block_realize->block;

  const StmtSRef& block_sref = blockize(sref, intrin_block_realize->exec_scope);
  const BlockRealize& block_realize = GetBlockRealize(block_sref);

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

  std::unordered_map<const VarNode*, const PrimExprNode*> var_map;
  auto update_var_map = [&var_map](const PrimExpr& lhs, const PrimExpr& rhs) {
    if (const auto* var = lhs.as<VarNode>()) {
      var_map[var] = rhs.get();
    }
  };

  for (const auto& pair : buffer_map) {
    update_var_map(pair.first->data, pair.second->data);
  }

  CHECK(intrin_block_realize);
  // Mutate description function
  Stmt new_stmt = BufferReplacer(buffer_map, var_map, std::move(comparator.extra_block_vars_),
                                 comparator.buffer_indices_)(intrin_block_realize->block);

  const auto* block_node = new_stmt.as<BlockNode>();

  std::unordered_map<const VarNode*, PrimExpr> element_offset;
  auto get_element_offset = [&element_offset](const Array<TensorRegion>& old_regions,
                                              const Array<TensorRegion>& new_regions) {
    CHECK_EQ(old_regions.size(), new_regions.size());
    for (size_t i = 0; i < old_regions.size(); ++i) {
      Array<PrimExpr> indices;
      const auto& old_region = old_regions[i];
      const auto& new_region = new_regions[i];
      for (const auto range : old_region->region) {
        indices.push_back(range->min);
      }
      if (const auto* var = new_region->buffer->elem_offset.as<VarNode>()) {
        PrimExpr call = Call(DataType::Int(32), builtin::get_elem_offset(),
                             {BufferLoad(old_region->buffer, indices)});
        auto it = element_offset.find(var);
        if (it != element_offset.end()) {
          CHECK(ExprDeepEqual()(it->second, call));
        } else {
          element_offset[var] = call;
        }
      }
    }
  };
  get_element_offset(block_node->reads, intrin_block->reads);
  get_element_offset(block_node->writes, intrin_block->writes);

  Block new_block = Downcast<Block>(Substitute(new_stmt, element_offset));

  // Replace
  Map<Block, Block> block_map;
  block_map.Set(new_block, block_realize->block);
  this->Replace(block_sref, new_block, block_map);
}

}  // namespace tir
}  // namespace tvm