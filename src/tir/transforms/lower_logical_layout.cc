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
 * \brief Lower logical layout into physical layout
 * \file lower_logical_layout.cc
 */
#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../../arith/pattern_match.h"
#include "../schedule/analysis.h"
#include "./ir_utils.h"

namespace tvm {
namespace tir {

using FLowerLogicalLayout = TypedPackedFunc<Array<PrimExpr>(Array<PrimExpr>)>;

class LogicalLayoutNode : public Object {
 public:
  FLowerLogicalLayout lower_func;
  int num_dims;
  TVM_DECLARE_FINAL_OBJECT_INFO(LogicalLayoutNode, Object);

  void VisitAttrs(AttrVisitor* v) {
    // Skip visiting lower_func
    v->Visit("num_dims", &num_dims);
  }
  static constexpr const char* _type_key = "tir.LogicalLayout";
};

class LogicalLayout : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(LogicalLayout, ObjectRef, LogicalLayoutNode);

  explicit LogicalLayout(FLowerLogicalLayout lower_func, int num_dims) {
    ObjectPtr<LogicalLayoutNode> n = make_object<LogicalLayoutNode>();
    n->lower_func = std::move(lower_func);
    n->num_dims = num_dims;
    this->data_ = std::move(n);
  }
};

TVM_REGISTER_NODE_TYPE(LogicalLayoutNode);
class LogicalLayoutRegistry {
 public:
  Map<String, LogicalLayout> reg;

  static LogicalLayoutRegistry* Global() {
    static LogicalLayoutRegistry* inst = new LogicalLayoutRegistry();
    return inst;
  }

  static void Register(const String& name, FLowerLogicalLayout lower_func, int num_dims) {
    Global()->reg.Set(name, LogicalLayout(lower_func, num_dims));
  }
};

class LogicalLayoutMutator : public StmtExprMutator {
 public:
  Stmt Rewrite(const Stmt& stmt) { return LogicalLayoutMutator()(stmt); }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    loop_map_.emplace(op->loop_var, GetRef<For>(op));
    auto new_stmt = StmtExprMutator::VisitStmt_(op);
    loop_map_.erase(op->loop_var);
    if (removed_loops.count(op->loop_var)) {
      return Downcast<For>(new_stmt)->body;
    }
    return new_stmt;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    BlockNode* n = block.CopyOnWrite();

    for (size_t i = 0; i < block->alloc_buffers.size(); i++) {
      if (buffer_map_.count(block->alloc_buffers[i])) {
        n->alloc_buffers.Set(i, buffer_map_.at(block->alloc_buffers[i]));
      }
    }
    return block;
  }

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
    const auto& reg = LogicalLayoutRegistry::Global()->reg;
    auto it = reg.find(store->buffer->scope);
    if (it == reg.end()) {
      return store;
    }

    BufferStoreNode* op = store.CopyOnWrite();
    auto& indices = op->indices;
    size_t orig_buffer_num_dims = indices.size();
    const LogicalLayout& logical_layout = (*it).second;
    CHECK_LE(logical_layout->num_dims, orig_buffer_num_dims)
        << "ValueError: The lower function of logical layout " << op->buffer->scope
        << " expects the buffer to be at least " << logical_layout->num_dims
        << "-D (actual: " << orig_buffer_num_dims << "-D).";

    Array<PrimExpr> leading_indices = GetLeadingIndices(indices, logical_layout->num_dims);
    // Collect related outer loops of the block
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> index_vars =
        CollectVars(leading_indices);

    if (!NeedRewriteOuterLoops(_op, index_vars)) {
      RewriteBufferIndices(logical_layout, &op->indices);
      auto leading_indices = GetLeadingIndices(op->indices, op->indices.size() - orig_buffer_num_dims);
      auto leading_shape = InferRange(leading_indices);
      ReallocOrValidateBuffer(&op->buffer, logical_layout, leading_shape);
      return store;
    }

    // mark the outer loops to remove
    for (const Var& index_var : index_vars) {
      removed_loops.insert(index_var);
    }

    // rewrite the indices and infer the extents of outer loops
    leading_indices = logical_layout->lower_func(leading_indices);
    auto leading_shape = InferRange(leading_indices);
    std::vector<Stmt> loop_nests = BuildLoopNests(leading_shape);
    Array<PrimExpr> new_loop_vars;
    new_loop_vars.reserve(loop_nests.size());
    for (const auto& loop_nest : loop_nests) {
      new_loop_vars.push_back(loop_nest.as<ForNode>()->loop_var);
    }
    // use the new loop vars as the new indices
    indices.resize(indices.size() - logical_layout->num_dims);
    indices.insert(indices.end(), new_loop_vars.begin(), new_loop_vars.end());

    auto inverse_var_map = GetInverseAffineIterMap(indices, index_vars, new_loop_vars);
    op->value = Substitute(op->value, inverse_var_map);
    ReallocOrValidateBuffer(&op->buffer, logical_layout, leading_shape);
    return MergeNest(loop_nests, store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    const auto& reg = LogicalLayoutRegistry::Global()->reg;
    auto it = reg.find(load->buffer->scope);
    if (it == reg.end()) {
      return load;
    }
    const LogicalLayout& logical_layout = (*it).second;
    size_t orig_buffer_num_dims = load->indices.size();
    CHECK(buffer_map_.count(load->buffer)) << "ValueError: Cannot find the producer of the buffer "
                                           << load->buffer->name << " with logical layout.";
    CHECK_LE(logical_layout->num_dims, orig_buffer_num_dims)
        << "ValueError: The lower function of logical layout " << load->buffer->scope
        << " expects the buffer to be at least " << logical_layout->num_dims
        << "-D (actual: " << orig_buffer_num_dims << "-D.";
    BufferLoadNode* op = load.CopyOnWrite();
    RewriteBufferIndices(logical_layout, &op->indices);
    op->buffer = buffer_map_.at(load->buffer);
    return std::move(load);
  }

  void ReallocOrValidateBuffer(Buffer *buffer, const LogicalLayout& logical_layout, const arith::NDIntSet& leading_shape) {
    Array<PrimExpr> new_shape(
        (*buffer)->shape.begin(),
        (*buffer)->shape.end() - logical_layout->num_dims);
    for (const auto& range : leading_shape) {
      new_shape.push_back(range.max() + 1);
    }
    new_shape.reserve(new_shape.size() + leading_shape.size());

    auto it = buffer_map_.find(*buffer);
    if (it != buffer_map_.end()) {
      const auto& new_buffer = it->second;
      ICHECK(new_buffer->shape.size() == new_shape.size());
      for (size_t i = 0; i < new_shape.size(); i++) {
        CHECK(analyzer_.CanProveEqual(new_buffer->shape[i], new_shape[i])) << "ValueError: Inconsistent buffer shape of the buffer after logical layout lowering.";
      }
      return;
    }
    ObjectPtr<BufferNode> n = make_object<BufferNode>(*(buffer->get()));
    std::string scope = n->scope;
    n->scope = scope.substr(0, scope.find('.'));  // remove the suffix of the logical layout
    Buffer new_buffer = Buffer(std::move(n));
    buffer_map_.emplace(*buffer, new_buffer);
    *buffer = new_buffer;
  }

  Map<Var, PrimExpr> GetInverseAffineIterMap(
      const Array<PrimExpr>& indices,
      std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& index_vars,
      const Array<PrimExpr>& loop_vars) {
    Map<Var, Range> input_iters;
    for (const Var& index_var : index_vars) {
      const auto& for_loop = loop_map_.at(index_var);
      input_iters.Set(index_var, Range::FromMinExtent(for_loop->min, for_loop->extent));
    }
    Array<arith::IterSumExpr> iter_map =
        arith::DetectIterMap(indices, input_iters, Bool(true), true, &analyzer_);
    return arith::InverseAffineIterMap(iter_map, loop_vars);
  }

  // Infer range of the indices by evaluating NDIntSet
  arith::NDIntSet InferRange(const Array<PrimExpr> indices) {
    arith::NDIntSet nd_int_set = arith::NDIntSetFromPoint(indices);
    std::unordered_map<const VarNode*, arith::IntSet> dom_map;
    for (const auto& index : indices) {
      PostOrderVisit(index, [&](const ObjectRef& obj) {
        if (obj.as<VarNode>()) {
          const For& for_loop = loop_map_.at(Downcast<Var>(obj));
          dom_map.emplace(obj.as<VarNode>(),
                          arith::IntSetFromMinExtent(for_loop->min, for_loop->extent));
        }
      });
    }
    arith::NDIntSet new_ranges = arith::EvalNDIntSet(nd_int_set, dom_map);
    // Validate the new ranges has zero as minumum
    for (const auto& range : new_ranges) {
      CHECK(is_zero(range.min())) << "ValueError: the transformed indices of the logical layout "
                                     "should have zero as minimum.";
      CHECK(!range.IsNothing() && !arith::is_pos_inf(range.max()))
          << "ValueError: Invalid range " << range << " of the transformed indices of the buffer with logical "
             "layout";
    }
    return new_ranges;
  }

  Array<PrimExpr> GetLeadingIndices(const Array<PrimExpr>& indices, int num_dims) {
    Array<PrimExpr> leading_indices(indices.end() - num_dims, indices.end());
    return leading_indices;
  }

  // Collect variables in the indices
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> CollectVars(
      const Array<PrimExpr>& indices) {
    std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> vars;
    for (const PrimExpr& index : indices) {
      PostOrderVisit(index, [&vars](const ObjectRef& obj) {
        if (obj.as<VarNode>()) {
          vars.insert(Downcast<Var>(obj));
        }
      });
    }
    return vars;
  }

  // Build loop nests to cover the given NDIntSet
  std::vector<Stmt> BuildLoopNests(const arith::NDIntSet& shape) {
    std::vector<Stmt> loop_vars;
    const auto nop = Evaluate(Integer(0));
    for (size_t i = 0; i < shape.size(); i++) {
      Range dom(shape[i].min(), shape[i].max() + 1);
      // TODO: how should we name these loop vars?
      if (i == 0) {
        // the outermost loop is automatically bound to thread axis
        IterVar iter_var(dom, Var(), IterVarType::kDataPar, "threadIdx.x");
        loop_vars.push_back(
            For(iter_var->var, dom->min, dom->extent, ForKind::kThreadBinding, nop, iter_var));
      } else {
        loop_vars.push_back(For(Var(), dom->min, dom->extent, ForKind::kSerial, nop));
      }
    }
    return loop_vars;
  }

  // Check whether need to rewrite outer loops to make access to the buffer continuous.
  bool NeedRewriteOuterLoops(
      const StmtNode* stmt,
      const std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>& loop_vars) {

    // We require the loop variables that appear in the buffer indices to be the loops that are direct
    // ancestors of the buffer store.
    // For example, we require the outer loops to have the
    // following pattern:
    // for (ax0, ...) {
    //   for (ax1, ...) {
    //     for (ax2, ...) {
    //       buf[ax0 * 4 + ax1, ax2] = ...
    //     }
    //   }
    // }
    // Note that these outer loops can be reordered, but no other statements can be their children.

    for (const Var& loop_var : loop_vars) {
      const auto& for_loop = loop_map_.at(loop_var);
      if (for_loop->body.get() == stmt || for_loop->body.as<BlockRealizeNode>()) {
        continue;
      } else if (const auto* child_loop = for_loop->body.as<ForNode>()) {
        if (loop_vars.count(child_loop->loop_var)) {
          continue;
        }
      }
      return false;
    }
    return true;
  }

  Buffer MakeNewBuffer(const Buffer& orig_buffer, const LogicalLayout& logical_layout,
                       const arith::NDIntSet& leading_shape) const {
    ObjectPtr<BufferNode> n = make_object<BufferNode>(*orig_buffer.get());
    Array<PrimExpr> new_shape(
        orig_buffer->shape.begin(),
        orig_buffer->shape.begin() + orig_buffer->shape.size() - logical_layout->num_dims);
    for (const auto& range : leading_shape) {
      new_shape.push_back(range.max() + 1);
    }
    n->shape = std::move(new_shape);
    std::string scope = n->scope;
    n->scope = scope.substr(0, scope.find('.'));  // remove the suffix of the logical layout
    return Buffer(std::move(n));
  }

  void RewriteBufferIndices(const LogicalLayout& logical_layout, Array<PrimExpr>* indices) {
    Array<PrimExpr> leading_indices = GetLeadingIndices(*indices, logical_layout->num_dims);
    leading_indices = logical_layout->lower_func(leading_indices);
    indices->resize(indices->size() - logical_layout->num_dims);
    indices->insert(indices->end(), leading_indices.begin(), leading_indices.end());
  }

  std::unordered_map<Var, For, ObjectPtrHash, ObjectPtrEqual> loop_map_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> removed_loops;
  arith::Analyzer analyzer_;
};

namespace transform {

Pass LowerLogicalLayout() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = LogicalLayoutMutator().Rewrite(std::move(f->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerLogicalLayout", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerLogicalLayout").set_body_typed(LowerLogicalLayout);
TVM_REGISTER_GLOBAL("tir.LogicalLayoutRegister").set_body_typed(LogicalLayoutRegistry::Register);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
