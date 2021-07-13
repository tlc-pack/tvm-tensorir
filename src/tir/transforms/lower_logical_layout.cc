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
 *  Lower logical layout
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
  size_t num_dims;
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
  Stmt Rewrite(const Stmt& stmt) {
    return LogicalLayoutMutator()(stmt);
  }

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    loop_map_.emplace(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    // ancestor_loops_.push_back(GetRef<For>(op));
    auto new_stmt = StmtExprMutator::VisitStmt_(op);
    loop_map_.erase(op->loop_var);
    if (removed_for_loops.count(op->loop_var)) {
      return Downcast<For>(new_stmt)->body;
    }
    // ancestor_loops_.pop_back();
    return new_stmt;
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    // Step 0: Collect target buffers
    for (const auto& buffer : op->alloc_buffers) {
      if (LogicalLayoutRegistry::Global()->reg.count(buffer->scope)) {
        target_buffers_.insert(buffer);
      }
    }
    // Step 1: Recursively rewrite children and collect access region of target buffers.
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    BlockNode* n = block.CopyOnWrite();

    // Step 2: Infer physical buffer shape of the target buffers.
    for (size_t i = 0; i < block->alloc_buffers.size(); i++) {
      if (buffer_map_.count(block->alloc_buffers[i])) {
        n->alloc_buffers.Set(i, buffer_map_.at(block->alloc_buffers[i]));
      }
    }
    // Step 3: Rewrite buffer allocation.
    return block;
  }

  // void RewriteBufferRegion(Buffer* buffer, Region* region) const {
  //   for (size_t i = 0; i < region->size(); i++) {
  //     CHECK((*region)[i].IsSinglePoint());
  //   }
  //   auto it = buffer_info_.find(*buffer);
  //   if (it == buffer_info_.end()) {
  //     // Skip if the buffer is parameter
  //     return;
  //   }
  //   const BufferAllocInfo& info = it->second;
  //   ICHECK_EQ(region->size(), info.region.size());
  //   Region new_region;
  //   new_region.reserve(info.region.size());
  //   for (size_t i = 0; i < info.region.size(); ++i) {
  //     const Range& range = (*region)[i];
  //     new_region.push_back(Range::FromMinExtent(range->min - info.region[i]->min, range->extent));
  //   }
  //   *buffer = info.new_buffer;
  //   *region = std::move(new_region);
  // }

  void RewriteBufferRegions(Array<BufferRegion>* regions) const {
    Array<BufferRegion> new_regions;
    new_regions.reserve(regions->size());
    for (const auto& region : *regions) {
      BufferRegion buffer_region = region;
      BufferRegionNode* p = buffer_region.CopyOnWrite();
      // RewriteBufferRegion(&p->buffer, &p->region);
      new_regions.push_back(buffer_region);
    }
    *regions = std::move(new_regions);
  }

  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> removed_for_loops;

  arith::NDIntSet InferNewShape(const Array<PrimExpr> indices) {
    arith::NDIntSet nd_int_set = arith::NDIntSetFromPoint(indices);
    std::unordered_map<const VarNode*, arith::IntSet> dom_map;
    for (const auto &index : indices) {
      PostOrderVisit(index, [&](const ObjectRef& obj) {
        if (obj.as<VarNode>()) {
          const auto&range = loop_map_.at(Downcast<Var>(obj));
          dom_map.emplace(obj.as<VarNode>(), arith::IntSetFromMinExtent(range->min, range->extent));
        }
      });
    }
    arith::NDIntSet new_ranges = arith::EvalNDIntSet(nd_int_set, dom_map);
    return new_ranges;
  }

  Stmt VisitStmt_(const BufferStoreNode* _op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_op));
    const auto& reg = LogicalLayoutRegistry::Global()->reg;
    auto it = reg.find(_op->buffer->scope);
    if (it == reg.end()) {
      return store;
    }
    BufferStoreNode* op = store.CopyOnWrite();
    auto& indices = op->indices;

    const LogicalLayout& logical_layout = (*it).second;
    // CHECK(_op->value.as<BufferLoadNode>()) << "ValueError: data copying between buffers expected";
    if (!_op->value.as<BufferLoadNode>()) {
      size_t offset = op->indices.size() - logical_layout->num_dims;
      RewriteBufferIndices(op->buffer, &op->indices);
      auto new_ranges = InferNewShape(Array<PrimExpr>(op->indices.begin() + offset, op->indices.end()));
      Buffer new_buffer = MakeNewBuffer(op->buffer, logical_layout, new_ranges);
      buffer_map_[op->buffer] = new_buffer;
      op->buffer = new_buffer;
      return store;
    }

    CHECK_LE(logical_layout->num_dims, indices.size());
    Array<PrimExpr> args;
    arith::NDIntSet nd_intset;
    for (size_t i = 0; i < logical_layout->num_dims; i++) {
      args.push_back(indices[indices.size() - logical_layout->num_dims + i]);
      CHECK(args.back().as<VarNode>()) << args.back() << " ORIG " << store;
      removed_for_loops.insert(Downcast<Var>(args.back()));
    }
    Array<PrimExpr> new_args = logical_layout->lower_func(args);

    auto new_ranges = InferNewShape(new_args);

    Array<PrimExpr> iter_vars;
    std::vector<Stmt> loop_nests;
    auto nop = Evaluate(Integer(0));
    for (size_t i = 0; i < new_ranges.size(); i++) {
      Range dom(new_ranges[i].min(), new_ranges[i].max());
      // TODO check int set exactly matches the range
      LOG(INFO) << "Make ForLoop range " << dom;
      IterVar iter_var(dom, Var(), IterVarType::kDataPar, i == 0 ? "threadIdx.x" : "");
      CHECK(is_zero(new_ranges[i].min()));
      loop_nests.push_back(For(iter_var->var, new_ranges[i].min(), new_ranges[i].max() - new_ranges[i].min() + 1, i == 0 ? ForKind::kThreadBinding : ForKind::kSerial, nop, iter_var));
      iter_vars.push_back(iter_var);
    }
    arith::Analyzer analyzer;
    Array<arith::IterSumExpr> iter_map = arith::DetectIterMap(iter_vars, loop_map_, Bool(true), true, &analyzer);
    // LOG(INFO)<<iter_map.size() << " " << iter_map;
    Array<PrimExpr> loop_vars;

    for (const auto& iter_var : iter_vars) {
      loop_vars.push_back(iter_var);
    }

    auto inverse_var_map = arith::InverseAffineIterMap(iter_map, loop_vars);
    indices.resize(indices.size() - args.size());
    std::copy(loop_vars.begin(), loop_vars.end(), std::back_inserter(indices));

    // rewrite op->value
    BufferLoad value = Downcast<BufferLoad>(op->value);
    value = Downcast<BufferLoad>(Substitute(value, inverse_var_map));
    LOG(INFO) << "New BufferLoad " << value;
    op->value = value;


    Buffer new_buffer = MakeNewBuffer(op->buffer, logical_layout, new_ranges);
    LOG(INFO) << "New Buffer Shape " << new_buffer->shape;
    buffer_map_[op->buffer] = new_buffer;
    op->buffer = new_buffer;
    return MergeNest(loop_nests, store);
  }

  Buffer MakeNewBuffer(const Buffer& orig_buffer, const LogicalLayout& logical_layout, const arith::NDIntSet& nd_int_set) const {
    ObjectPtr<BufferNode> n = make_object<BufferNode>(*orig_buffer.get());
    Array<PrimExpr> new_shape(orig_buffer->shape.begin(), orig_buffer->shape.begin() + orig_buffer->shape.size() - logical_layout->num_dims);
    for (size_t i = 0; i < nd_int_set.size(); i++) {
      ICHECK(is_zero(nd_int_set[i].min()));
      new_shape.push_back(nd_int_set[i].max() - nd_int_set[i].min() + 1);
    }
    n->shape = std::move(new_shape);
    std::string scope = n->scope;
    n->scope = scope.substr(0, scope.find('.')); // remove logical layout suffix
    return Buffer(std::move(n));
  }

  void RewriteBufferIndices(const Buffer& orig_buffer, Array<PrimExpr> *indices) {
    const LogicalLayout& logical_layout = LogicalLayoutRegistry::Global()->reg.at(orig_buffer->scope);
    CHECK_LE(logical_layout->num_dims, indices->size());
    Array<PrimExpr> args;
    for (size_t i = 0; i < logical_layout->num_dims; i++) {
      args.push_back(indices->operator[](indices->size() - logical_layout->num_dims + i));
    }
    Array<PrimExpr> new_args = logical_layout->lower_func(args);
    indices->resize(indices->size() - args.size());
    for (size_t i = 0; i < new_args.size(); i++) {
      indices->push_back(new_args[i]);
    }
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_op));
    if (!buffer_map_.count(load->buffer)) {
      return load;
    }
    BufferLoadNode* op = load.CopyOnWrite();
    RewriteBufferIndices(op->buffer, &(op->indices));
    op->buffer = buffer_map_.at(load->buffer);
    return std::move(load);
  }

  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> target_buffers_;
  std::unordered_map<Buffer, Region, ObjectPtrHash, ObjectPtrEqual> access_region_;
  std::unordered_map<Var, Range, ObjectPtrHash, ObjectPtrEqual> loop_map_;
  Array<For> ancestor_loops_;
};

Stmt LowerLogicalLayout(Stmt stmt, const std::string& target) {
  return LogicalLayoutMutator().Rewrite(std::move(stmt));
}

namespace transform {

Pass LowerLogicalLayout() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    // auto regions = CollectBufferAccessRegion(f);
    // // auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    // // ICHECK(target.defined()) << "LowerLogicalLayout: Require the target attribute";
    // auto mtriple = target.value()->GetAttr<runtime::String>("mtriple", "");
    // LOG(INFO) << f;
    n->body =
        LowerLogicalLayout(std::move(n->body), "");
        LOG(INFO) << "LowerLogicalLayout Result:\n"<<n->body;
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerLogicalLayout", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerLogicalLayout").set_body_typed(LowerLogicalLayout);
TVM_REGISTER_GLOBAL("tir.LogicalLayoutRegister").set_body_typed(LogicalLayoutRegistry::Register);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
