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
#include "./access_analysis.h"

#include "./auto_scheduler_utils.h"
#include "./loop_tree.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(RangeNode);
TVM_REGISTER_NODE_TYPE(ScopeAccessNode);

Range::Range(PrimExpr min, Optional<PrimExpr> extent) {
  ObjectPtr<RangeNode> n = make_object<RangeNode>();
  n->min = std::move(min);
  n->extent = std::move(extent);
  data_ = std::move(n);
}

ScopeAccess::ScopeAccess(
    Map<tir::Var, Array<Domain>> read_doms, Map<tir::Var, Array<Domain>> write_doms,
    std::unordered_map<const tir::VarNode*, std::vector<const LoopTreeNode*>> producer_siblings,
    std::unordered_map<const tir::VarNode*, std::vector<const LoopTreeNode*>> consumer_siblings) {
  ObjectPtr<ScopeAccessNode> n = make_object<ScopeAccessNode>();
  n->read_doms = std::move(read_doms);
  n->write_doms = std::move(write_doms);
  n->producer_siblings = std::move(producer_siblings);
  n->consumer_siblings = std::move(consumer_siblings);
  data_ = std::move(n);
}

ScopeAccess ScopeAccess::FromLoopTreeLeaf(const LoopTree& leaf) {
  std::unordered_map<tir::Var, Array<Domain>, ObjectPtrHash, ObjectPtrEqual> read_doms;
  std::unordered_map<tir::Var, Array<Domain>, ObjectPtrHash, ObjectPtrEqual> write_doms;

  auto add_read = [&read_doms](const tir::Buffer& buffer, const Array<PrimExpr>& indices) {
    Domain domain;
    for (const PrimExpr& expr : indices) {
      domain.push_back(Range(expr, NullOpt));
    }
    read_doms[buffer->data].push_back(domain);
  };

  auto add_write = [&write_doms](const tir::Buffer& buffer, const Array<PrimExpr>& indices) {
    Domain domain;
    for (const PrimExpr& expr : indices) {
      domain.push_back(Range(expr, NullOpt));
    }
    write_doms[buffer->data].push_back(domain);
  };

  auto gather_buffer_load = [&add_read](const PrimExpr& expr) {
    tir::PostOrderVisit(expr, [&add_read](const ObjectRef& obj) {
      if (const auto* buffer_load = obj.as<tir::BufferLoadNode>()) {
        add_read(buffer_load->buffer, buffer_load->indices);
      }
    });
  };

  for (const ObjectRef& obj : leaf->children) {
    if (const auto* buffer_store = obj.as<tir::BufferStoreNode>()) {
      add_write(buffer_store->buffer, buffer_store->indices);
      gather_buffer_load(buffer_store->value);
    } else if (const auto* reduce_step = obj.as<tir::ReduceStepNode>()) {
      // lhs should be buffer write instead
      if (const auto* buffer_load = reduce_step->lhs.as<tir::BufferLoadNode>()) {
        add_write(buffer_load->buffer, buffer_load->indices);
      } else {
        LOG(FATAL) << "InternalError: lhs of reduce step should be BufferLoadNode, but get type: "
                   << reduce_step->lhs->GetTypeKey();
      }
      gather_buffer_load(reduce_step->rhs);
    } else {
      LOG(FATAL) << "TypeError: Cannot recognize the type of a statement in the leaf node of the "
                    "loop tree: "
                 << obj->GetTypeKey();
    }
  }
  return ScopeAccess(read_doms, write_doms, {}, {});
}

Map<LoopTree, ScopeAccess> AccessAnalysis(const LoopTree& root) {
  Map<LoopTree, ScopeAccess> result;
  std::function<void(const LoopTreeNode* root)> f_analysis =
      [&f_analysis, &result](const LoopTreeNode* root) -> void {
    const LoopTree& root_ref = GetRef<LoopTree>(root);
    int n_children = 0;
    for (const ObjectRef& obj : root->children) {
      if (const auto* loop = obj.as<LoopTreeNode>()) {
        f_analysis(loop);
        ++n_children;
      }
      // TODO(@junrushao1994): handle the case for both loop and stmt as children
    }
    if (n_children == 0) {
      result.Set(root_ref, ScopeAccess::FromLoopTreeLeaf(root_ref));
    } else {
      // TODO(@junrushao1994): access analysis for internal non-leaf nodes
      result.Set(root_ref, ScopeAccess({}, {}, {}, {}));
    }
  };
  f_analysis(root.get());
  return result;
}

TVM_REGISTER_GLOBAL("auto_scheduler.access_analysis.AccessAnalysis").set_body_typed(AccessAnalysis);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RangeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* node = ref.as<RangeNode>();
      CHECK(node);
      arith::Analyzer analyzer;
      if (node->extent.defined()) {
        PrimExpr left_inclusive = analyzer.Simplify(node->min);
        PrimExpr right_exclusive = analyzer.Simplify(node->min + node->extent.value());
        p->stream << '[' << left_inclusive << ", " << right_exclusive << ')';
      } else {
        PrimExpr point = analyzer.Simplify(node->min);
        p->stream << '[' << point << ']';
      }
    });

}  // namespace auto_scheduler
}  // namespace tvm
