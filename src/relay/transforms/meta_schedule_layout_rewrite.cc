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
 * \file meta_schedule_layout_rewrite.h
 * \brief Rewrite the layout of "layout free" tensors (e.g., the weight tensors in
 * conv2d and dense layers) according to the tile structure generated by the meta-scheduler.
 */

#include "meta_schedule_layout_rewrite.h"

#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <deque>
#include <functional>
#include <utility>
#include <vector>

#include "../backend/compile_engine.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

// Two global variables for receiving layout information from python

std::mutex MetaScheduleLayoutRewriter::mutex;
std::deque<meta_schedule::LayoutRewriteHint>
    MetaScheduleLayoutRewriter::global_layout_rewrite_queue;

// Copy an Attrs but with a new meta_schedule_rewritten_layout filed.
template <typename T>
Attrs CopyAttrsWithNewLayout(const T* ptr, const Array<PrimExpr>& original_shape) {
  auto n = make_object<T>(*ptr);
  n->meta_schedule_original_shape = original_shape;
  return Attrs(n);
}

// Mutate ops in a function
class MetaScheduleFuncMutator : public ExprMutator {
 public:
  MetaScheduleFuncMutator(std::deque<meta_schedule::LayoutRewriteHint> layout_rewrite_queue)
      : ExprMutator(), layout_rewrite_queue_(std::move(layout_rewrite_queue)) {}

  Expr VisitExpr_(const CallNode* n) {
    auto new_n = ExprMutator::VisitExpr_(n);

    const auto* call = new_n.as<CallNode>();

    if (call && call->op.as<OpNode>() &&
        (std::find(target_ops_.begin(), target_ops_.end(), n->op.as<OpNode>()->name) !=
         target_ops_.end()) &&
        !layout_rewrite_queue_.empty()) {
      // Pop a new layout from the queue
      LOG(INFO) << n->op.as<OpNode>()->name;
      const meta_schedule::LayoutRewriteHint hint = layout_rewrite_queue_.front();
      layout_rewrite_queue_.pop_front();

      Array<Integer> extents(hint.extents);

      Array<Integer> reorder(hint.reorder);
      auto var = call->args[1].as<VarNode>();
      auto type = var->type_annotation.as<TensorTypeNode>();
      LOG(INFO) << "extents:" << extents;
      LOG(INFO) << "reorder:" << reorder;
      //      auto original_shape=call->args[1].as<Var>()->get()->type_annotation;
      // Insert a new op to do layout transform. (This will be simplified by FoldConstant later).
      Expr updated_kernel = MakeMetaScheduleLayoutTransform(call->args[1], extents, reorder);
      Array<Expr> updated_args = {call->args[0], updated_kernel};

      // Update the attrs
      Attrs updated_attrs;
      if (auto pattr = call->attrs.as<Conv2DAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, type->shape);
      } else if (auto pattr = call->attrs.as<DenseAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, type->shape);
      } else if (auto pattr = call->attrs.as<BatchMatmulAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, type->shape);
      } else if (auto pattr = call->attrs.as<Conv2DWinogradAttrs>()) {
        updated_attrs = CopyAttrsWithNewLayout(pattr, type->shape);
      } else {
        LOG(FATAL) << "Unhandled attribute: " << call->attrs;
      }
      new_n = Call(call->op, updated_args, updated_attrs);
    }
    return new_n;
  }

 public:
  std::deque<meta_schedule::LayoutRewriteHint> layout_rewrite_queue_;

  std::vector<std::string> target_ops_{"nn.conv2d", "nn.dense", "nn.batch_matmul",
                                       "nn.contrib_conv2d_winograd_without_weight_transform"};
};

Expr MetaScheduleLayoutRewriter::VisitExpr_(const CallNode* n) {
  auto new_n = ExprMutator::VisitExpr_(n);

  if (const auto* call = new_n.as<CallNode>()) {
    if (const auto* func = call->op.as<FunctionNode>()) {
      global_layout_rewrite_queue.clear();

      // Use ScheduleGetter to call python lower functions.
      // This is used to get the layout transform information.
      // The layout transformation will be recorded to global_ori_layout_queue
      // and global_new_layouts_queue in ComputeDAG::RewriteLayout.
      CreateSchedule(GetRef<Function>(func), Target::Current());

      // Mutate the called function
      if (!global_layout_rewrite_queue.empty()) {
        auto ret = MetaScheduleFuncMutator(global_layout_rewrite_queue).VisitExpr(new_n);
        return ret;
      }
    }
  }

  return new_n;
}

Expr MetaScheduleLayoutRewrite(const Expr& expr) {
  return MetaScheduleLayoutRewriter().Mutate(expr);
}

namespace transform {

Pass MetaScheduleLayoutRewrite() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(relay::MetaScheduleLayoutRewrite(f));
      };
  return CreateFunctionPass(pass_func, 3, "MetaScheduleLayoutRewrite", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.MetaScheduleLayoutRewrite")
    .set_body_typed(MetaScheduleLayoutRewrite);

TVM_REGISTER_GLOBAL("relay.attrs.get_meta_schedule_original_layout")
    .set_body_typed([](const Attrs& attrs) {
      if (attrs->IsInstance<Conv2DAttrs>()) {
        return attrs.as<Conv2DAttrs>()->meta_schedule_original_shape;
      } else if (attrs->IsInstance<DenseAttrs>()) {
        return attrs.as<DenseAttrs>()->meta_schedule_original_shape;
      } else if (attrs->IsInstance<BatchMatmulAttrs>()) {
        return attrs.as<BatchMatmulAttrs>()->meta_schedule_original_shape;
      } else if (attrs->IsInstance<Conv2DWinogradAttrs>()) {
        return attrs.as<Conv2DWinogradAttrs>()->meta_schedule_original_shape;
      } else {
        LOG(FATAL) << "Unhandled attribute: " << attrs;
      }
      return Array<PrimExpr>();
    });
}  // namespace transform

}  // namespace relay
}  // namespace tvm
