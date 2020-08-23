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
#include "./meta_schedule.h"  // NOLINT(build/include)

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(MetaScheduleNode);

MetaSchedule::MetaSchedule(MetaIR meta_ir, MetaIR cursor, Array<Instruction> instructions,
                           std::vector<MetaScheduleNode::RevokeFunc> revokes,
                           Array<tir::Var> declared_vars) {
  ObjectPtr<MetaScheduleNode> n = make_object<MetaScheduleNode>();
  n->meta_ir = std::move(meta_ir);
  n->cursor = std::move(cursor);
  n->instructions = std::move(instructions);
  n->revokes = std::move(revokes);
  n->declared_vars = std::move(declared_vars);
  data_ = std::move(n);
}

inline MetaIR FindLeftistChildren(const MetaIR& meta_ir) {
  const auto* cursor = meta_ir.as<LoopTreeNode>();
  CHECK(cursor != nullptr) << "TypeError: meta_ir expects to be a loop tree, but gets type: "
                           << meta_ir->GetTypeKey();
  for (;;) {
    const Array<MetaIR> children = cursor->children;
    CHECK(!children.empty()) << "ValueError: loop tree should have non-empty children";
    const auto* left_child = children[0].as<LoopTreeNode>();
    if (left_child == nullptr) {
      break;
    }
    cursor = left_child;
  }
  return GetRef<MetaIR>(cursor);
}

MetaSchedule::MetaSchedule(MetaIR meta_ir)
    : MetaSchedule(meta_ir, FindLeftistChildren(meta_ir), {}, {}, Array<tir::Var>()) {}

inline String FindValidName(const Array<tir::Var>& declared_vars) {
  for (int i = 0;; ++i) {
    String result = "v." + std::to_string(i);
    bool name_valid = true;
    for (const tir::Var& var : declared_vars) {
      if (var->name_hint == result) {
        name_valid = false;
        break;
      }
    }
    if (name_valid) {
      return result;
    }
  }
  LOG(FATAL) << "InternalError: Unreachable";
  throw;
}

inline bool IsVarDeclared(const Array<tir::Var>& declared_vars, const tir::Var& var) {
  for (const tir::Var declared_var : declared_vars) {
    if (declared_var.same_as(var)) {
      return true;
    }
  }
  return false;
}

inline const MetaIRNode* MoveWithOffset(const MetaIRNode* p, int offset) {
  if (offset == 0) {
    return p;
  }
  if (offset > 0) {
    for (int i = offset; p != nullptr && i > 0; --i) {
      p = p->right_sibling;
    }
  } else {
    for (int i = -offset; p != nullptr && i > 0; --i) {
      p = p->left_sibling;
    }
  }
  CHECK(p != nullptr) << "ValueError: Not enough siblings given the offset = " << offset;
  return p;
}

inline Array<MetaIR> GetBlockUnderLoop(const LoopTreeNode* p, int loop_id) {
  int n = p->iters.size();
  std::vector<MetaIR> children;
  for (const MetaIR& child : p->children) {
    if (const auto* loop_tree = child.as<LoopTreeNode>()) {
      children.push_back(LoopTree(make_object<LoopTreeNode>(*loop_tree)));
    } else if (const auto* leaf_stmt = child.as<LeafStmtNode>()) {
      children.push_back(LeafStmt(make_object<LeafStmtNode>(*leaf_stmt)));
    } else {
      LOG(FATAL) << "TypeError: Unknown type: " << child->GetTypeKey();
    }
    MetaIR& back = children.back();
    back->parent = back->left_sibling = back->right_sibling = nullptr;
  }
  if (n == loop_id) {
    return children;
  }
  return {LoopTree(/*iters=*/{p->iters.begin() + loop_id, p->iters.end()},
                   /*block_realize=*/p->block_realize, /*children=*/children)};
}

tir::Var MetaScheduleNode::DeclIntVarNode(Array<Integer> choices, String name_hint) {
  if (name_hint == "") {
    name_hint = FindValidName(this->declared_vars);
  }
  // Create the result variable
  tir::Var var(name_hint, runtime::DataType::Int(32));
  this->declared_vars.push_back(var);
  // IR is not changed under this instruction
  // Create the instructions
  this->instructions.push_back(auto_scheduler::DeclIntVar(var, choices));
  // Create revocation of the instrcutions
  this->revokes.push_back([this]() {
    this->instructions.pop_back();
    this->declared_vars.pop_back();
  });
  return var;
}

Optional<tir::Var> MetaScheduleNode::SplitInnerToOuter(int loop_id,
                                                       Array<Optional<tir::Var>> factors,
                                                       String name_hint) {
  Optional<tir::Var> inferred = NullOpt;
  bool none_exists = false;
  for (const Optional<tir::Var>& factor : factors) {
    if (factor.defined()) {
      CHECK(IsVarDeclared(declared_vars, factor.value()))
          << "ValueError: In SplitInnerToOuter, there is an undefined variable: " << factor.value();
      continue;
    }
    CHECK(!none_exists) << "ValueError: SplitInnerToOuter expects at most one None in factors";
    if (name_hint == "") {
      name_hint = FindValidName(declared_vars);
    }
    inferred = tir::Var(name_hint, runtime::DataType::Int(32));
    none_exists = true;
  }
  // Prepare for IR manipulation: replace LoopTreeNode::iters with split ones
  const auto* loop_tree = this->cursor.as<LoopTreeNode>();
  CHECK(loop_tree != nullptr) << "TypeError: SplitInnerToOuter expects LoopTree, but get type: "
                              << this->cursor->GetTypeKey();
  Array<Iterator> iters = loop_tree->iters;
  int n = iters.size();
  CHECK(0 <= loop_id && loop_id < n) << "IndexError: SplitInnerToOuter gets loop_id = " << loop_id
                                     << ", but the cursor points to a block with " << n << " loops";
  std::vector<Iterator> new_iters;
  for (int i = 0; i < n; ++i) {
    if (i != loop_id) {
      new_iters.push_back(iters[i]);
      continue;
    }
    const Iterator& iter = iters[i];
    int n_factors = factors.size();
    for (int j = 0; j < n_factors; ++j) {
      new_iters.emplace_back(/*name=*/iter->name + "." + std::to_string(j),
                             /*extent=*/factors[j].value_or(inferred.value()),
                             /*kind=*/iter->kind,
                             /*annotation=*/iter->annotation);
    }
  }
  // Do the mutation
  loop_tree->iters = new_iters;
  if (none_exists) {
    this->declared_vars.push_back(inferred.value());
  }
  // Create the instructions
  this->instructions.push_back(auto_scheduler::SplitInnerToOuter(loop_id, factors, inferred));
  // Create revocation of the instrcutions
  this->revokes.push_back([this, none_exists, loop_tree, iters]() {
    loop_tree->iters = iters;
    if (none_exists) {
      this->declared_vars.pop_back();
    }
    this->instructions.pop_back();
  });
  return inferred;
}

void MetaScheduleNode::Reorder(Array<Integer> after_ids) {
  // IR manipulation
  const auto* loop_tree = this->cursor.as<LoopTreeNode>();
  CHECK(loop_tree != nullptr) << "TypeError: Reorder expects LoopTree, but get type: "
                              << this->cursor->GetTypeKey();
  Array<Iterator> iters = loop_tree->iters;
  CHECK_EQ(iters.size(), after_ids.size())
      << "ValueError: Mismatched number of elements in Reorder";
  int n = after_ids.size();
  std::vector<int> loop_id_used(n, 0);
  std::vector<Iterator> new_iters;
  for (const Integer& _loop_id : after_ids) {
    int loop_id = _loop_id;
    CHECK_EQ(loop_id_used[loop_id], 0)
        << "ValueError: Reorder takes duplicated arguments in `after_ids`: " << loop_id;
    loop_id_used[loop_id] = 1;
    new_iters.push_back(iters[loop_id]);
  }
  // Do the mutation
  loop_tree->iters = new_iters;
  // Create the instructions
  this->instructions.push_back(auto_scheduler::Reorder(after_ids));
  // Create revocation of the instrcutions
  this->revokes.push_back([this, loop_tree, iters]() {
    loop_tree->iters = iters;
    this->instructions.pop_back();
  });
}

void MetaScheduleNode::ComputeAtOffset(int offset, int loop_id) {
  arith::Analyzer analyzer;
  CHECK_GE(offset, 0) << "ValueError: ComputeAt requires positive offset, but get offset = "
                      << offset;
  // Get `src` and `tgt`
  const auto* src = this->cursor.as<LoopTreeNode>();
  CHECK(src != nullptr) << "TypeError: ComputeAt expects LoopTree, but get type: "
                        << this->cursor->GetTypeKey();
  const MetaIRNode* _tgt = MoveWithOffset(src, offset);
  CHECK(_tgt->IsInstance<LoopTreeNode>())
      << "TypeError: ComputeAt expects sibling as LoopTree, but get type: " << _tgt->GetTypeKey();
  const auto* tgt = static_cast<const LoopTreeNode*>(_tgt);
  CHECK(src->parent != nullptr && tgt->parent != nullptr)
      << "ValueError: Cannot compute_at on the root of the LoopTree";
  CHECK(tgt->parent->IsInstance<LoopTreeNode>())
      << "TypeError: Expects parent node to be LoopTreeNode, but gets: "
      << tgt->parent->GetTypeKey();
  const auto* parent = static_cast<const LoopTreeNode*>(tgt->parent);
  // Check the first `loop_id` iters between `src` and `tgt` are consistent
  CHECK(0 <= loop_id && loop_id < static_cast<int>(src->iters.size()))
      << "ValueError: ComputeAt uses loop_id to index the sibling loop tree, but number of loops "
         "in the block is: "
      << src->iters.size() << ", but loop_id = " << loop_id;
  CHECK(0 <= loop_id && loop_id < static_cast<int>(tgt->iters.size()))
      << "ValueError: ComputeAt uses loop_id to index the sibling loop tree, but number of loops "
         "in the sibling is: "
      << tgt->iters.size() << ", but loop_id = " << loop_id;
  for (int i = 0; i <= loop_id; ++i) {
    CHECK(analyzer.CanProve(src->iters[i]->extent == tgt->iters[i]->extent))
        << "ValueError: The extents of " << i
        << "-th iterators are not equal between src and tgt, where src->iters[" << i
        << "]->extent = " << src->iters[i]->extent << ", and tgt->iters[" << i
        << "]->extent = " << tgt->iters[i]->extent;
  }
  // Create the new children of tgt
  std::vector<MetaIR> new_children_of_tgt;
  for (const MetaIR& child : GetBlockUnderLoop(src, loop_id + 1)) {
    new_children_of_tgt.push_back(child);
  }
  for (const MetaIR& child : GetBlockUnderLoop(tgt, loop_id + 1)) {
    new_children_of_tgt.push_back(child);
  }
  SetAsChildrenOf(new_children_of_tgt.begin(), new_children_of_tgt.end(), tgt);
  // Create the new children of parent
  std::vector<MetaIR> new_children_of_parent;
  for (const MetaIR& child : parent->children) {
    if (child.get() != src) {
      new_children_of_parent.push_back(child);
      CHECK(child->parent == parent);
    }
  }
  // Backup
  Array<MetaIR> parent_children = parent->children;
  Array<MetaIR> tgt_children = tgt->children;
  Array<Iterator> tgt_iters = tgt->iters;
  // Do the manipulation
  parent->children = new_children_of_parent;
  tgt->children = new_children_of_tgt;
  tgt->iters = {tgt_iters.begin(), tgt_iters.begin() + (loop_id + 1)};
  if (src->left_sibling != nullptr) {
    src->left_sibling->right_sibling = src->right_sibling;
  }
  if (src->right_sibling != nullptr) {
    src->right_sibling->left_sibling = src->left_sibling;
  }
  this->cursor = GetRef<MetaIR>(tgt);
  // Create the instructions
  this->instructions.push_back(auto_scheduler::ComputeAtOffset(offset, loop_id));
  // Create revocation of the instrcutions
  this->revokes.push_back([this, src, tgt, tgt_children, tgt_iters, parent, parent_children]() {
    parent->children = parent_children;
    tgt->children = tgt_children;
    tgt->iters = tgt_iters;
    if (src->left_sibling != nullptr) {
      src->left_sibling->right_sibling = src;
    }
    if (src->right_sibling != nullptr) {
      src->right_sibling->left_sibling = src;
    }
    this->cursor = GetRef<MetaIR>(src);
    this->instructions.pop_back();
  });
}

MetaIR MetaScheduleNode::CursorMoveOffset(int offset) {
  MetaIR cursor = this->cursor;
  const MetaIRNode* next_cursor = MoveWithOffset(cursor.get(), offset);
  // Do the mutation
  this->cursor = GetRef<MetaIR>(next_cursor);
  // Create the instructions
  this->instructions.push_back(auto_scheduler::CursorMoveOffset(offset));
  // Create revocation of the instrcutions
  this->revokes.push_back([this, cursor]() {
    this->cursor = cursor;
    this->instructions.pop_back();
  });
  return this->cursor;
}

TVM_REGISTER_GLOBAL("auto_scheduler.meta_schedule.FromMetaIR").set_body_typed([](MetaIR meta_ir) {
  return MetaSchedule(meta_ir);
});

TVM_REGISTER_GLOBAL("auto_scheduler.meta_schedule.DeclIntVarNode")
    .set_body_typed([](MetaSchedule schedule, Array<Integer> choices, String name_hint) {
      return schedule->DeclIntVarNode(choices, name_hint);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.meta_schedule.SplitInnerToOuter")
    .set_body_typed([](MetaSchedule schedule, int loop_id, Array<Optional<tir::Var>> factors,
                       String name_hint) {
      return schedule->SplitInnerToOuter(loop_id, factors, name_hint);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.meta_schedule.Reorder")
    .set_body_typed([](MetaSchedule schedule, Array<Integer> after_ids) -> void {
      schedule->Reorder(after_ids);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.meta_schedule.ComputeAtOffset")
    .set_body_typed([](MetaSchedule schedule, int offset, int loop_id) -> void {
      schedule->ComputeAtOffset(offset, loop_id);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.meta_schedule.CursorMoveOffset")
    .set_body_typed([](MetaSchedule schedule, int offset) {
      return schedule->CursorMoveOffset(offset);
    });

}  // namespace auto_scheduler
}  // namespace tvm
