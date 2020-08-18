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

#include "./loop_tree.h"  // NOLINT(build/include)

#include "./auto_scheduler_utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(IteratorNode);
TVM_REGISTER_NODE_TYPE(MetaIRNode);
TVM_REGISTER_NODE_TYPE(LoopTreeNode);
TVM_REGISTER_NODE_TYPE(LeafStmtNode);

Iterator::Iterator(String name, PrimExpr extent, IterKind kind, IterAnnotation annotation) {
  ObjectPtr<IteratorNode> n = make_object<IteratorNode>();
  n->name = std::move(name);
  n->extent = std::move(extent);
  n->kind = std::move(kind);
  n->annotation = std::move(annotation);
  data_ = std::move(n);
}

LoopTree::LoopTree(Array<Iterator> iters, Optional<tir::BlockRealize> block_realize,
                   Array<MetaIR> children) {
  ObjectPtr<LoopTreeNode> n = make_object<LoopTreeNode>();
  n->parent = nullptr;
  n->left_sibling = nullptr;
  n->right_sibling = nullptr;
  n->iters = std::move(iters);
  n->block_realize = std::move(block_realize);
  n->children = std::move(children);
  const MetaIRNode* left_sibling = nullptr;
  for (const MetaIR& child : n->children) {
    CHECK(child->parent == nullptr);
    CHECK(child->left_sibling == nullptr);
    CHECK(child->right_sibling == nullptr);
    child->parent = n.get();
    if (left_sibling != nullptr) {
      child->left_sibling = left_sibling;
      left_sibling->right_sibling = child.get();
    }
    left_sibling = child.get();
  }
  data_ = std::move(n);
}

LeafStmt::LeafStmt(const tir::Stmt& stmt) {
  ObjectPtr<LeafStmtNode> n = make_object<LeafStmtNode>();
  n->parent = nullptr;
  n->left_sibling = nullptr;
  n->right_sibling = nullptr;
  if (const auto* reduce_step = stmt.as<tir::ReduceStepNode>()) {
    const auto* buffer_update = reduce_step->lhs.as<tir::BufferLoadNode>();
    CHECK(buffer_update != nullptr)
        << "InternalError: unknown type in ReduceStep: " << reduce_step->lhs->GetTypeKey();
    std::vector<tir::BufferLoad> reads;
    tir::PostOrderVisit(reduce_step->rhs, [&reads](const ObjectRef& obj) {
      if (const auto* load = obj.as<tir::BufferLoadNode>()) {
        reads.push_back(GetRef<tir::BufferLoad>(load));
      }
    });
    n->kind = LeafStmtKind::kReduceStep;
    n->write = GetRef<tir::BufferLoad>(buffer_update);
    n->reads = reads;
    n->stmt = stmt;
  } else if (const auto* buffer_store = stmt.as<tir::BufferStoreNode>()) {
    std::vector<tir::BufferLoad> reads;
    tir::PostOrderVisit(buffer_store->value, [&reads](const ObjectRef& obj) {
      if (const auto* load = obj.as<tir::BufferLoadNode>()) {
        reads.push_back(GetRef<tir::BufferLoad>(load));
      }
    });
    n->kind = LeafStmtKind::kBufferStore;
    n->write = tir::BufferLoad(buffer_store->buffer, buffer_store->indices);
    n->reads = reads;
    n->stmt = stmt;
  } else {
    LOG(FATAL) << "TypeError: A leaf statement is supposed to be ReduceStep, or BufferStore, but "
                  "get type: "
               << stmt->GetTypeKey();
    throw;
  }
  data_ = std::move(n);
}

/*!
 * \brief Figure out IterKind of a loop variable using the information provided in TIR
 * \param loop The TIR loop that contains the loop variable
 * \param sch The TIR schedule
 * \return IterKind indicating the kind of iteration variable
 */
IterKind IterKindFromTIRLoop(const tir::Loop& loop, const tir::Schedule& sch) {
  // TODO(@junrushao1994): introduce kOpaque?
  // Step 0. Check if the subtree under loop satisfies one-way fine-grained dataflow condition
  const tir::StmtSRef& loop_sref = sch->stmt2ref.at(loop.get());
  bool is_compact_dataflow = sch->GetParentScope(loop_sref).IsCompactDataFlow(loop_sref, sch.get());
  // Step 1. Collect direct child blocks under the loop
  std::vector<const tir::BlockRealizeNode*> child_block_realizes;
  tir::PreOrderVisit(loop, [&child_block_realizes](const ObjectRef& node) -> bool {
    if (const auto* realize = node.as<tir::BlockRealizeNode>()) {
      child_block_realizes.push_back(realize);
      return false;
    }
    return true;
  });
  // For non-compact dataflow, right now we only support a single child block
  if (!is_compact_dataflow && child_block_realizes.size() > 1) {
    return IterKind::kSpecial;
  }
  // Step 2. Check if bindings of all child blocks are validated
  for (const tir::BlockRealizeNode* realize : child_block_realizes) {
    if (!sch->stmt2ref.at(realize->block.get())->binding_valid) {
      return IterKind::kSpecial;
    }
  }
  // Step 3. Check the loop variable is bound to what kind of block variables
  bool bind_data_par = false;
  bool bind_reduction = false;
  bool bind_other = false;
  for (const tir::BlockRealizeNode* realize : child_block_realizes) {
    // Enumerate child blocks
    const tir::BlockNode* block = realize->block.get();
    CHECK_EQ(realize->binding_values.size(), block->iter_vars.size())
        << "InternalError: BlockRealize is inconsistent with its Block";
    int n = realize->binding_values.size();
    for (int i = 0; i < n; ++i) {
      const tir::IterVar& iter_var = block->iter_vars[i];
      const PrimExpr& binding = realize->binding_values[i];
      // If loop variable is bound in the current binding
      if (ExprContainsVar(binding, loop->loop_var)) {
        if (iter_var->iter_type == tir::kDataPar) {
          bind_data_par = true;
        } else if (iter_var->iter_type == tir::kCommReduce) {
          bind_reduction = true;
        } else {
          bind_other = true;
        }
      }
    }
  }
  // Step 4. Check if the loop variable can be data parallel or reduction
  if (!bind_reduction && !bind_other) {
    // TODO(@junrushao1994): if it is not bound to anything, do we really consider it as data
    // parallel?
    return IterKind::kDataPar;
  }
  if (bind_reduction && !bind_other) {
    return IterKind::kReduction;
  }
  return IterKind::kSpecial;
}

/*!
 * \brief Create the loop tree recursively from a TIR AST
 * \param root The root of the AST. Should be tir::BlockRealize or tir::Loop
 * \return The corresponding loop tree created
 */
LoopTree LoopTreeFromTIR(const tir::StmtNode* root, const tir::Schedule& sch) {
  CHECK(root->IsInstance<tir::LoopNode>() || root->IsInstance<tir::BlockRealizeNode>())
      << "InternalError: Cannot create LoopTree because the TIR root provided starts with neither "
         "Loop nor BlockRealize";
  // Step 1. Collect all loops
  std::vector<Iterator> iters;
  arith::Analyzer analyzer;
  while (root->IsInstance<tir::LoopNode>()) {
    const auto* loop = static_cast<const tir::LoopNode*>(root);
    CHECK(analyzer.CanProve(loop->min == 0))
        << "ValueError: Auto scheduler requires normalized loop range (starting from 0), but get: "
        << GetRef<tir::Loop>(loop);
    iters.emplace_back(
        /*name=*/loop->loop_var->name_hint,
        /*extent=*/analyzer.Simplify(loop->extent),
        /*kind=*/IterKindFromTIRLoop(GetRef<tir::Loop>(loop), sch),
        /*annotation=*/IterAnnotation::kNone);
    root = loop->body.get();
  }
  // Step 2. Check if there is a BlockRealize under the nested loop
  // It is possible that BlockRealize is not the direct child
  Optional<tir::BlockRealize> block_realize = NullOpt;
  if (root->IsInstance<tir::BlockRealizeNode>()) {
    const auto* block_realize_ptr = static_cast<const tir::BlockRealizeNode*>(root);
    block_realize = GetRef<tir::BlockRealize>(block_realize_ptr);
    CHECK(block_realize_ptr->block->annotations.empty())
        << "InternalError: block with pre-defined annotations are not supported";
    root = block_realize_ptr->block->body.get();
  }
  // Step 3. Collect all children of the block
  std::vector<MetaIR> children;
  for (const tir::Stmt& stmt : root->IsInstance<tir::SeqStmtNode>()
                                   ? static_cast<const tir::SeqStmtNode*>(root)->seq
                                   : Array<tir::Stmt>{GetRef<tir::Stmt>(root)}) {
    // BlockRealizeNode should be contained in a BlockRealizeNode
    CHECK(!stmt->IsInstance<tir::BlockNode>())
        << "InternalError: the IR is invalid because a BlockNode is not contained in "
           "BlockRealizeNode";
    // Check: no nested SeqStmt
    CHECK(!stmt->IsInstance<tir::SeqStmtNode>())
        << "InternalError: the IR is invalid because there are nested SeqStmt";
    if (stmt->IsInstance<tir::LoopNode>() || stmt->IsInstance<tir::BlockRealizeNode>()) {
      children.push_back(LoopTreeFromTIR(stmt.get(), sch));
    } else {
      children.push_back(LeafStmt(stmt));
    }
  }
  return LoopTree(iters, block_realize, children);
}

LoopTree LoopTree::FromPrimFunc(const tir::PrimFunc& func) {
  const auto* realize = func->body.as<tir::BlockRealizeNode>();
  CHECK(realize != nullptr)
      << "InternalError: the PrimFunc is invalid because its body is not BlockRealizeNode";
  return LoopTreeFromTIR(realize, tir::ScheduleNode::Create(func));
}

class LoopTreePrinter {
 public:
  /*!
   * \brief Entry function to stringify LoopTreeNode
   * \param root The object to be stringified
   * \return The human readable string representation
   */
  std::string Run(const LoopTreeNode* root) {
    Cout() << "LoopTree(" << root << "):\n";
    RecursivePrint(root);
    return os.str();
  }
  /*!
   * \brief Recursively print the LoopTreeNode
   * \param root The root of the tree to be stringified
   */
  void RecursivePrint(const LoopTreeNode* root) {
    constexpr int kIndentWidth = 2;
    int indent_delta = 0;
    // Step 1. Print loops with proper indentation
    for (const Iterator& iter : root->iters) {
      Cout() << "for " << iter << std::endl;
      // add one level of indentation
      indent_delta += kIndentWidth;
      this->indent += kIndentWidth;
    }
    // Step 2. Print its children
    for (const ObjectRef& child : root->children) {
      // Case 1. the child is another node in the loop tree
      if (const auto* node = child.as<LoopTreeNode>()) {
        RecursivePrint(node);
      } else {
        // Case 2: the child is a leaf statement
        CHECK(child->IsInstance<LeafStmtNode>())
            << "InternalError: Expect type LeafStmtNode, get: " << child->GetTypeKey();
        Cout() << child << std::endl;
      }
    }
    // Cancel the indentation
    this->indent -= indent_delta;
  }
  /*!
   * \brief Prefixed function to print indented line
   * \return The ostream that can be used to continue the current line
   */
  std::ostream& Cout() { return os << std::string(indent, ' '); }
  /*! \brief The current indentation */
  int indent = 0;
  /*! \brief The ostream used to store the printed result temporarily */
  std::ostringstream os;
};

String LoopTreeNode::ToString() const { return LoopTreePrinter().Run(this); }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LoopTreeNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* node = obj.as<LoopTreeNode>();
      CHECK(node);
      p->stream << node->ToString();
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IteratorNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* node = obj.as<IteratorNode>();
      CHECK(node);
      p->stream  // Print name
          << node->name
          << ' '
          // Print kind
          << IterKind2String(node->kind)
          // Print loop domain
          << "(" << node->extent
          << ")"
          // Print loop annotation
          << (node->annotation == IterAnnotation::kNone
                  ? ""
                  : " # " + IterAnnotation2String(node->annotation));
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LeafStmtNode>([](const ObjectRef& obj, ReprPrinter* p) {
      const auto* leaf = obj.as<LeafStmtNode>();
      CHECK(leaf);
      p->stream << LeafStmtKind2String(leaf->kind) << '(' << leaf->write->buffer->name << ") from ";
      std::vector<std::string> read_names;
      for (const tir::BufferLoad& buffer_load : leaf->reads) {
        read_names.push_back(buffer_load->buffer->name);
      }
      std::sort(read_names.begin(), read_names.end());
      p->stream << "(";
      bool is_first = true;
      for (const std::string& str : read_names) {
        if (is_first) {
          is_first = false;
        } else {
          p->stream << ", ";
        }
        p->stream << str;
      }
      p->stream << ")";
    });

TVM_REGISTER_GLOBAL("auto_scheduler.loop_tree.FromPrimFunc").set_body_typed(LoopTree::FromPrimFunc);

}  // namespace auto_scheduler
}  // namespace tvm
