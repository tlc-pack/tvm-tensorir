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
#include "./loop_tree.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ir/error.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(IteratorNode);
TVM_REGISTER_NODE_TYPE(LoopTreeNode);

Iterator::Iterator(String name, PrimExpr min, PrimExpr extent, IterKind kind,
                   IterAnnotation annotation) {
  ObjectPtr<IteratorNode> n = make_object<IteratorNode>();
  n->name = std::move(name);
  n->min = std::move(min);
  n->extent = std::move(extent);
  n->kind = std::move(kind);
  n->annotation = std::move(annotation);
  data_ = std::move(n);
}

LoopTree::LoopTree(Array<ObjectRef> children, Array<Iterator> iters,
                   const tir::BlockRealizeNode* block_realize) {
  ObjectPtr<LoopTreeNode> n = make_object<LoopTreeNode>();
  n->children = std::move(children);
  n->iters = std::move(iters);
  n->block_realize = std::move(block_realize);
  data_ = std::move(n);
}

/*!
 * \brief Convert the tir::IterVarType to auto_scheduler::IterKind
 * \param iter_var_type The input iter variable type in TIR
 * \return The corresponding loop tree created
 */
inline IterKind IterKindFromTir(tir::IterVarType iter_var_type) {
  if (iter_var_type == tir::kDataPar) {
    return IterKind::kSpatial;
  } else if (iter_var_type == tir::kCommReduce) {
    return IterKind::kReduction;
  }
  LOG(FATAL) << "TypeError: iter_var type is not supported: "
             << tir::IterVarType2String(iter_var_type);
  throw;
}

/*!
 * \brief Create the loop tree recursively from the BlockRealizeNode
 * \param realize The BlockRealizeNode used to create the loop tree
 * \return The corresponding loop tree created
 */
LoopTree LoopTreeFromBlockRealize(const tir::BlockRealizeNode* realize) {
  CHECK(realize->block->annotations.empty())
      << "InternalError: block with pre-defined annotations are not supported";
  // Step 1. Collect all children of the block
  std::vector<ObjectRef> children;
  if (const auto* seq = realize->block->body.as<tir::SeqStmtNode>()) {
    // The node has many children
    for (const auto& stmt : seq->seq) {
      children.emplace_back(stmt);
    }
  } else {
    // The node has only one child
    children.emplace_back(realize->block->body);
  }
  // Create children subtree first
  for (auto& child : children) {
    if (const auto* sub_block = child.as<tir::BlockRealizeNode>()) {
      // Case 1. There is no additional loops between two `BlockRealizeNode`
      child = LoopTreeFromBlockRealize(sub_block);
    } else if (const auto* loop = child.as<tir::LoopNode>()) {
      // Case 2. Nested loops
      // loop until `loop->body` is not a LoopNode
      while (const tir::StmtNode* body = loop->body.as<tir::LoopNode>()) {
        loop = static_cast<const tir::LoopNode*>(body);
      }
      const auto* sub_block = loop->body.as<tir::BlockRealizeNode>();
      CHECK(sub_block) << "InternalError: the IR is invalid because nested loop does not contain a "
                          "BlockRealizeNode as its direct body statement";
      child = LoopTreeFromBlockRealize(sub_block);
    } else {
      // Case 3. Leaf statement
      // Check: Any BlockNode should be contained inside `BlockRealizeNode`
      CHECK(!child->IsInstance<tir::BlockNode>())
          << "InternalError: the IR is invalid because a BlockNode is not contained in "
             "BlockRealizeNode";
    }
  }
  // TODO(@junrushao1994): do we need to check if the node's children are of the same type?
  // Step 2. Create iter vars using the info in block
  std::vector<Iterator> iters;
  for (const auto& iter : realize->block->iter_vars) {
    iters.emplace_back(
        /*name=*/iter->var->name_hint,
        /*min=*/iter->dom->min,
        /*extent=*/iter->dom->extent,
        /*kind=*/IterKindFromTir(iter->iter_type),
        /*annotation=*/IterAnnotation::kNone);
  }
  return LoopTree(children, iters, realize);
}

LoopTree LoopTree::FromPrimFunc(const tir::PrimFunc& func) {
  const auto* realize = func->body.as<tir::BlockRealizeNode>();
  CHECK(realize != nullptr)
      << "InternalError: the PrimFunc is invalid because its body is not BlockRealizeNode";
  return LoopTreeFromBlockRealize(realize);
}

class LoopTreeNode::Stringifier : public tir::StmtFunctor<void(const tir::Stmt&)> {
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
    for (const auto& iter : root->iters) {
      PrimExpr left_inclusive = analyzer.Simplify(iter->min);
      PrimExpr right_exclusive = analyzer.Simplify(iter->min + iter->extent);
      Cout()  // Print name of the iter_var
          << "for " << iter->name
          << ' '
          // Print kind
          << IterKind2String(iter->kind)
          // Print loop domain
          << '[' << left_inclusive << ", " << right_exclusive
          << ')'
          // Print loop annotation
          << (iter->annotation == IterAnnotation::kNone
                  ? ""
                  : " # " + IterAnnotation2String(iter->annotation))
          << std::endl;
      // add one level of indentation
      indent_delta += kIndentWidth;
      this->indent += kIndentWidth;
    }
    // Step 2. Print its children
    for (const auto& child : root->children) {
      // Case 1. the child is another node in the loop tree
      if (const auto* node = child.as<LoopTreeNode>()) {
        RecursivePrint(node);
        continue;
      }
      // Case 2: the child is a leaf, and contains a certain computation
      const auto* stmt = child.as<tir::StmtNode>();
      CHECK(stmt) << "InternalError: Expect type tir::Stmt, get: " << child->GetTypeKey();
      try {
        VisitStmt(GetRef<tir::Stmt>(stmt));
      } catch (const dmlc::Error& e) {
        // If the printing function is not defined, then printing is not supported and the developer
        // need to fix this
        LOG(FATAL) << "InternalError: printing is not well supported for type: "
                   << child->GetTypeKey() << "\n"
                   << e.what();
        throw;
      }
    }
    // Cancel the indentation
    this->indent -= indent_delta;
  }
  /*! \brief Print ReduceStep */
  void VisitStmt_(const tir::ReduceStepNode* reduce) override {
    if (const auto* buffer_load = reduce->lhs.as<tir::BufferLoadNode>()) {
      Cout() << buffer_load->buffer->name << " = ..." << std::endl;
    } else {
      LOG(FATAL) << "InternalError: unknown type in ReduceStep: " << reduce->lhs->GetTypeKey();
    }
  }
  /*! \brief Print BufferStore */
  void VisitStmt_(const tir::BufferStoreNode* buffer_store) override {
    Cout() << buffer_store->buffer->name << " = ..." << std::endl;
  }
  /*!
   * \brief Prefixed function to print indented line
   * \return The ostream that can be used to continue the current line
   */
  std::ostream& Cout() { return os << std::string(indent, ' '); }
  /*! \brief The current indentation */
  int indent = 0;
  /*! \brief The ostream used to store the stringfying result temporarily */
  std::ostringstream os;
  /*! \brief Analyzer to simplify loop domain */
  arith::Analyzer analyzer;
};

String LoopTreeNode::ToString() const { return LoopTreeNode::Stringifier().Run(this); }

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LoopTreeNode>([](const ObjectRef& ref, ReprPrinter* p) {
      const auto* node = ref.as<LoopTreeNode>();
      CHECK(node);
      p->stream << node->ToString();
    });

TVM_REGISTER_GLOBAL("auto_scheduler.loop_tree.FromPrimFunc").set_body_typed(LoopTree::FromPrimFunc);

}  // namespace auto_scheduler
}  // namespace tvm
