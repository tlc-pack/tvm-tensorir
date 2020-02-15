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
 * \file printer/tir_meta_mutator.cc
 * \brief Mutator to mutate meta node to bed embedded into TIR
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>
#include "../tir/ir/functor_common.h"

namespace tvm {
namespace tir {

class TIRMetaMutator:
    public StmtExprMutator {
 public:
  using SymbolTable = const ObjectRef&;

  explicit TIRMetaMutator(SymbolTable symbol_table_)
    : symbol_table(symbol_table_) {}

  /*! \brief mutate the meta_node */
  ObjectRef Mutate(const ObjectRef& meta_node) {
    if (meta_node.as<StmtNode>()) {
      return VisitStmt(Downcast<Stmt>(meta_node));
    } else if (meta_node.as<PrimExprNode>()) {
      return VisitExpr(Downcast<PrimExpr>(meta_node));
    } else {
      static const FType& f = vtable();
      CHECK(f.can_dispatch(meta_node));
      return f(meta_node, this);
    }
  }

  /*! \brief Look up node by name_hint in symbol table */
  ObjectRef Lookup(std::string name) {
    auto *array = symbol_table.as<ArrayNode>();
    CHECK(array != nullptr);
    for (size_t i = array->data.size() - 1; i >= 0; i--) {
      auto* map = array->data[i].as<StrMapNode>();
      if (map && map->data.find(name) != map->data.end()) {
        return map->data.at(name);
      }
    }
    CHECK(false) << name << " not found in symbol table";
  }

  // Allow registration to mutator.
  using FType = NodeFunctor<ObjectRef(const ObjectRef&, TIRMetaMutator*)>;
  static FType& vtable();

  /*! \brief symbol table for lookup use */
  SymbolTable symbol_table;

 private:
  PrimExpr VisitExpr_(const VarNode* op) final;
  PrimExpr VisitExpr_(const BufferLoadNode* op) final;

  Stmt VisitStmt_(const BlockNode* op) final;
  Stmt VisitStmt_(const BufferStoreNode* op) final;
};

PrimExpr TIRMetaMutator::VisitExpr_(const VarNode* op) {
  ObjectRef symbol = Lookup(op->name_hint);
  return Downcast<Var>(symbol);
}

PrimExpr TIRMetaMutator::VisitExpr_(const BufferLoadNode* op) {
  auto* node = ExprMutator::VisitExpr_(op).as<BufferLoadNode>();
  auto node_ptr = runtime::GetObjectPtr<BufferLoadNode>(const_cast<BufferLoadNode*>(node));

  ObjectRef symbol = Lookup(op->buffer->name);
  node_ptr->buffer = Downcast<Buffer>(symbol);
  return BufferLoad(node_ptr);
}

Stmt TIRMetaMutator::VisitStmt_(const BlockNode* op) {
  auto* node = StmtMutator::VisitStmt_(op).as<BlockNode>();
  auto node_ptr = runtime::GetObjectPtr<BlockNode>(const_cast<BlockNode*>(node));

  auto fmutate = [this](const TensorRegion& e) {
    return Downcast<TensorRegion>(this->Mutate(e));
  };
  Array<TensorRegion> reads = MutateArray(op->reads, fmutate);
  Array<TensorRegion> writes = MutateArray(op->writes, fmutate);
  node_ptr->reads = std::move(reads);
  node_ptr->writes = std::move(writes);
  return Block(node_ptr);
}

Stmt TIRMetaMutator::VisitStmt_(const BufferStoreNode* op) {
  auto* node = StmtMutator::VisitStmt_(op).as<BufferStoreNode>();
  auto node_ptr = runtime::GetObjectPtr<BufferStoreNode>(const_cast<BufferStoreNode*>(node));

  ObjectRef symbol = Lookup(op->buffer->name);
  node_ptr->buffer = Downcast<Buffer>(symbol);
  return BufferStore(node_ptr);
}

TVM_STATIC_IR_FUNCTOR(TIRMetaMutator, vtable)
.set_dispatch<TensorRegionNode>([](const ObjectRef& node, TIRMetaMutator* p) -> ObjectRef{
  auto *op = node.as<TensorRegionNode>();
  auto node_ptr = runtime::GetObjectPtr<TensorRegionNode>(const_cast<TensorRegionNode*>(op));

  ObjectRef symbol = p->Lookup(op->buffer->name);
  node_ptr->buffer = Downcast<Buffer>(symbol);
  auto fmutate = [p](const Range& e){ return Downcast<Range>(p->Mutate(e)); };
  node_ptr->region = MutateArray(op->region, fmutate);
  return TensorRegion(node_ptr);
});

TVM_STATIC_IR_FUNCTOR(TIRMetaMutator, vtable)
.set_dispatch<RangeNode>([](const ObjectRef& node, TIRMetaMutator* p) -> ObjectRef{
  auto *op = node.as<RangeNode>();
  auto node_ptr = runtime::GetObjectPtr<RangeNode>(const_cast<RangeNode*>(op));

  node_ptr->min = Downcast<PrimExpr>(p->Mutate(op->min));
  node_ptr->extent = Downcast<PrimExpr>(p->Mutate(op->extent));
  return Range(node_ptr);
});


TIRMetaMutator::FType& TIRMetaMutator::vtable() {
  static FType inst;
  return inst;
}

TVM_REGISTER_GLOBAL("tir.hybrid.parser.Mutate_Meta")
.set_body_typed<void(TIRMetaMutator::SymbolTable, const ObjectRef&)>(
[](TIRMetaMutator::SymbolTable symbol_table, const ObjectRef& meta_node){
  TIRMetaMutator(symbol_table).Mutate(meta_node);
});

}  // namespace tir
}  // namespace tvm
