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
 * \file src/tir/ir/function.cc
 * \brief The function data structure.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace tir {

LinkedParam::LinkedParam(int64_t id, ::tvm::runtime::NDArray param) {
  auto n = make_object<LinkedParamNode>();
  n->id = id;
  n->param = param;
  data_ = std::move(n);
}

// Get the function type of a PrimFunc
PrimFunc::PrimFunc(Array<tir::Var> params, Stmt body, Type ret_type,
                   Map<tir::Var, Buffer> buffer_map, DictAttrs attrs, Span span) {
  // Assume void-return type for now
  // TODO(tvm-team) consider type deduction from body.
  if (!ret_type.defined()) {
    ret_type = VoidType();
  }
  auto n = make_object<PrimFuncNode>();
  n->params = std::move(params);
  n->body = std::move(body);
  n->ret_type = std::move(ret_type);
  n->buffer_map = std::move(buffer_map);
  n->attrs = std::move(attrs);
  n->checked_type_ = n->func_type_annotation();
  n->span = std::move(span);
  data_ = std::move(n);
}

FuncType PrimFuncNode::func_type_annotation() const {
  Array<Type> param_types;
  for (auto param : this->params) {
    param_types.push_back(GetType(param));
  }
  return FuncType(param_types, ret_type, {}, {});
}

TVM_REGISTER_NODE_TYPE(PrimFuncNode);

Array<PrimExpr> IndexMapNode::Apply(const Array<PrimExpr>& inputs) const {
  CHECK_EQ(inputs.size(), this->src_iters.size());
  int n = inputs.size();
  std::unordered_map<const VarNode*, PrimExpr> var_map;
  var_map.reserve(n);
  for (int i = 0; i < n; ++i) {
    var_map.emplace(this->src_iters[i].get(), inputs[i]);
  }
  Array<PrimExpr> results;
  results.reserve(this->tgt_iters.size());
  for (PrimExpr result : this->tgt_iters) {
    results.push_back(Substitute(std::move(result), var_map));
  }
  return results;
}

IndexMap::IndexMap(Array<Var> src_iters, Array<PrimExpr> tgt_iters) {
  ObjectPtr<IndexMapNode> n = make_object<IndexMapNode>();
  n->src_iters = std::move(src_iters);
  n->tgt_iters = std::move(tgt_iters);
  data_ = std::move(n);
}

IndexMap IndexMap::FromFunc(int ndim, runtime::TypedPackedFunc<Array<PrimExpr>(Array<Var>)> func) {
  Array<Var> src_iters;
  src_iters.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    src_iters.push_back(Var("i" + std::to_string(i), DataType::Int(32)));
  }
  return IndexMap(src_iters, func(src_iters));
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IndexMapNode>([](const ObjectRef& node, ReprPrinter* p) {
      const auto* n = node.as<IndexMapNode>();
      ICHECK(n);
      p->stream << "IndexMap: (";
      for (int i = 0, total = n->src_iters.size(); i < total; ++i) {
        if (i != 0) {
          p->stream << ", ";
        }
        p->stream << n->src_iters[i];
      }
      p->stream << ") => ";
      p->stream << "(";
      for (int i = 0, total = n->tgt_iters.size(); i < total; ++i) {
        if (i != 0) {
          p->stream << ", ";
        }
        p->stream << n->tgt_iters[i];
      }
      p->stream << ")";
    });

TVM_REGISTER_NODE_TYPE(IndexMapNode);
TVM_REGISTER_GLOBAL("tir.IndexMap")
    .set_body_typed([](Array<Var> src_iters, Array<PrimExpr> tgt_iters) {
      return IndexMap(src_iters, tgt_iters);
    });
TVM_REGISTER_GLOBAL("tir.IndexMapFromFunc").set_body_typed(IndexMap::FromFunc);
TVM_REGISTER_GLOBAL("tir.IndexMapApply").set_body_method<IndexMap>(&IndexMapNode::Apply);

TensorIntrin::TensorIntrin(PrimFunc desc_func, PrimFunc intrin_func) {
  // check the number of func var is equal
  CHECK_EQ(desc_func->params.size(), intrin_func->params.size());
  CHECK_EQ(desc_func->buffer_map.size(), intrin_func->buffer_map.size());

  // check both functions' bodies are directly block
  const auto* desc_realize =
      Downcast<BlockRealize>(desc_func->body)->block->body.as<BlockRealizeNode>();
  const auto* intrin_realize =
      Downcast<BlockRealize>(intrin_func->body)->block->body.as<BlockRealizeNode>();
  CHECK(desc_realize != nullptr) << "description function's body expect a directly block";
  CHECK(intrin_realize != nullptr) << "intrinsic function's body expect a directly block";

  const Block& desc_block = desc_realize->block;
  const Block& intrin_block = intrin_realize->block;

  // check block var number and iter type
  CHECK_EQ(desc_block->iter_vars.size(), intrin_block->iter_vars.size())
      << "Two blocks should have the same number of block vars";
  for (size_t i = 0; i < desc_block->iter_vars.size(); i++) {
    const IterVar& desc_var = desc_block->iter_vars[i];
    const IterVar& intrin_var = intrin_block->iter_vars[i];
    CHECK(desc_var->iter_type == intrin_var->iter_type)
        << "Block iter_type mismatch between " << desc_var->iter_type << " and "
        << intrin_var->iter_type;
  }

  auto n = make_object<TensorIntrinNode>();
  n->description = std::move(desc_func);
  n->implementation = std::move(intrin_func);
  data_ = std::move(n);
}

class TensorIntrinManager {
 public:
  Map<String, tir::TensorIntrin> reg;

  static TensorIntrinManager* Global() {
    static TensorIntrinManager* inst = new TensorIntrinManager();
    return inst;
  }
};

TensorIntrin TensorIntrin::Register(String name, PrimFunc desc_func, PrimFunc intrin_func) {
  TensorIntrinManager* manager = TensorIntrinManager::Global();
  ICHECK_EQ(manager->reg.count(name), 0)
      << "ValueError: TensorIntrin '" << name << "' has already been registered";
  TensorIntrin intrin(desc_func, intrin_func);
  manager->reg.Set(name, intrin);
  return intrin;
}

TensorIntrin TensorIntrin::Get(String name) {
  const TensorIntrinManager* manager = TensorIntrinManager::Global();
  ICHECK_EQ(manager->reg.count(name), 1)
      << "ValueError: TensorIntrin '" << name << "' is not registered";
  return manager->reg.at(name);
}

TVM_REGISTER_NODE_TYPE(TensorIntrinNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimFuncNode>([](const ObjectRef& ref, ReprPrinter* p) {
      // TODO(tvm-team) redirect to Text printer once we have a good text format.
      auto* node = static_cast<const PrimFuncNode*>(ref.get());
      p->stream << "PrimFunc(" << node->params << ") ";
      if (node->attrs.defined()) {
        p->stream << "attrs=" << node->attrs;
      }
      p->stream << " {\n";
      p->indent += 2;
      p->Print(node->body);
      p->indent -= 2;
      p->stream << "}\n";
    });

TVM_REGISTER_GLOBAL("tir.PrimFunc")
    .set_body_typed([](Array<tir::Var> params, Stmt body, Type ret_type,
                       Map<tir::Var, Buffer> buffer_map, DictAttrs attrs, Span span) {
      return PrimFunc(params, body, ret_type, buffer_map, attrs, span);
    });

TVM_REGISTER_GLOBAL("tir.TensorIntrin")
    .set_body_typed([](PrimFunc desc_func, PrimFunc intrin_func) {
      return TensorIntrin(desc_func, intrin_func);
    });

TVM_REGISTER_GLOBAL("tir.TensorIntrinRegister").set_body_typed(TensorIntrin::Register);
TVM_REGISTER_GLOBAL("tir.TensorIntrinGet").set_body_typed(TensorIntrin::Get);

}  // namespace tir
}  // namespace tvm
