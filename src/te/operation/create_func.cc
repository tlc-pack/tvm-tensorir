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

#include <tvm/runtime/registry.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/analysis.h>

#include "../schedule/graph.h"

namespace tvm {
namespace tir {

class Translator : public ExprMutator {
 public:
  explicit Translator(
      const std::unordered_map<te::Operation, Buffer, ObjectPtrHash, ObjectPtrEqual>& buffers)
      : buffers_(buffers) {}

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    const auto& tensor = Downcast<te::Tensor>(op->producer);
    auto it = buffers_.find(tensor->op);
    CHECK(it != buffers_.end()) << "Cannot find the tensor " << tensor;
    Buffer buffer = it->second;
    return BufferLoad(buffer, op->indices);
  }

 private:
  const std::unordered_map<te::Operation, Buffer, ObjectPtrHash, ObjectPtrEqual>& buffers_;
};

std::string GetUniqueName(const std::string& prefix,
                          std::unordered_map<std::string, int>* name_map) {
  std::string unique_prefix = prefix;
  auto it = name_map->find(prefix);
  if (it != name_map->end()) {
    while (name_map->count(unique_prefix = prefix + "_" + std::to_string(++it->second)) > 0);
  }
  (*name_map)[unique_prefix] = 0;
  return unique_prefix;
}

PrimFunc create_tir(const Array<te::Tensor>& tensors) {
  Array<te::Operation> ops;
  for (const auto& tensor : tensors) {
    ops.push_back(tensor->op);
  }
  auto g = te::CreateReadGraph(ops);
  Array<te::Operation> order = te::PostDFSOrder(ops, g);

  // output set.
  std::unordered_set<te::Operation> output_set;
  for (const te::Operation& x : ops) {
    output_set.insert(x);
  }

  // Buffer_map and params for PrimFunc
  Map<Var, Buffer> buffer_map;
  Array<Var> parameters;

  // translator for rewrite tensor to buffer
  std::unordered_map<te::Operation, Buffer, ObjectPtrHash, ObjectPtrEqual> op2buffers;
  Translator translator(op2buffers);

  // root allocation and body(seq_stmt) for root block
  Array<Buffer> allocations;
  Array<Stmt> seq;

  // name map for unique block name
  std::unordered_map<std::string, int> name_map;

  for (const auto& op : order) {
    CHECK_EQ(op->num_outputs(), 1);
    const te::Tensor& tensor = op.output(0);
    if (const auto& placeholder = op.as<te::PlaceholderOpNode>()) {
      Var arg("var_" + placeholder->name, PrimType(DataType::Handle()));
      Buffer input_buffer = decl_buffer(placeholder->shape, placeholder->dtype, placeholder->name);
      op2buffers[op] = input_buffer;
      parameters.push_back(arg);
      buffer_map.Set(arg, input_buffer);
    } else if (const auto& compute_op = op.as<te::ComputeOpNode>()) {
      Array<IterVar> block_vars;
      arith::Analyzer analyzer;

      auto push_block_vars = [&analyzer, &block_vars](const Array<IterVar>& iters) {
        for (const auto& iter_var : iters) {
          auto new_var = make_object<IterVarNode>(*iter_var.get());
          new_var->dom = Range::FromMinExtent(analyzer.Simplify(iter_var->dom->min),
                                              analyzer.Simplify(iter_var->dom->extent));
          block_vars.push_back(IterVar(new_var));
        }
      };

      push_block_vars(compute_op->axis);
      push_block_vars(compute_op->reduce_axis);

      CHECK_EQ(compute_op->body.size(), 1);
      const PrimExpr& expr = compute_op->body[0];

      // Declare buffer
      Buffer buffer = decl_buffer(tensor->shape, tensor->dtype, compute_op->name);
      op2buffers[op] = buffer;

      // Calculate indices for BufferStore
      Array<PrimExpr> indices;
      for (const auto& iter_var : compute_op->axis) indices.push_back(iter_var->var);

      Optional<Stmt> init = NullOpt;
      Stmt body;
      Array<PrimExpr> simplified_indices;

      for (const auto& index : indices) {
        simplified_indices.push_back(analyzer.Simplify(index));
      }
      if (const auto* reduce = expr.as<ReduceNode>()) {
        CHECK_EQ(reduce->source.size(), 1);
        PrimExpr lhs = BufferLoad(buffer, simplified_indices);
        PrimExpr rhs = analyzer.Simplify(translator(reduce->source[0]));
        CHECK(lhs->dtype == rhs->dtype);
        body = BufferStore(buffer, reduce->combiner.get()->operator()({lhs}, {rhs})[0],
                           simplified_indices);
        init = BufferStore(buffer, reduce->combiner->identity_element[0], simplified_indices);
      } else {
        body = BufferStore(buffer, analyzer.Simplify(translator(expr)), simplified_indices);
      }

      if (output_set.count(op)) {
        // Update Prim function's args
        Var arg("var_" + op->name, PrimType(DataType::Handle()));
        parameters.push_back(arg);
        buffer_map.Set(arg, buffer);
      } else {
        // Add allocation
        allocations.push_back(buffer);
      }
      Map<String, ObjectRef> annotations = op->attrs;
      annotations.Set("script_detect_access", IntImm(DataType::Int(32), 3));
      Block block(/*iter_vars=*/block_vars,
                  /*reads=*/{}, /*writes=*/{}, /*name_hint=*/GetUniqueName(op->name, &name_map),
                  /*body=*/body, /*init=*/init, /*alloc_buffers=*/{}, /*match_buffers=*/{},
                  /*annotations=*/annotations);

      Array<PrimExpr> nan_bindings;
      for (size_t i = 0; i < block_vars.size(); i++)
        nan_bindings.push_back(FloatImm(DataType::Float(32), std::nan("")));
      BlockRealize realize(nan_bindings, Bool(true), block);
      seq.push_back(realize);
    } else {
      LOG(FATAL) << "Unsupported OperationNode";
    }
  }

  PrimFunc func = PrimFunc(parameters, SeqStmt::Flatten(seq), VoidType(), buffer_map);

  const auto* complete = runtime::Registry::Get("script.Complete");
  ICHECK(complete);

  return (*complete)(func, allocations);
}

TVM_REGISTER_GLOBAL("te.CreateFunc").set_body_typed([](const Array<te::Tensor>& tensors) {
  return create_tir(tensors);
});

}  // namespace tir
}  // namespace tvm