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

#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

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
  Array<BufferAllocate> allocations;
  Array<Stmt> seq;

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
      Array<IterVar> block_vars = compute_op->axis;
      block_vars.insert(block_vars.end(), compute_op->reduce_axis.begin(),
                        compute_op->reduce_axis.end());

      CHECK_EQ(compute_op->body.size(), 1);
      const PrimExpr& expr = compute_op->body[0];

      // Declare buffer
      Buffer buffer = decl_buffer(tensor->shape, tensor->dtype, compute_op->name);
      op2buffers[op] = buffer;

      // Calculate indices for BufferStore
      Array<PrimExpr> indices;
      for (const auto& iter_var : compute_op->axis) indices.push_back(iter_var->var);

      Stmt body;
      if (const auto* reduce = expr.as<ReduceNode>()) {
        CHECK_EQ(reduce->source.size(), 1);
        body = ReduceStep(reduce->combiner, BufferLoad(buffer, indices),
                          translator(reduce->source[0]));
      } else {
        body = BufferStore(buffer, translator(expr), indices);
      }

      if (output_set.count(op)) {
        // Update Prim function's args
        Var arg("var_" + op->name, PrimType(DataType::Handle()));
        parameters.push_back(arg);
        buffer_map.Set(arg, buffer);
      } else {
        // Add allocation
        allocations.push_back(BufferAllocate(buffer, ""));
      }

      Block block(block_vars, NullValue<Array<TensorRegion>>(), NullValue<Array<TensorRegion>>(),
                  body, {}, {}, op->name);
      Array<PrimExpr> null_bindings;
      for (size_t i = 0; i < block_vars.size(); i++) null_bindings.push_back(NullValue<PrimExpr>());
      BlockRealize realize(null_bindings, Bool(true), block, "");
      seq.push_back(realize);
    } else {
      LOG(FATAL) << "Unsupported OperationNode";
    }
  }

  Stmt root = auto_complete(SeqStmt::Flatten(seq), allocations);

  auto func = make_object<PrimFuncNode>();
  func->params = parameters;
  func->buffer_map = buffer_map;
  func->body = root;
  func->attrs = DictAttrs(Map<String, ObjectRef>());
  func->ret_type = TupleType(Array<Type>());
  return PrimFunc(func);
}

TVM_REGISTER_GLOBAL("te.CreateFunc").set_body_typed([](const Array<te::Tensor>& tensors) {
  return create_tir(tensors);
});

}  // namespace tir
}  // namespace tvm