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

#include "../schedule/graph.h"

namespace tvm {
namespace tir {

/*! \brief The helper mutator that transforms ProducerLoad to BufferLoad */
class ProducerToBufferTransformer : public ExprMutator {
 public:
  explicit ProducerToBufferTransformer(
      const std::unordered_map<te::Operation, Buffer, ObjectPtrHash, ObjectPtrEqual>& op2buffers)
      : op2buffers_(op2buffers) {}

  PrimExpr VisitExpr_(const ProducerLoadNode* op) final {
    const auto& tensor = Downcast<te::Tensor>(op->producer);
    auto it = op2buffers_.find(tensor->op);
    ICHECK(it != op2buffers_.end()) << "Cannot find the tensor " << tensor;
    const Buffer& buffer = it->second;
    return BufferLoad(buffer, op->indices);
  }

 private:
  /*! \brief The Map from Operations to buffers */
  const std::unordered_map<te::Operation, Buffer, ObjectPtrHash, ObjectPtrEqual>& op2buffers_;
};

String GetUniqueName(const String& prefix, std::unordered_map<String, int>* name_count) {
  String unique_prefix = prefix;
  auto it = name_count->find(prefix);
  if (it != name_count->end()) {
    while (name_count->count(unique_prefix = prefix + "_" + std::to_string(++it->second)) > 0)
      ;
  }
  (*name_count)[unique_prefix] = 0;
  return unique_prefix;
}

PrimFunc create_tir(const Array<te::Tensor>& tensors) {
  // Step 1. Create tensor read graph.
  Array<te::Operation> ops;
  for (const auto& tensor : tensors) {
    ops.push_back(tensor->op);
  }
  const te::ReadGraph& g = te::CreateReadGraph(ops);
  const Array<te::Operation>& order = te::PostDFSOrder(ops, g);

  // Step 2. Mark output OPs.
  std::unordered_set<te::Operation> output_ops;
  for (const te::Operation& x : ops) {
    output_ops.insert(x);
  }

  // Buffer_map and params for PrimFunc.
  Map<Var, Buffer> buffer_map;
  Array<Var> parameters;
  // Transformer to rewrite Operation to Buffer.
  std::unordered_map<te::Operation, Buffer, ObjectPtrHash, ObjectPtrEqual> op2buffers;
  ProducerToBufferTransformer transformer(op2buffers);
  // Root allocation and its body stmts.
  Array<Buffer> root_alloc;
  Array<Stmt> root_stmts;
  // Name count map to make block name unique.
  std::unordered_map<String, int> name_count;
  // Analyzer for simplification.
  arith::Analyzer analyzer;

  // Step 3. Rewrite compute stages into blocks.
  for (const te::Operation& op : order) {
    ICHECK_EQ(op->num_outputs(), 1);
    const te::Tensor& tensor = op.output(0);
    if (const auto* placeholder = op.as<te::PlaceholderOpNode>()) {
      // Case 1. Input stage (te.placeholder)
      Var arg("var_" + placeholder->name, PrimType(DataType::Handle()));
      // Declear buffer and set to func buffer_map
      const Buffer& input_buffer =
          decl_buffer(placeholder->shape, placeholder->dtype, placeholder->name);
      parameters.push_back(arg);
      buffer_map.Set(arg, input_buffer);
      // Update map from OPs to Buffers
      op2buffers[op] = input_buffer;
    } else if (const auto* compute_op = op.as<te::ComputeOpNode>()) {
      // Case 2. Compute stage (te.compute)
      // Step 3.1. Push_back data_par axis and reduce_axis into block_vars.
      Array<IterVar> block_vars;
      block_vars.reserve(compute_op->axis.size() + compute_op->reduce_axis.size());
      auto f_push_block_vars = [&analyzer, &block_vars](const Array<IterVar>& iters) {
        for (IterVar iter_var : iters) {
          iter_var.CopyOnWrite()->dom = Range::FromMinExtent(
              analyzer.Simplify(iter_var->dom->min), analyzer.Simplify(iter_var->dom->extent));
          block_vars.push_back(IterVar(iter_var));
        }
      };
      f_push_block_vars(compute_op->axis);
      f_push_block_vars(compute_op->reduce_axis);

      // Step 3.2. Push_back data_par axis and reduce_axis into block_vars.
      ICHECK_EQ(compute_op->body.size(), 1);
      const PrimExpr& expr = compute_op->body[0];

      // Step 3.3. Declare buffer and update op2buffers
      Buffer buffer = decl_buffer(tensor->shape, tensor->dtype, compute_op->name);
      op2buffers[op] = buffer;

      // Step 3.4. Calculate indices for BufferStore
      Array<PrimExpr> indices;
      indices.reserve(compute_op->axis.size());
      for (const IterVar& iter_var : compute_op->axis)
        indices.push_back(analyzer.Simplify(iter_var->var));

      // Step 3.5. Create block body.
      Optional<Stmt> init = NullOpt;
      Stmt body;
      if (const auto* reduce = expr.as<ReduceNode>()) {
        // Case 2.1. Reduce compute
        ICHECK_EQ(reduce->source.size(), 1);
        const PrimExpr& lhs = BufferLoad(buffer, indices);
        const PrimExpr& rhs = analyzer.Simplify(transformer(reduce->source[0]));
        ICHECK(lhs->dtype == rhs->dtype);
        body = BufferStore(buffer, reduce->combiner.get()->operator()({lhs}, {rhs})[0], indices);
        init = BufferStore(buffer, reduce->combiner->identity_element[0], indices);
      } else {
        // Case 2.2. Data parallel compute
        body = BufferStore(buffer, analyzer.Simplify(transformer(expr)), indices);
      }

      // Step 3.6. Update func buffer_map or root allocation.
      if (output_ops.count(op)) {
        // Case 1. It's a output stage then update Prim function's args.
        Var arg("var_" + op->name, PrimType(DataType::Handle()));
        parameters.push_back(arg);
        buffer_map.Set(arg, buffer);
      } else {
        // Case 2. It's an intermediate stage then alloc the buffer under root.
        root_alloc.push_back(buffer);
      }

      // Step 3.7. Add script_parsing_detect_access attr for auto complete the whole IR.
      Map<String, ObjectRef> annotations = op->attrs;
      annotations.Set(tir::attr::script_parsing_detect_access, IntImm(DataType::Int(32), 3));

      // Step 3.8. Create nan iter_values for BlockRealize, which can be determined during
      // completing.
      Array<PrimExpr> nan_bindings(block_vars.size(), FloatImm(DataType::Float(32), std::nan("")));

      // Step 3.9. Create Block and BlockRealize and push_back to root stmts.
      BlockRealize realize(/*iter_values=*/std::move(nan_bindings),
                           /*predicate=*/Bool(true),
                           /*block=*/
                           Block(/*iter_vars=*/std::move(block_vars),
                                 /*reads=*/{},
                                 /*writes=*/{},
                                 /*name_hint=*/GetUniqueName(op->name, &name_count),
                                 /*body=*/std::move(body),
                                 /*init=*/std::move(init),
                                 /*alloc_buffers=*/{},
                                 /*match_buffers=*/{},
                                 /*annotations=*/annotations));

      root_stmts.push_back(realize);
    } else {
      ICHECK(false) << "Unsupported OperationNode";
    }
  }

  // Step 4. Create func and complete it.
  PrimFunc func = PrimFunc(/*params=*/std::move(parameters),
                           /*body=*/SeqStmt::Flatten(root_stmts),
                           /*ret_type=*/VoidType(),
                           /*buffer_map=*/std::move(buffer_map));

  const auto* complete = runtime::Registry::Get("script.Complete");
  ICHECK(complete);

  return (*complete)(func, root_alloc);
}

TVM_REGISTER_GLOBAL("te.CreateTIR").set_body_typed([](const Array<te::Tensor>& tensors) {
  return create_tir(tensors);
});

}  // namespace tir
}  // namespace tvm
