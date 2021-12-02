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
#include "../utils.h"

namespace tvm {
namespace tir {
std::pair<Optional<tir::StmtSRef>,String> FindTensorized(const tir::Schedule& sch, const String&
                                                                                     number) {
  Optional<tir::StmtSRef> result = NullOpt;
  String name;
  IRModule mod = sch->mod();
  for (const auto& kv : mod->functions) {
    const GlobalVar& g_var = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* prim_func = base_func.as<tir::PrimFuncNode>()) {
      tir::PreOrderVisit(
          prim_func->body,
          [&result, &sch, &number, &g_var, &name](const ObjectRef& obj) -> bool {
            if (const auto* block = obj.as<tir::BlockNode>()) {
              tir::StmtSRef block_sref = sch->GetSRef(block);
              if (HasAnn(block_sref, tir::attr::auto_tensor_core, number)) {
                result = block_sref;
                name = g_var->name_hint;
                return false;
              }
            }
            return true;
          });
      return std::make_pair(result,name);
    }
  }
  return std::make_pair(result,name);
}

int CanTensorize(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                 const tir::TensorIntrin& intrin) {
  Optional<tir::TensorizeInfo> opt_tensorize_info =
      GetTensorizeLoopMapping(sch->state(), block_sref, intrin->description);
  if (!opt_tensorize_info.defined()) {
    return 0;
  }
  const auto* info = opt_tensorize_info.value().get();
  arith::Analyzer analyzer;
  for (const auto& kv : info->loop_map) {
    const tir::StmtSRef& block_loop_sref = kv.first;
    const auto* block_loop = block_loop_sref->StmtAs<tir::ForNode>();
    const tir::For& desc_loop = kv.second;
    if (!analyzer.CanProve(block_loop->extent == desc_loop->extent)) {
      return 0;
    }
  }
  return info->loop_map.size();
}

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

class RewriteTensorizeNode : public PostprocNode {
 public:


  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  /*! \brief The names of intrinsic relatede to tensor core */
  std::vector<String> intrin_names;

  static constexpr const char* _type_key = "meta_schedule.RewriteTensorize";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteTensorizeNode, PostprocNode);
};
bool RewriteTensorizeNode::Apply(const tvm::tir::Schedule& sch) {
  for (int intrin = 0; intrin < 5; ++intrin) {
    auto kv = FindTensorized(sch, std::to_string(intrin));
    Optional<tir::StmtSRef> opt_block_sref = kv.first;
    String g_var_name = kv.second;
    if (opt_block_sref) {
      tir::StmtSRef block_sref = opt_block_sref.value();
      tir::BlockRV block_rv = GetRVFromSRef(sch, block_sref, g_var_name);
      // Remove the annotation
      sch->Unannotate(block_rv, tir::attr::auto_tensor_core);
      // Get the surrounding loops
      auto loops = sch->GetLoops(sch->GetBlock(block_sref->StmtAs<tir::BlockNode>()->name_hint));
      // Tensorize
      if (int number_of_loops =
              CanTensorize(sch, block_sref, tir::TensorIntrin::Get(intrin_names.at(intrin)))) {
        sch->Tensorize(loops[loops.size() - number_of_loops], intrin_names.at(intrin));
      }
    }
  }
  return true;
}

Postproc Postproc::RewriteTensorize(const String& compute_intrin,
                                    const String& load_intrin_A,
                                    const String& load_intrin_B, const String& store_intrin,
                                    const String& init_intrin) {
  ObjectPtr<RewriteTensorizeNode> n = make_object<RewriteTensorizeNode>();
  n->intrin_names.push_back(compute_intrin);
  n->intrin_names.push_back(load_intrin_A);
  n->intrin_names.push_back(load_intrin_B);
  n->intrin_names.push_back(store_intrin);
  n->intrin_names.push_back(init_intrin);
  return Postproc(n);
}
TVM_REGISTER_NODE_TYPE(RewriteTensorizeNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteTensorize")
    .set_body_typed(Postproc::RewriteTensorize);

}  // namespace meta_schedule
}  // namespace tvm
