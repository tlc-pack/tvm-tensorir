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
#include "./analysis.h"
#include "./search.h"
#include "./utils.h"

namespace tvm {
namespace meta_schedule {

String AsHybrid(const Schedule& sch) {
  const auto* f = runtime::Registry::Get("hybrid.AsHybrid");
  String s = (*f)(sch->sch->func, false);
  return s;
}

class MultiLevelTiling {
 public:
  String tiling_structure;

  explicit MultiLevelTiling(String tiling_structure) : tiling_structure(tiling_structure) {}

  RulePackedArgs operator()(Schedule sch, BlockRV block_rv) {
    // Right now it only works with a leaf block with a single statement
    if (!IsTrivialBinding(sch, block_rv)) {
      return RulePackedArgs(sch);
    }
    Optional<Array<tir::Var>> block_vars = BlockVarsUsedInStore(sch, block_rv);
    if (!block_vars.defined()) {
      return RulePackedArgs(sch);
    }
    Array<Integer> iter_types = GetBlockVarTypes(sch, block_rv);
    // Check if multi-level-tiling is needed
    {
      Array<tir::BufferLoad> loads = GetBufferLoad(sch, block_rv);
      int n_missing = 0;
      for (const tir::BufferLoad& load : loads) {
        n_missing += CountMissingBlockVars(load, block_vars.value());
      }
      bool has_reduce = false;
      for (const Integer& iter_type : iter_types) {
        int iter_var_type = iter_type;
        if (iter_var_type == tir::IterVarType::kCommReduce) {
          has_reduce = true;
          break;
        }
      }
      if (n_missing < 1 || (n_missing < 2 && !has_reduce)) {
        return RulePackedArgs(sch);
      }
    }
    // Do the multi-level tiling
    std::vector<int> s_idx = FindCharPos(this->tiling_structure, 'S');
    std::vector<int> r_idx = FindCharPos(this->tiling_structure, 'R');
    std::vector<std::vector<LoopRV>> order(this->tiling_structure.size());
    Array<LoopRV> axes = sch->GetAxes(block_rv);
    {
      LOG(INFO) << "block = " << sch->Eval(block_rv)->GetStmt<tir::BlockNode>()->tag;
      Array<tir::Var> loop_vars;
      for (const LoopRV& loop_rv : axes) {
        loop_vars.push_back(sch->Eval(loop_rv)->GetStmt<tir::LoopNode>()->loop_var);
      }
      LOG(INFO) << "axes = " << loop_vars;
    }
    CHECK_EQ(axes.size(), iter_types.size());
    int n = axes.size();
    for (int i = 0; i < n; ++i) {
      std::vector<int>* idx = nullptr;
      if (iter_types[i] == tir::IterVarType::kDataPar) {
        idx = &s_idx;
      } else if (iter_types[i] == tir::IterVarType::kCommReduce) {
        idx = &r_idx;
      } else {
        continue;
      }
      int n_tiles = idx->size();
      Array<tir::Var> factors =
          sch->SampleTileFactor(/*n=*/n_tiles, /*loop=*/axes[i], /*where=*/{1, 2, 4});
      Array<LoopRV> splits =
          sch->Split(/*loop=*/axes[i], /*factors=*/{factors.begin(), factors.end()});
      for (int j = 0; j < n_tiles; ++j) {
        order[idx->at(j)].push_back(splits[j]);
      }
    }
    sch->Reorder(ConcatArray(order));
    LOG(INFO) << "sch =\n" << AsHybrid(sch);
    return RulePackedArgs(/*proceed=*/{}, /*ignored=*/{sch});
  }

  static SearchRule MakeRule(String tiling_structure) {
    auto invoke = [tiling_structure](Schedule sch, BlockRV block) -> RulePackedArgs {
      MultiLevelTiling rule(tiling_structure);
      return rule(sch, block);
    };
    return SearchRule("multi_level_tiling", invoke);
  }
};

TVM_REGISTER_GLOBAL("meta_schedule.rule.MultiLevelTiling")
    .set_body_typed(MultiLevelTiling::MakeRule);

}  // namespace meta_schedule
}  // namespace tvm
